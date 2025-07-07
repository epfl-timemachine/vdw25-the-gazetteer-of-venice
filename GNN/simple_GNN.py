import os
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.explain import Explainer, GNNExplainer
from sentence_transformers import SentenceTransformer
import networkx as nx
import numpy as np
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from transformers import BertTokenizer, BertModel

from NER.build_graph_for_GNN import build_graph_for_GNN, build_qa_pairs

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- User's Provided Graph Building Script ---
# Load pre-trained model and tokenizer for entity embedding
print("Loading BERT model for entity embedding...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
print("BERT model loaded.")


# Model-architecture-related section#


# --- GNN Model Definition ---
class GNNQAModel(nn.Module):
    def __init__(self,
                 num_entity_features: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_heads: int,
                 dropout_rate: float,
                 question_embedding_dim: int):
        super(GNNQAModel, self).__init__()

        # GATConv layers for message passing and aggregation
        self.conv1 = GATConv(num_entity_features, hidden_channels, heads=num_heads, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, dropout=dropout_rate)

        # Attention layer to focus on entities relevant to the question
        self.attn_q = nn.Linear(out_channels + question_embedding_dim, 1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                entity_features: torch.Tensor,
                edge_index: torch.Tensor,
                question_embedding: torch.Tensor,
                question_entity_mask: torch.Tensor,  # Mask for entities mentioned in the question
                place_entity_mask: torch.Tensor  # Mask for entities that are 'place' type (potential answers)
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # GNN layers
        x = self.conv1(entity_features, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        entity_representations = self.conv2(x, edge_index)
        entity_representations = F.elu(entity_representations)

        # Expand question embedding to match number of entities for concatenation
        question_embedding_expanded = question_embedding.unsqueeze(0).expand(
            entity_representations.size(0), -1
        )

        # Combine entity representations with question embedding for attention scoring
        combined_for_attn = torch.cat([entity_representations, question_embedding_expanded], dim=-1)

        # Compute attention scores
        attn_logits = self.attn_q(combined_for_attn).squeeze(-1)  # Shape: (num_entities,)

        # Mask attention: Only consider entities that are either in the question
        # or are 'place' entities (potential answers).
        relevant_entities_for_attn_mask = (question_entity_mask | place_entity_mask).float()
        # Apply a large negative value to logits of irrelevant entities
        attn_logits = attn_logits + (relevant_entities_for_attn_mask - 1) * 1e9

        attn_weights = F.softmax(attn_logits, dim=0)  # Shape: (num_entities,)

        # Compute a weighted sum of entity representations to get the question-focused graph representation
        question_focused_graph_representation = torch.sum(
            attn_weights.unsqueeze(-1) * entity_representations, dim=0
        )

        return question_focused_graph_representation, entity_representations


class AnswerClassifier(nn.Module):

    def __init__(self, graph_representation_dim: int, num_place_entities: int):
        super(AnswerClassifier, self).__init__()
        # Simple linear layer to project the graph representation to logits for each place entity
        self.classifier_head = nn.Linear(graph_representation_dim, num_place_entities)

    def forward(self, question_focused_graph_representation: torch.Tensor) -> torch.Tensor:
        # Add a batch dimension for the linear layer if it expects one, then remove it
        logits = self.classifier_head(question_focused_graph_representation.unsqueeze(0)).squeeze(0)
        return logits


# --- Main Execution Flow ---
if __name__ == "__main__":
    # --- 0. Configuration ---
    BERT_EMBEDDING_DIM = bert_model.config.hidden_size  # 768 for 'bert-base-uncased'
    question_embedding_dim = BERT_EMBEDDING_DIM  # Question is average of its entity embeddings
    hidden_channels = 128
    out_channels = 256
    num_heads = 4
    dropout_rate = 0.1
    num_epochs = 20
    learning_rate = 0.005  # Adjusted learning rate for more epochs

    # --- 1. Build NetworkX Graph and Embed Entities ---
    graph_pickle_path = "..\\NER\\graph.pkl"
    if os.path.exists(graph_pickle_path):
        print(f"Loading graph from '{graph_pickle_path}'...")
        with open(graph_pickle_path, 'rb') as f:
            G = pickle.load(f)
    else:
        G = build_graph_for_GNN("..\\NER\\NER_results.txt")
        print(f"Saving graph to '{graph_pickle_path}'...")
        with open(graph_pickle_path, 'wb') as f:
            pickle.dump(G, f)
        print("Graph saved.")

    # --- 2. Map NetworkX nodes to PyTorch Geometric IDs and prepare initial features ---
    # Create mappings for node IDs
    original_node_name_to_idx = {name: i for i, name in enumerate(G.nodes())}
    idx_to_original_node_name = {i: name for name, i in original_node_name_to_idx.items()}
    num_total_graph_entities = len(G.nodes())
    print(f"Total nodes in PyG graph (all entities): {num_total_graph_entities}")

    # Prepare initial entity features and masks based on NetworkX node attributes
    initial_entity_features_list = []
    entity_roles = []  # To store the role ('place' or 'other') for each PyG ID
    place_entity_indices = []  # Indices of 'place' entities for classification

    for original_name, node_attrs in G.nodes(data=True):
        idx = original_node_name_to_idx[original_name]
        entity_embedding = node_attrs['embedding'].to(device)  # Ensure embedding is on device
        role = node_attrs['role']
        entity_roles.append(role)

        is_place_entity = 1.0 if role == 'place' else 0.0
        if role == 'place':
            place_entity_indices.append(idx)

        # is_question_entity will be set per question later
        # For initial features, we only need the fixed embedding and role flag
        # The question_entity_mask is dynamic per question, not part of static features
        # Concatenate embedding with is_place_entity flag. is_question_entity flag added later dynamically.
        initial_feature_vector = torch.cat([entity_embedding,
                                            torch.tensor([is_place_entity], device=device)], dim=0)
        initial_entity_features_list.append(initial_feature_vector)

    # All initial entity features for the graph (embedding + is_place_entity flag)
    # Note: is_question_entity flag will be concatenated dynamically per question in the forward pass.
    # So `num_entity_features` in GNNQAModel will be BERT_EMBEDDING_DIM + 2 (for is_place_entity + is_question_entity)
    base_initial_entity_features = torch.stack(initial_entity_features_list).to(device)

    # Map place entity indices to a contiguous range for classification targets
    # For example, if place_entity_indices = [0, 5, 8], then original ID 0 maps to classifier_idx 0, etc.
    place_idx_to_classifier_idx = {original_idx: i for i, original_idx in enumerate(place_entity_indices)}
    classifier_idx_to_place_idx = {i: original_idx for original_idx, i in place_idx_to_classifier_idx.items()}
    num_place_entities = len(place_entity_indices)
    print(f"Number of 'place' entities (potential answers): {num_place_entities}")

    # Convert node names in edge_index to integer IDs
    pyg_edge_index = torch.tensor([[original_node_name_to_idx[u], original_node_name_to_idx[v]] for u, v in G.edges()],
                                  dtype=torch.long).t().contiguous().to(device)
    # Ensure undirected edges by adding reverse edges and removing duplicates
    pyg_edge_index = torch.cat([pyg_edge_index, pyg_edge_index.flip([0])], dim=1)
    pyg_edge_index = torch.unique(pyg_edge_index, dim=1)
    print(f"PyTorch Geometric edge index shape: {pyg_edge_index.shape}")

    # --- 3. Prepare Training Samples (Question entities -> Answer place entity) ---
    training_data_samples = []

    # Example questions and their ground truth answers ('place' entities)
    # Question is a list of 'other' entities, answer is a 'place' entity
    qa_pairs = build_qa_pairs("..\\NER\\NER_results.txt")

    for qa_pair in qa_pairs:
        q_entities_texts = qa_pair["question_entities"]
        answer_place_text = qa_pair["answer_place"]

        # Validate if all entities exist in the graph
        if not all(entity_text in original_node_name_to_idx for entity_text in q_entities_texts):
            print(f"Skipping QA pair: Question entities {q_entities_texts} not all in graph.")
            continue
        if answer_place_text not in original_node_name_to_idx or \
                G.nodes[answer_place_text]['role'] != 'place':
            print(f"Skipping QA pair: Answer place '{answer_place_text}' not found or not a 'place' entity.")
            continue

        # Get question embedding by averaging embeddings of question entities
        q_entity_embeddings = [G.nodes[e_text]['embedding'] for e_text in q_entities_texts]
        if not q_entity_embeddings:  # Handle empty question_entities
            q_embedding = torch.zeros(BERT_EMBEDDING_DIM, device=device)
        else:
            q_embedding = torch.mean(torch.stack(q_entity_embeddings), dim=0)

        # Create question_entity_mask for this specific question
        current_q_mask = torch.zeros(num_total_graph_entities, dtype=torch.bool, device=device)
        for q_ent_text in q_entities_texts:
            current_q_mask[original_node_name_to_idx[q_ent_text]] = True

        # Create place_entity_mask (identifies all potential answer 'place' entities)
        # This mask is constant across all questions but needs to be passed.
        current_place_mask = torch.zeros(num_total_graph_entities, dtype=torch.bool, device=device)
        current_place_mask[place_entity_indices] = True

        # Get true answer place entity's classifier index
        true_answer_original_idx = original_node_name_to_idx[answer_place_text]
        true_answer_classifier_idx = place_idx_to_classifier_idx[true_answer_original_idx]

        training_data_samples.append({
            "question_text": " ".join(q_entities_texts),  # For logging
            "question_embedding": q_embedding,
            "question_entity_mask": current_q_mask,
            "place_entity_mask": current_place_mask,  # All place entities as potential answers
            "true_answer_place_text": answer_place_text,
            "true_answer_classifier_idx": torch.tensor(true_answer_classifier_idx, dtype=torch.long, device=device),
        })
    print(f"Prepared {len(training_data_samples)} training samples.")

    # --- 4. Initialize Models, Loss, and Optimizer ---
    # `num_entity_features` for GNN: BERT embedding dim + is_place_entity flag + is_question_entity flag
    num_gnn_entity_features = BERT_EMBEDDING_DIM + 2
    gnn_model = GNNQAModel(
        num_entity_features=num_gnn_entity_features,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        question_embedding_dim=question_embedding_dim
    ).to(device)

    answer_classifier = AnswerClassifier(
        graph_representation_dim=out_channels,
        num_place_entities=num_place_entities
    ).to(device)

    optimizer = torch.optim.Adam(
        list(gnn_model.parameters()) + list(answer_classifier.parameters()),
        lr=learning_rate
    )
    criterion = nn.CrossEntropyLoss()

    # --- 5. Training Loop ---
    print("\nStarting training loop...")
    gnn_model.train()
    answer_classifier.train()

    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        for i, sample in enumerate(training_data_samples):
            optimizer.zero_grad()

            # Dynamically add the is_question_entity flag to the initial features for this sample
            is_question_entity_flag_tensor = sample["question_entity_mask"].float().unsqueeze(-1)
            # The 'base_initial_entity_features' already contains embedding + is_place_entity_flag
            current_entity_features = torch.cat([base_initial_entity_features, is_question_entity_flag_tensor], dim=-1)

            # Forward pass through GNN
            q_focused_graph_rep, _ = gnn_model(
                entity_features=current_entity_features,
                edge_index=pyg_edge_index,
                question_embedding=sample["question_embedding"],
                question_entity_mask=sample["question_entity_mask"],
                place_entity_mask=sample["place_entity_mask"]
            )

            # Forward pass through Answer Classifier
            logits = answer_classifier(q_focused_graph_rep)

            # Calculate loss
            # Add unsqueeze(0) to logits and target to simulate batch of 1 for CrossEntropyLoss
            loss = criterion(logits.unsqueeze(0), sample["true_answer_classifier_idx"].unsqueeze(0))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Calculate accuracy
            predicted_classifier_idx = torch.argmax(logits).item()
            if predicted_classifier_idx == sample["true_answer_classifier_idx"].item():
                correct_predictions += 1

        epoch_accuracy = correct_predictions / len(training_data_samples)
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(training_data_samples):.4f}, Accuracy: {epoch_accuracy:.4f}")
    print("Training complete.")

    # # --- 6. Inference Example ---
    # print("\n--- Inference Example ---")
    # gnn_model.eval()
    # answer_classifier.eval()
    #
    # with torch.no_grad():
    #     # Define a new question (list of 'other' entities)
    #     inference_question_entities = ["capital", "Germany"]
    #     # Embed the new question
    #     inference_q_entity_embeddings = [G.nodes[e_text]['embedding'] for e_text in inference_question_entities if
    #                                      e_text in original_node_name_to_idx]
    #     if not inference_q_entity_embeddings:
    #         inference_q_embedding = torch.zeros(BERT_EMBEDDING_DIM, device=device)
    #     else:
    #         inference_q_embedding = torch.mean(torch.stack(inference_q_entity_embeddings), dim=0)
    #
    #     # Create question_entity_mask for inference question
    #     inference_q_mask = torch.zeros(num_total_graph_entities, dtype=torch.bool, device=device)
    #     for q_ent_text in inference_question_entities:
    #         if q_ent_text in original_node_name_to_idx:
    #             inference_q_mask[original_node_name_to_idx[q_ent_text]] = True
    #
    #     # Ensure place_entity_mask is available
    #     inference_place_mask = torch.zeros(num_total_graph_entities, dtype=torch.bool, device=device)
    #     inference_place_mask[place_entity_indices] = True
    #
    #     # Dynamically add the is_question_entity flag for inference
    #     inference_entity_features = torch.cat([base_initial_entity_features, inference_q_mask.float().unsqueeze(-1)],
    #                                           dim=-1)
    #
    #     print(f"Question: Entities are '{'#'.join(inference_question_entities)}'")
    #
    #     # Forward pass through GNN
    #     q_focused_graph_rep_inf, _ = gnn_model(
    #         entity_features=inference_entity_features,
    #         edge_index=pyg_edge_index,
    #         question_embedding=inference_q_embedding,
    #         question_entity_mask=inference_q_mask,
    #         place_entity_mask=inference_place_mask
    #     )
    #
    #     # Forward pass through Answer Classifier
    #     logits_inf = answer_classifier(q_focused_graph_rep_inf)
    #
    #     # Get predicted answer index
    #     predicted_classifier_idx = torch.argmax(logits_inf).item()
    #     predicted_original_idx = classifier_idx_to_place_idx[predicted_classifier_idx]
    #     predicted_answer_text = idx_to_original_node_name[predicted_original_idx]
    #
    #     print(f"Predicted Answer Place: '{predicted_answer_text}'")
    #
    # # --- 7. GNNExplainer Example ---
    # print("\n--- GNNExplainer Example ---")
    #
    # # Pick a sample for explanation (e.g., the first training sample)
    # explain_sample = training_data_samples[0]
    # explain_q_text = explain_sample["question_text"]
    # explain_true_answer_text = explain_sample["true_answer_place_text"]
    #
    # print(f"Explaining the prediction for question: '{explain_q_text}' (True answer: '{explain_true_answer_text}')")
    #
    #
    # # The model for explainer needs to return the logits of the final classification layer.
    # # It also needs to have the question-specific inputs fixed.
    # def model_for_explainer_classification(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    #     """
    #     A wrapper for GNNExplainer. It takes node features and edge index,
    #     and uses fixed question-specific inputs to return the final classification logits.
    #     """
    #     # Fixed inputs for the specific explanation context
    #     fixed_question_embedding = explain_sample["question_embedding"]
    #     fixed_question_entity_mask = explain_sample["question_entity_mask"]
    #     fixed_place_entity_mask = explain_sample["place_entity_mask"]
    #
    #     # The 'x' passed to explainer is 'initial_entity_features' (embedding + is_place_entity_flag)
    #     # We need to concatenate the 'is_question_entity_flag' for the GNN's forward pass
    #     is_question_entity_flag_tensor = fixed_question_entity_mask.float().unsqueeze(-1)
    #     full_entity_features_for_gnn = torch.cat([x, is_question_entity_flag_tensor], dim=-1)
    #
    #     q_focused_rep, _ = gnn_model(
    #         entity_features=full_entity_features_for_gnn,
    #         edge_index=edge_index,
    #         question_embedding=fixed_question_embedding,
    #         question_entity_mask=fixed_question_entity_mask,
    #         place_entity_mask=fixed_place_entity_mask
    #     )
    #     return answer_classifier(q_focused_rep).unsqueeze(0)  # Add batch dimension for Explainer
    #
    #
    # explainer = Explainer(
    #     model=model_for_explainer_classification,
    #     algorithm=GNNExplainer(epochs=200, lr=0.01),
    #     explanation_type='model',
    #     node_mask_type='attributes',  # Explains attribute importance for nodes
    #     edge_mask_type='complementary',  # Explains edge importance
    #     model_config=dict(
    #         mode='classification',  # We are explaining a classification task
    #         return_type='raw',  # The model returns raw logits
    #         task_level='graph',  # We are explaining a graph-level prediction (which place entity to choose)
    #     )
    # )
    #
    # # Generate explanation for the prediction of 'explain_sample'
    # # We pass the base_initial_entity_features which is (embedding + is_place_entity_flag)
    # # The is_question_entity_flag is added dynamically inside model_for_explainer_classification
    # explanation = explainer(
    #     x=base_initial_entity_features,  # Pass the features excluding the dynamic question flag
    #     edge_index=pyg_edge_index,
    #     target=explain_sample["true_answer_classifier_idx"]  # Explain w.r.t. the true answer class
    # )
    #
    # # Print top 5 important nodes (entities) and edges
    # print("\nTop 5 Important Nodes for the Answer Prediction:")
    # # node_mask indicates importance of features per node. Summing gives overall node importance.
    # # Note: GNNExplainer's node_mask can be interpreted per feature or summed for overall node.
    # node_mask_summed = explanation.node_mask.squeeze().sum(
    #     dim=-1).cpu().numpy()  # Sum across feature dimension for overall node importance
    # top_node_indices = np.argsort(node_mask_summed)[::-1][:5]
    #
    # for idx in top_node_indices:
    #     node_name = idx_to_original_node_name[idx]
    #     node_role = G.nodes[node_name]['role']
    #     print(f"  Entity '{node_name}' (ID: {idx}, Role: {node_role}): Importance Score: {node_mask_summed[idx]:.4f}")
    #
    # print("\nTop 5 Important Edges for the Answer Prediction:")
    # if explanation.edge_mask is not None:
    #     edge_importance = explanation.edge_mask.squeeze().cpu().numpy()
    #     original_edges_tuple_list = [(u.item(), v.item()) for u, v in pyg_edge_index.t()]
    #
    #     # Create a set to store unique undirected edges (smaller_idx, larger_idx)
    #     unique_undirected_edges = set()
    #     edge_importance_map = {}  # Map (u,v) tuple to importance score
    #
    #     for i, (u_orig, v_orig) in enumerate(original_edges_tuple_list):
    #         # Sort the tuple to treat (u,v) and (v,u) as the same undirected edge
    #         sorted_edge = tuple(sorted((u_orig, v_orig)))
    #         if sorted_edge not in unique_undirected_edges:
    #             unique_undirected_edges.add(sorted_edge)
    #             edge_importance_map[sorted_edge] = edge_importance[i]
    #         else:
    #             # If already exists, take the maximum importance from both directions (or sum, depending on interpretation)
    #             edge_importance_map[sorted_edge] = max(edge_importance_map[sorted_edge], edge_importance[i])
    #
    #     # Sort the unique undirected edges by their importance scores
    #     sorted_unique_edges = sorted(edge_importance_map.items(), key=lambda item: item[1], reverse=True)[:5]
    #
    #     for (u, v), score in sorted_unique_edges:
    #         node_u_name = idx_to_original_node_name[u]
    #         node_v_name = idx_to_original_node_name[v]
    #         print(f"  Edge ('{node_u_name}' ({u}), '{node_v_name}' ({v})): Importance Score: {score:.4f}")
    # else:
    #     print("Edge mask not generated by GNNExplainer (check model_config or algorithm).")
