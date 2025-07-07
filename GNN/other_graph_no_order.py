import os
import pickle
import random
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from transformers import BertModel, BertTokenizer

from NER.build_graph_for_GNN import build_qa_pairs, build_graph_for_GNN

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- User's Provided Graph Building Script ---
# Load pre-trained model and tokenizer for entity embedding
print("Loading BERT model for entity embedding...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
print("BERT model loaded.")

train_or_explain = sys.argv[1]


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

        self.conv1 = GATConv(num_entity_features, hidden_channels, heads=num_heads, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, dropout=dropout_rate)

        self.attn_q = nn.Linear(out_channels + question_embedding_dim, 1)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                entity_features: torch.Tensor,
                edge_index: torch.Tensor,
                question_embedding: torch.Tensor,
                question_entity_mask: torch.Tensor,
                place_entity_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(entity_features, edge_index)
        x = F.elu(x)
        x = self.dropout(x)

        entity_representations = self.conv2(x, edge_index)
        entity_representations = F.elu(entity_representations)

        question_embedding_expanded = question_embedding.unsqueeze(0).expand(
            entity_representations.size(0), -1
        )

        combined_for_attn = torch.cat([entity_representations, question_embedding_expanded], dim=-1)

        attn_logits = self.attn_q(combined_for_attn).squeeze(-1)

        relevant_entities_for_attn_mask = question_entity_mask.float()
        attn_logits = attn_logits + (relevant_entities_for_attn_mask - 1) * 1e9  # Mask irrelevant entities

        attn_weights = F.softmax(attn_logits, dim=0)

        question_focused_graph_representation = torch.sum(
            attn_weights.unsqueeze(-1) * entity_representations, dim=0
        )

        return question_focused_graph_representation, entity_representations, attn_weights


class AnswerClassifier(nn.Module):
    """
    A classification head that predicts the answer 'place' entity using
    a similarity-based approach, now adapted for binary classification
    with negative sampling.
    """

    def __init__(self, graph_representation_dim: int, hidden_dim_mlp: int = 512, dropout_rate: float = 0.2):
        super(AnswerClassifier, self).__init__()
        self.answer_query_projection = nn.Sequential(
            nn.Linear(graph_representation_dim, hidden_dim_mlp),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim_mlp, graph_representation_dim)
        )

    def forward(self,
                question_focused_graph_representation: torch.Tensor,
                candidate_place_representations: torch.Tensor) -> torch.Tensor:
        answer_query_embedding = self.answer_query_projection(question_focused_graph_representation)

        # Compute similarity scores (dot product)
        # Ensure answer_query_embedding is (1, dim) and candidate_place_representations is (batch_size, dim)
        # Result will be (1, batch_size), then squeeze to (batch_size,)
        logits = torch.matmul(answer_query_embedding.unsqueeze(0),
                              candidate_place_representations.transpose(0, 1)).squeeze(0)
        return logits


# --- Main Execution Flow ---
if __name__ == "__main__":
    # --- 0. Configuration ---
    BERT_EMBEDDING_DIM = bert_model.config.hidden_size
    question_embedding_dim = BERT_EMBEDDING_DIM
    hidden_channels = 128
    out_channels = 256
    num_heads = 4
    dropout_rate = 0.1
    num_epochs = 10
    learning_rate = 0.001  # Reduced learning rate for stability
    weight_decay = 1e-5  # L2 regularization
    negative_samples_per_positive = 5  # Number of negative place entities to sample per positive
    # Removed batch_size as per request

    answer_classifier_mlp_hidden_dim = 512

    # Paths for dummy files and graph pickle
    entity_file_path = "NER\\NER_results.txt"
    graph_pickle_path = "NER\\graph.pkl"

    # --- 1. Build or Load NetworkX Graph ---
    G = None
    if os.path.exists(graph_pickle_path):
        print(f"Loading graph from '{graph_pickle_path}'...")
        try:
            with open(graph_pickle_path, 'rb') as f:
                G = pickle.load(f)
            print(f"Graph loaded with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
            for node_name, node_attrs in G.nodes(data=True):
                if 'embedding' in node_attrs and isinstance(node_attrs['embedding'], torch.Tensor):
                    G.nodes[node_name]['embedding'] = node_attrs['embedding'].to(device)
        except Exception as e:
            print(f"Error loading graph from pickle: {e}. Rebuilding graph.")
            G = build_graph_for_GNN(entity_file_path)
            print(f"Saving graph to '{graph_pickle_path}'...")
            with open(graph_pickle_path, 'wb') as f:
                pickle.dump(G, f)
            print("Graph saved.")
    else:
        G = build_graph_for_GNN(entity_file_path)
        print(f"Saving graph to '{graph_pickle_path}'...")
        with open(graph_pickle_path, 'wb') as f:
            pickle.dump(G, f)
        print("Graph saved.")

    # --- 2. Map NetworkX nodes to PyTorch Geometric IDs and prepare initial features ---
    original_node_name_to_idx = {name: i for i, name in enumerate(G.nodes())}
    idx_to_original_node_name = {i: name for name, i in original_node_name_to_idx.items()}
    num_total_graph_entities = len(G.nodes())
    print(f"Total nodes in PyG graph (all entities): {num_total_graph_entities}")

    initial_entity_features_list = []
    entity_roles = []
    place_entity_indices = []

    for original_name, node_attrs in G.nodes(data=True):
        idx = original_node_name_to_idx[original_name]
        entity_embedding = node_attrs['embedding'].to(device)
        role = node_attrs['role']
        entity_roles.append(role)

        is_place_entity = 1.0 if role == 'place' else 0.0
        if role == 'place':
            place_entity_indices.append(idx)

        initial_feature_vector = torch.cat([entity_embedding,
                                            torch.tensor([is_place_entity], device=device)], dim=0)
        initial_entity_features_list.append(initial_feature_vector)

    base_initial_entity_features = torch.stack(initial_entity_features_list).to(device)

    # Convert place entity indices to a list of original names for easier sampling
    all_place_names = [idx_to_original_node_name[idx] for idx in place_entity_indices]
    num_place_entities = len(all_place_names)
    print(f"Number of 'place' entities (potential answers): {num_place_entities}")

    pyg_edge_index = torch.tensor([[original_node_name_to_idx[u], original_node_name_to_idx[v]] for u, v in G.edges()],
                                  dtype=torch.long).t().contiguous().to(device)
    pyg_edge_index = torch.cat([pyg_edge_index, pyg_edge_index.flip([0])], dim=1)
    pyg_edge_index = torch.unique(pyg_edge_index, dim=1)
    print(f"PyTorch Geometric edge index shape: {pyg_edge_index.shape}")

    # --- 3. Prepare Training Samples (Question entities -> Answer place entity) ---
    qa_pairs_raw = build_qa_pairs(entity_file_path)

    # Filter out QA pairs where entities are missing from the graph or answer is not a place
    filtered_qa_pairs = []
    for qa_pair in qa_pairs_raw:
        q_entities_texts = qa_pair["question_entities"]
        answer_place_text = qa_pair["answer_place"]

        if not all(entity_text in original_node_name_to_idx for entity_text in q_entities_texts):
            continue
        if answer_place_text not in original_node_name_to_idx or \
                G.nodes[answer_place_text]['role'] != 'place':
            continue
        filtered_qa_pairs.append(qa_pair)

    train_qa_pairs = filtered_qa_pairs
    print(f"Total QA pairs after filtering: {len(filtered_qa_pairs)}")
    print(f"Train QA pairs: {len(train_qa_pairs)}")


    # Pre-process QA pairs for faster training iteration
    def prepare_qa_samples(qa_list, all_place_names_list, original_node_name_to_idx_map, G_nodes_data, device_in):
        processed_samples = []
        for qa_pair in qa_list:
            q_entities_texts = qa_pair["question_entities"]
            answer_place_text = qa_pair["answer_place"]

            q_entity_embeddings = [G_nodes_data[e_text]['embedding'] for e_text in q_entities_texts]
            if not q_entity_embeddings:
                q_embedding = torch.zeros(BERT_EMBEDDING_DIM, device=device_in)
            else:
                q_embedding = torch.mean(torch.stack(q_entity_embeddings), dim=0)

            current_q_mask = torch.zeros(num_total_graph_entities, dtype=torch.bool, device=device_in)
            for q_ent_text in q_entities_texts:
                current_q_mask[original_node_name_to_idx_map[q_ent_text]] = True

            current_place_mask = torch.zeros(num_total_graph_entities, dtype=torch.bool, device=device_in)
            current_place_mask[[original_node_name_to_idx_map[name] for name in all_place_names_list]] = True

            true_answer_original_idx = original_node_name_to_idx_map[answer_place_text]

            processed_samples.append({
                "question_text": " ".join(q_entities_texts),
                "question_embedding": q_embedding,
                "question_entity_mask": current_q_mask,
                "place_entity_mask": current_place_mask,
                "true_answer_place_text": answer_place_text,
                "true_answer_original_idx": true_answer_original_idx,
            })
        return processed_samples


    training_data_samples = prepare_qa_samples(train_qa_pairs, all_place_names, original_node_name_to_idx, G.nodes,
                                               device)

    # --- 4. Initialize Models, Loss, and Optimizer ---
    num_gnn_entity_features = BERT_EMBEDDING_DIM + 2  # Retained original feature count
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
        hidden_dim_mlp=answer_classifier_mlp_hidden_dim,
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = torch.optim.Adam(
        list(gnn_model.parameters()) + list(answer_classifier.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    # --- 5. Training Loop ---
    if train_or_explain == '1':
        print("\nStarting training loop...")
        best_accuracy = -1

        for epoch in range(num_epochs):
            all_att_weights = None
            gnn_model.train()
            answer_classifier.train()
            total_loss = 0
            correct_predictions_train = 0

            # Shuffle training samples at the beginning of each epoch
            random.shuffle(training_data_samples)

            for i, sample in enumerate(training_data_samples):
                optimizer.zero_grad()

                is_question_entity_flag_tensor = sample["question_entity_mask"].float().unsqueeze(-1)
                current_entity_features = torch.cat([base_initial_entity_features,
                                                     is_question_entity_flag_tensor], dim=-1)

                q_focused_graph_rep, entity_reps_from_gnn, att_weights = gnn_model(
                    entity_features=current_entity_features,
                    edge_index=pyg_edge_index,
                    question_embedding=sample["question_embedding"],
                    question_entity_mask=sample["question_entity_mask"],
                    place_entity_mask=sample["place_entity_mask"]
                )
                if all_att_weights is not None:
                    all_att_weights = torch.cat((all_att_weights, att_weights.unsqueeze(0)), dim=0)
                else:
                    all_att_weights = att_weights.unsqueeze(0)

                # --- Negative Sampling for Training ---
                true_answer_original_idx = sample["true_answer_original_idx"]

                # Get representation of the true answer place
                positive_candidate_rep = entity_reps_from_gnn[true_answer_original_idx]  # (dim,)

                # Sample negative place entities
                negative_candidate_original_indices = []
                available_negative_indices = [
                    idx for idx in place_entity_indices if idx != true_answer_original_idx
                ]
                if len(available_negative_indices) > negative_samples_per_positive:
                    negative_candidate_original_indices = random.sample(
                        available_negative_indices, negative_samples_per_positive
                    )
                else:  # If not enough, take all available negatives
                    negative_candidate_original_indices = available_negative_indices

                negative_candidate_reps = entity_reps_from_gnn[negative_candidate_original_indices]  # (num_neg, dim)

                # Combine positive and negative candidates along with their labels
                # List of (representation, label) tuples
                candidates_with_labels = [(positive_candidate_rep, 1.0)]
                for neg_rep in negative_candidate_reps:
                    candidates_with_labels.append((neg_rep, 0.0))

                # Shuffle candidates and their labels together
                random.shuffle(candidates_with_labels)

                # Separate shuffled candidates and labels
                shuffled_candidate_reps = torch.stack([item[0] for item in candidates_with_labels])  # (1+num_neg, dim)
                shuffled_labels = torch.tensor([item[1] for item in candidates_with_labels], device=device)  # (1+num_neg,)

                # Get logits for all shuffled candidates
                logits = answer_classifier(q_focused_graph_rep, shuffled_candidate_reps)

                # Calculate loss
                loss = criterion(logits, shuffled_labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # For training accuracy, find the index of the true positive (label 1.0)
                # and check if it has the highest logit.
                # `true_positive_idx_in_shuffled` will be the index of the element with label 1.0
                # in the `shuffled_labels` tensor.
                true_positive_idx_in_shuffled = (shuffled_labels == 1.0).nonzero(as_tuple=True)[0].item()

                if len(logits) > 0 and torch.argmax(logits) == true_positive_idx_in_shuffled:
                    correct_predictions_train += 1

            avg_train_loss = total_loss / len(training_data_samples)
            train_accuracy = correct_predictions_train / len(training_data_samples)

            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}")

            if train_accuracy > best_accuracy:
                best_accuracy = train_accuracy
                torch.save(gnn_model.state_dict(), 'best_gnn_model.pth')
                torch.save(answer_classifier.state_dict(), 'best_answer_classifier.pth')
                best_indices = torch.topk(all_att_weights, 5, dim=-1).indices
                best_list = best_indices.tolist()
                best_indices = [{idx_to_original_node_name[training_data_samples[ind]["true_answer_original_idx"]]
                                 :[idx_to_original_node_name[i] for i in row]} for ind, row in enumerate(best_list)]

        print("Training complete.")
        with open("Chatbot\\mapped_names.pkl", "wb") as f:
            pickle.dump(best_indices, f)

    if train_or_explain == '2':

        # --- Explainability Section ---
        print("\n--- Starting Explainability Analysis ---")

        # Load the best models for evaluation
        try:
            gnn_model.load_state_dict(torch.load('GNN\\best_gnn_model.pth'))
            answer_classifier.load_state_dict(torch.load('GNN\\best_answer_classifier.pth'))
            print("Loaded best models for explainability.")
        except FileNotFoundError:
            print("Error: Best models not found. Run training first to save them.")
            exit()

        gnn_model.eval()
        answer_classifier.eval()

