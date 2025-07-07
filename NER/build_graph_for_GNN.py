import networkx as nx
from transformers import BertModel, BertTokenizer

# Load pre-trained model and tokenizer for entity embedding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def embed_entity(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach()


def build_graph_for_GNN(entity_file):
    G = nx.Graph()
    with open(entity_file, 'r') as ef:
        for line in ef.readlines():
            line = line.strip().lstrip('%').strip()
            split_line = line.split('%')
            place = split_line[0]
            entities = split_line[1].split('#')

            # add place and entities as nodes
            if place not in G:
                G.add_node(place, role='place', embedding=embed_entity(place))
            else:
                G.nodes[place]['role'] = 'place'
            for entity in entities:
                if entity not in G:
                    G.add_node(entity, role='other', embedding=embed_entity(entity))
                if not G.has_edge(place, entity) and not G.has_edge(entity, place):
                    G.add_edge(place, entity)

            #  we have edges between: place and entities, one for each connection
    return G


# build_graph_for_GNN("NER\\NER_results.txt")

def build_qa_pairs(entity_file):
    qa_pairs = []
    with open(entity_file, 'r') as ef:
        for line in ef.readlines():
            line = line.strip().lstrip('%').strip()
            split_line = line.split('%')
            place = split_line[0]
            entities = split_line[1].split('#')
            qa_pairs.append({"question_entities": entities, "answer_place": place})
    return qa_pairs


# build_qa_pairs("NER\\NER_results.txt")
