import networkx as nx
from collections import defaultdict
from itertools import combinations

with open("NER\entities_to_remove.txt", 'r', encoding='utf-8') as f:
    entities_to_remove = []
    for line in f.readlines():
        print(line)
        entities_to_remove.append(line.strip())

print(entities_to_remove)


def build_graph(entity_file):
    G = nx.Graph()
    with open(entity_file, 'r', encoding='utf-8') as ef:
        for line in ef.readlines():
            line = line.strip().lstrip('%').strip()
            split_line = line.split('%')
            place = split_line[0]
            entities = split_line[1].split('#')

            # add place and entities as nodes
            if place not in G:
                G.add_node(place, type="place")

            for entity in entities:
                if entity not in entities_to_remove:
                    if entity not in G:
                        G.add_node(entity, type="other")
                    if not G.has_edge(place, entity) and not G.has_edge(entity, place):
                        G.add_edge(place, entity)

            #  we have edges between: place and entities, one for each connection
    return G

def collapse_graph_to_places(original_graph):
    # Dictionary to keep track of which places are connected to each "other" entity
    entity_to_places = defaultdict(set)

    for node, data in original_graph.nodes(data=True):
        if data.get("type") == "place":
            for neighbor in original_graph.neighbors(node):
                if original_graph.nodes[neighbor].get("type") == "other":
                    entity_to_places[neighbor].add(node)

    # Now, for each "other" entity, look at all place-pairs that share it
    collapsed_graph = nx.Graph()

    for place in (n for n, d in original_graph.nodes(data=True) if d.get("type") == "place"):
        collapsed_graph.add_node(place, type="place")

    edge_weights = defaultdict(int)

    for places in entity_to_places.values():
        for place1, place2 in combinations(sorted(places), 2):
            edge_weights[(place1, place2)] += 1

    for (place1, place2), weight in edge_weights.items():
        collapsed_graph.add_edge(place1, place2, weight=weight)

    return collapsed_graph


graph = build_graph("NER/NER_results.txt")
nx.write_graphml(graph, "out/graph.graphml")


collapsed_graph = collapse_graph_to_places(graph)
nx.write_graphml(collapsed_graph, "out/collapsed_graph.graphml")
