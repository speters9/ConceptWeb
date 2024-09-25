"""
This script processes lesson readings and objectives to extract key concepts and their relationships for export to graph.

It generates a concept map, visualizes it as a graph, and also creates a word cloud of the concepts.
The process involves:
    summarizing text,
    extracting relationships between concepts,
    normalizing and processing these relationships, and
    building and visualizing a graph.

Workflow:
1. **Summarize Text**: Summarizes the lesson readings based on provided objectives.
2. **Extract Relationships**: Extracts key concepts and the relationships between them.
3. **Normalize and Process**: Normalizes concepts and processes relationships to handle similar concepts.
4. **Build Graph**: Constructs an undirected graph from the processed relationships.
5. **Visualize Graph**: Visualizes the graph, adjusting node sizes based on centrality and edge thickness based on relationship frequency.
6. **Generate Word Cloud**: Creates a word cloud of the concepts.

Dependencies:
- Requires access to OpenAI API for generating summaries and extracting relationships.
- Uses NetworkX and Matplotlib for graph construction and visualization.
- WordCloud library is used for generating the word cloud.
"""

import logging
# base libraries
import os
from pathlib import Path
from typing import List, Tuple

# graph setup
import networkx as nx
import networkx.algorithms.community as nx_comm
from cdlib import algorithms
# env setup
from dotenv import load_dotenv
from sklearn.cluster import SpectralClustering

# self-defined utils
from src.concept_web.concept_extraction import process_relationships

load_dotenv()

# Path definitions
projectDir = Path(os.getenv('projectDir'))
dataDir = projectDir / "data/"

# %%

# %%

# ---------------------------------------------------------------

# Clean relationships, build network

# ---------------------------------------------------------------


# %%

def build_graph(relationships: List[Tuple[str, str, str]]) -> nx.Graph:
    """
    Build an undirected graph from the processed relationships.

    Args:
        relationships (List[Tuple[str, str, str]]): List of tuples representing relationships between concepts.

    Returns:
        networkx.Graph: The constructed graph.

    Raises:
        ValueError: If the relationships are not correctly formatted.
    """
    # Initialize an undirected graph
    G = nx.Graph()

    processed_relationships = process_relationships(relationships)

    # Add nodes and edges from relationships
    for concept1, relationship, concept2 in processed_relationships:
        if relationship != "None":
            if G.has_edge(concept1, concept2):
                G[concept1][concept2]['relation'].add(relationship)  # f"{concept1} -> {relationship} -> {concept2}")
                G[concept1][concept2]['weight'] += 1
            else:
                G.add_edge(concept1, concept2, weight=1, relation={relationship})  # [f"{concept1} -> {relationship} -> {concept2}"])

    # Normalize edge weights and centrality
    edge_weights = nx.get_edge_attributes(G, 'weight').values()

    # Calculate min and max weights
    max_weight = max(edge_weights) if edge_weights else 1  # Avoid division by zero
    min_weight = min(edge_weights) if edge_weights else 1  # Avoid division by zero

    # Normalize edge weights
    min_normalized_weight = 0.5
    max_normalized_weight = 4

    try:
        for u, v, d in G.edges(data=True):
            normalized_weight = min_normalized_weight + (max_normalized_weight - min_normalized_weight) * \
                (d['weight'] - min_weight) / (max_weight - min_weight)
            G[u][v]['normalized_weight'] = normalized_weight

        # Calculate degree centrality for each node
        centrality = nx.degree_centrality(G)

        # Normalize centrality to a range suitable for text size (e.g., 10 to 50)
        min_size = 6
        max_size = 24
        max_centrality = max(centrality.values())
        min_centrality = min(centrality.values())

        for node, centrality_value in centrality.items():
            normalized_size = min_size + (max_size - min_size) * (centrality_value - min_centrality) / (max_centrality - min_centrality)
            G.nodes[node]['text_size'] = normalized_size
            G.nodes[node]['centrality'] = centrality_value

    except ZeroDivisionError:
        # Log a warning that the graph could not be normalized
        logging.warning("Normalization of weights and centrality skipped due to lack of variation in the graph.\nReturning unnormalized edge weight and text size")
        # Fall back to default sizes if normalization fails
        for node in G.nodes():
            G.nodes[node]['text_size'] = 12  # Default text size
            G.nodes[node]['centrality'] = 0.5  # Default centrality

    return G


def detect_communities(G: nx.Graph, method: str = "leiden", num_clusters: int = None) -> nx.Graph:
    """
    Detects communities in the graph using the specified method.

    Args:
        G (networkx.Graph): The graph for which to detect communities.
        method (str): The method to use for community detection. Options are "leiden", "louvain", or "spectral".
        num_clusters (int, optional): The number of clusters for spectral clustering (only required for "spectral").

    Returns:
        networkx.Graph: The graph with community labels assigned to nodes.

    Raises:
        ValueError: If the specified method is not recognized.
    """
    G_copy = G.copy()

    if method == "leiden":
        # Use Louvain method for community detection
        communities_obj = algorithms.leiden(G)
        communities = communities_obj.communities  # extract communities from 'nodeclustering' object
    elif method == "louvain":
        # Use Louvain method for community detection
        communities = nx_comm.louvain_communities(G)
    elif method == "spectral":
        # Create a list of node names to maintain the order
        nodes = list(G.nodes())

        # Create the adjacency matrix for the graph
        adj_matrix = nx.to_numpy_array(G, nodelist=nodes)

        # Apply spectral clustering
        sc = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', assign_labels='kmeans')
        labels = sc.fit_predict(adj_matrix)

        # Group nodes by their cluster labels using node names instead of indices
        communities = [set() for _ in range(num_clusters)]
        for node, label in zip(nodes, labels):
            communities[label].add(node)
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'louvain' or 'spectral'.")

    # Assign each node to its community for visualization
    for i, community in enumerate(communities):
        for node in community:
            G_copy.nodes[node]['community'] = i  # Assign a group ID for use in visualization

    return G_copy


if __name__ == "__main__":
    import json

    with open(dataDir / 'interim/conceptlist_test.json', 'r') as f:
        conceptlist = json.load(f)

    with open(dataDir / 'interim/relationship_list_test.json', 'r') as f:
        relationship_list = json.load(f)
    # Build the graph
    G_base = build_graph(relationship_list)
    # Detect communities using Louvain method
    G = detect_communities(G_base, method="leiden")

    print(f"representation node: {G.nodes['representation']}")
