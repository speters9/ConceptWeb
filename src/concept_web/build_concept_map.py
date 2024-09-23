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

# base libraries
import json
import os
from pathlib import Path
from typing import List, Set, Tuple

# graph setup
import inflect
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
# self-defined utils
from BeamerBot.src_code.slide_pipeline_utils import (extract_lesson_objectives,
                                                     load_readings)
# env setup
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# llm chain setup
from langchain_openai import ChatOpenAI
from pyvis.network import Network
from sklearn.cluster import SpectralClustering
from wordcloud import WordCloud

load_dotenv()

OPENAI_KEY = os.getenv('openai_key')
OPENAI_ORG = os.getenv('openai_org')

# Path definitions
readingDir = Path(os.getenv('readingsDir'))
slideDir = Path(os.getenv('slideDir'))
syllabus_path = Path(os.getenv('syllabus_path'))

projectDir = Path(os.getenv('projectDir'))
dataDir = projectDir / "BeamerBot/data"

# %%

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_KEY,
    organization=OPENAI_ORG,
)


def summarize_text(text: str, prompt: str, objectives: str, parser=StrOutputParser()) -> str:
    """
    Summarize the provided text using the specified prompt and objectives.

    Args:
        text (str): The text to be summarized.
        prompt (str): The prompt template to guide the summarization.
        objectives (str): Lesson objectives to focus the summary.
        parser (StrOutputParser): The parser to handle the output.

    Returns:
        str: The summary generated by the language model.
    """
    summary_template = ChatPromptTemplate.from_template(prompt)
    chain = summary_template | llm | parser
    summary = chain.invoke({'text': text,
                            'objectives': objectives})

    return summary


def extract_relationships(text: str, objectives: str, prompt: str, parser=StrOutputParser()) -> List[Tuple[str, str, str]]:
    """
    Extract key concepts and their relationships from the provided text.

    Args:
        text (str): The summarized text.
        objectives (str): Lesson objectives to guide the relationship extraction.
        prompt (str): The prompt template to extract relationships.
        parser (StrOutputParser): The parser to handle the output.

    Returns:
        list: A list of tuples representing the relationships between concepts.
    """
    combined_template = ChatPromptTemplate.from_template(prompt)
    chain = combined_template | llm | parser

    response = chain.invoke({'objectives': objectives,
                            'text': text})

    try:
        # Clean and parse the JSON output
        response_cleaned = response.replace("```json", "").replace("```", "")
        data = json.loads(response_cleaned)

        # Extract concepts and relationships
        relationships = [tuple(relationship) for relationship in data["relationships"]]

    except json.JSONDecodeError as e:
        print(f"Error parsing the response: {e}")
        return []

    return relationships


def extract_concepts_from_relationships(relationships: List[Tuple[str, str, str]]) -> List[str]:
    """
    Extract unique concepts from the list of relationships.

    Args:
        relationships (list): List of tuples representing relationships between concepts.

    Returns:
        list: A list of unique concepts.
    """
    concepts = set()  # Use a set to avoid duplicates
    for concept1, _, concept2 in relationships:
        concepts.add(concept1)
        concepts.add(concept2)
    return list(concepts)


# %%
# summary_prompt = """You are a political science professor specializing in American government.
#                     You will be given a text and asked to summarize this text in light of your lesson objectives.
#                     Your lesson objectives are: \n {objectives} \n
#                     Summarize the following text: \n {text}"""


summary_prompt = """You are a political science professor specializing in American government.
                    You will be given a text and asked to summarize this text in light of your expertise.
                    Summarize the following text: \n {text}"""

relationship_prompt = """You are a political science professor specializing in American government.
                        You are instructing an introductory undergraduate American government class.
                        You will be mapping relationships between the concepts this class addresses.
                        The objectives for this lesson are: \n {objectives} \n

                        From the following text for this lesson, extract the key concepts and the relationships between them.
                        Identify the key concepts and then explain how each relates to the others.
                        \n
                        {text}
                        \n

                        Extract the most important and generally applicable key concepts and themes from the following summary.
                        Focus on high-level concepts or overarching themes relevant to an undergraduate American Politics course and the lesson objectives.
                        Examples of such concepts might include things like "Separation of Powers", "Federalism", "Standing Armies", or "Representation".

                        Avoid overly specific or narrow topics.

                        Provide the relationships between each concept with the other discovered concepts in the format:
                            "relationships": [
                              ["Concept 1", "relationship_type", "Concept 2"],
                              ["Concept 1", "relationship_type", "Concept 3"],
                              ...
                            ]

                        If there is no meaningful relationship from the standpoint of lesson objectives and your expertise as a professor of American Government, \
                        return "None" in the "relationship_type" field.

                        Because you are comparing a lot of concepts, the json may be long. That's fine.

                        Ensure results are returned in a valid json.
                        """

parser = StrOutputParser()
relationship_list = []
conceptlist = []

for lsn in range(1, 15):
    print(f"Extracting Lesson {lsn}")
    lsn_summaries = []
    readings = []
    objectives = ['']
    inputDir = readingDir / f'L{lsn}/'
    # load readings from the lesson folder
    if os.path.exists(inputDir):
        for file in inputDir.iterdir():
            if file.suffix in ['.pdf', '.txt']:
                readings_text = load_readings(file)
                readings.append(readings_text)

    if not readings:
        continue

    lsn_objectives = extract_lesson_objectives(syllabus_path, lsn, only_current=True)

    for reading in readings:
        summary = summarize_text(reading, prompt=summary_prompt, objectives=lsn_objectives, parser=parser)
        relationships = extract_relationships(summary, lsn_objectives, relationship_prompt, parser=parser)
        relationship_list.extend(relationships)

        concepts = extract_concepts_from_relationships(relationships)
        conceptlist.extend(concepts)


# %%

with open(dataDir / 'conceptlist_test.json', 'w') as f:
    json.dump(conceptlist, f)

with open(dataDir / 'relationship_list_test.json', 'w') as f:
    json.dump(relationship_list, f)

# %%

with open(dataDir / 'conceptlist_test.json', 'r') as f:
    conceptlist = json.load(f)

with open(dataDir / 'relationship_list_test.json', 'r') as f:
    relationship_list = json.load(f)


# %%

# ---------------------------------------------------------------

# Clean relationships, build network

# ---------------------------------------------------------------


# %%

def normalize_concept(concept: str) -> str:
    """
    Normalize a concept by converting it to lowercase, replacing spaces with underscores, and converting plural forms to singular.

    Args:
        concept (str): The concept to normalize.

    Returns:
        str: The normalized concept.
    """
    p = inflect.engine()

    # Normalize case, remove extra spaces, and split on spaces and underscores
    words = concept.lower().strip().replace('_', ' ').split()

    normalized_words = [
        p.singular_noun(word) if word != 'is' and p.singular_noun(word) else word
        for word in words
    ]
    return "_".join(normalized_words)


def jaccard_similarity(concept1: str, concept2: str, threshold: float = 0.85) -> bool:
    """
    Calculate the Jaccard similarity between two concepts.

    Args:
        concept1 (str): The first concept.
        concept2 (str): The second concept.
        threshold (float): The similarity threshold.

    Returns:
        bool: True if the similarity exceeds the threshold, False otherwise.
    """
    set1 = set(concept1)
    set2 = set(concept2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union
    return similarity >= threshold


def replace_similar_concepts(existing_concepts: Set[str], new_concept: str) -> str:
    """
    Replace a new concept with an existing similar concept if found.

    Args:
        existing_concepts (set): Set of existing concepts.
        new_concept (str): The new concept to check.

    Returns:
        str: The existing concept if a match is found, otherwise the new concept.
    """
    for existing_concept in existing_concepts:
        # If concepts are too similar, consolidate naming
        if jaccard_similarity(existing_concept, new_concept):
            return existing_concept
    return new_concept


def process_relationships(relationships: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """
    Process and normalize relationships by consolidating similar concepts.

    Args:
        relationships (list): List of tuples representing relationships between concepts.

    Returns:
        list: Processed relationships with normalized concepts.
    """
    # Initialize a set to keep track of all unique concepts
    unique_concepts = set()
    processed_relationships = []

    if not isinstance(relationships[0], tuple):
        relationships = [tuple(relation) for relation in relationships]

    for c1, relationship, c2 in relationships:
        # Normalize concepts
        clean_concept1 = normalize_concept(c1)
        clean_concept2 = normalize_concept(c2)
        clean_relation = normalize_concept(relationship)

        # Replace similar concepts with existing ones
        concept1 = replace_similar_concepts(unique_concepts, clean_concept1)
        concept2 = replace_similar_concepts(unique_concepts, clean_concept2)

        # Add concepts to the unique set
        unique_concepts.add(concept1)
        unique_concepts.add(concept2)

        # Add the relationship to the processed list
        processed_relationships.append((concept1, clean_relation, concept2))

    return processed_relationships


def build_graph(relationships: List[Tuple[str, str, str]]) -> nx.Graph:
    """
    Build an undirected graph from the processed relationships.

    Args:
        relationships (list): List of tuples representing relationships between concepts.

    Returns:
        networkx.Graph: The constructed graph.
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

    for u, v, d in G.edges(data=True):
        # Normalize each edge weight
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
        # Normalize the centrality to be within the desired range
        normalized_size = min_size + (max_size - min_size) * (centrality_value - min_centrality) / (max_centrality - min_centrality)
        G.nodes[node]['text_size'] = normalized_size  # Add a text_size attribute based on centrality
        G.nodes[node]['centrality'] = centrality_value

    return G


def detect_communities(G: nx.Graph, method: str = "louvain", num_clusters: int = None) -> List[Set[str]]:
    """
    Detects communities in the graph using the specified method.

    Args:
        G (networkx.Graph): The graph for which to detect communities.
        method (str): The method to use for community detection. Options are "louvain" or "spectral".
        num_clusters (int): The number of clusters for spectral clustering (only required for "spectral").

    Returns:
        List[Set[str]]: A list of sets, each containing the nodes in a detected community.
    """
    if method == "louvain":
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

    return communities


# Build the graph
G = build_graph(relationship_list)


# Detect communities using Louvain method
communities = detect_communities(G, method="louvain")

# Alternative: spectral clustering
# communities = detect_communities(G, method="spectral", num_clusters=10)

# Assign each node to its community for visualization
for i, community in enumerate(communities):
    for node in community:
        G.nodes[node]['community'] = i  # Assign a group ID for use in visualization

# %%
# nx.write_gexf(G, dataDir/"concept_graph_concept_mod.gexf")


# %%

def visualize_graph_interactive(G: nx.Graph, output_path: str) -> None:
    """
    Visualizes the graph interactively using pyvis and saves it as an HTML file.
    Includes options for physics simulations, node filtering, and clustering.

    Args:
        G (networkx.Graph): The graph to visualize.
        output_path (str): The file path where the HTML file will be saved.
    """
    net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white')

    # Generate a color map based on the number of communities
    communities = set(nx.get_node_attributes(G, 'community').values())
    color_map = plt.colormaps['tab20']  # 'tab20' is a colormap with 20 distinct colors
    community_colors = {community: mcolors.to_hex(color_map(i)) for i, community in enumerate(communities)}

    # Assign colors to nodes based on their community group
    for node in G.nodes():
        community = G.nodes[node]['community']
        G.nodes[node]['color'] = community_colors[community]  # Set the color attribute

    # Convert the NetworkX graph to a pyvis graph and add text size
    net.from_nx(G)
    for node in net.nodes:
        node["size"] = node['text_size']
        node["font"].update({"size": node['text_size']})

    for edge in net.edges:
        edge['relation'] = list(edge['relation'])
        edge['title'] = ", ".join(edge['relation'])
        edge['width'] = edge['normalized_weight']

    # Add physics controls for a dynamic layout
    net.show_buttons(filter_=['layout'])  # ['physics'])

    # Save the network as an HTML file
    net.save_graph(output_path)

    # Optionally, you can also open it directly in a browser
    # net.show(output_path)


# Create and save the interactive graph as an HTML file
output_path = str(dataDir / "interactive_concept_map_test.html")
visualize_graph_interactive(G, output_path)


# %%
# Create a string with each concept repeated according to its frequency
concept_string = " ".join(conceptlist)

# Generate the word cloud
wordcloud = WordCloud(width=1500, height=1000, background_color='white', max_font_size=150, max_words=250).generate(concept_string)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
