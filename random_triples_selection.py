import random
import torch
from rdflib import Graph

# Set seed for reproducibility
SEED = 42
random.seed(SEED)

# Define file path
graph_file_path = "Eurostat KG.ttl"  # Change this to your actual file

# Load RDF Graph
def load_graph(file_path):
    g = Graph()
    g.parse(file_path, format="turtle")
    return g

# Function to preprocess data and extract triples
def preprocess_data(graph):
    entities = list(set(s for s, _, _ in graph) | set(o for _, _, o in graph))
    relations = list(set(p for _, p, _ in graph))

    entity2idx = {entity: idx for idx, entity in enumerate(entities)}
    relation2idx = {relation: idx for idx, relation in enumerate(relations)}

    triples = [(entity2idx[s], relation2idx[p], entity2idx[o]) for s, p, o in graph]

    return triples, entity2idx, relation2idx

# Load graph and extract data
graph = load_graph(graph_file_path)
triples, entity2idx, relation2idx = preprocess_data(graph)

# Sample 200 random triples
num_samples = 200
sampled_triples = random.sample(triples, min(num_samples, len(triples)))

# Convert to PyTorch tensor
test_triples_tensor = torch.tensor(sampled_triples, dtype=torch.long)

# Save to file for later use
torch.save(test_triples_tensor, "models/data/test_triples.pt")

print(f"Successfully selected {len(sampled_triples)} random triples for testing.")
print(f"Saved to: models/data/test_triples.pt")
