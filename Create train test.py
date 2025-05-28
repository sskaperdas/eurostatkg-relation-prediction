import os
import random
import numpy as np
import torch
from rdflib import Graph
from sklearn.model_selection import train_test_split

# Ensure deterministic behavior & set device
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Create directories for saving results
os.makedirs("models/data", exist_ok=True)  # Store train, hybrid loss (worked), val splits
os.makedirs("models", exist_ok=True)  # Store model metadata
os.makedirs("logs", exist_ok=True)  # Store logs
os.makedirs("plots", exist_ok=True)  # Store visualizations


# Load and preprocess the knowledge graph
def load_graph(file_path):
    g = Graph()
    g.parse(file_path, format="turtle")
    return g


def preprocess_data(graph):
    entities = list(set(s for s, _, _ in graph) | set(o for _, _, o in graph))
    relations = list(set(p for _, p, _ in graph))

    entity2idx = {entity: idx for idx, entity in enumerate(entities)}
    relation2idx = {relation: idx for idx, relation in enumerate(relations)}

    triples = [(entity2idx[s], relation2idx[p], entity2idx[o]) for s, p, o in graph]

    return len(entities), len(relations), triples, entity2idx, relation2idx


# Load and preprocess dataset
graph_file_path = "Eurostat_KG.ttl"
num_entities, num_relations, triples, entity2idx, relation2idx = preprocess_data(load_graph(graph_file_path))

# Generate negative samples
rng = np.random.default_rng(SEED)
negative_triples = [(s, p, (o + rng.integers(1, num_entities)) % num_entities) for s, p, o in triples]

# Convert to numpy arrays
all_triples = np.array(triples + negative_triples)
labels = np.array([1] * len(triples) + [0] * len(negative_triples))

# **Split the dataset into Train, Validation, and Test**
X_train, X_temp, y_train, y_temp = train_test_split(all_triples, labels, test_size=0.3, random_state=SEED)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# **Save Data to Files in `models/data/`**
torch.save(X_train, "models/data/X_train.pt")
torch.save(y_train, "models/data/y_train.pt")

torch.save(X_val, "models/data/X_val.pt")
torch.save(y_val, "models/data/y_val.pt")

torch.save(X_test, "models/data/X_test.pt")
torch.save(y_test, "models/data/y_test.pt")

# Save metadata for correct model loading
torch.save(num_entities, "models/num_entities.pt")
torch.save(num_relations, "models/num_relations.pt")

print("\n Train/Validation/Test splits saved successfully in 'models/data/'.")
