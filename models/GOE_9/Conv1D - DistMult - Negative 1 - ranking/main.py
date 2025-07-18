###############################################
# CONFIGURATION
###############################################
import os
import random
import time
import numpy as np
import torch
import os

from torch_geometric.nn import MessagePassing

# General parameters
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory parameters
BASE_DIR = "models"
PROJECT_DIR = "GOE_9"
MODEL_NAME = "Conv1D - DistMult - Negative 1 - ranking"
FULL_MODEL_DIR = os.path.join(BASE_DIR, PROJECT_DIR, MODEL_NAME)

# Training parameters
EMBEDDING_DIM = 512
EPOCHS = 70
BATCH_SIZE = 256
LEARNING_RATE = 1e-5
PATIENCE = 5

# Graph parameters
GRAPH_FILE_PATH = "Eurostat_KG.ttl"
###############################################
# END CONFIGURATION
###############################################

# Ensure deterministic behavior & set device
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

print(f"Using device: {DEVICE}")

# Create directories for saving results
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(FULL_MODEL_DIR, exist_ok=True)

###############################################
# Rest of your code (unchanged) using config variables
###############################################

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC (no display)
import matplotlib.pyplot as plt

import seaborn as sns
import torch.nn as nn
import torch.optim as optim
from rdflib import Graph
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from tqdm import tqdm

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

import torch.nn.functional as F
from torch_geometric.utils import softmax

class OntologyAwareGATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_types, type_embedding_dim=32, heads=4, dropout=0.3):
        super().__init__(aggr='add')  # We'll use attention-weighted sum

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.type_embeddings = nn.Embedding(num_types, type_embedding_dim)
        self.gate = nn.Linear(in_channels, type_embedding_dim)

        # For attention
        self.att_src = nn.Parameter(torch.Tensor(heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(heads, out_channels))

        # Projection layer
        self.linear = nn.Linear(in_channels + type_embedding_dim, heads * out_channels, bias=False)

        self.residual_proj = nn.Linear(in_channels + type_embedding_dim, heads * out_channels, bias=False)

        # Normalization & dropout
        self.norm = nn.BatchNorm1d(heads * out_channels)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, edge_index, entity_types):
        type_info = self.type_embeddings(entity_types)  # (N, type_dim)
        gate = torch.sigmoid(self.gate(x))  # (N, type_dim)
        x_typed = torch.cat([x, gate * type_info], dim=1)  # (N, in + type_dim)

        self._x_residual = x_typed  # Save for residual
        return self.propagate(edge_index=edge_index, x=x_typed)

    def message(self, x_i, x_j, index):
        x_i_proj = self.linear(x_i).view(-1, self.heads, self.out_channels)
        x_j_proj = self.linear(x_j).view(-1, self.heads, self.out_channels)

        alpha = (x_i_proj * self.att_src).sum(dim=-1) + (x_j_proj * self.att_dst).sum(dim=-1)
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        alpha = softmax(alpha, index)
        alpha = self.dropout(alpha)

        # Apply attention
        x_j_weighted = x_j_proj * alpha.unsqueeze(-1)  # [E, H, C]

        # ===== FIX IS HERE =====
        return x_j_weighted.view(-1, self.heads * self.out_channels)  # [E, H*C]

    def update(self, aggr_out):
        out = aggr_out.view(-1, self.heads * self.out_channels)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)

        residual = self.residual_proj(self._x_residual)

        if residual.shape == out.shape:
            out = out + residual

        return out

# --- RANKING Version of the DistMult + GAT Model ---
class DistMultGATModel_Ranking(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, num_types, dropout_rate=0.5):
        super(DistMultGATModel_Ranking, self).__init__()
        self.embedding_dim = embedding_dim
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        self.heads = 4  # Define number of heads here

        # === Ontology-aware GAT ===
        self.gat_layer = OntologyAwareGATLayer(
            in_channels=embedding_dim,
            # --- THIS IS THE FIX ---
            # The output of each head is embedding_dim / heads.
            # After concatenation in the update step, the final dimension will be embedding_dim.
            out_channels=embedding_dim // self.heads,
            num_types=num_types,
            heads=self.heads,
            dropout=0.3
        )
        self.edge_index = None
        self.entity_type_tensor = None

        # === Conv1D & Dense Head (from your design) ===
        self.conv1d_1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv1d_2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dense1 = nn.Linear(256 * 1, 512)
        self.batch_norm3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(512, 1)

    def forward(self, h, r, t):
        s_idx, p_idx, o_idx = h, r, t

        if self.edge_index is None or self.entity_type_tensor is None:
            raise ValueError("GAT requires edge_index and entity_type_tensor to be set.")

        # Apply GAT to refine entity embeddings
        raw_entity_emb = self.entity_embeddings.weight
        refined_entity_emb = self.gat_layer(
            raw_entity_emb,
            edge_index=self.edge_index,
            entity_types=self.entity_type_tensor.to(raw_entity_emb.device)
        )

        # Look up embeddings for the current batch
        s_embed = refined_entity_emb[s_idx]
        p_embed = self.relation_embeddings(p_idx)
        o_embed = refined_entity_emb[o_idx]

        # DistMult scoring
        score = torch.sum(s_embed * p_embed * o_embed, dim=-1)

        # Pass score through your custom Conv1D head
        x = score.unsqueeze(1).unsqueeze(2)
        x = self.conv1d_1(x);
        x = self.batch_norm1(x);
        x = F.relu(x);
        x = self.dropout1(x)
        x = self.conv1d_2(x);
        x = self.batch_norm2(x);
        x = F.relu(x);
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.dense1(x);
        x = self.batch_norm3(x);
        x = F.relu(x);
        x = self.dropout3(x)

        # --- KEY CHANGE: Return the raw logit, not the sigmoid output ---
        final_score = self.dense2(x)
        return final_score

# Load and preprocess dataset
graph_file_path = GRAPH_FILE_PATH
num_entities, num_relations, triples, entity2idx, relation2idx = preprocess_data(load_graph(graph_file_path))

# Generate negative samples
rng = np.random.default_rng(SEED)
negative_triples = [(s, p, (o + rng.integers(1, num_entities)) % num_entities) for s, p, o in triples]

# Convert to tensors and send to GPU
all_triples = np.array(triples + negative_triples)
labels = np.array([1] * len(triples) + [0] * len(negative_triples))

# Load the dataset from saved files
X_train = torch.load("models/data/X_train.pt").to(DEVICE)
y_train = torch.load("models/data/y_train.pt").to(DEVICE)
X_val = torch.load("models/data/X_val.pt").to(DEVICE)
y_val = torch.load("models/data/y_val.pt").to(DEVICE)
X_test = torch.load("models/data/X_test.pt").to(DEVICE)
y_test = torch.load("models/data/y_test.pt").to(DEVICE)

# Training function

def train_ranking_model(model, X_train, y_train, X_val, y_val,
                        epochs, batch_size, learning_rate,
                        patience, weight_decay=0.01, margin=1.0):
    """
    Train a ranking model, keeping the structure of the original classification trainer.
    """
    # 1. Use MarginRankingLoss for the ranking task
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MarginRankingLoss(margin=margin)
    model.to(DEVICE)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        perm = torch.randperm(X_train.shape[0])
        X_train_shuffled, y_train_shuffled = X_train[perm], y_train[perm]

        tqdm_bar = tqdm(range(0, len(X_train_shuffled), batch_size), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
        for i in tqdm_bar:
            batch_X = X_train_shuffled[i:i + batch_size].to(DEVICE)
            batch_y = y_train_shuffled[i:i + batch_size].to(DEVICE)

            # 2. Separate batch into positive and negative triples
            pos_triples = batch_X[batch_y == 1]
            neg_triples = batch_X[batch_y == 0]

            # Ensure we have pairs to compare for the margin loss
            if len(pos_triples) == 0 or len(neg_triples) == 0:
                continue

            min_len = min(len(pos_triples), len(neg_triples))
            pos_triples = pos_triples[:min_len]
            neg_triples = neg_triples[:min_len]

            optimizer.zero_grad()

            # 3. Get scores for positive and negative triples separately
            # This assumes the model's forward pass takes (h, r, t)
            pos_scores = model(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
            neg_scores = model(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])

            # Target is a tensor of 1s, indicating we want pos_scores > neg_scores
            target = torch.ones_like(pos_scores).to(DEVICE)

            loss = loss_fn(pos_scores.view(-1), neg_scores.view(-1), target.view(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            tqdm_bar.set_postfix(loss=epoch_loss / num_batches)

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for j in range(0, len(X_val), batch_size):
                batch_X_val = X_val[j:j + batch_size].to(DEVICE)
                batch_y_val = y_val[j:j + batch_size].to(DEVICE)

                # Replicate the same pairing logic for validation
                pos_val = batch_X_val[batch_y_val == 1]
                neg_val = batch_X_val[batch_y_val == 0]

                if len(pos_val) == 0 or len(neg_val) == 0:
                    continue

                min_len_val = min(len(pos_val), len(neg_val))
                pos_val = pos_val[:min_len_val]
                neg_val = neg_val[:min_len_val]

                pos_scores_val = model(pos_val[:, 0], pos_val[:, 1], pos_val[:, 2])
                neg_scores_val = model(neg_val[:, 0], neg_val[:, 1], neg_val[:, 2])

                target_val = torch.ones_like(pos_scores_val).to(DEVICE)
                batch_loss = loss_fn(pos_scores_val.view(-1), neg_scores_val.view(-1), target_val.view(-1))
                val_loss += batch_loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # --- Early Stopping Logic (Unchanged) ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save with a distinct "ranking" name to avoid overwriting
            torch.save(model.state_dict(), os.path.join(FULL_MODEL_DIR, "best_ranking_model.pth"))
            print(f" New best ranking model saved at epoch {epoch + 1} with validation loss {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Early Stopping Counter: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print(f" Early stopping at epoch {epoch + 1}")
            break

    # --- Post-Training (Unchanged structure) ---
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    with open(os.path.join(FULL_MODEL_DIR, "training_time_ranking.txt"), "w") as f:
        f.write(f"Training Time: {training_time:.2f} seconds ({training_time / 60:.2f} minutes)\n")

    plots_dir = os.path.join(FULL_MODEL_DIR, "plots_ranking")
    os.makedirs(plots_dir, exist_ok=True)
    np.save(os.path.join(FULL_MODEL_DIR, "train_losses_ranking.npy"), np.array(train_losses))
    np.save(os.path.join(FULL_MODEL_DIR, "val_losses_ranking.npy"), np.array(val_losses))

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss", color='blue', linestyle="-", linewidth=2)
    plt.plot(val_losses, label="Validation Loss", color='red', linestyle="--", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Margin Ranking Loss")  # Updated label
    plt.legend()
    plt.title(f"Training vs Validation Loss (Ranking) - {MODEL_NAME}")
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "loss_plot_ranking.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Training & Validation loss plot saved to {os.path.join(plots_dir, 'loss_plot_ranking.png')}")

    # Load the best model
    best_model_path = os.path.join(FULL_MODEL_DIR, "best_ranking_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f" Best ranking model loaded for evaluation.")
    else:
        print("Warning: Best model file not found. Returning model with last weights.")

    return model

# Train Classification Model

# Train Classification Model
from rdflib.namespace import RDF
from rdflib.namespace import RDFS, OWL

def build_type_indices(entity_type_map, constraints, entity2idx, relation2idx):
    """
    Automatically build type2idx, constraints_idx, and entity_type_map_idx.

    Args:
        entity_type_map (dict[str → set[str]]): RDF types per entity URI.
        constraints (dict[str → (domain_uri, range_uri)]): constraints from ontology.
        entity2idx (dict[str → int])
        relation2idx (dict[str → int])

    Returns:
        type2idx: dict[type_uri → int]
        constraints_idx: dict[relation_idx → expected_range_type_idx]
        entity_type_map_idx: dict[entity_idx → set[type_idx]]
    """
    # === 1. Build type2idx from all seen types ===
    all_types = set()
    for types in entity_type_map.values():
        all_types.update(types)
    type2idx = {t: i for i, t in enumerate(sorted(all_types))}

    # === 2. Map constraints to idx form ===
    constraints_idx = {}
    for rel_uri, (_, range_uri) in constraints.items():
        if rel_uri in relation2idx and range_uri in type2idx:
            constraints_idx[relation2idx[rel_uri]] = type2idx[range_uri]

    # === 3. Map entity types to idx form ===
    entity_type_map_idx = {}
    for ent_uri, types in entity_type_map.items():
        if ent_uri in entity2idx:
            ent_idx = entity2idx[ent_uri]
            class_ids = {type2idx[t] for t in types if t in type2idx}
            if class_ids:
                entity_type_map_idx[ent_idx] = class_ids

    return type2idx, constraints_idx, entity_type_map_idx

def extract_domain_range_constraints(graph):
    """
    Extracts rdfs:domain and rdfs:range constraints for relations.
    Returns a dictionary: {relation_uri: (domain_class_uri, range_class_uri)}
    """
    constraints = {}
    for r in set(p for _, p, _ in graph):
        domain = next((o for s, p, o in graph.triples((r, RDFS.domain, None))), None)
        range_ = next((o for s, p, o in graph.triples((r, RDFS.range, None))), None)
        if domain or range_:
            constraints[r] = (domain, range_)
    return constraints

def build_entity_type_map(graph):
    """
    Builds a mapping from each entity to its rdf:type(s)
    """
    type_map = {}
    for s, _, o in graph.triples((None, RDF.type, None)):
        type_map.setdefault(s, set()).add(o)
    return type_map


g = load_graph(graph_file_path)
constraints = extract_domain_range_constraints(g)
entity_type_map = build_entity_type_map(g)
idx2entity = [None] * num_entities
for ent_uri, idx in entity2idx.items():
    idx2entity[idx] = ent_uri

idx2relation = [None] * num_relations
for rel_uri, idx in relation2idx.items():
    idx2relation[idx] = rel_uri

type2idx, constraints_idx, entity_type_map_idx = build_type_indices(
    entity_type_map, constraints, entity2idx, relation2idx
)

# Build the set of all rdf:type URIs
all_types = set()
for types in entity_type_map.values():
    all_types.update(types)

type2idx = {t: i for i, t in enumerate(sorted(all_types))}
idx2type = {i: t for t, i in type2idx.items()}
num_types = len(type2idx)

classifier = DistMultGATModel_Ranking    (
    num_entities=num_entities,
    num_relations=num_relations,
    embedding_dim=EMBEDDING_DIM,
    num_types=num_types  # from your `type2idx`
).to(DEVICE)

from torch_geometric.utils import from_networkx
import networkx as nx
import torch_geometric

# 1. Build a graph from your triples
def build_edge_index(triples, num_entities):
    edge_list = []
    for h, _, t in triples:
        edge_list.append((h, t))
        edge_list.append((t, h))  # bidirectional for GCN
    G = nx.Graph()
    G.add_edges_from(edge_list)
    data = from_networkx(G)
    return data.edge_index

# 2. Convert entity types to tensor
def build_entity_type_tensor(entity_type_map_idx, num_entities):
    default_type = 0  # assume 0 is safe if entity has no type
    type_tensor = torch.zeros(num_entities, dtype=torch.long)
    for ent_id, type_ids in entity_type_map_idx.items():
        type_tensor[ent_id] = next(iter(type_ids))  # use first type (or average logic if needed)
    return type_tensor

# === Build GCN inputs ===
edge_index = build_edge_index(triples, num_entities).to(DEVICE)
entity_type_tensor = build_entity_type_tensor(entity_type_map_idx, num_entities).to(DEVICE)

# === Inject into model ===
classifier.edge_index = edge_index
classifier.entity_type_tensor = entity_type_tensor

trained_classifier = train_ranking_model(
    model=classifier,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    patience=PATIENCE,
    margin=1.0
)
os.makedirs(FULL_MODEL_DIR, exist_ok=True)
torch.save(trained_classifier.state_dict(), os.path.join(FULL_MODEL_DIR, "classifier.pth"))
trained_classifier.load_state_dict(torch.load(os.path.join(FULL_MODEL_DIR, "classifier.pth"), map_location=DEVICE))

# Evaluate Model
import json
from sklearn.metrics import accuracy_score, matthews_corrcoef, balanced_accuracy_score, average_precision_score
from sklearn.metrics import confusion_matrix

def generate_evaluation_plots(y_true, y_scores, train_losses, val_losses, model_name=MODEL_NAME):
    """
    Generate and save all evaluation plots including:
    - Precision-Recall Curve
    - ROC Curve
    - Learning Curve
    - Prediction Score Distribution
    - Confusion Matrix
    """
    plots_dir = os.path.join(FULL_MODEL_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    from sklearn.metrics import precision_recall_curve, roc_curve
    # **Precision-Recall Curve**
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "precision_recall_curve.png"))
    plt.close()
    print(f" Precision-Recall Curve saved to {os.path.join(plots_dir, 'precision_recall_curve.png')}")
    # **ROC Curve**
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.', label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "roc_curve.png"))
    plt.close()
    print(f" ROC Curve saved to {os.path.join(plots_dir, 'roc_curve.png')}")
    # **Learning Curve (Loss per Epoch)**
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Learning Curve - {model_name}")
    plt.savefig(os.path.join(plots_dir, "learning_curve.png"))
    plt.close()
    print(f" Learning Curve saved to {os.path.join(plots_dir, 'learning_curve.png')}")
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for threshold in thresholds:
        y_pred = (y_scores > threshold).astype(int)
        # **Confusion Matrix**
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix")
        plt.savefig(os.path.join(plots_dir, f"confusion_matrix_{float(threshold)}.png"))
        plt.close()
        print(f" Confusion Matrix (Threshold {threshold}) saved to {os.path.join(plots_dir, f'confusion_matrix_{float(threshold)}.png')}")
        # **Prediction Score Distribution at Different Thresholds**
        plt.figure(figsize=(8, 6))
        pos_color = "#FF4500"  # Bright Orange-Red
        neg_color = "#1E90FF"  # Vibrant Royal Blue
        pos_scores = y_scores[y_true == 1]
        neg_scores = y_scores[y_true == 0]
        plt.hist(pos_scores, bins=30, alpha=0.6, color=pos_color, edgecolor="black", linewidth=1.2,
                 label="Positive Class")
        plt.hist(neg_scores, bins=30, alpha=0.6, color=neg_color, edgecolor="black", linewidth=1.2,
                 label="Negative Class")
        plt.xlabel("Predicted Probability")
        plt.ylabel("Frequency")
        plt.title(f"Prediction Score Distribution")
        plt.legend(fontsize=12, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
        plt.subplots_adjust(right=0.75)
        plt.savefig(os.path.join(plots_dir, f"prediction_distribution_{float(threshold)}.png"), dpi=300, bbox_inches="tight")
        plt.close()

from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import math

def generate_advanced_plots(y_true, y_scores, train_losses, val_losses, model, model_name=MODEL_NAME):
    """
    Generate and save multiple evaluation plots including:
    - Smoothed Loss Curve
    - Precision-Recall Tradeoff
    - F1 Score by Threshold
    - Distribution of Predictions
    - t-SNE Visualization of Entity Embeddings with Clustering
    """
    plots_dir = os.path.join(FULL_MODEL_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    # ---- Smoothed Loss Curve ----
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss", alpha=0.6, linewidth=2, color="#1f77b4")
    plt.plot(val_losses, label="Validation Loss", alpha=0.6, linewidth=2, color="#ff7f0e")
    if len(train_losses) > 5:
        smoothed_train = np.convolve(train_losses, np.ones(5) / 5, mode="valid")
        smoothed_val = np.convolve(val_losses, np.ones(5) / 5, mode="valid")
        plt.plot(smoothed_train, linestyle="--", linewidth=2, color="blue", label="Smoothed Train Loss")
        plt.plot(smoothed_val, linestyle="--", linewidth=2, color="orange", label="Smoothed Val Loss")
    plt.xlabel("Epochs", fontsize=14, fontweight="bold")
    plt.ylabel("Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=12, loc="upper right")
    plt.title("Smoothed Learning Curve", fontsize=16, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(plots_dir, "smoothed_learning_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()
    # ---- Precision-Recall Tradeoff ----
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], label="Precision", linewidth=2, color="blue")
    plt.plot(thresholds, recall[:-1], label="Recall", linewidth=2, color="green")
    plt.axvline(best_threshold, linestyle="--", color="red", label=f"Optimal Threshold: {best_threshold:.2f}")
    plt.xlabel("Threshold", fontsize=14, fontweight="bold")
    plt.ylabel("Score", fontsize=14, fontweight="bold")
    plt.title("Precision-Recall Tradeoff", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(plots_dir, "precision_recall_tradeoff.png"), dpi=300, bbox_inches="tight")
    plt.close()
    # ---- F1 Score by Threshold ----
    threshold_samples = np.linspace(0, 1, 100)
    f1_scores = [f1_score(y_true, y_scores >= t) for t in threshold_samples]
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_samples, f1_scores, label="F1 Score", color="purple", linewidth=2)
    plt.axvline(best_threshold, linestyle="--", color="red", label=f"Optimal: {best_threshold:.2f}")
    plt.xlabel("Threshold", fontsize=14, fontweight="bold")
    plt.ylabel("F1 Score", fontsize=14, fontweight="bold")
    plt.title("F1 Score by Classification Threshold", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(plots_dir, "f1_by_threshold.png"), dpi=300, bbox_inches="tight")
    plt.close()
    # ---- t-SNE Visualization of Entity Embeddings ----
    entity_embeddings = model.entity_embeddings.weight.cpu().detach().numpy()
    max_samples = 5000
    if entity_embeddings.shape[0] > max_samples:
        sampled_indices = np.random.choice(entity_embeddings.shape[0], max_samples, replace=False)
        entity_embeddings = entity_embeddings[sampled_indices]
    perplexity_value = 30 if entity_embeddings.shape[0] > 100 else 5
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    reduced_embeddings = tsne.fit_transform(entity_embeddings)
    num_clusters = min(10, len(reduced_embeddings) // 500)
    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap="viridis", alpha=0.7)
    plt.xlabel("t-SNE Component 1", fontsize=14, fontweight="bold")
    plt.ylabel("t-SNE Component 2", fontsize=14, fontweight="bold")
    plt.title("t-SNE Visualization of Entity Embeddings", fontsize=16, fontweight="bold")
    plt.colorbar(scatter, label="Cluster Index")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(plots_dir, "tsne_embeddings.png"), dpi=300, bbox_inches="tight")
    plt.close()

def evaluate_model(model, X_test, y_test, model_name="TransEModel", batch_size=512):
    """
    Evaluates the trained model on the hybrid loss (worked) set.
    Computes classification metrics for different thresholds.
    Saves results and evaluation time.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    start_time = time.time()
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i + batch_size].to(DEVICE)
            batch_y = y_test[i:i + batch_size].to(DEVICE)
            batch_preds = model(batch_X).cpu()

            all_predictions.append(batch_preds)
            all_labels.append(batch_y.cpu())
    predictions = torch.cat(all_predictions, dim=0)
    y_test_cpu = torch.cat(all_labels, dim=0)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    metrics_results = {}
    for threshold in thresholds:
        y_pred = (predictions > threshold).float()
        accuracy = accuracy_score(y_test_cpu, y_pred)
        balanced_acc = balanced_accuracy_score(y_test_cpu, y_pred)
        precision = precision_score(y_test_cpu, y_pred)
        recall = recall_score(y_test_cpu, y_pred)
        f1 = f1_score(y_test_cpu, y_pred)
        roc_auc = roc_auc_score(y_test_cpu, predictions)
        pr_auc = average_precision_score(y_test_cpu, predictions)
        mcc = matthews_corrcoef(y_test_cpu, y_pred)
        metrics_results[threshold] = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "mcc": mcc
        }
        print(f"\n Evaluation Metrics for {model_name} (Threshold = {threshold}):")
        print(f" Accuracy: {accuracy:.4f}")
        print(f" Balanced Accuracy: {balanced_acc:.4f}")
        print(f" Precision: {precision:.4f}")
        print(f" Recall: {recall:.4f}")
        print(f" F1 Score: {f1:.4f}")
        print(f" ROC AUC: {roc_auc:.4f}")
        print(f" PR AUC: {pr_auc:.4f}")
        print(f" MCC: {mcc:.4f}")
    evaluation_time = time.time() - start_time
    with open(os.path.join(FULL_MODEL_DIR, "evaluation_time.txt"), "w") as f:
        f.write(f"Evaluation Time: {evaluation_time:.2f} seconds ({evaluation_time/60:.2f} minutes)\n")
    model_results_dir = FULL_MODEL_DIR
    logs_dir = os.path.join(model_results_dir, "logs")
    plots_dir = os.path.join(model_results_dir, "plots")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    metrics_file = os.path.join(logs_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_results, f, indent=4)
    print(f" Metrics saved to {metrics_file}")
    train_losses = np.load(os.path.join(model_results_dir, "train_losses.npy"))
    val_losses = np.load(os.path.join(model_results_dir, "val_losses.npy"))
    generate_evaluation_plots(
        y_true=y_test_cpu.numpy(),
        y_scores=predictions.numpy(),
        train_losses=train_losses,
        val_losses=val_losses,
        model_name=model_name
    )
    generate_advanced_plots(
        y_true=y_test_cpu.numpy(),
        y_scores=predictions.numpy(),
        train_losses=train_losses,
        val_losses=val_losses,
        model=model,
        model_name=model_name
    )
    return metrics_results

# Call the evaluation function and save results
model_metrics = evaluate_model(trained_classifier, X_test, y_test, model_name=MODEL_NAME)

def calculate_hits_metrics(model, X_test, y_test, epsilons=None, batch_size=512):
    """
    Computes strict and soft Hits@1, @5, @10 for multiple epsilon values.
    This version is corrected to be compatible with models that expect
    (h, r, t) as separate arguments in their forward pass.

    Args:
        model (nn.Module): Trained model.
        X_test (Tensor): Test triples [N, 3].
        y_test (Tensor): Labels (1 = true, 0 = false).
        epsilons (list[float]): List of epsilon margins for soft hits.
        batch_size (int): Batch size for evaluation.

    Returns:
        dict: All strict and soft hit results across all epsilon levels.
    """
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1]

    model.eval()
    num_relations = model.relation_embeddings.num_embeddings

    strict_hits = {1: 0, 5: 0, 10: 0}
    soft_hits = {eps: {1: 0, 5: 0, 10: 0} for eps in epsilons}
    total_positives = 0

    with torch.no_grad():
        test_tqdm_bar = tqdm(range(0, len(X_test), batch_size), desc="Calculating Hits@k")
        for i in test_tqdm_bar:
            batch_X = X_test[i:i + batch_size].to(DEVICE)
            batch_y = y_test[i:i + batch_size].to(DEVICE)

            positive_triples = batch_X[batch_y == 1]

            if len(positive_triples) == 0:
                continue

            for j in range(len(positive_triples)):
                h_id, r_id, t_id = positive_triples[j].tolist()

                all_r = torch.arange(num_relations, device=DEVICE)
                h_expand = torch.full_like(all_r, h_id)
                t_expand = torch.full_like(all_r, t_id)

                # --- CORRECTED MODEL CALL ---
                # Pass h, r, t as separate arguments
                all_scores = model(h_expand, all_r, t_expand).squeeze()

                true_score = all_scores[r_id].item()
                sorted_scores, sorted_indices = torch.sort(all_scores, descending=True)
                rank = (sorted_indices == r_id).nonzero(as_tuple=True)[0].item() + 1

                # Strict Hits@K
                for k in [1, 5, 10]:
                    if rank <= k:
                        strict_hits[k] += 1

                # Soft Hits@K per epsilon
                for eps in epsilons:
                    for k in [1, 5, 10]:
                        top_k_scores = sorted_scores[:k]
                        if any(true_score >= (score.item() - eps) for score in top_k_scores):
                            soft_hits[eps][k] += 1

                total_positives += 1

    if total_positives == 0:
        return {}

    # Compile and return the final results, normalized by the number of positive triples
    results = {f"hits@{k}": strict_hits[k] / total_positives for k in [1, 5, 10]}
    for eps in epsilons:
        for k in [1, 5, 10]:
            results[f"soft_hits@{k}_eps={eps}"] = soft_hits[eps][k] / total_positives

    return results

def calculate_mrr_metrics(model, X_test, y_test, epsilons=None, batch_size=512):
    """
    Computes strict MRR and soft MRR for multiple epsilon thresholds.
    This version is corrected to be compatible with models that expect
    (h, r, t) as separate arguments in their forward pass.

    Args:
        model (nn.Module): Trained model.
        X_test (Tensor): Test triples (shape: [N, 3]).
        y_test (Tensor): Binary labels (only positives are used).
        epsilons (list of float): List of tolerances for soft MRR.
        batch_size (int): Prediction batch size.

    Returns:
        dict: {
            "mrr": ...,
            "soft_mrr@{ε1}": ...,
            "soft_mrr@{ε2}": ...
        }
    """
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1]

    model.eval()
    num_relations = model.relation_embeddings.num_embeddings

    mrr_total = 0.0
    soft_mrr_totals = {eps: 0.0 for eps in epsilons}
    total_positives = 0

    with torch.no_grad():
        test_tqdm_bar = tqdm(range(0, len(X_test), batch_size), desc="Calculating MRR")
        for i in test_tqdm_bar:
            batch_X = X_test[i:i + batch_size].to(DEVICE)
            batch_y = y_test[i:i + batch_size].to(DEVICE)

            positive_triples = batch_X[batch_y == 1]

            if len(positive_triples) == 0:
                continue

            for j in range(len(positive_triples)):
                h_id, r_id, t_id = positive_triples[j].tolist()

                all_r = torch.arange(num_relations, device=DEVICE)
                h_expand = torch.full_like(all_r, h_id)
                t_expand = torch.full_like(all_r, t_id)

                # --- CORRECTED MODEL CALL ---
                all_scores = model(h_expand, all_r, t_expand).squeeze()

                true_score = all_scores[r_id].item()

                sorted_scores, sorted_indices = torch.sort(all_scores, descending=True)
                rank = (sorted_indices == r_id).nonzero(as_tuple=True)[0].item() + 1
                mrr_total += 1.0 / rank

                for eps in epsilons:
                    # A soft rank is the number of other relations with scores that are
                    # significantly (by more than eps) better than the true relation's score.
                    soft_rank = (all_scores > true_score + eps).sum().item() + 1
                    soft_mrr_totals[eps] += 1.0 / soft_rank

                total_positives += 1

    if total_positives == 0:
        return {}

    results = {"mrr": mrr_total / total_positives}
    for eps in epsilons:
        key = f"soft_mrr_eps={eps}"
        results[key] = soft_mrr_totals[eps] / total_positives

    return results

def calculate_mean_rank_metrics(model, X_test, y_test, epsilons=None, batch_size=512):
    """
    Computes strict and soft Mean Rank (MR) for relation prediction.
    This version is corrected to be compatible with models that expect
    (h, r, t) as separate arguments in their forward pass.

    Args:
        model (nn.Module): Trained model.
        X_test (Tensor): Test triples [N, 3].
        y_test (Tensor): Labels (1 for positive only).
        epsilons (list of float): List of tolerances for soft rank.
        batch_size (int): Batch size for evaluation.

    Returns:
        dict: A dictionary containing the strict and soft mean rank metrics.
    """
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1]

    model.eval()
    num_relations = model.relation_embeddings.num_embeddings

    strict_ranks = []
    soft_ranks_dict = {eps: [] for eps in epsilons}

    with torch.no_grad():
        test_tqdm_bar = tqdm(range(0, len(X_test), batch_size), desc="Calculating Mean Rank")
        for i in test_tqdm_bar:
            batch_X = X_test[i:i + batch_size].to(DEVICE)
            batch_y = y_test[i:i + batch_size].to(DEVICE)

            positive_triples = batch_X[batch_y == 1]

            if len(positive_triples) == 0:
                continue

            for j in range(len(positive_triples)):
                h_id, r_id, t_id = positive_triples[j].tolist()

                all_r = torch.arange(num_relations, device=DEVICE)
                h_expand = torch.full_like(all_r, h_id)
                t_expand = torch.full_like(all_r, t_id)

                # --- CORRECTED MODEL CALL ---
                scores = model(h_expand, all_r, t_expand).squeeze()

                true_score = scores[r_id].item()

                # Strict rank: how many scores are strictly greater than true
                rank_strict = (scores > true_score).sum().item() + 1
                strict_ranks.append(rank_strict)

                # Soft ranks: count how many scores are within epsilon margin of the true score
                for eps in epsilons:
                    # Your original logic was slightly different, this is a more standard soft rank:
                    # Number of scores better than or almost as good as the true score.
                    rank_soft = (scores >= true_score - eps).sum().item()
                    soft_ranks_dict[eps].append(rank_soft)

    if not strict_ranks:
        return {}

    results = {
        "mean_rank_strict": np.mean(strict_ranks)
    }
    for eps in epsilons:
        results[f"mean_rank_soft_eps={eps}"] = np.mean(soft_ranks_dict[eps])

    return results

def calculate_ndcg_metrics(model, X_test, y_test, epsilons=None, batch_size=512, k=10):
    """
    Calculates Strict and Soft NDCG@k for relation prediction.
    This version is corrected to be compatible with models that expect
    (h, r, t) as separate arguments in their forward pass.
    """
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1]

    model.eval()
    num_relations = model.relation_embeddings.num_embeddings

    strict_ndcgs = []
    soft_ndcgs_dict = {eps: [] for eps in epsilons}

    with torch.no_grad():
        test_tqdm_bar = tqdm(range(0, len(X_test), batch_size), desc="Calculating NDCG@k")
        for i in test_tqdm_bar:
            batch_X = X_test[i:i + batch_size].to(DEVICE)
            batch_y = y_test[i:i + batch_size].to(DEVICE)

            positive_triples = batch_X[batch_y == 1]

            if len(positive_triples) == 0:
                continue

            for j in range(len(positive_triples)):
                h_id, r_id, t_id = positive_triples[j].tolist()

                all_r = torch.arange(num_relations, device=DEVICE)
                h_expand = torch.full_like(all_r, h_id)
                t_expand = torch.full_like(all_r, t_id)

                # --- CORRECTED MODEL CALL ---
                scores = model(h_expand, all_r, t_expand).squeeze()

                sorted_scores, sorted_indices = torch.sort(scores, descending=True)
                true_score = scores[r_id].item()

                # Strict NDCG
                strict_rels = [1 if rel_id == r_id else 0 for rel_id in sorted_indices[:k].tolist()]
                strict_dcg = sum((2 ** rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(strict_rels))
                strict_idcg = (2 ** 1 - 1) / math.log2(2) if any(strict_rels) else 1.0
                strict_ndcg = strict_dcg / strict_idcg
                strict_ndcgs.append(strict_ndcg)

                # Soft NDCG per epsilon
                for eps in epsilons:
                    soft_rels = [1 if abs(true_score - score.item()) <= eps else 0 for score in sorted_scores[:k]]
                    soft_dcg = sum((2 ** rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(soft_rels))

                    soft_hits = sum(soft_rels)
                    if soft_hits == 0:
                        soft_ndcg = 0.0
                    else:
                        # The ideal gain should be based on the number of "relevant" items found
                        soft_idcg = sum((2 ** 1 - 1) / math.log2(i + 2) for i in range(min(soft_hits, k)))
                        soft_ndcg = soft_dcg / soft_idcg if soft_idcg > 0 else 0.0
                    soft_ndcgs_dict[eps].append(soft_ndcg)

    if not strict_ndcgs:
        return {}

    results = {
        f"strict_ndcg@{k}": np.mean(strict_ndcgs)
    }
    for eps in epsilons:
        results[f"soft_ndcg@{k}_eps={eps}"] = np.mean(soft_ndcgs_dict[eps])

    return results

def precompute_gnn_embeddings(model):
    """
    Helper function to compute GNN-refined embeddings once for evaluation.
    This is more robust and works for both GCN and GAT layers.
    """
    print("Pre-computing GNN-refined embeddings for evaluation...")
    model.eval()

    gnn_layer = None
    if hasattr(model, 'gcn_layer'):
        gnn_layer = model.gcn_layer
        print("Found GCN layer.")
    elif hasattr(model, 'gat_layer'):
        gnn_layer = model.gat_layer
        print("Found GAT layer.")
    else:
        # If no GNN layer, just return the base embeddings
        print("No GNN layer found. Using base entity embeddings.")
        return model.entity_embeddings.weight.detach()

    with torch.no_grad():
        base_entity_emb = model.entity_embeddings.weight
        # Ensure all necessary tensors for GNN are on the correct device
        edge_index = model.edge_index.to(base_entity_emb.device)
        entity_types = model.entity_type_tensor.to(base_entity_emb.device)

        refined_entity_emb = gnn_layer(
            base_entity_emb,
            edge_index=edge_index,
            entity_types=entity_types
        )
    print("GNN embeddings computed.")
    return refined_entity_emb.detach()

def calculate_median_rank_metrics(model, X_test, y_test, epsilons=None, batch_size=512):
    """
    Computes strict and soft Median Rank(s) for relation prediction using multiple epsilon values.
    This version is corrected for the optimized GNN-based model.
    """
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1]

    model.eval()
    # Pre-compute the GNN-refined embeddings once before starting the evaluation.
    refined_entity_emb = precompute_gnn_embeddings(model)
    num_relations = model.relation_embeddings.num_embeddings

    strict_ranks = []
    soft_ranks_dict = {eps: [] for eps in epsilons}

    with torch.no_grad():
        test_tqdm_bar = tqdm(range(0, len(X_test), batch_size), desc="Calculating Median Rank")
        for i in test_tqdm_bar:
            batch_X = X_test[i:i + batch_size].to(DEVICE)
            batch_y = y_test[i:i + batch_size].to(DEVICE)

            # Only evaluate on the positive triples from the test set
            positive_triples = batch_X[batch_y == 1]

            if len(positive_triples) == 0:
                continue

            for j in range(len(positive_triples)):
                h_id, r_id, t_id = positive_triples[j].tolist()

                # Create a batch of all possible relations for the given (h, t) pair
                all_r = torch.arange(num_relations, device=DEVICE)
                h_expand = torch.full_like(all_r, h_id)
                t_expand = torch.full_like(all_r, t_id)

                # --- CORRECTED MODEL CALL ---
                # Pass the pre-computed embeddings as the fourth argument
                scores = model(h_expand, all_r, t_expand, refined_entity_emb).squeeze()

                true_score = scores[r_id].item()

                # --- Strict Rank ---
                # The rank is the position of the true relation in the sorted list of scores.
                sorted_scores, sorted_indices = torch.sort(scores, descending=True)
                strict_rank = (sorted_indices == r_id).nonzero(as_tuple=True)[0].item() + 1
                strict_ranks.append(strict_rank)

                # --- Soft Ranks ---
                for eps in epsilons:
                    # The soft rank is the number of relations with scores plausibly close to the true score.
                    soft_rank = (scores >= (true_score - eps)).sum().item()
                    soft_ranks_dict[eps].append(soft_rank)

    if not strict_ranks:
        return {}  # Return empty dict if no positive samples were found

    # Compile and return the final results
    results = {
        "strict_median_rank": float(np.median(strict_ranks))
    }
    for eps in epsilons:
        results[f"soft_median_rank_eps={eps}"] = float(np.median(soft_ranks_dict[eps]))

    return results

def plot_rank_distributions_multi_eps(model, X_test, y_test, epsilons=None, batch_size=512, k=43, save_dir=f"{FULL_MODEL_DIR}/plots"):
    """
    Plots strict rank distribution once and soft rank distributions for multiple epsilon values.

    Args:
        model (nn.Module): Trained model.
        X_test (Tensor): Test triples.
        y_test (Tensor): Labels (1 = positive).
        epsilons (list of float): Tolerance values for soft rank.
        batch_size (int): Batch size.
        k (int): Max rank to plot.
        save_dir (str): Folder to save plots.
    """
    import os
    model.eval()
    num_relations = model.relation_embeddings.num_embeddings
    os.makedirs(save_dir, exist_ok=True)

    strict_ranks = []
    soft_ranks_dict = {eps: [] for eps in epsilons}

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i + batch_size].to(DEVICE)
            batch_y = y_test[i:i + batch_size].to(DEVICE)

            for j in range(len(batch_X)):
                if batch_y[j] != 1:
                    continue

                h_id, r_id, t_id = batch_X[j].tolist()
                all_r = torch.arange(num_relations).to(DEVICE)
                h_expand = h_id * torch.ones_like(all_r)
                t_expand = t_id * torch.ones_like(all_r)

                batch_inputs = torch.stack([h_expand, all_r, t_expand], dim=1)  # Shape: (num_relations, 3)
                scores = model(batch_inputs).squeeze()
                sorted_scores, sorted_indices = torch.sort(scores, descending=True)
                true_score = scores[r_id].item()

                # Strict rank
                strict_rank = (sorted_indices == r_id).nonzero(as_tuple=True)[0].item() + 1
                strict_ranks.append(strict_rank)

                # Soft ranks for each ε
                for eps in epsilons:
                    soft_rank = None
                    for idx, score in enumerate(sorted_scores):
                        if abs(score.item() - true_score) <= eps:
                            soft_rank = idx + 1
                            break
                    if soft_rank is not None:
                        soft_ranks_dict[eps].append(soft_rank)

    # Plot strict rank histogram
    plt.figure(figsize=(10, 6))
    plt.hist(strict_ranks, bins=range(1, k + 2), alpha=0.75, color="skyblue", edgecolor="black")
    plt.title("Strict Relation Rank Distribution")
    plt.xlabel("Rank of True Relation")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(os.path.join(save_dir, "strict_rank_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Strict rank histogram saved: strict_rank_distribution.png")

    # Plot soft rank histograms for each epsilon
    for eps in epsilons:
        ranks = soft_ranks_dict[eps]
        if ranks:
            plt.figure(figsize=(10, 6))
            plt.hist(ranks, bins=range(1, k + 2), alpha=0.75, color="salmon", edgecolor="black")
            plt.title(f"Soft Relation Rank Distribution (ε = {eps})")
            plt.xlabel("Soft Rank of True Relation")
            plt.ylabel("Frequency")
            plt.grid(True, linestyle="--", alpha=0.6)
            fname = f"soft_rank_distribution_eps_{str(eps).replace('.', '_')}.png"
            plt.savefig(os.path.join(save_dir, fname), dpi=300, bbox_inches="tight")
            plt.close()
            print(f" Soft rank histogram saved: {fname}")

# === Evaluate Metrics ===
hits_results = calculate_hits_metrics(trained_classifier, X_test, y_test, epsilons=[0.01, 0.05, 0.1])
mrr_results = calculate_mrr_metrics(trained_classifier, X_test, y_test, epsilons=[0.01, 0.05, 0.1])
mean_rank_results = calculate_mean_rank_metrics(trained_classifier, X_test, y_test, epsilons=[0.01, 0.05, 0.1])
ndcg_results = calculate_ndcg_metrics(trained_classifier, X_test, y_test, epsilons=[0.01, 0.05, 0.1], k=10)
median_rank_results = calculate_median_rank_metrics(trained_classifier, X_test, y_test, epsilons=[0.01, 0.05, 0.1])

# === Print Results ===
for metric_name, value in hits_results.items():
    print(f"{metric_name}: {value:.4f}")

for metric_name, value in mrr_results.items():
    print(f"{metric_name}: {value:.4f}")

for metric_name, value in mean_rank_results.items():
    print(f"{metric_name}: {value:.4f}")

print(ndcg_results)
print(median_rank_results)

# === Combine All Metrics ===
all_metrics = {
    "hits": {**hits_results},
    "mrr": {**mrr_results},
    "mean_rank": {**mean_rank_results},
    "ndcg": {**ndcg_results},
    "median_rank": {**median_rank_results}
}

# === Save to evaluation_metrics.json ===
logs_dir = os.path.join(FULL_MODEL_DIR, "logs")
os.makedirs(logs_dir, exist_ok=True)

metrics_file = os.path.join(logs_dir, "evaluation_metrics.json")
with open(metrics_file, "w") as f:
    json.dump(all_metrics, f, indent=4)

plot_rank_distributions_multi_eps(
    model=trained_classifier,
    X_test=X_test,
    y_test=y_test,
    epsilons=[0.01, 0.05, 0.1],
    save_dir=os.path.join(FULL_MODEL_DIR, "plots")
)
