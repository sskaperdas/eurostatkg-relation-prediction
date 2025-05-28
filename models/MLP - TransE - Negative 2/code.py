import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from rdflib import Graph
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, matthews_corrcoef)
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedShuffleSplit

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --------------------------
# Data loading and processing
# --------------------------
def load_graph(file_path):
    g = Graph()
    g.parse(file_path, format="turtle")
    return g

def preprocess_data(graph):
    entities = list(set(s for s, _, _ in graph) | set(o for _, _, o in graph))
    relations = list(set(p for _, p, _ in graph))

    entity2idx = {entity: idx for idx, entity in enumerate(entities)}
    idx2entity = {idx: entity for idx, entity in enumerate(entities)}
    relation2idx = {relation: idx for idx, relation in enumerate(relations)}
    idx2relation = {idx: relation for idx, relation in enumerate(relations)}

    num_entities = len(entities)
    num_relations = len(relations)

    triples = [(entity2idx[s], relation2idx[p], entity2idx[o]) for s, p, o in graph]

    return num_entities, num_relations, triples, entity2idx, idx2entity, relation2idx, idx2relation

# --------------------------
# Loss and utility functions
# --------------------------
class MarginRankingLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(MarginRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, y_true, y_pred):
        # y_true is assumed to be 1 for positive, 0 for negative.
        pos_scores = y_pred[y_true == 1]
        neg_scores = y_pred[y_true == 0]
        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return torch.tensor(0.0, device=y_pred.device)
        pos_scores = pos_scores.unsqueeze(0)  # shape (1, num_pos)
        neg_scores = neg_scores.unsqueeze(1)  # shape (num_neg, 1)
        margin_matrix = self.margin - pos_scores + neg_scores
        loss = torch.sum(torch.clamp(margin_matrix, min=0))
        return loss

def combine_hole_embeddings(embed_real, embed_img):
    # As in the original code, perform elementwise multiplication and duplicate
    combined_real = embed_real * embed_img
    combined_img = embed_real * embed_img
    return torch.cat([combined_real, combined_img], dim=-1)

def flip_entities(triples):
    flipped_triples = []
    for triple in triples:
        s, p, o = triple
        # Flip subject and object entities
        flipped_triples.append((o, p, s))
    return np.array(flipped_triples)

def popularity_based_sampling(triples, num_entities, num_relations, batch_size=1000):
    entity_counter = Counter(triple[0] for triple in triples)
    relation_counter = Counter(triple[1] for triple in triples)

    # Rank entities and relations based on their frequency
    entity_ranks = rankdata([-entity_counter[entity] for entity in range(num_entities)])
    relation_ranks = rankdata([-relation_counter[relation] for relation in range(num_relations)])

    # Calculate probability distribution using ranks
    entity_probs = {entity: 1 / (entity_ranks[entity] + 1) for entity in range(num_entities)}
    relation_probs = {relation: 1 / (relation_ranks[relation] + 1) for relation in range(num_relations)}

    # Normalize probabilities
    entity_probs_sum = sum(entity_probs.values())
    relation_probs_sum = sum(relation_probs.values())
    entity_probs = {entity: prob / entity_probs_sum for entity, prob in entity_probs.items()}
    relation_probs = {relation: prob / relation_probs_sum for relation, prob in relation_probs.items()}

    sampled_entities = np.random.choice(num_entities, size=batch_size, p=list(entity_probs.values()))
    sampled_relations = np.random.choice(num_relations, size=batch_size, p=list(relation_probs.values()))

    return [(entity, relation, entity) for entity, relation in zip(sampled_entities, sampled_relations)]

# --------------------------
# Model definitions
# --------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, activation='relu'):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = None

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class ModelWithHoIE(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, hoie_embedding_dim, gcn_units, dropout_rate=0.5, l2_reg=0.01):
        super(ModelWithHoIE, self).__init__()
        # Embedding layers
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.hoie_embeddings = nn.Embedding(num_entities, hoie_embedding_dim * 2)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)
        self.hoie_embedding_dim = hoie_embedding_dim

        # Dense layers for HoIE
        self.hoie_dense1 = nn.Linear(hoie_embedding_dim * 2, hoie_embedding_dim)
        self.hoie_dense2 = nn.Linear(hoie_embedding_dim, hoie_embedding_dim)

        # GCN layers. The input dimension is the concatenation of:
        # s_embed, p_embed, o_embed (3 * embedding_dim) and s_hoie, o_hoie (2 * hoie_embedding_dim).
        gcn_input_dim = 3 * embedding_dim + 2 * hoie_embedding_dim
        self.gcn_layer1 = GCNLayer(gcn_input_dim, gcn_units, activation='relu')
        self.gcn_layer2 = GCNLayer(gcn_units, gcn_units, activation='relu')

        # 1D Convolution layers.
        # In PyTorch, Conv1d expects input shape (batch, channels, sequence_length).
        self.conv1d_1 = nn.Conv1d(in_channels=gcn_units, out_channels=128, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv1d_2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)

        # After the conv layers the feature map size is (batch, 256, 1)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(256, 512)
        self.batch_norm3 = nn.BatchNorm1d(512)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()
        # Note: L2 regularization can be applied via optimizer's weight_decay.

    def forward(self, inputs):
        # inputs: tensor of shape (batch, 3) where each row is (s_idx, p_idx, o_idx)
        s_idx = inputs[:, 0].long()
        p_idx = inputs[:, 1].long()
        o_idx = inputs[:, 2].long()

        s_embed = self.entity_embeddings(s_idx)
        p_embed = self.relation_embeddings(p_idx)
        o_embed = self.entity_embeddings(o_idx)

        # HoIE for subject
        hoie_embed_s = self.hoie_embeddings(s_idx)
        hoie_real_s = hoie_embed_s[:, :self.hoie_embedding_dim]
        hoie_img_s = hoie_embed_s[:, self.hoie_embedding_dim:]
        s_hoie = self.hoie_dense1(combine_hole_embeddings(hoie_real_s, hoie_img_s))
        s_hoie = self.hoie_dense2(s_hoie)

        # HoIE for object
        hoie_embed_o = self.hoie_embeddings(o_idx)
        hoie_real_o = hoie_embed_o[:, :self.hoie_embedding_dim]
        hoie_img_o = hoie_embed_o[:, self.hoie_embedding_dim:]
        o_hoie = self.hoie_dense1(combine_hole_embeddings(hoie_real_o, hoie_img_o))
        o_hoie = self.hoie_dense2(o_hoie)

        concatenated_embed = torch.cat([s_embed, p_embed, o_embed, s_hoie, o_hoie], dim=1)
        gcn_output1 = self.gcn_layer1(concatenated_embed)
        gcn_output2 = self.gcn_layer2(gcn_output1)  # shape: (batch, gcn_units)

        # Prepare for Conv1d: add a dimension so shape becomes (batch, channels, sequence_length)
        x = gcn_output2.unsqueeze(2)  # (batch, gcn_units, 1)

        x = self.conv1d_1(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)

        x = self.conv1d_2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)

        x = self.flatten(x)  # (batch, 256)
        x = self.dense1(x)
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        prediction = self.sigmoid(x)
        return prediction

# --------------------------
# Data preparation and splitting
# --------------------------
graph_file_path = "Eurostat_KG.ttl"
num_entities, num_relations, triples, entity2idx, idx2entity, relation2idx, idx2relation = preprocess_data(
    load_graph(graph_file_path))

# Generate negative examples
negative_triples = popularity_based_sampling(triples, num_entities, num_relations, batch_size=len(triples))
all_triples = np.vstack((triples, np.array(negative_triples)))
labels = np.concatenate([np.ones(len(triples)), np.zeros(len(negative_triples))])

# Augment by flipping entities
flipped_triples = flip_entities(all_triples)
augmented_triples = np.vstack((all_triples, flipped_triples))
positive_triples_set = set(map(tuple, triples))
augmented_labels = np.array([1 if tuple(triple) in positive_triples_set else 0 for triple in augmented_triples])

# Split data: train, validation, and hybrid loss (worked) using stratified splitting
stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_index, temp_index = next(stratified_splitter.split(augmented_triples, augmented_labels))
X_train_augmented = augmented_triples[train_index]
y_train_augmented = augmented_labels[train_index]
X_temp_augmented = augmented_triples[temp_index]
y_temp_augmented = augmented_labels[temp_index]

stratified_splitter_test_val = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
test_index, val_index = next(stratified_splitter_test_val.split(X_temp_augmented, y_temp_augmented))
X_test_augmented = X_temp_augmented[test_index]
y_test_augmented = y_temp_augmented[test_index]
X_val_augmented = X_temp_augmented[val_index]
y_val_augmented = y_temp_augmented[val_index]

# --------------------------
# Training configuration
# --------------------------
embedding_dim = 256
initial_learning_rate = 1e-4
num_epochs = 30
batch_size = 256

# Create the model. Note: we pass hoie_embedding_dim=embedding_dim and gcn_units=embedding_dim
model = ModelWithHoIE(num_entities, num_relations, embedding_dim, embedding_dim, gcn_units=embedding_dim)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Use binary cross entropy loss
criterion = nn.MarginRankingLoss()
optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-5)

# Create DataLoaders
train_dataset = TensorDataset(torch.tensor(X_train_augmented, dtype=torch.long),
                                torch.tensor(y_train_augmented, dtype=torch.float))
val_dataset = TensorDataset(torch.tensor(X_val_augmented, dtype=torch.long),
                              torch.tensor(y_val_augmented, dtype=torch.float))
test_dataset = TensorDataset(torch.tensor(X_test_augmented, dtype=torch.long),
                               torch.tensor(y_test_augmented, dtype=torch.float))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# --------------------------
# Training loop with early stopping
# --------------------------
best_val_loss = float("inf")
epochs_no_improve = 5
patience = 8

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    train_loss = running_loss / len(train_dataset)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            outputs = outputs.view(-1)
            loss = criterion(outputs, targets)
            running_val_loss += loss.item() * inputs.size(0)
    val_loss = running_val_loss / len(val_dataset)
    val_losses.append(val_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    scheduler.step(val_loss)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        # Save best model state
        best_model_state = model.state_dict()
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered!")
            break

# Restore best model weights
model.load_state_dict(best_model_state)

# --------------------------
# Evaluation on hybrid loss (worked) set
# --------------------------
model.eval()
all_test_outputs = []
all_test_targets = []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = outputs.view(-1).cpu().numpy()
        all_test_outputs.extend(outputs)
        all_test_targets.extend(targets.numpy())
all_test_outputs = np.array(all_test_outputs)
all_test_targets = np.array(all_test_targets)

# Threshold predictions at 0.5
y_pred = (all_test_outputs > 0.5).astype(int)
y_true = all_test_targets.astype(int)

# Calculate evaluation metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, all_test_outputs)
pr_auc = average_precision_score(y_true, all_test_outputs)
mcc = matthews_corrcoef(y_true, y_pred)

print("\nEvaluation Metrics:")
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'ROC AUC: {roc_auc:.4f}')
print(f'PR AUC: {pr_auc:.4f}')
print(f'MCC: {mcc:.4f}')

# --------------------------
# Plot training vs validation loss
# --------------------------
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
