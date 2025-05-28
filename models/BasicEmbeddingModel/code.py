###############################################
# CONFIGURATION
###############################################
import os
import random
import time
import numpy as np
import torch

# General parameters
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directory parameters
BASE_DIR = "models"
LOGS_DIR = "logs"
PLOTS_DIR = "plots"
MODEL_NAME = "BasicEmbeddingModel"
FULL_MODEL_DIR = os.path.join(BASE_DIR, MODEL_NAME)

# Training parameters
EPOCHS = 10
BATCH_SIZE = 512
LEARNING_RATE = 0.00001  # same as 1e-5

# Graph parameters
GRAPH_FILE_PATH = "Eurostat_KG.ttl"
###############################################
# END CONFIGURATION
###############################################

import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from rdflib import Graph
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from tqdm import tqdm

# Ensure deterministic behavior & set device
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

print(f"Using device: {DEVICE}")

# Create directories for saving results using configuration
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(FULL_MODEL_DIR, exist_ok=True)


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


# **Modified TransE Model for Classification**
class BasicEmbeddingModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim):
        super(BasicEmbeddingModel, self).__init__()

        self.embedding_dim = embedding_dim

        # Simple embeddings (no TransE transformations)
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Initialize embeddings using Xavier initialization
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)

        # Basic MLP classifier (No Dropout)
        self.fc1 = nn.Linear(embedding_dim * 3, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, h, r, t):
        h_embed = self.entity_embeddings(h)
        r_embed = self.relation_embeddings(r)
        t_embed = self.entity_embeddings(t)

        # Concatenate embeddings instead of TransE distance
        x = torch.cat([h_embed, r_embed, t_embed], dim=1)

        # Simple MLP classifier (No Dropout)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        output = torch.sigmoid(self.fc3(x))  # Binary classification output

        return output


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
def train_model(model, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    model.to(DEVICE)
    model.train()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        perm = torch.randperm(X_train.shape[0])
        X_train, y_train = X_train[perm], y_train[perm]

        tqdm_bar = tqdm(range(0, len(X_train), batch_size), desc=f"Epoch {epoch + 1}/{epochs}", leave=False)

        for i in tqdm_bar:
            batch_X = X_train[i:i + batch_size].to(DEVICE)
            batch_y = y_train[i:i + batch_size].to(DEVICE)

            optimizer.zero_grad()
            h, r, t = batch_X[:, 0], batch_X[:, 1], batch_X[:, 2]
            preds = model(h, r, t)

            loss = loss_fn(preds.view(-1), batch_y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            tqdm_bar.set_postfix(loss=epoch_loss)

        # Compute validation loss
        with torch.no_grad():
            val_preds = model(X_val[:, 0].to(DEVICE), X_val[:, 1].to(DEVICE), X_val[:, 2].to(DEVICE))
            val_loss = loss_fn(val_preds.view(-1), y_val.to(DEVICE).view(-1)).item()

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

    model_results_dir = FULL_MODEL_DIR
    os.makedirs(model_results_dir, exist_ok=True)
    plots_dir = os.path.join(model_results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # **Save training and validation losses**
    np.save(os.path.join(model_results_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(model_results_dir, "val_losses.npy"), np.array(val_losses))

    print(f"Training & Validation losses saved to {model_results_dir}/")

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Training vs Validation Loss - {MODEL_NAME}")
    loss_plot_path = os.path.join(plots_dir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.close()

    return model


# Train Classification Model
embedding_dim = 128
classifier = BasicEmbeddingModel(num_entities, num_relations, embedding_dim).to(DEVICE)
trained_classifier = train_model(classifier, X_train, y_train, X_val, y_val)

os.makedirs(FULL_MODEL_DIR, exist_ok=True)
torch.save(trained_classifier.state_dict(), os.path.join(FULL_MODEL_DIR, "classifier.pth"))
trained_classifier.load_state_dict(torch.load(os.path.join(FULL_MODEL_DIR, "classifier.pth"), map_location=DEVICE))


# Evaluate Model
import json
from sklearn.metrics import accuracy_score, matthews_corrcoef, balanced_accuracy_score, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
from sklearn.manifold import TSNE

def generate_evaluation_plots(y_true, y_scores, y_pred, train_losses, val_losses, model_name=MODEL_NAME):
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

    # **Prediction Score Distribution**
    plt.figure(figsize=(8, 6))
    sns.histplot(y_scores, bins=30, kde=True, color="blue")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title(f"Prediction Score Distribution - {model_name}")
    plt.savefig(os.path.join(plots_dir, "prediction_distribution.png"))
    plt.close()
    print(f" Prediction Distribution saved to {os.path.join(plots_dir, 'prediction_distribution.png')}")

    # **Confusion Matrix**
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
    plt.close()
    print(f" Confusion Matrix saved to {os.path.join(plots_dir, 'confusion_matrix.png')}")


def generate_advanced_plots(y_true, y_scores, y_pred, train_losses, val_losses, model, model_name=MODEL_NAME):
    """
    Generate and save multiple evaluation plots including:
    - Smoothed Loss Curve
    - Precision-Recall Tradeoff (Optimized)
    - F1 Score by Threshold (Optimized)
    - Distribution of Predictions (Optimized)
    - Embedding Visualization (Optimized t-SNE)
    """
    plots_dir = os.path.join(FULL_MODEL_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    ### ** Smoothed Loss Curve**
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss", alpha=0.7)
    plt.plot(val_losses, label="Validation Loss", alpha=0.7)

    # Apply Moving Average Smoothing
    smoothed_train = np.convolve(train_losses, np.ones(3)/3, mode="valid")
    smoothed_val = np.convolve(val_losses, np.ones(3)/3, mode="valid")
    plt.plot(smoothed_train, label="Smoothed Train Loss", linestyle="--")
    plt.plot(smoothed_val, label="Smoothed Val Loss", linestyle="--")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Smoothed Learning Curve - {model_name}")
    plt.savefig(os.path.join(plots_dir, "smoothed_learning_curve.png"))
    plt.close()
    print(f" Smoothed Loss Curve saved to {os.path.join(plots_dir, 'smoothed_learning_curve.png')}")

    ### ** Precision-Recall Tradeoff**
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"Precision-Recall Tradeoff - {model_name}")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "precision_recall_tradeoff.png"))
    plt.close()
    print(f" Precision-Recall Tradeoff saved to {os.path.join(plots_dir, 'precision_recall_tradeoff.png')}")

    ### ** Optimized F1 Score by Threshold**
    num_thresholds = 50
    threshold_samples = np.linspace(0, 1, num_thresholds)
    f1_scores = [f1_score(y_true, y_scores >= t) for t in threshold_samples]

    plt.figure(figsize=(8, 6))
    plt.plot(threshold_samples, f1_scores, label="F1 Score", color="green")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title(f"F1 Score by Classification Threshold - {model_name}")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "f1_by_threshold.png"))
    plt.close()
    print(f" F1 Score by Threshold saved to {os.path.join(plots_dir, 'f1_by_threshold.png')}")

    ### ** Optimized Prediction Score Distribution**
    plt.figure(figsize=(10, 6))

    pos_color = "#FF5733"
    neg_color = "#1F77B4"
    kde_color_pos = "#C70039"
    kde_color_neg = "#0E4D92"

    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]

    plt.hist(pos_scores, bins=30, alpha=0.5, color=pos_color, edgecolor="black", linewidth=1.2, label="Positive Class")
    plt.hist(neg_scores, bins=30, alpha=0.5, color=neg_color, edgecolor="black", linewidth=1.2, label="Negative Class")

    sns.kdeplot(pos_scores, color=kde_color_pos, linewidth=2, linestyle="--", label="Positive Density")
    sns.kdeplot(neg_scores, color=kde_color_neg, linewidth=2, linestyle="--", label="Negative Density")

    plt.xlabel("Predicted Probability", fontsize=14, fontweight="bold")
    plt.ylabel("Frequency", fontsize=14, fontweight="bold")
    plt.title(f"Prediction Score Distribution - {model_name}", fontsize=16, fontweight="bold", pad=15)
    plt.legend(fontsize=12, loc="upper center")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(os.path.join(plots_dir, "prediction_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f" Fixed Prediction Distribution saved to {os.path.join(plots_dir, 'prediction_distribution.png')}")

    ### ** Optimized t-SNE Embeddings**
    entity_embeddings = model.entity_embeddings.weight.cpu().detach().numpy()
    max_samples = 5000
    if entity_embeddings.shape[0] > max_samples:
        sampled_indices = np.random.choice(entity_embeddings.shape[0], max_samples, replace=False)
        entity_embeddings = entity_embeddings[sampled_indices]

    tsne = TSNE(n_components=2, perplexity=30 if entity_embeddings.shape[0] > 100 else 5, random_state=SEED)
    reduced_embeddings = tsne.fit_transform(entity_embeddings)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.6)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title(f"t-SNE Visualization of Entity Embeddings - {model_name}")
    plt.savefig(os.path.join(plots_dir, "tsne_embeddings.png"))
    plt.close()
    print(f" t-SNE Entity Embeddings saved to {os.path.join(plots_dir, 'tsne_embeddings.png')}")


def evaluate_model(model, X_test, y_test, model_name=MODEL_NAME):
    model.eval()
    with torch.no_grad():
        h, r, t = X_test[:, 0].to(DEVICE), X_test[:, 1].to(DEVICE), X_test[:, 2].to(DEVICE)
        predictions = model(h, r, t).cpu()
        y_test_cpu = y_test.cpu()
        y_pred = (predictions > 0.5).float().cpu()

        accuracy = accuracy_score(y_test_cpu, y_pred)
        balanced_acc = balanced_accuracy_score(y_test_cpu, y_pred)
        precision = precision_score(y_test_cpu, y_pred)
        recall = recall_score(y_test_cpu, y_pred)
        f1 = f1_score(y_test_cpu, y_pred)
        roc_auc = roc_auc_score(y_test_cpu, predictions)
        pr_auc = average_precision_score(y_test_cpu, predictions)
        mcc = matthews_corrcoef(y_test_cpu, y_pred)

        conf_matrix = confusion_matrix(y_test_cpu, y_pred)

        print(f"\nEvaluation Metrics for {model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        print(f"MCC: {mcc:.4f}")

        model_results_dir = FULL_MODEL_DIR
        logs_dir = os.path.join(model_results_dir, "logs")
        plots_dir = os.path.join(model_results_dir, "plots")
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        metrics_dict = {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "mcc": mcc
        }

        metrics_file = os.path.join(logs_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics_dict, f, indent=4)

        print(f"Metrics saved to {metrics_file}")

        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix - {model_name}")
        conf_matrix_path = os.path.join(plots_dir, "confusion_matrix.png")
        plt.savefig(conf_matrix_path)
        plt.close()
        print(f"Confusion Matrix saved to {conf_matrix_path}")

        train_losses = np.load(os.path.join(model_results_dir, "train_losses.npy"))
        val_losses = np.load(os.path.join(model_results_dir, "val_losses.npy"))

        generate_evaluation_plots(
            y_true=y_test_cpu.numpy(),
            y_scores=predictions.numpy(),
            y_pred=y_pred.numpy(),
            train_losses=train_losses,
            val_losses=val_losses,
            model_name=model_name
        )

        generate_advanced_plots(
            y_true=y_test_cpu.numpy(),
            y_scores=predictions.numpy(),
            y_pred=y_pred.numpy(),
            train_losses=train_losses,
            val_losses=val_losses,
            model=trained_classifier,
            model_name=model_name
        )

        return metrics_dict


# Call the function and save results
model_metrics = evaluate_model(trained_classifier, X_test, y_test, model_name=MODEL_NAME)
