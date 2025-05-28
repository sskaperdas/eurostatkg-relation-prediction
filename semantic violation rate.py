import os
import json
import torch
from model_file import TransEModel  # Replace with your actual model class

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load mappings
with open("entity_to_types.json") as f:
    entity_to_types = {k: set(v) for k, v in json.load(f).items()}
with open("relation_to_domain.json") as f:
    relation_to_domain = json.load(f)
with open("relation_to_range.json") as f:
    relation_to_range = json.load(f)

# Define violation checker
def is_semantic_violation(h_id, r_id, t_id):
    h_types = entity_to_types.get(str(h_id), set())
    t_types = entity_to_types.get(str(t_id), set())
    domain = relation_to_domain.get(str(r_id))
    range_ = relation_to_range.get(str(r_id))
    return (domain and domain not in h_types) or (range_ and range_ not in t_types)

# Load dataset
X_test = torch.load("models/data/X_test.pt").to(DEVICE)
y_test = torch.load("models/data/y_test.pt").to(DEVICE)
X_cpu = X_test.cpu()
y_true = y_test.cpu()

# Search root
MODELS_ROOT = "models"

# Walk and find all model folders
for root, dirs, files in os.walk(MODELS_ROOT):
    if "classifier.pth" in files:
        model_path = os.path.join(root, "classifier.pth")
        print(f"📦 Evaluating model: {model_path}")

        # === Load model dynamically ===
        try:
            # You may need to set this manually depending on model variant
            model = TransEModel(num_entities=59938, num_relations=43, embedding_dim=128)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
        except Exception as e:
            print(f"⚠️ Failed to load model at {model_path}: {e}")
            continue

        # === Batch predictions ===
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(X_test), 512):
                batch = X_test[i:i + 512]
                h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
                pred = model(h, r, t).cpu()
                all_preds.append(pred)

        predictions = torch.cat(all_preds, dim=0)

        # === Evaluate thresholds ===
        from sklearn.metrics import *
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        results = {}
        for threshold in thresholds:
            y_pred = (predictions > threshold).float()
            acc = accuracy_score(y_true, y_pred)
            bal_acc = balanced_accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            roc = roc_auc_score(y_true, predictions)
            pr = average_precision_score(y_true, predictions)
            mcc = matthews_corrcoef(y_true, y_pred)

            # Compute semantic violation rate
            positive_triples = X_cpu[y_pred.bool()]
            violations = sum(is_semantic_violation(h, r, t) for h, r, t in positive_triples.tolist())
            svr = violations / max(1, len(positive_triples))

            results[threshold] = {
                "accuracy": acc,
                "balanced_accuracy": bal_acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1,
                "roc_auc": roc,
                "pr_auc": pr,
                "mcc": mcc,
                "semantic_violation_rate": svr
            }

        # Save results
        out_dir = os.path.join(root, "logs")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "eval_with_violation_rate.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"✅ Saved SVR results to: {out_path}")
