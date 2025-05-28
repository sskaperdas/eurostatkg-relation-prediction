import pickle
import random
import numpy as np
import torch

SEED = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def score_triples(model, triples):
    model.eval()
    with torch.no_grad():
        h = torch.tensor([h for h, _, _ in triples], dtype=torch.long).to(model.entity_embeddings.weight.device)
        r = torch.tensor([r for _, r, _ in triples], dtype=torch.long).to(model.entity_embeddings.weight.device)
        t = torch.tensor([t for _, _, t in triples], dtype=torch.long).to(model.entity_embeddings.weight.device)
        scores = model.score_triple(h, r, t).cpu().numpy()
    return scores

def select_top_and_bottom_triples(X_test, y_test, model, top_k=3, bottom_k=3):
    triples = [(int(h), int(r), int(t)) for (h, r, t) in X_test.cpu().numpy()]
    labels = y_test.cpu().numpy()

    # Score all triples once
    scores = score_triples(model, triples)

    # Pair each triple with its score and label
    scored = list(zip(triples, scores, labels))

    # Filter and sort by score
    correct_sorted = sorted([x for x in scored if x[2] == 1], key=lambda x: -x[1])
    wrong_sorted = sorted([x for x in scored if x[2] == 0], key=lambda x: x[1])

    correct = [x[0] for x in correct_sorted[:top_k]]
    wrong = [x[0] for x in wrong_sorted[:bottom_k]]

    return correct, wrong

def sample_violating_triples(X_test, y_test, idx2entity, idx2relation, entity_type_map, constraints, max_samples=3):
    violating = []
    for (h, r, t), y in zip(X_test, y_test):
        if y != 1:
            continue
        h_uri = idx2entity[int(h)]
        r_uri = idx2relation[int(r)]
        t_uri = idx2entity[int(t)]

        domain, range_ = constraints.get(r_uri, (None, None))
        h_types = entity_type_map.get(h_uri, set())
        t_types = entity_type_map.get(t_uri, set())

        is_violation = (domain and domain not in h_types) or (range_ and range_ not in t_types)
        if is_violation:
            violating.append((int(h), int(r), int(t)))
        if len(violating) >= max_samples:
            break

    return violating

def to_uris(triples, idx2entity, idx2relation):
    return [(str(idx2entity[h]), str(idx2relation[r]), str(idx2entity[t])) for (h, r, t) in triples]

def ensure_no_overlap(a, b, c):
    set_a = set(a)
    set_b = set(b)
    set_c = set(c)
    assert set_a.isdisjoint(set_b), "Overlap between correct and wrong!"
    assert set_a.isdisjoint(set_c), "Overlap between correct and violating!"
    assert set_b.isdisjoint(set_c), "Overlap between wrong and violating!"

# === Inputs: assume all of these are already loaded/defined ===
# X_test, y_test, idx2entity, idx2relation, entity_type_map, constraints, trained_classifier

# === Main Selection Logic ===
correct_triples, wrong_triples = select_top_and_bottom_triples(X_test, y_test, trained_classifier, top_k=3, bottom_k=3)
violating_triples = sample_violating_triples(X_test, y_test, idx2entity, idx2relation, entity_type_map, constraints, max_samples=3)

correct_uris = to_uris(correct_triples, idx2entity, idx2relation)
wrong_uris = to_uris(wrong_triples, idx2entity, idx2relation)
violating_uris = to_uris(violating_triples, idx2entity, idx2relation)

ensure_no_overlap(correct_triples, wrong_triples, violating_triples)

with open("cached_triples.pkl", "wb") as f:
    pickle.dump({
        "correct": correct_uris,
        "wrong": wrong_uris,
        "violating": violating_uris,
        "idx2entity": idx2entity,
        "idx2relation": idx2relation
    }, f)

print("\n✅ Saved top correct, bottom wrong, and violating triples to cached_triples.pkl")
