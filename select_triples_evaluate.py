import torch
import pickle

def clean_for_paper(text, max_words=4):
    if text.startswith("http"):
        return text.split("/")[-1].split("#")[-1]
    words = text.strip().split()
    return "_".join(words[:max_words]) + ("..." if len(words) > max_words else "")

def soft_rank(score, all_scores, epsilon=0.01, k=1):
    better_scores = (all_scores > score + epsilon).sum().item()
    rank = better_scores + 1
    return rank <= k

def evaluate_triples(model, triples, idx2entity, idx2relation, device='cuda', epsilon=0.01):
    model.eval()
    results, ranks, soft_hits = [], [], 0
    num_relations = model.relation_embeddings.num_embeddings

    with torch.no_grad():
        for h, r, t in triples:
            h = torch.tensor(h).to(device)
            r = torch.tensor(r).to(device)
            t = torch.tensor(t).to(device)

            all_r = torch.arange(num_relations).to(device)
            h_batch = h * torch.ones_like(all_r)
            t_batch = t * torch.ones_like(all_r)

            all_scores = model(h_batch, all_r, t_batch).squeeze()
            true_score = all_scores[r].item()
            rank = (torch.argsort(all_scores, descending=True) == r).nonzero(as_tuple=True)[0].item() + 1

            soft_hit = soft_rank(true_score, all_scores, epsilon, k=1)
            soft_hits += int(soft_hit)

            h_str = str(idx2entity[h.item()])
            r_str = str(idx2relation[r.item()])
            t_str = str(idx2entity[t.item()])
            results.append((h_str, r_str, t_str, true_score, rank))
            ranks.append(rank)

    avg_rank = sum(ranks) / len(ranks) if ranks else float('nan')
    soft_hits_ratio = soft_hits / len(triples) if triples else 0.0
    return results, avg_rank, soft_hits_ratio

def print_evaluation(triples_with_scores, avg_rank, soft_ratio, epsilon, label):
    print(f"\n{'=' * 15} {label.upper()} TRIPLES {'=' * 15}")
    for h, r, t, score, rank in triples_with_scores:
        print(f"{clean_for_paper(h)} --[{clean_for_paper(r)}]--> {clean_for_paper(t)}  | score={score:.4f}, rank={rank}")
    print(f"📊 Avg Rank: {avg_rank:.2f} | Soft@1 (ε={epsilon}): {soft_ratio:.2%}")

# === Load triples from disk ===
with open("cached_triples.pkl", "rb") as f:
    cached = pickle.load(f)

idx2entity = cached["idx2entity"]
idx2relation = cached["idx2relation"]
entity2idx = {str(uri): i for i, uri in enumerate(idx2entity)}
relation2idx = {str(uri): i for i, uri in enumerate(idx2relation)}

epsilon = 0.01

# === Evaluate all categories ===
for label in ["correct", "wrong", "violating"]:
    if label not in cached:
        continue
    print(f"\n=== Evaluating {label.upper()} ===")
    triples_uris = cached[label]
    indexed_triples = [(entity2idx[h], relation2idx[r], entity2idx[t]) for (h, r, t) in triples_uris]
    results, avg_rank, soft_hits = evaluate_triples(
        trained_classifier, indexed_triples, idx2entity, idx2relation, device='cuda', epsilon=epsilon
    )
    print_evaluation(results, avg_rank, soft_hits, epsilon, label)
