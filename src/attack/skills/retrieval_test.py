"""Evaluate adversarial skill retrieval."""

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/processed"
CANDIDATES = ROOT / "results/adversarial_skills/generated_skills_candidates.json"
OUTPUT = ROOT / "results/retrieval"
MODELS = ("text-embedding-3-small", "nomic-embed-text", "all-MiniLM-L6-v2")
COUNTS = (1, 3, 5, 10, 15, 30)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def embed_separately(texts, model):
    """Keep PyTorch and FAISS in different processes to avoid macOS OpenMP conflicts."""
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "texts.json", Path(tmp) / "vectors.npy"
        save(inp, texts)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
        subprocess.run([sys.executable, str(Path(__file__).with_name("create_embeddings.py")),
                        "--model", model, "--input", str(inp), "--output", str(out)],
                       check=True, env=env)
        vectors = np.ascontiguousarray(np.load(out), dtype=np.float32)
    faiss.normalize_L2(vectors)
    return vectors


def index(vectors):
    result = faiss.IndexFlatIP(vectors.shape[1])
    result.add(np.ascontiguousarray(vectors))
    return result


def greedy(attack, train, clean_scores):
    scores = train @ attack.T
    captured = scores > clean_scores[:, None]
    order, covered = [], np.zeros(len(train), dtype=bool)
    while True:
        options = []
        for i in range(len(attack)):
            if i in order:
                continue
            new = captured[:, i] & ~covered
            gain = int(new.sum())
            if gain:
                options.append((gain, float((scores[new, i] - clean_scores[new]).sum()), -i))
        if not options:
            return order
        i = -max(options)[2]
        order.append(i)
        covered |= captured[:, i]


def evaluate(model, catalog, candidates, train, test):
    folder = OUTPUT / model
    folder.mkdir(parents=True, exist_ok=True)
    texts = ([f"{s['name']}: {s['description']}" for s in catalog + candidates]
             + [q for _, q in train + test])
    vectors = embed_separately(texts, model)
    n, a, t = len(catalog), len(candidates), len(train)
    clean, attack = vectors[:n], vectors[n:n+a]
    train_vec, test_vec = vectors[n+a:n+a+t], vectors[n+a+t:]
    baseline = index(clean)
    train_scores, _ = baseline.search(train_vec, 1)
    clean_scores, clean_ids = baseline.search(test_vec, 1)
    order = greedy(attack, train_vec, train_scores[:, 0])
    rows, summaries, selections = [], [], {}

    for count in COUNTS:
        selected = order[:count]
        selections[str(count)] = [candidates[i] for i in selected]
        attacked = index(np.concatenate((clean, attack[selected])))
        scores, ids = attacked.search(test_vec, 1)
        wins = int((ids[:, 0] >= n).sum())
        train_wins = int(((train_vec @ attack[selected].T).max(axis=1)
                          > train_scores[:, 0]).sum()) if selected else 0
        for j, (task_id, query) in enumerate(test):
            winner = int(ids[j, 0])
            source = "adversarial" if winner >= n else "legitimate"
            skill = candidates[selected[winner-n]] if winner >= n else catalog[winner]
            rows.append(dict(skill_count=count, actual_skill_count=len(selected),
                             task_id=task_id, query=query,
                             clean_top1=catalog[int(clean_ids[j, 0])]["name"],
                             clean_similarity=round(float(clean_scores[j, 0]), 6),
                             attacked_top1=skill["name"], attacked_source=source,
                             attacked_similarity=round(float(scores[j, 0]), 6),
                             hijacked=source == "adversarial"))
        summaries.append(dict(skill_count=count, selected=len(selected),
                              training_hijacked=train_wins, test_hijacked=wins,
                              test_rhr_at_1=wins / len(test)))
        print(f"Skills {count:>2}: selected {len(selected):>2}, "
              f"train {train_wins}/{len(train)}, "
              f"test RHR@1 {wins}/{len(test)} = {wins/len(test):.1%}", flush=True)

    # Persist only the largest index; selections for smaller counts are in JSON.
    selected = order[:max(COUNTS)]
    faiss.write_index(index(np.concatenate((clean, attack[selected]))), str(folder / "index.faiss"))
    metadata = ([dict(source="legitimate", **{k: s.get(k) for k in ("id", "name", "description")})
                 for s in catalog]
                + [dict(source="adversarial", candidate_index=i, **candidates[i])
                   for i in selected])
    save(folder / "metadata.json", dict(embedding_model=model, skills=metadata))
    save(folder / "selected_skills.json", selections)
    save(folder / "summary.json", dict(embedding_model=model, train_queries=len(train),
                                       test_queries=len(test), skill_counts=summaries))
    with (folder / "retrieval_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {folder}")


def main():
    tasks = read(DATA / "retrieval_queries.json")["tasks"][:10]
    raw = read(DATA / "skills_catalog.json")
    catalog = raw.get("skills", raw.get("catalog", [])) if isinstance(raw, dict) else raw
    candidates = read(CANDIDATES)
    train = [(t["task_id"], q) for t in tasks for q in t["train_queries"]]
    test = [(t["task_id"], q) for t in tasks for q in t["test_queries"]]
    if not catalog or not candidates or not train or not test:
        raise ValueError("Empty catalog, candidates, or query set")
    if {q.strip().casefold() for _, q in train} & {q.strip().casefold() for _, q in test}:
        raise ValueError("Train/test query overlap")
    print(f"Using {len(candidates)} cached candidates")
    for model in MODELS:
        print(f"\n=== {model} ===", flush=True)
        evaluate(model, catalog, candidates, train, test)


if __name__ == "__main__":
    main()
