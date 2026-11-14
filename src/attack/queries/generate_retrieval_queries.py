"""Generate train/test skill-retrieval queries for SkillsBench tasks."""

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[3]

load_dotenv(ROOT / ".env")

GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")

CATALOG = ROOT / "data/processed/skills_catalog.json"
RAW = ROOT / "data/raw/skillsbench"
OUTPUT = ROOT / "data/processed/retrieval_queries.json"

SEED = 42
N_TASKS = 10
N_QUERIES = 10


def task_file(task_id):
    """Support both tasks/<id>/task.md and <id>/task.md layouts."""
    for path in (RAW / "tasks" / task_id / "task.md", RAW / task_id / "task.md"):
        if path.is_file():
            return path
    return None


def main():
    if not CATALOG.is_file():
        raise FileNotFoundError(f"Missing {CATALOG}; run python src/load_skills.py")
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    if isinstance(catalog, dict):
        catalog = catalog.get("skills", catalog.get("catalog", []))

    # Deduplicate task IDs; several skills may belong to the same task.
    tasks = {}
    for skill in catalog:
        task_id = skill["task"]
        tasks.setdefault(task_id, []).append({
            "name": skill["name"], "description": skill["description"]
        })
    selected = list(tasks.items())[:N_TASKS]
    if len(selected) < N_TASKS:
        raise ValueError(f"Found only {len(selected)} distinct tasks")

    client = OpenAI()
    rng = random.Random(SEED)
    results = []

    for task_id, skills in selected:
        path = task_file(task_id)
        if path is None:
            raise FileNotFoundError(f"Cannot find task.md for {task_id} under {RAW}")
        instruction = path.read_text(encoding="utf-8")
        prompt = (
            "Generate exactly 10 distinct, realistic user requests for discovering an agent skill "
            "that would help accomplish the task below. Preserve its actual capabilities; "
            "vary wording and specificity without inventing new capabilities. "
            "Write user requests, not skill descriptions. Do not mention SkillsBench. "
            'Return JSON only: {"queries": ["...", ...]}\n\n'
            f"TASK ID: {task_id}\nTASK INSTRUCTIONS:\n{instruction[:10000]}\n\n"
            f"ASSOCIATED SKILLS:\n{json.dumps(skills, ensure_ascii=False)[:5000]}"
        )
        response = client.chat.completions.create(
            model=GEN_MODEL,
            temperature=0.7,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You generate realistic, diverse benchmark user queries."},
                {"role": "user", "content": prompt},
            ],
        )
        raw_queries = json.loads(response.choices[0].message.content)["queries"]
        queries = list(dict.fromkeys(
            q.strip() for q in raw_queries if isinstance(q, str) and q.strip()
        ))
        if len(queries) != N_QUERIES:
            raise ValueError(f"{task_id}: expected 10 unique queries, got {len(queries)}; rerun")
        rng.shuffle(queries)
        results.append({
            "task_id": task_id,
            "task_file": str(path.relative_to(ROOT)),
            "skills": skills,
            "train_queries": queries[:5],
            "test_queries": queries[5:],
        })
        print(f"{len(results)}/{N_TASKS}: {task_id} — 5 train, 5 test", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "model": GEN_MODEL, "seed": SEED, "query_type": "skill_retrieval", "tasks": results
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {N_TASKS * N_QUERIES} queries to {OUTPUT}")


if __name__ == "__main__":
    main()
