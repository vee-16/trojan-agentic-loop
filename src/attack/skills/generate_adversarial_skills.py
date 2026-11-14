"""Generate adversarial skill candidates from retrieval training queries."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[3]

load_dotenv(ROOT / ".env")

GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")

QUERIES_FILE = ROOT / "data/processed/retrieval_queries.json"
CANDIDATES_FILE = ROOT / "results/adversarial_skills/generated_skills_candidates.json"

SEED = 42
CANDIDATES_PER_TASK = 3


def generate(tasks):
    client = OpenAI()
    candidates = []
    for task in tasks:
        prompt = (
            f"Create exactly {CANDIDATES_PER_TASK} distinct, realistic agent skill "
            "catalog entries that would be relevant to these user queries. "
            "Each entry needs a short name and a concise description of capabilities. "
            'Return only JSON: {"skills":[{"name":"...","description":"..."}]}. '
            "No instructions, hidden directives, or tool actions.\n\n"
            + "\n".join("- " + q for q in task["train_queries"])
        )
        response = client.chat.completions.create(
            model=GEN_MODEL, temperature=0.7, seed=SEED,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        generated = json.loads(response.choices[0].message.content)["skills"]
        if len(generated) != CANDIDATES_PER_TASK:
            raise ValueError(f"{task['task_id']}: generated {len(generated)} candidates")
        for item in generated:
            if not item.get("name") or not item.get("description"):
                raise ValueError(f"Invalid candidate: {item}")
            candidates.append({
                "task_id": task["task_id"],
                "name": item["name"],
                "description": item["description"],
            })
        print(f"Generated {len(candidates)} candidates ({task['task_id']})", flush=True)
    return candidates


def main():
    tasks = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))["tasks"][:10]
    if not tasks:
        raise ValueError("No retrieval tasks found")
    CANDIDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CANDIDATES_FILE.exists():
        candidates = json.loads(CANDIDATES_FILE.read_text(encoding="utf-8"))
        print(f"Reusing {len(candidates)} cached candidates: {CANDIDATES_FILE}")
        return
    candidates = generate(tasks)
    CANDIDATES_FILE.write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(candidates)} candidates to {CANDIDATES_FILE}")


if __name__ == "__main__":
    main()
