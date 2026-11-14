"""Generate task-execution queries from SkillsBench, separate from retrieval queries."""

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
OUTPUT = ROOT / "data/processed/usage_queries.json"

SEED = 42
N_TASKS = 10
N_QUERIES = 10


def task_file(task_id):
    for path in (RAW / 'tasks' / task_id / 'task.md', RAW / task_id / 'task.md'):
        if path.is_file():
            return path
    return None


def main():
    if not CATALOG.is_file():
        raise FileNotFoundError(f'Missing {CATALOG}; run python src/load_skills.py')
    catalog = json.loads(CATALOG.read_text(encoding='utf-8'))
    if isinstance(catalog, dict):
        catalog = catalog.get('skills', catalog.get('catalog', []))

    tasks = {}
    for skill in catalog:
        tasks.setdefault(skill['task'], []).append({
            'name': skill['name'], 'description': skill['description']
        })

    client = OpenAI()
    rng = random.Random(SEED)
    results = []

    for task_id, skills in list(tasks.items())[:N_TASKS]:
        path = task_file(task_id)
        if path is None:
            raise FileNotFoundError(f'Missing task.md for {task_id}')
        original = path.read_text(encoding='utf-8')
        prompt = (
            'Generate exactly 10 distinct USER TASK REQUESTS for an AI agent to EXECUTE '
            'the task below, not to discover or recommend a skill. Each request must '
            'preserve the original task goal, essential constraints, named inputs, '
            'file paths, and expected outputs. Do not invent resources, files, data, '
            'or credentials. Vary wording, not the benchmark problem. '
            'Do not mention SkillsBench. '
            'Return JSON only: {"queries": ["...", "..."]}\n\n'
            f'TASK ID: {task_id}\nORIGINAL TASK:\n{original[:16000]}\n\n'
            f'ASSOCIATED SKILLS (context only):\n{json.dumps(skills, ensure_ascii=False)[:5000]}'
        )
        response = client.chat.completions.create(
            model=GEN_MODEL, temperature=0.7,
            response_format={'type': 'json_object'},
            messages=[
                {'role': 'system', 'content': 'Write faithful, executable paraphrases of benchmark user tasks. Never change task requirements.'},
                {'role': 'user', 'content': prompt},
            ],
        )
        queries = json.loads(response.choices[0].message.content)['queries']
        queries = list(dict.fromkeys(q.strip() for q in queries if isinstance(q, str) and q.strip()))
        if len(queries) != N_QUERIES:
            raise ValueError(f'{task_id}: expected 10 unique queries, got {len(queries)}')
        rng.shuffle(queries)
        results.append({
            'task_id': task_id,
            'task_file': str(path.relative_to(ROOT)),
            'skills': skills,
            'original_task': original,
            'train_queries': queries[:5],
            'test_queries': queries[5:],
        })
        print(f'{len(results)}/{N_TASKS}: {task_id} — 5 train, 5 test', flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        'model': GEN_MODEL, 'seed': SEED, 'query_type': 'skill_usage', 'tasks': results
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Saved {len(results) * N_QUERIES} queries to {OUTPUT}')


if __name__ == '__main__':
    main()
