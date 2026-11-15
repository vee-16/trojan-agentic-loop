"""Adversarial skill metadata generation."""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from create_embeddings import embed

ROOT = Path(__file__).resolve().parents[3]

load_dotenv(ROOT / ".env")

GEN_MODEL = os.getenv("GEN_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("ATTACK_EMBED_MODEL", "text-embedding-3-small")

QUERIES_FILE = ROOT / "data/processed/retrieval_queries.json"
CANDIDATES_FILE = ROOT / "results/adversarial_skills/generated_skills_candidates.json"
METRICS_FILE = ROOT / "results/adversarial_skills/skill_coverage_metrics.json"

SEED = 42
NUM_TASKS = 10

class SkillEntry(BaseModel):
    name: str
    description: str


class GeneratedSkills(BaseModel):
    skills: list[SkillEntry]


@dataclass
class AttackConfig:
    num_skills_per_query: int = 3
    query_sample_size: int = 10
    num_skills_per_sample: int | None = None
    max_generation_iterations: int = 20
    max_embedding_distance: float = 0.4
    total_skill_budget: int | None = None

    def __post_init__(self):
        if self.num_skills_per_sample is None:
            self.num_skills_per_sample = self.num_skills_per_query


@dataclass
class SkillCandidate:
    name: str
    description: str
    min_distance_to_queries: float
    embedding: np.ndarray = field(repr=False)
    iteration: int = -1


def embed_normalized(texts):
    """Embed via the shared pipeline encoder, then L2-normalise for cosine."""
    arr = np.asarray(embed(list(texts), EMBED_MODEL), dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr


GEN_SYSTEM = "You design agent skill catalog entries for a retrieval benchmark."


class AdversarialSkillAttack:
    def __init__(self, queries, config: AttackConfig):
        self.queries = queries
        self.config = config
        self.client = OpenAI()
        # Cache query embeddings
        print(f"Embedding {len(queries)} target queries with {EMBED_MODEL} ...",
              flush=True)
        self.query_embeddings = embed_normalized(queries)

    def _generate_skills(self, iteration, sample_indices):
        sample = [self.queries[i] for i in sample_indices]
        num_skills = self.config.num_skills_per_sample
        joined = "\n".join(f"- {q}" for q in sample)
        prompt = f"Generate {num_skills} skills relevant to ALL these queries:\n\n{joined}\n\nRequirements:\n- Each skill must work across all the queries above, not be specific to one query.\n- Give each skill a 1-2 word name, lowercase, underscores only.\n- Each description must include exact phrases from the queries."
        try:
            response = self.client.beta.chat.completions.parse(
                model=GEN_MODEL,
                temperature=0.7,
                seed=SEED + iteration,
                response_format=GeneratedSkills,
                messages=[
                    {"role": "system", "content": GEN_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
            parsed = response.choices[0].message.parsed
        except Exception as exc:  # log and skip generation failures
            print(f"  [iter {iteration:>2}] generation failed: {exc}", flush=True)
            return []

        if parsed is None or not parsed.skills:
            return []
        skills = parsed.skills

        skill_embeddings = embed_normalized(s.description for s in skills)
        sample_embeddings = self.query_embeddings[sample_indices]
        distances = 1.0 - skill_embeddings @ sample_embeddings.T  # (skills, sample)
        candidates = [
            SkillCandidate(
                name=s.name,
                description=s.description,
                min_distance_to_queries=float(row.min()),
                embedding=emb,
                iteration=iteration,
            )
            for s, emb, row in zip(skills, skill_embeddings, distances)
        ]
        print(f"  [iter {iteration:>2}] +{len(candidates)} candidate skills "
              f"(min dist {min(c.min_distance_to_queries for c in candidates):.3f})",
              flush=True)
        return candidates

    def _run_phase1(self):
        pool = []
        for iteration in range(self.config.max_generation_iterations):
            rng = random.Random(SEED + iteration)
            k = min(self.config.query_sample_size, len(self.queries))
            sample_indices = rng.sample(range(len(self.queries)), k)
            pool.extend(self._generate_skills(iteration, sample_indices))
        pool.sort(key=lambda c: c.min_distance_to_queries)
        return pool

    def _run_phase2(self, pool):
        cfg = self.config
        num_queries = len(self.queries)
        if not pool:
            return [], []

        candidate_embeddings = np.vstack([c.embedding for c in pool])
        # Cosine distance from every candidate skill to every target query.
        distances = 1.0 - candidate_embeddings @ self.query_embeddings.T
        within = distances <= cfg.max_embedding_distance  # (skills, queries)

        coverage = np.zeros(num_queries, dtype=int)
        selected, rounds = [], []
        remaining = set(range(len(pool)))

        while remaining:
            needy = coverage < cfg.num_skills_per_query
            if not needy.any():
                break

            best, best_gain, best_tie = None, 0, None
            for i in remaining:
                newly = within[i] & needy
                gain = int(newly.sum())
                if gain == 0:
                    continue
                # Tie-break: strictly smaller min_distance_to_queries wins
                # (else the first candidate encountered keeps its place).
                tie = -pool[i].min_distance_to_queries
                if gain > best_gain or (gain == best_gain and
                                        (best_tie is None or tie > best_tie)):
                    best, best_gain, best_tie = i, gain, tie

            if best is None:
                break

            newly = within[best] & needy
            coverage[newly] += 1
            selected.append(best)
            remaining.discard(best)
            rounds.append({
                "skill": f"adversarial_skill_{len(selected)}",
                "candidate_name": pool[best].name,
                "newly_covered": best_gain,
                "min_distance": round(pool[best].min_distance_to_queries, 6),
                "queries_fully_covered": int(
                    (coverage >= cfg.num_skills_per_query).sum()),
                "pool_remaining": len(remaining),
            })

            if cfg.total_skill_budget and len(selected) >= cfg.total_skill_budget:
                break

        return selected, rounds

    def attack(self):
        pool = self._run_phase1()
        print(f"\nPhase 1: generated {len(pool)} candidate skills across "
              f"{self.config.max_generation_iterations} iterations", flush=True)
        selected_indices, rounds = self._run_phase2(pool)
        num_queries = len(self.queries)
        fully_covered = rounds[-1]["queries_fully_covered"] if rounds else 0


        candidates = [{
            "name": c.name,
            "description": c.description,
            "min_distance_to_queries": round(c.min_distance_to_queries, 6),
            "iteration": c.iteration,
        } for c in pool]

        attack_results = {
            "gen_model": GEN_MODEL,
            "embed_model": EMBED_MODEL,
            "config": asdict(self.config),
            "num_target_queries": num_queries,
            "num_generated_candidates": len(pool),
            "num_selected_skills": len(selected_indices),
            "queries_fully_covered": fully_covered,
            "coverage_fraction": round(fully_covered / num_queries, 4)
            if num_queries else 0.0,
            "selected_skills": [
                {"name": pool[i].name, "description": pool[i].description,
                 "min_distance_to_queries": round(pool[i].min_distance_to_queries, 6)}
                for i in selected_indices
            ],
            "selection_rounds": rounds,
        }
        print(f"Phase 2: set-cover selected {len(selected_indices)} skills; "
              f"{fully_covered}/{num_queries} queries fully covered "
              f"(>= {self.config.num_skills_per_query} within "
              f"{self.config.max_embedding_distance})", flush=True)
        return candidates, attack_results


def load_queries():
    data = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    tasks = data["tasks"][:NUM_TASKS]
    if not tasks:
        raise ValueError("No retrieval tasks found")
    queries = [q for task in tasks for q in task["train_queries"]]
    if not queries:
        raise ValueError("No training queries found")
    return queries


def main():
    CANDIDATES_FILE.parent.mkdir(parents=True, exist_ok=True)
    if CANDIDATES_FILE.exists():
        candidates = json.loads(CANDIDATES_FILE.read_text(encoding="utf-8"))
        print(f"Reusing {len(candidates)} cached candidates: {CANDIDATES_FILE}")
        return

    queries = load_queries()
    attack = AdversarialSkillAttack(queries, AttackConfig())
    candidates, attack_results = attack.attack()

    if not candidates:
        raise RuntimeError("Generation produced no candidate skills")

    CANDIDATES_FILE.write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    METRICS_FILE.write_text(
        json.dumps(attack_results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"\nSaved {len(candidates)} candidate skills to {CANDIDATES_FILE}")
    print(f"Saved coverage metrics to {METRICS_FILE}")


if __name__ == "__main__":
    main()
