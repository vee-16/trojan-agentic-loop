"""Download and index SkillsBench v1.1 skills."""

import json
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data/raw/skillsbench"
CATALOG_PATH = ROOT / "data/processed/skills_catalog.json"

REPO_URL = "https://github.com/benchflow-ai/skillsbench.git"
VERSION = "1.1"

SPARSE_PATHS = [
    "/tasks/",
    "/tasks-extra/",
    "/registry.json",
]


def sync_skills():
    """Keep only SkillsBench task directories and registry.json."""

    if not (RAW_DIR / ".git").exists():
        if (RAW_DIR / "registry.json").is_file():
            return

        if RAW_DIR.exists():
            raise FileNotFoundError(
                f"Incomplete SkillsBench directory: {RAW_DIR}"
            )

        RAW_DIR.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [
                "git", "clone",
                "--depth", "1",
                "--branch", f"v{VERSION}",
                "--filter=blob:none",
                "--sparse",
                REPO_URL,
                str(RAW_DIR),
            ],
            check=True,
        )

    # Apply root-anchored patterns to new and existing clones.
    subprocess.run(
        [
            "git", "-C", str(RAW_DIR),
            "sparse-checkout", "set",
            "--no-cone",
            *SPARSE_PATHS,
        ],
        check=True,
    )

    if not (RAW_DIR / "registry.json").is_file():
        raise FileNotFoundError("SkillsBench registry.json not found")


def build_catalog():
    """Index skills belonging to registered SkillsBench v1.1 tasks."""

    sync_skills()

    registry = json.loads(
        (RAW_DIR / "registry.json").read_text(encoding="utf-8")
    )

    release = next(
        (
            entry
            for entry in registry
            if entry["name"] == "skillsbench"
            and entry["version"] == VERSION
        ),
        None,
    )

    if release is None:
        raise ValueError(f"SkillsBench v{VERSION} not found")

    catalog = []

    for task in release["tasks"]:
        skills_dir = RAW_DIR / task["path"] / "environment/skills"

        if not skills_dir.is_dir():
            continue

        for path in sorted(skills_dir.rglob("SKILL.md")):
            content = path.read_text(encoding="utf-8")
            metadata = {}

            if content.startswith("---"):
                parts = content.split("---", 2)

                if len(parts) == 3:
                    metadata = yaml.safe_load(parts[1]) or {}

            relative_path = path.relative_to(RAW_DIR).as_posix()

            catalog.append({
                "id": relative_path,
                "task": task["name"],
                "name": metadata.get("name", path.parent.name),
                "description": metadata.get("description", ""),
                "path": relative_path,
                "source": f"skillsbench-v{VERSION}",
            })

    CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    CATALOG_PATH.write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return catalog


def load_skills():
    """Load the processed skill catalog."""

    if not CATALOG_PATH.is_file():
        return build_catalog()

    return json.loads(
        CATALOG_PATH.read_text(encoding="utf-8")
    )


def get_skill(skill_id: str):
    """Read the original SKILL.md using its catalog ID."""

    catalog = load_skills()

    if skill_id not in {skill["id"] for skill in catalog}:
        raise ValueError(f"Unknown skill: {skill_id}")

    path = (RAW_DIR / skill_id).resolve()

    if not path.is_relative_to(RAW_DIR.resolve()):
        raise ValueError("Invalid skill path")

    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    catalog = build_catalog()

    print(f"SkillsBench v{VERSION}")
    print(f"Indexed {len(catalog)} skills")
    print(f"Catalog saved to: {CATALOG_PATH}")
