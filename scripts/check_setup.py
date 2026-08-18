#!/usr/bin/env python3
"""User-facing setup check: key, dependencies, benchmark, and config."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from craft_debate.api import load_api_key  # noqa: E402


def main() -> int:
    problems = []

    if sys.version_info < (3, 9):
        problems.append(f"Python {sys.version.split()[0]} is too old (need >=3.9)")

    try:
        import openai  # noqa: F401
    except ImportError:
        problems.append("package 'openai' is missing — run: python3 -m pip install -e .")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        problems.append("package 'matplotlib' is missing — run: python3 -m pip install -e .")

    benchmark = PROJECT_ROOT / "benchmark" / "craft_structures_20.json"
    if not benchmark.is_file():
        problems.append(f"benchmark file missing: {benchmark}")
    else:
        try:
            data = json.loads(benchmark.read_text(encoding="utf-8"))
            if not isinstance(data, list) or len(data) != 20:
                problems.append(f"benchmark should contain 20 structures, found {len(data)}")
        except json.JSONDecodeError:
            problems.append(f"benchmark file is not valid JSON: {benchmark}")

    config_path = PROJECT_ROOT / "config" / "debate_config.json"
    if not config_path.is_file():
        problems.append(f"config missing: {config_path}")
    else:
        try:
            json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            problems.append(f"config file is not valid JSON: {config_path}")

    key = load_api_key(PROJECT_ROOT)
    if not key:
        problems.append(
            "no API key found — put it in .secret/openai_api_key or export OPENAI_API_KEY"
        )

    if problems:
        print("Setup issues found:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("Setup OK: python, dependencies, benchmark, config, and API key are all present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
