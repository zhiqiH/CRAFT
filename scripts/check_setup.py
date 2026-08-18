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
    config = None
    if not config_path.is_file():
        problems.append(f"config missing: {config_path}")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            problems.append(f"config file is not valid JSON: {config_path}")

    if config is not None:
        providers = {"openai"}
        stages = config.get("stages") or {}
        for stage in ("proposers", "critics", "judge"):
            providers.add((stages.get(stage) or {}).get("provider", "openai"))
        if config.get("judges", {}).get("enabled"):
            providers.add((stages.get("judges") or {}).get("provider", "openai"))
        key_sources = {
            "openai": ("openai_api_key", "OPENAI_API_KEY"),
            "deepseek": ("deepseek_api_key", "DEEPSEEK_API_KEY"),
        }
        for provider in sorted(providers):
            file_name, env_var = key_sources.get(provider, (None, None))
            if file_name and not load_api_key(PROJECT_ROOT, file_name, env_var):
                problems.append(
                    f"no {provider} API key found — put it in .secret/{file_name} "
                    f"or export {env_var}"
                )

    if problems:
        print("Setup issues found:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("Setup OK: python, dependencies, benchmark, config, and required API keys are all present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
