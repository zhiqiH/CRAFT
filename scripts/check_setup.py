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

    config_path = PROJECT_ROOT / "config" / "paper_config.json"
    config = None
    if not config_path.is_file():
        problems.append(f"config missing: {config_path}")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            problems.append(f"config file is not valid JSON: {config_path}")

    benchmark_path_value = "benchmark/craft_structures_20.json"
    if config is not None:
        benchmark_path_value = config.get("benchmark", {}).get("path", benchmark_path_value)
    benchmark_path = Path(benchmark_path_value)
    benchmark = benchmark_path if benchmark_path.is_absolute() else PROJECT_ROOT / benchmark_path
    if not benchmark.is_file():
        problems.append(f"benchmark file missing: {benchmark}")
    else:
        try:
            data = json.loads(benchmark.read_text(encoding="utf-8"))
            if not isinstance(data, list) or not data:
                problems.append(f"benchmark should be a non-empty list, found {len(data)}")
        except json.JSONDecodeError:
            problems.append(f"benchmark file is not valid JSON: {benchmark}")

    if config is not None:
        providers = set()
        for stage in ("director", "builder"):
            providers.add((config.get(stage) or {}).get("provider", "openai"))
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
