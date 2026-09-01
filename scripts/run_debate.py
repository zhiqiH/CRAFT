#!/usr/bin/env python3
"""User-facing entry point: run one Debate experiment end to end."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from craft_debate.api import LLMClient, MockLLM, load_api_key  # noqa: E402
from craft_debate.benchmark import load_structures  # noqa: E402
from craft_debate.io import build_summary, experiment_name, write_json  # noqa: E402
from craft_debate.plotting import plot_score_curves  # noqa: E402
from craft_debate.topology import Debate  # noqa: E402

PROVIDERS = {
    "openai": {
        "env_var": "OPENAI_API_KEY",
        "secret_file": "openai_api_key",
        "base_url": None,
    },
    "deepseek": {
        "env_var": "DEEPSEEK_API_KEY",
        "secret_file": "deepseek_api_key",
        "base_url": "https://api.deepseek.com",
    },
    "ollama": {
        "env_var": "OLLAMA_API_KEY",
        "secret_file": "ollama_api_key",
        "base_url": "http://localhost:11434/v1",
    },
}


def parse_list_arg(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def resolve_stage_config(stage: str, config: dict) -> dict:
    """Merge global defaults with the optional per-stage override."""
    base = {
        "model": config["model"],
        "provider": "openai",
        "temperature": config.get("temperature", 0.7),
        "max_tokens": int(config.get("max_completion_tokens", 2000)),
        "api_key": "openai_api_key",
        "thinking": None,
    }
    legacy_stage = {
        "phase1": "proposers",
        "reconciliation": "critics",
        "builder": "judge",
    }[stage]
    stages = config.get("stages", {}) or {}
    override = stages.get(stage) or stages.get(legacy_stage) or {}
    base.update({key: value for key, value in override.items() if value is not None})
    return base


def make_client(stage_cfg: dict, mock: bool, api_cfg: dict) -> Any:
    if mock:
        return MockLLM(model=stage_cfg["model"])

    provider = PROVIDERS.get(stage_cfg["provider"], PROVIDERS["openai"])
    api_key = load_api_key(
        PROJECT_ROOT,
        key_name=stage_cfg.get("api_key", provider["secret_file"]),
        env_var=provider["env_var"],
    )
    if not api_key:
        raise RuntimeError(
            f"no API key for {stage_cfg['provider']} — set {provider['env_var']} "
            f"or put it in .secret/{stage_cfg.get('api_key', provider['secret_file'])}"
        )

    extra_body = {}
    if stage_cfg["provider"] == "deepseek" and stage_cfg.get("thinking") == "disabled":
        extra_body = {"thinking": {"type": "disabled"}}
    elif stage_cfg["provider"] == "deepseek" and stage_cfg.get("thinking") == "enabled":
        extra_body = {"thinking": {"type": "enabled"}}

    return LLMClient(
        model=stage_cfg["model"],
        temperature=float(stage_cfg["temperature"]),
        max_completion_tokens=int(stage_cfg["max_tokens"]),
        api_key=api_key,
        timeout_seconds=float(api_cfg.get("timeout_seconds", 120)),
        max_retries=int(api_cfg.get("max_retries", 6)),
        backoff_seconds=float(api_cfg.get("backoff_seconds", 2)),
        base_url=provider["base_url"],
        provider=stage_cfg["provider"],
        extra_body=extra_body,
    )


def debate_model_label(stage_configs: dict) -> str:
    """Headline model label for the experiment name."""
    labels = [
        f"{stage_configs[stage]['model']}-{stage}"
        for stage in ("phase1", "reconciliation", "builder")
    ]
    models = {
        stage_configs[stage]["model"]
        for stage in ("phase1", "reconciliation", "builder")
    }
    if len(models) == 1:
        return next(iter(models))
    return "+".join(labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run oracle-free 3+3+1 Debate on CRAFT")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "debate_config.json"),
        help="Path to the JSON config (default: config/debate_config.json)",
    )
    parser.add_argument(
        "--benchmark",
        help="Benchmark dataset: a name like craft-80 (resolves to benchmark/craft-80.json) or a JSON path",
    )
    parser.add_argument("--structures", help="Comma-separated structure indices, e.g. 0,1,2")
    parser.add_argument("--runs", help="Comma-separated run indices, e.g. 1,2,3")
    parser.add_argument("--mock", action="store_true", help="Dry run with a deterministic mock LLM")
    parser.add_argument("--name", help="Explicit experiment name (overrides YYYYMMDDHHMM-<model>)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if args.benchmark:
        value = args.benchmark
        if not value.endswith(".json"):
            if "/" in value or "\\" in value:
                value = f"{value}.json"
            else:
                value = f"benchmark/{value}.json"
        config["benchmark"]["path"] = value

    if args.structures:
        config["benchmark"]["structures"] = parse_list_arg(args.structures)
    if args.runs:
        config["benchmark"]["runs"] = parse_list_arg(args.runs)
    stage_configs = {
        stage: resolve_stage_config(stage, config)
        for stage in ("phase1", "reconciliation", "builder")
    }
    display_model = debate_model_label(stage_configs)
    if args.mock:
        display_model = f"{display_model}-mock"
    timestamp = datetime.now()
    name = args.name or experiment_name(timestamp, display_model)

    structures = load_structures(PROJECT_ROOT / config["benchmark"]["path"])
    indices = config["benchmark"]["structures"]
    runs = config["benchmark"]["runs"]
    unknown = [i for i in indices if i < 0 or i >= len(structures)]
    if unknown:
        print(f"ERROR: structure index out of range (0-{len(structures) - 1}): {unknown}")
        return 1

    try:
        clients = {}
        for stage in ("phase1", "reconciliation", "builder"):
            clients[stage] = make_client(
                stage_configs[stage], mock=args.mock, api_cfg=config.get("api", {})
            )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    model_summary = ", ".join(
        f"{stage}={stage_configs[stage]['model']}"
        for stage in ("phase1", "reconciliation", "builder")
    )

    print(f"Experiment: {name} | agents={model_summary} | "
          f"structures={indices} runs={runs} rounds={config['debate']['max_rounds']}")
    print(f"Benchmark: {len(structures)} structures loaded from {config['benchmark']['path']}")

    async def run_experiment():
        games = []
        for structure_index in indices:
            structure_data = structures[structure_index]
            for run_index in runs:
                print(
                    f"\n=== {structure_data['id']} (complexity={structure_data['complexity']}) "
                    f"run={run_index} ==="
                )
                debate = Debate(
                    config=config,
                    structure_data=structure_data,
                    structure_index=structure_index,
                    run_index=run_index,
                    clients=clients,
                )
                game = await debate.run()
                games.append(game)
                print(
                    f"  final progress={game['final_progress']:.4f} "
                    f"completed={game['completed']} rounds={game['rounds_completed']}"
                )
        return games

    games = asyncio.run(run_experiment())
    experiment = {
        "experiment": {
            "name": name,
            "created_at": timestamp.isoformat(timespec="seconds"),
            "model": display_model,
            "mock": bool(args.mock),
            "topology": "Oracle-free Debate (3 Directors -> 3 reconciliations -> 1 Builder)",
            "paper": "CRAFT: arXiv:2603.25268v2",
            "config": config,
        },
        "games": games,
    }

    trajectories_dir = PROJECT_ROOT / config["output"]["trajectories_dir"]
    results_dir = PROJECT_ROOT / config["output"]["results_dir"]
    trajectory_path = trajectories_dir / f"{name}.json"
    summary_path = results_dir / f"{name}.json"
    plot_path = results_dir / f"{name}.png"

    write_json(trajectory_path, experiment)
    summary = build_summary(experiment)
    write_json(summary_path, summary)
    plot_score_curves(summary, plot_path, ymax=(config.get("plot") or {}).get("ymax"))

    print("\nWrote:")
    print(f"  trajectory: {trajectory_path}")
    print(f"  summary:    {summary_path}")
    print(f"  plot:       {plot_path}")
    agg = summary["aggregate"]
    print(
        f"Summary: {agg['n_games']} game(s), {agg['n_completed']} completed, "
        f"mean final progress={agg['mean_final_progress']:.4f}"
    )
    quality = agg.get("protocol_quality")
    if quality:
        print(
            "Protocol validity: "
            f"phase1={quality['phase1_valid_rate']:.1%}, "
            f"reconciliation={quality['reconciliation_valid_rate']:.1%}, "
            f"builder={quality['builder_valid_rate']:.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
