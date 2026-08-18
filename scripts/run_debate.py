#!/usr/bin/env python3
"""User-facing entry point: run one Debate experiment end to end."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from craft_debate.api import LLMClient, MockLLM, load_api_key  # noqa: E402
from craft_debate.benchmark import load_structures  # noqa: E402
from craft_debate.io import build_summary, experiment_name, write_json  # noqa: E402
from craft_debate.plotting import plot_score_curves  # noqa: E402
from craft_debate.topology import Debate  # noqa: E402


def parse_list_arg(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 7-agent Debate on CRAFT")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "debate_config.json"),
        help="Path to the JSON config (default: config/debate_config.json)",
    )
    parser.add_argument("--structures", help="Comma-separated structure indices, e.g. 0,1,2")
    parser.add_argument("--runs", help="Comma-separated run indices, e.g. 1,2,3")
    parser.add_argument("--rounds", type=int, help="Number of debate rounds (default 20)")
    parser.add_argument("--model", help="Model for all 7 agents (default: gpt-4o-mini)")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")
    parser.add_argument("--judges", action="store_true", help="Also run the paper's SG/MM/PS judges")
    parser.add_argument("--mock", action="store_true", help="Dry run with a deterministic mock LLM")
    parser.add_argument("--name", help="Explicit experiment name (overrides YYYYMMDDHHMM-<model>)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))

    if args.model:
        config["model"] = args.model
    if args.temperature is not None:
        config["temperature"] = args.temperature
    if args.rounds is not None:
        config["debate"]["max_rounds"] = args.rounds
    if args.structures:
        config["benchmark"]["structures"] = parse_list_arg(args.structures)
    if args.runs:
        config["benchmark"]["runs"] = parse_list_arg(args.runs)
    if args.judges:
        config["judges"]["enabled"] = True

    model = config["model"]
    display_model = f"{model}-mock" if args.mock else model
    timestamp = datetime.now()
    name = args.name or experiment_name(timestamp, display_model)

    structures = load_structures(PROJECT_ROOT / config["benchmark"]["path"])
    indices = config["benchmark"]["structures"]
    runs = config["benchmark"]["runs"]
    unknown = [i for i in indices if i < 0 or i >= len(structures)]
    if unknown:
        print(f"ERROR: structure index out of range (0-{len(structures) - 1}): {unknown}")
        return 1

    if args.mock:
        client = MockLLM(model=display_model)
        judges_client = client
    else:
        api_key = load_api_key(PROJECT_ROOT)
        if not api_key:
            print(
                "ERROR: no OpenAI API key found. Put it in .secret/openai_api_key "
                "(one line, no quotes) or export OPENAI_API_KEY."
            )
            return 1
        api_cfg = config.get("api", {})
        client = LLMClient(
            model=model,
            temperature=float(config.get("temperature", 0.7)),
            max_completion_tokens=int(config.get("max_completion_tokens", 2000)),
            api_key=api_key,
            timeout_seconds=float(api_cfg.get("timeout_seconds", 120)),
            max_retries=int(api_cfg.get("max_retries", 6)),
            backoff_seconds=float(api_cfg.get("backoff_seconds", 2)),
        )
        judge_cfg = config.get("judges", {})
        judges_client = (
            LLMClient(
                model=judge_cfg.get("model", model),
                temperature=float(judge_cfg.get("temperature", 0.0)),
                max_completion_tokens=int(config.get("max_completion_tokens", 2000)),
                api_key=api_key,
                timeout_seconds=float(api_cfg.get("timeout_seconds", 120)),
                max_retries=int(api_cfg.get("max_retries", 6)),
                backoff_seconds=float(api_cfg.get("backoff_seconds", 2)),
            )
            if config["judges"].get("enabled")
            else None
        )

    print(f"Experiment: {name} | model={display_model} | "
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
                    client=client,
                    judges_client=judges_client,
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
            "topology": "Debate (3 proposers -> 3 critics -> 1 judge)",
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
    plot_score_curves(summary, plot_path)

    print("\nWrote:")
    print(f"  trajectory: {trajectory_path}")
    print(f"  summary:    {summary_path}")
    print(f"  plot:       {plot_path}")
    agg = summary["aggregate"]
    print(
        f"Summary: {agg['n_games']} game(s), {agg['n_completed']} completed, "
        f"mean final progress={agg['mean_final_progress']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
