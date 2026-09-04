#!/usr/bin/env python3
"""Run the CRAFT paper protocol with the paper's fixed five-move Oracle."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from craft_debate.api import LLMClient, MockLLM, load_api_key  # noqa: E402
from craft_debate.benchmark import load_structures  # noqa: E402
from craft_debate.io import experiment_name, write_json  # noqa: E402
from craft_debate.paper_protocol import (  # noqa: E402
    DIRECTOR_SCHEDULES,
    PAPER_ORACLE_N,
    PaperGame,
    validate_director_schedule,
    validate_paper_oracle_config,
)

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


def parse_list_arg(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def make_client(stage_cfg: Dict[str, Any], mock: bool, api_cfg: Dict[str, Any]) -> Any:
    if mock:
        return MockLLM(model=stage_cfg["model"])

    provider_name = stage_cfg.get("provider", "openai")
    provider = PROVIDERS.get(provider_name, PROVIDERS["openai"])
    api_key = load_api_key(
        PROJECT_ROOT,
        key_name=stage_cfg.get("api_key", provider["secret_file"]),
        env_var=provider["env_var"],
    )
    if not api_key:
        raise RuntimeError(
            f"no API key for {provider_name} — set {provider['env_var']} "
            f"or put it in .secret/{stage_cfg.get('api_key', provider['secret_file'])}"
        )

    return LLMClient(
        model=stage_cfg["model"],
        temperature=float(stage_cfg.get("temperature", 0.0)),
        max_completion_tokens=int(stage_cfg.get("max_tokens", 2000)),
        api_key=api_key,
        timeout_seconds=float(api_cfg.get("timeout_seconds", 300)),
        max_retries=int(api_cfg.get("max_retries", 6)),
        backoff_seconds=float(api_cfg.get("backoff_seconds", 2)),
        base_url=provider["base_url"],
        provider=provider_name,
    )


def build_summary(experiment: Dict[str, Any]) -> Dict[str, Any]:
    games = experiment.get("games", [])
    max_turns = experiment["experiment"]["config"]["turns"]
    per_game = []
    for game in games:
        series = [turn["score"]["overall_progress"] for turn in game["turns"]]
        while len(series) < max_turns:
            series.append(series[-1] if series else game.get("baseline_progress", 0.0))
        turns = game["turns"]
        director_calls = sum(len(turn.get("director_order", [])) for turn in turns)
        director_tokens = sum(
            int(response.get("usage", {}).get("total_tokens") or 0)
            for turn in turns
            for response in turn.get("director_responses", {}).values()
        )
        builder_tokens = sum(
            int(turn.get("builder_response", {}).get("usage", {}).get("total_tokens") or 0)
            for turn in turns
        )
        invalid_actions = sum(
            1 for turn in turns if turn.get("execution", {}).get("ok") is False
        )
        per_game.append(
            {
                "structure_id": game["structure_id"],
                "structure_index": game["structure_index"],
                "run_index": game["run_index"],
                "complexity": game["complexity"],
                "archetypes": game["archetypes"],
                "turns_completed": len(game["turns"]),
                "director_schedule": game.get(
                    "director_schedule",
                    experiment.get("experiment", {})
                    .get("config", {})
                    .get("director_schedule", "original"),
                ),
                "director_calls": director_calls,
                "director_calls_per_turn": round(director_calls / len(turns), 6)
                if turns
                else 0.0,
                "director_tokens": director_tokens,
                "builder_tokens": builder_tokens,
                "invalid_actions": invalid_actions,
                "invalid_action_rate": round(invalid_actions / len(turns), 6)
                if turns
                else 0.0,
                "baseline_progress": game.get("baseline_progress", 0.0),
                "final_progress": game["final_progress"],
                "completed": game["completed"],
                "overall_progress_by_turn": series,
                "metrics_by_turn": [turn["score"] for turn in game["turns"]],
            }
        )

    mean_curve = []
    for idx in range(max_turns):
        values = [g["overall_progress_by_turn"][idx] for g in per_game]
        if not values:
            continue
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        mean_curve.append(
            {
                "turn": idx + 1,
                "overall_progress": mean,
                "sem": (variance ** 0.5) / (len(values) ** 0.5),
                "min": min(values),
                "max": max(values),
                "n_games": len(values),
            }
        )

    final_scores = [g["final_progress"] for g in per_game]
    total_turns = sum(g["turns_completed"] for g in per_game)
    total_director_calls = sum(g["director_calls"] for g in per_game)
    total_director_tokens = sum(g["director_tokens"] for g in per_game)
    total_builder_tokens = sum(g["builder_tokens"] for g in per_game)
    total_invalid_actions = sum(g["invalid_actions"] for g in per_game)
    return {
        "experiment": experiment.get("experiment", {}),
        "max_turns": max_turns,
        "per_game": per_game,
        "mean_curve": mean_curve,
        "aggregate": {
            "n_games": len(per_game),
            "n_completed": sum(1 for g in per_game if g["completed"]),
            "completion_rate": round(
                sum(1 for g in per_game if g["completed"]) / len(per_game), 6
            )
            if per_game
            else 0.0,
            "mean_final_progress": round(sum(final_scores) / len(final_scores), 6)
            if final_scores
            else 0.0,
            "total_turns": total_turns,
            "total_director_calls": total_director_calls,
            "mean_director_calls_per_turn": round(
                total_director_calls / total_turns, 6
            )
            if total_turns
            else 0.0,
            "total_director_tokens": total_director_tokens,
            "total_builder_tokens": total_builder_tokens,
            "invalid_action_rate": round(total_invalid_actions / total_turns, 6)
            if total_turns
            else 0.0,
            "final_overall_progress_by_turn": mean_curve[-1] if mean_curve else None,
        },
    }


def plot_score_curves(summary: Dict[str, Any], output_png: Path, ymax: float = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_game = summary["per_game"]
    turns = list(range(1, summary["max_turns"] + 1))
    mean_curve = summary["mean_curve"]
    means = [row["overall_progress"] for row in mean_curve]
    sems = [row["sem"] for row in mean_curve]
    data_max = max(
        (max(game["overall_progress_by_turn"], default=0.0) for game in per_game),
        default=0.0,
    )
    if ymax is None:
        top = 1.04
    else:
        # Use the requested cap as a zoom hint, but never clip curves above it.
        top = max(float(ymax), min(1.04, 1.02 * data_max + 0.02))

    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.suptitle(
        f"Paper protocol score curve — {summary['experiment'].get('model', '?')} "
        f"({summary['experiment'].get('name', '')})",
        fontsize=12,
    )
    if len(per_game) > 1:
        for game in per_game:
            ax.plot(
                turns,
                game["overall_progress_by_turn"],
                color="#9aa5b1",
                linewidth=0.8,
                alpha=0.75,
                zorder=1,
            )
    ax.fill_between(
        turns,
        [m - s for m, s in zip(means, sems)],
        [m + s for m, s in zip(means, sems)],
        color="#0b6ee0",
        alpha=0.18,
        linewidth=0,
        label="mean ± SEM",
    )
    ax.plot(
        turns,
        means,
        color="#0b6ee0",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label="mean overall progress",
        zorder=2,
    )
    if 0.95 < top:
        ax.axhline(0.95, color="#b91c1c", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.text(
            summary["max_turns"] + 0.25,
            0.95,
            "completion threshold (0.95)",
            va="center",
            fontsize=8,
            color="#b91c1c",
        )
    ax.set_xlabel("turn")
    ax.set_ylabel("overall progress")
    ax.set_ylim(-0.02, top)
    ax.set_xlim(0.5, summary["max_turns"] + 1.5)
    if summary["max_turns"] <= 20:
        ticks = turns
    else:
        ticks = list(range(1, summary["max_turns"] + 1, 5))
        if ticks[-1] != summary["max_turns"]:
            ticks.append(summary["max_turns"])
    ax.set_xticks(ticks)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=False)

    agg = summary["aggregate"]
    footer = (
        f"games={agg['n_games']} | completed={agg['n_completed']} | "
        f"mean final progress={agg['mean_final_progress']:.3f}"
    )
    fig.text(0.01, 0.01, footer, fontsize=8, color="#4b5563")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    return output_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CRAFT paper protocol with Oracle fixed at n=5"
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "paper_config.json"),
        help="Path to the paper-protocol config",
    )
    parser.add_argument(
        "--benchmark",
        help="Benchmark dataset: a name like craft-100-hollow (resolves to "
        "benchmark/craft-100-hollow.json) or a JSON path",
    )
    parser.add_argument("--structures", help="Comma-separated structure indices, e.g. 0")
    parser.add_argument("--runs", help="Comma-separated run indices, e.g. 1,2,3")
    parser.add_argument("--turns", type=int, help="Override the number of turns per game")
    parser.add_argument(
        "--director-schedule",
        choices=DIRECTOR_SCHEDULES,
        help="Director baseline: original random 1-3, or exactly 1/2/3",
    )
    parser.add_argument("--mock", action="store_true", help="Deterministic dry run (no LLM calls)")
    parser.add_argument("--quiet", action="store_true", help="Hide per-turn model output")
    parser.add_argument("--name", help="Explicit experiment name")
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
    if args.turns:
        config["turns"] = args.turns
    if args.director_schedule:
        config["director_schedule"] = args.director_schedule
    try:
        validate_paper_oracle_config(config)
        validate_director_schedule(config)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    structures = load_structures(PROJECT_ROOT / config["benchmark"]["path"])
    indices = config["benchmark"]["structures"]
    runs = config["benchmark"]["runs"]
    unknown = [i for i in indices if i < 0 or i >= len(structures)]
    if unknown:
        print(f"ERROR: structure index out of range (0-{len(structures) - 1}): {unknown}")
        return 1

    try:
        director_client = make_client(
            config["director"], mock=args.mock, api_cfg=config.get("api", {})
        )
        builder_client = make_client(
            config["builder"], mock=args.mock, api_cfg=config.get("api", {})
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    display_model = (
        f"{config['director']['model']}-directors+{config['builder']['model']}-builder"
    )
    if args.mock:
        display_model += "-mock"
    timestamp = datetime.now()
    name = args.name or experiment_name(timestamp, display_model)

    print(
        f"Experiment: {name} | directors={config['director']['model']} "
        f"builder={config['builder']['model']} | structures={indices} runs={runs} "
        f"turns={config['turns']} schedule={config.get('director_schedule', 'original')} "
        f"oracle_n={PAPER_ORACLE_N}"
    )
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
                game = PaperGame(
                    config=config,
                    structure_data=structure_data,
                    structure_index=structure_index,
                    run_index=run_index,
                    director_client=director_client,
                    builder_client=builder_client,
                    verbose=not args.quiet,
                )
                record = await game.run()
                games.append(record)
                print(
                    f"  final progress={record['final_progress']:.4f} "
                    f"completed={record['completed']} turns={record['turns_completed']}"
                )
        return games

    games = asyncio.run(run_experiment())
    experiment = {
        "experiment": {
            "name": name,
            "created_at": timestamp.isoformat(timespec="seconds"),
            "model": display_model,
            "mock": bool(args.mock),
            "protocol": (
                "CRAFT baseline protocol: Director schedule="
                f"{config.get('director_schedule', 'original')}; "
                f"Builder selects from up to {PAPER_ORACLE_N} Oracle-verified candidates"
            ),
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
