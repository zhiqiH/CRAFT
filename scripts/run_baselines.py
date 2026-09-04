#!/usr/bin/env python3
"""Run and compare the four matched fixed-horizon Director baselines."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULES = (
    ("Original", "original"),
    ("Fixed-1", "fixed-1"),
    ("Fixed-2", "fixed-2"),
    ("Fixed-3", "fixed-3"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Original, Fixed-1, Fixed-2, and Fixed-3 under matched controls"
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "config" / "paper_config.json"),
        help="Base experiment config",
    )
    parser.add_argument("--benchmark", help="Benchmark name or JSON path")
    parser.add_argument("--structures", help="Comma-separated structure indices")
    parser.add_argument("--runs", help="Comma-separated run indices")
    parser.add_argument("--turns", type=int, default=20, help="Fixed turn horizon")
    parser.add_argument("--mock", action="store_true", help="Offline pipeline validation")
    parser.add_argument("--quiet", action="store_true", help="Hide per-turn model output")
    parser.add_argument("--name-prefix", help="Output prefix shared by all four methods")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def plot_comparison(comparison: Dict[str, Any], output_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/craft-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/craft-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = comparison["methods"]
    labels = [row["method"] for row in rows]
    progress = [row["mean_final_progress"] for row in rows]
    sems = [row["final_progress_sem"] for row in rows]
    calls = [row["mean_director_calls_per_turn"] for row in rows]
    colors = ["#334155", "#2563eb", "#0f766e", "#9333ea"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    axes[0].bar(labels, progress, yerr=sems, capsize=4, color=colors)
    axes[0].set_ylabel("mean final progress")
    axes[0].set_ylim(0, max(1.0, max(progress, default=0.0) * 1.15))
    axes[0].set_title("Task quality (error bars: SEM)")
    axes[0].grid(True, axis="y", linewidth=0.5, alpha=0.3)

    axes[1].bar(labels, calls, color=colors)
    axes[1].set_ylabel("Director calls per turn")
    axes[1].set_ylim(0, 3.25)
    axes[1].set_title("Communication budget")
    axes[1].grid(True, axis="y", linewidth=0.5, alpha=0.3)

    controls = comparison["controls"]
    fig.suptitle(
        f"CRAFT Director baselines — fixed {controls['turns']} turns, "
        f"Oracle n={controls['oracle_n']}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prefix = args.name_prefix or f"{datetime.now():%Y%m%d%H%M%S}-baseline"
    results_dir = resolve_path(config["output"]["results_dir"])
    trajectories_dir = resolve_path(config["output"]["trajectories_dir"])

    common: List[str] = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_paper.py"),
        "--config",
        str(config_path),
        "--turns",
        str(args.turns),
    ]
    for flag, value in (
        ("--benchmark", args.benchmark),
        ("--structures", args.structures),
        ("--runs", args.runs),
    ):
        if value:
            common.extend((flag, value))
    if args.mock:
        common.append("--mock")
    if args.quiet:
        common.append("--quiet")

    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", "/tmp/craft-matplotlib")
    environment.setdefault("XDG_CACHE_HOME", "/tmp/craft-cache")
    summary_paths: List[Path] = []
    for method, schedule in SCHEDULES:
        name = f"{prefix}-{schedule}"
        command = common + [
            "--director-schedule",
            schedule,
            "--name",
            name,
        ]
        print(f"\n### Running {method} ({schedule}) ###", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)
        summary_paths.append(results_dir / f"{name}.json")

    methods = []
    for (method, schedule), summary_path in zip(SCHEDULES, summary_paths):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        aggregate = summary["aggregate"]
        last_curve = aggregate.get("final_overall_progress_by_turn") or {}
        methods.append(
            {
                "method": method,
                "director_schedule": schedule,
                "turn_horizon": args.turns,
                "n_games": aggregate["n_games"],
                "n_completed": aggregate["n_completed"],
                "completion_rate": aggregate["completion_rate"],
                "mean_final_progress": aggregate["mean_final_progress"],
                "final_progress_sem": round(float(last_curve.get("sem", 0.0)), 6),
                "mean_director_calls_per_turn": aggregate[
                    "mean_director_calls_per_turn"
                ],
                "total_director_calls": aggregate["total_director_calls"],
                "total_director_tokens": aggregate["total_director_tokens"],
                "total_builder_tokens": aggregate["total_builder_tokens"],
                "invalid_action_rate": aggregate["invalid_action_rate"],
                "summary_path": str(summary_path),
                "trajectory_path": str(
                    trajectories_dir / f"{prefix}-{schedule}.json"
                ),
            }
        )

    benchmark = args.benchmark or config["benchmark"]["path"]
    structures = args.structures or ",".join(
        str(value) for value in config["benchmark"]["structures"]
    )
    runs = args.runs or ",".join(str(value) for value in config["benchmark"]["runs"])
    comparison = {
        "name": prefix,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mock": bool(args.mock),
        "controls": {
            "benchmark": benchmark,
            "structures": structures,
            "runs": runs,
            "turns": args.turns,
            "oracle_n": config.get("oracle", {}).get("n"),
            "director_model": config["director"]["model"],
            "builder_model": config["builder"]["model"],
            "seed": config.get("seed"),
            "stop_on_complete": config.get("stop_on_complete", False),
        },
        "methods": methods,
    }
    comparison_json = results_dir / f"{prefix}-comparison.json"
    comparison_png = results_dir / f"{prefix}-comparison.png"
    comparison_json.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_comparison(comparison, comparison_png)

    print("\n### Baseline comparison ###")
    for row in methods:
        print(
            f"{row['method']:<8} progress={row['mean_final_progress']:.4f} "
            f"completion={row['completion_rate']:.1%} "
            f"calls/turn={row['mean_director_calls_per_turn']:.3f}"
        )
    print(f"comparison: {comparison_json}")
    print(f"plot:       {comparison_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
