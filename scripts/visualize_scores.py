#!/usr/bin/env python3
"""User-facing entry point: (re)draw the score-curve plot for one experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from craft_debate.io import build_summary, read_json  # noqa: E402
from craft_debate.plotting import plot_score_curves  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot the 20-round score curve")
    parser.add_argument("--name", help="Experiment base name, e.g. 202608172330-gpt-4o-mini")
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Pick the most recently created experiment automatically",
    )
    parser.add_argument(
        "--trajectories-dir", default="trajectories", help="Directory with trajectory JSONs"
    )
    parser.add_argument("--results-dir", default="results", help="Directory for summary/plot")
    args = parser.parse_args()

    trajectories_dir = PROJECT_ROOT / args.trajectories_dir
    results_dir = PROJECT_ROOT / args.results_dir
    if not args.name and not args.latest:
        parser.error("provide --name or --latest")

    if args.latest:
        candidates = sorted(trajectories_dir.glob("*.json")) + sorted(results_dir.glob("*.json"))
        candidates = [p for p in candidates if p.stem != "README"]
        if not candidates:
            print("No experiment files found.")
            return 1
        args.name = max(candidates, key=lambda p: p.stat().st_mtime).stem

    name = args.name
    summary_path = results_dir / f"{name}.json"
    trajectory_path = trajectories_dir / f"{name}.json"
    if summary_path.is_file():
        summary = read_json(summary_path)
    elif trajectory_path.is_file():
        print(f"{summary_path} not found; rebuilding the summary from the trajectory.")
        summary = build_summary(read_json(trajectory_path))
    else:
        print(f"Nothing found for experiment '{name}' in results/ or trajectories/.")
        return 1

    output = results_dir / f"{name}.png"
    plot_score_curves(summary, output)
    print(f"Plot written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
