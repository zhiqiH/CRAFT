"""Experiment naming, JSON I/O, and result summarization."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def experiment_name(timestamp: datetime, model: str) -> str:
    """``YYYYMMDDHHMM-<model>``, with the model part filesystem-safe."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-").lower()
    return f"{timestamp:%Y%m%d%H%M}-{slug}"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _score_series(game: Dict[str, Any], max_rounds: int) -> List[float]:
    """Overall progress after each round, padded to max_rounds by repetition."""
    values = [round_item["score"]["overall_progress"] for round_item in game["rounds"]]
    while len(values) < max_rounds:
        values.append(values[-1] if values else game.get("baseline_progress", 0.0))
    return values[:max_rounds]


def build_summary(experiment: Dict[str, Any]) -> Dict[str, Any]:
    """Condense a full trajectory into the per-round score table used for plots."""
    games = experiment.get("games", [])
    max_rounds = max((len(g["rounds"]) for g in games), default=0)
    per_game = []
    for game in games:
        series = _score_series(game, max_rounds)
        per_game.append(
            {
                "structure_id": game["structure_id"],
                "structure_index": game["structure_index"],
                "run_index": game["run_index"],
                "complexity": game["complexity"],
                "rounds_completed": len(game["rounds"]),
                "baseline_progress": game.get("baseline_progress", 0.0),
                "final_progress": game["final_progress"],
                "completed": game["completed"],
                "overall_progress_by_round": series,
                "metrics_by_round": [
                    round_item["score"] for round_item in game["rounds"]
                ],
            }
        )

    means: List[Dict[str, float]] = []
    for idx in range(max_rounds):
        rows = [g["metrics_by_round"][idx] for g in per_game if idx < len(g["metrics_by_round"])]
        if not rows:
            continue
        means.append(
            {
                "round": idx + 1,
                "overall_progress": sum(r["overall_progress"] for r in rows) / len(rows),
                "completion_percentage": sum(r["completion_percentage"] for r in rows) / len(rows),
                "iou_score": sum(r["iou_score"] for r in rows) / len(rows),
                "position_accuracy": sum(r["position_accuracy"] for r in rows) / len(rows),
                "n_games": len(rows),
            }
        )

    final_scores = [g["final_progress"] for g in per_game]
    return {
        "experiment": experiment.get("experiment", {}),
        "max_rounds": max_rounds,
        "per_game": per_game,
        "mean_curve": means,
        "aggregate": {
            "n_games": len(per_game),
            "n_completed": sum(1 for g in per_game if g["completed"]),
            "mean_final_progress": round(sum(final_scores) / len(final_scores), 6)
            if final_scores
            else 0.0,
            "final_overall_progress_by_round": means[-1] if means else None,
        },
    }
