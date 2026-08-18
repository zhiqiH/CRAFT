"""Render the 20-round score curves to a PNG."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def plot_score_curves(
    summary: Dict[str, Any],
    output_png: Path,
    dpi: int = 150,
    ymax: float = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_game = summary["per_game"]
    max_rounds = summary["max_rounds"]
    rounds = list(range(1, max_rounds + 1))

    metric_keys = ["overall_progress", "completion_percentage", "iou_score", "position_accuracy"]
    data_max = max(
        (row[key] for row in summary["mean_curve"] for key in metric_keys),
        default=0.0,
    )
    if ymax is None:
        top = 1.04
    else:
        # Zoom to the requested cap, but never clip data above it.
        top = max(float(ymax), min(1.04, 1.02 * data_max + 0.02))

    fig, (ax_main, ax_metrics) = plt.subplots(
        2, 1, figsize=(9, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0]}
    )
    fig.suptitle(
        f"Debate score curve — {summary['experiment'].get('model', '?')} "
        f"({summary['experiment'].get('name', '')})",
        fontsize=13,
    )

    mean_values = [row["overall_progress"] for row in summary["mean_curve"]]
    if len(per_game) > 1:
        for game in per_game:
            ax_main.plot(
                rounds,
                game["overall_progress_by_round"],
                color="#9aa5b1",
                linewidth=0.8,
                alpha=0.75,
                zorder=1,
            )
    ax_main.plot(
        rounds,
        mean_values,
        color="#0b6ee0",
        linewidth=2.2,
        marker="o",
        markersize=3.5,
        label="mean overall progress",
        zorder=2,
    )
    if 0.95 < top:
        ax_main.axhline(0.95, color="#b91c1c", linestyle="--", linewidth=0.9, alpha=0.7)
        ax_main.text(
            max_rounds + 0.25,
            0.95,
            "completion threshold (0.95)",
            va="center",
            fontsize=8,
            color="#b91c1c",
        )
    ax_main.set_ylabel("overall progress")
    ax_main.set_ylim(-0.02, top)
    ax_main.set_xlim(0.5, max_rounds + 2.5)
    ax_main.grid(True, axis="y", linewidth=0.5, alpha=0.3)
    ax_main.legend(loc="lower right", fontsize=8, frameon=False)

    metrics = {
        "completion %": "completion_percentage",
        "IoU": "iou_score",
        "position accuracy": "position_accuracy",
        "overall progress": "overall_progress",
    }
    colors = ["#0b6ee0", "#7c3aed", "#047857", "#d97706"]
    for (label, key), color in zip(metrics.items(), colors):
        values = [row[key] for row in summary["mean_curve"]]
        ax_metrics.plot(rounds, values, color=color, linewidth=1.6, label=label)
    ax_metrics.set_xlabel("round")
    ax_metrics.set_ylabel("metric value (mean)")
    ax_metrics.set_ylim(-0.02, top)
    ax_metrics.set_xticks(rounds)
    ax_metrics.grid(True, axis="y", linewidth=0.5, alpha=0.3)
    ax_metrics.legend(loc="lower right", fontsize=8, frameon=False, ncol=2)

    agg = summary["aggregate"]
    footer = (
        f"games={agg['n_games']} | completed={agg['n_completed']} | "
        f"mean final progress={agg['mean_final_progress']:.3f}"
    )
    fig.text(0.01, 0.005, footer, fontsize=8, color="#4b5563")

    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)
    return output_png
