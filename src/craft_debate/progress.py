"""Task progress metrics, matching the official CRAFT TaskProgressTracker."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def normalize_structure(structure: Dict[str, List[str]]) -> Dict[Tuple[int, int], List[str]]:
    normalized = {}
    for i in range(3):
        for j in range(3):
            key = f"({i},{j})"
            normalized[(i, j)] = list(structure.get(key, []) or [])
            alt = f"({i}, {j})"
            if alt in structure and key not in structure:
                normalized[(i, j)] = list(structure[alt])
    return normalized


def calculate_progress(
    current_structure: Dict[str, List[str]], target_structure: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Compute IoU / distance / completion / position accuracy for two boards."""
    current = normalize_structure(current_structure)
    target = normalize_structure(target_structure)

    intersection = 0
    union = 0
    for coord in current:
        current_blocks = set(current[coord])
        target_blocks = set(target[coord])
        intersection += len(current_blocks & target_blocks)
        union += len(current_blocks | target_blocks)
    iou_score = intersection / union if union > 0 else 0.0

    total_distance = 0
    total_possible_distance = 0
    for coord in current:
        current_blocks = set(current[coord])
        target_blocks = set(target[coord])
        total_distance += len(current_blocks - target_blocks) + len(target_blocks - current_blocks)
        total_possible_distance += len(current_blocks) + len(target_blocks)
    if total_possible_distance == 0:
        distance_score = 1.0
    else:
        distance_score = 1.0 - (total_distance / total_possible_distance)

    correct_blocks = 0
    total_target_blocks = 0
    for coord in target:
        for idx, block in enumerate(target[coord]):
            total_target_blocks += 1
            if idx < len(current[coord]) and current[coord][idx] == block:
                correct_blocks += 1
    completion_percentage = (
        correct_blocks / total_target_blocks if total_target_blocks > 0 else 0.0
    )

    correct_positions = 0
    for coord in target:
        if set(target[coord]) == set(current[coord]):
            correct_positions += 1
    position_accuracy = correct_positions / 9.0

    return {
        "iou_score": iou_score,
        "distance_score": distance_score,
        "completion_percentage": completion_percentage,
        "position_accuracy": position_accuracy,
        "overall_progress": (iou_score + completion_percentage + position_accuracy) / 3.0,
        "blocks_placed_correctly": correct_blocks,
        "blocks_total_target": total_target_blocks,
        "blocks_total_current": sum(len(v) for v in current.values()),
    }
