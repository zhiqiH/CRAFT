#!/usr/bin/env python3
"""Programmatic CRAFT structure generator.

Faithful port of the official ``structure_generator_v2.py`` from
https://github.com/csu-signal/CRAFT, plus an optional reachability filter.

Examples:
    python3 benchmark/generate_benchmark.py --count 80
    # writes benchmark/craft-80.json

    python3 benchmark/generate_benchmark.py --count 60 --out benchmark/craft-60.json --seed 7
    python3 benchmark/generate_benchmark.py --count 60 --min-ceiling 0.7
    python3 benchmark/generate_benchmark.py --count 100 --empty-hidden-cells \
        --out benchmark/craft-100-hollow.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from craft_debate.domain import ALL_COORDS, get_director_views, orthogonal_neighbors  # noqa: E402

COLORS = ["g", "b", "r", "y", "o"]
BLOCK_TYPES = [f"{c}{s}" for c in COLORS for s in ["s", "l"]]
REQUIRED_FULL = [c for c in ALL_COORDS if c not in ["(1,1)", "(2,1)"]]
OPTIONAL = ["(1,1)", "(2,1)"]


def generate_layer_tiling(
    rng: random.Random,
    positions_needed: List[str],
    prev_layer: Optional[Dict[str, str]] = None,
) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    grid: Dict[str, str] = {}
    spans: List[Tuple[str, str]] = []
    needed = set(positions_needed)
    order = list(needed)
    rng.shuffle(order)

    for coord in order:
        if coord in grid:
            continue
        prev_block = (prev_layer or {}).get(coord)

        if rng.random() < 0.5:
            free_nbrs = [n for n in orthogonal_neighbors(coord) if n in needed and n not in grid]
            if free_nbrs:
                neighbor = rng.choice(free_nbrs)
                prev_nbr = (prev_layer or {}).get(neighbor)
                color = rng.choice(COLORS)
                block = f"{color}l"
                attempts = 0
                while attempts < 10 and (block == prev_block or block == prev_nbr):
                    color = rng.choice(COLORS)
                    block = f"{color}l"
                    attempts += 1
                grid[coord] = block
                grid[neighbor] = block
                spans.append((coord, neighbor))
                continue

        color = rng.choice(COLORS)
        block = f"{color}s"
        attempts = 0
        while attempts < 10 and block == prev_block:
            color = rng.choice(COLORS)
            block = f"{color}s"
            attempts += 1
        grid[coord] = block

    return grid, spans


def generate_valid_structure(
    rng: Optional[random.Random] = None,
    empty_hidden_cells: bool = False,
) -> Tuple[Dict[str, List[str]], Dict[int, List[Tuple[str, str]]]]:
    """Generate one valid CRAFT target.

    The seven visible positions always have height 3. By default, the two
    positions hidden from every Director have independently sampled heights in
    [0, 2]. ``empty_hidden_cells`` fixes both hidden positions at height 0.
    """
    if rng is None:
        rng = random.Random()

    opt_heights = (
        {coord: 0 for coord in OPTIONAL}
        if empty_hidden_cells
        else {coord: rng.randint(0, 2) for coord in OPTIONAL}
    )
    structure = {coord: [] for coord in ALL_COORDS}
    all_spans: Dict[int, List[Tuple[str, str]]] = {}
    prev_layer: Optional[Dict[str, str]] = None

    for layer in range(3):
        positions_this_layer = list(REQUIRED_FULL)
        for coord in OPTIONAL:
            if opt_heights[coord] > layer:
                positions_this_layer.append(coord)
        layer_grid, spans = generate_layer_tiling(rng, positions_this_layer, prev_layer)
        all_spans[layer] = spans
        for coord in positions_this_layer:
            structure[coord].append(layer_grid[coord])
        prev_layer = layer_grid

    return structure, all_spans


def validate_structure(
    structure: Dict[str, List[str]],
    spans: Dict[int, List[Tuple[str, str]]],
    strict: bool = True,
) -> Tuple[bool, List[str]]:
    """Official structure validation (domino consistency + height rules)."""
    errors: List[str] = []

    for coord in ALL_COORDS:
        if coord not in structure:
            errors.append(f"Missing coordinate: {coord}")
            continue
        stack = structure[coord]
        if len(stack) > 3:
            errors.append(f"{coord}: stack height {len(stack)} exceeds 3")
        for layer, block in enumerate(stack):
            if block not in BLOCK_TYPES:
                errors.append(f"{coord} layer {layer}: invalid block '{block}'")
        if strict and coord in REQUIRED_FULL and len(stack) != 3:
            errors.append(f"{coord}: required position has {len(stack)} blocks (need 3)")
        if strict and coord in OPTIONAL and len(stack) > 2:
            errors.append(f"{coord}: optional position has {len(stack)} blocks (max 2)")

    for layer, layer_spans in spans.items():
        seen_in_span = set()
        for coord_a, coord_b in layer_spans:
            if coord_b not in orthogonal_neighbors(coord_a):
                errors.append(f"Layer {layer}: span {coord_a}<->{coord_b} not orthogonal")
            stack_a = structure.get(coord_a, [])
            stack_b = structure.get(coord_b, [])
            if layer < len(stack_a) and layer < len(stack_b):
                if stack_a[layer] != stack_b[layer]:
                    errors.append(
                        f"Layer {layer}: span {coord_a}={stack_a[layer]} vs {coord_b}={stack_b[layer]}"
                    )
                if not stack_a[layer].endswith("l"):
                    errors.append(f"Layer {layer}: span block '{stack_a[layer]}' is not large")
            seen_in_span.add(coord_a)
            seen_in_span.add(coord_b)
        for coord in ALL_COORDS:
            stack = structure.get(coord, [])
            if layer < len(stack) and stack[layer].endswith("l") and coord not in seen_in_span:
                errors.append(f"Layer {layer}: large block at {coord} has no span partner")

    return len(errors) == 0, errors


def compute_ceiling(structure: Dict[str, List[str]], spans: Dict[int, List[Tuple[str, str]]]) -> float:
    """Greedy upper bound on overall_progress for this structure under the engine rules."""
    from craft_debate.environment import GameState
    from craft_debate.oracle import enumerate_correct_actions
    from craft_debate.progress import calculate_progress

    env = GameState(structure, spans)
    for _ in range(30):  # any target needs at most 25 moves
        ok = [e for e in enumerate_correct_actions(env) if e["flag"] == "ok"]
        if not ok:
            break
        best = max(ok, key=lambda e: e["overall_progress"])
        env.execute_move(best["move"])
    return calculate_progress(env.current_structure, env.target_structure)["overall_progress"]


def complexity_of(structure: Dict[str, List[str]]) -> str:
    total = sum(len(stack) for stack in structure.values())
    return "simple" if total <= 22 else "medium" if total <= 24 else "complex"


def build_structure_item(index: int, count: int, structure: Dict[str, List[str]], spans: Dict[int, List[Tuple[str, str]]]) -> Dict[str, Any]:
    views = get_director_views(structure, spans)
    total = sum(len(stack) for stack in structure.values())
    filled = sum(1 for stack in structure.values() if stack)
    return {
        "id": f"craft-{count}-{index + 1:03d}",
        "complexity": complexity_of(structure),
        "structure": structure,
        "spans": {str(layer): value for layer, value in spans.items()},
        "director_views": views,
        "metadata": {
            "total_blocks": total,
            "filled_positions": filled,
            "optional_heights": {coord: len(structure[coord]) for coord in OPTIONAL},
        },
    }


def build_dataset(
    count: int,
    seed: int,
    min_ceiling: Optional[float] = None,
    empty_hidden_cells: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(seed)
    structures: List[Dict[str, Any]] = []
    ceilings: List[float] = []
    attempts = 0
    max_attempts = count * 50

    while len(structures) < count and attempts < max_attempts:
        attempts += 1
        structure, spans = generate_valid_structure(
            rng=rng,
            empty_hidden_cells=empty_hidden_cells,
        )
        valid, errors = validate_structure(structure, spans, strict=True)
        if not valid:
            continue
        if min_ceiling is not None:
            ceiling = compute_ceiling(structure, spans)
            if ceiling < min_ceiling:
                continue
        else:
            ceiling = compute_ceiling(structure, spans)
        ceilings.append(ceiling)
        structures.append(build_structure_item(len(structures), count, structure, spans))

    if len(structures) < count:
        raise RuntimeError(
            f"only generated {len(structures)}/{count} valid structures "
            f"after {attempts} attempts — relax --min-ceiling"
        )

    report = {
        "complexity": dict(Counter(item["complexity"] for item in structures)),
        "ceiling_min": round(min(ceilings), 4),
        "ceiling_max": round(max(ceilings), 4),
        "ceiling_mean": round(sum(ceilings) / len(ceilings), 4),
        "ceiling_ge_0.9": sum(1 for c in ceilings if c >= 0.9),
        "ceiling_ge_0.7": sum(1 for c in ceilings if c >= 0.7),
    }
    return structures, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a programmatic CRAFT dataset")
    parser.add_argument("--count", type=int, default=80, help="Number of structures (default 80)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument(
        "--out",
        help="Output JSON path (default: benchmark/craft-<count>.json)",
    )
    parser.add_argument(
        "--min-ceiling",
        type=float,
        help="Discard structures whose theoretical progress ceiling is below this value",
    )
    parser.add_argument(
        "--empty-hidden-cells",
        action="store_true",
        help="Force the Director-invisible positions (1,1) and (2,1) to be empty",
    )
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be positive")

    out_path = Path(args.out) if args.out else PROJECT_ROOT / "benchmark" / f"craft-{args.count}.json"
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    structures, report = build_dataset(
        args.count,
        args.seed,
        args.min_ceiling,
        empty_hidden_cells=args.empty_hidden_cells,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(structures, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(structures)} structures to {out_path}")
    print(f"Complexity mix: {report['complexity']}")
    print(
        f"Ceiling: min={report['ceiling_min']} max={report['ceiling_max']} "
        f"mean={report['ceiling_mean']} | >=0.9: {report['ceiling_ge_0.9']} "
        f">=0.7: {report['ceiling_ge_0.7']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
