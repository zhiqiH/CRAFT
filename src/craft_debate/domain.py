"""Shared domain constants and geometry helpers for the CRAFT 3x3 board."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

ALL_COORDS = [f"({i},{j})" for i in range(3) for j in range(3)]
INVISIBLE_CELLS = {"(1,1)", "(2,1)"}

COLOR_NAMES = {"g": "green", "b": "blue", "r": "red", "y": "yellow", "o": "orange", "n": "none"}
AVAILABLE_BLOCKS = ["gs", "gl", "bs", "bl", "rs", "rl", "ys", "yl", "os", "ol"]

WALLS = {
    "D1": ["(0,0)", "(1,0)", "(2,0)"],
    "D2": ["(0,0)", "(0,1)", "(0,2)"],
    "D3": ["(0,2)", "(1,2)", "(2,2)"],
}

PERSPECTIVE_DESCRIPTIONS = {
    "D1": "From left to right, you see cells (0,0), (1,0), (2,0) across all layers.",
    "D2": "From left to right, you see cells (0,0), (0,1), (0,2) across all layers.",
    "D3": "From left to right, you see cells (0,2), (1,2), (2,2) across all layers.",
}


def coord_ij(coord: str) -> Tuple[int, int]:
    """``(i,j)`` -> ``(i, j)`` as ints."""
    i, j = re.findall(r"\d", coord)
    return int(i), int(j)


def ij_coord(i: int, j: int) -> str:
    return f"({i},{j})"


def norm_pos(pos: Any) -> Optional[str]:
    if not isinstance(pos, str) or "(" not in pos:
        return None
    digits = re.findall(r"\d", pos)
    if len(digits) != 2:
        return None
    return f"({digits[0]},{digits[1]})"


def orthogonal_neighbors(coord: str) -> List[str]:
    i, j = coord_ij(coord)
    result = []
    if j < 2:
        result.append(ij_coord(i, j + 1))
    if i < 2:
        result.append(ij_coord(i + 1, j))
    if j > 0:
        result.append(ij_coord(i, j - 1))
    if i > 0:
        result.append(ij_coord(i - 1, j))
    return result


def get_director_views(
    structure: Dict[str, List[str]], spans: Optional[Dict[int, List[Tuple[str, str]]]] = None
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Recompute the private 2D projections of a structure for D1/D2/D3."""

    def cell(coord: str, layer: int, visible_coords: List[str]) -> Dict[str, Any]:
        stack = structure.get(coord, [])
        if layer >= len(stack):
            return {"color": "none", "size": 1}
        block = stack[layer]
        color = COLOR_NAMES.get(block[0], "none")
        if block.endswith("l") and spans:
            layer_spans = spans.get(layer, [])
            partner = next(
                (b if a == coord else a for a, b in layer_spans if coord in (a, b)), None
            )
            size = 2 if (partner and partner in visible_coords) else 1
        else:
            size = 1
        return {"color": color, "size": size}

    views = {}
    for did, coords in WALLS.items():
        views[did] = {f"row_{layer}": [cell(c, layer, coords) for c in coords] for layer in range(3)}
    return views
