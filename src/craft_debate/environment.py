"""Physics-constrained CRAFT game engine.

Faithful port of the validation rules from the official CRAFT implementation
(https://github.com/csu-signal/CRAFT):

* 3x3 grid, up to three vertical layers per cell
* five colors (g/b/r/y/o) and two sizes (s = 1 cell, l = 2 adjacent cells)
* placing always targets the next free layer of a stack
* large blocks must span two orthogonal neighbours of equal height
* large blocks may not touch the invisible cells (1,1) and (2,1)
* only the top block of a stack can be removed
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from .domain import (
    ALL_COORDS,
    AVAILABLE_BLOCKS,
    INVISIBLE_CELLS,
    WALLS,
    get_director_views,
    norm_pos,
    orthogonal_neighbors,
)

__all__ = ["GameState", "get_director_views"]


class GameState:
    """Board state plus move validation/execution, mirroring EnhancedGameState."""

    def __init__(
        self,
        target_structure: Dict[str, List[str]],
        target_spans: Optional[Dict[int, List[Tuple[str, str]]]] = None,
        available_blocks: Optional[List[str]] = None,
        start_from_empty: bool = True,
    ) -> None:
        self.target_structure = {c: list(target_structure.get(c, [])) for c in ALL_COORDS}
        self.target_spans = {
            int(k): [tuple(pair) for pair in v] for k, v in (target_spans or {}).items()
        }
        self.current_structure = {c: [] for c in ALL_COORDS}
        self.current_spans: Dict[int, List[Tuple[str, str]]] = {}
        self.available_blocks = list(available_blocks or AVAILABLE_BLOCKS)
        self.invisible_cells = set(INVISIBLE_CELLS)
        self.turn = 0
        if not start_from_empty:
            self.current_structure = copy.deepcopy(self.target_structure)
            self.current_spans = copy.deepcopy(self.target_spans)

    # ------------------------------------------------------------------ utils
    def snapshot(self) -> Dict[str, Any]:
        return {
            "structure": copy.deepcopy(self.current_structure),
            "spans": {str(k): [list(p) for p in v] for k, v in self.current_spans.items()},
        }

    def board_complete(self) -> bool:
        for coord in ALL_COORDS:
            if self.current_structure[coord] != self.target_structure[coord]:
                return False
        return True

    def simulate_move(self, move: Dict[str, Any]) -> Dict[str, Any]:
        """Dry-run a move on a copy of the board (used by the oracle)."""
        from .progress import calculate_progress  # deferred: avoids an import cycle

        clone = copy.deepcopy(self)
        result = clone.execute_move(move)
        if not result["ok"]:
            return {"ok": False, "error": result["error"]}
        return {
            "ok": True,
            "structure_placement": result["structure_placement"],
            "side_placement": result["side_placement"],
            "overall_progress": calculate_progress(
                clone.current_structure, clone.target_structure
            )["overall_progress"],
        }

    # ------------------------------------------------------------- validation
    def validate_move(self, move: Dict[str, Any]) -> Tuple[bool, str]:
        action = move.get("action")
        position = norm_pos(move.get("position"))
        block = move.get("block")
        layer = move.get("layer", 0)
        span_to = norm_pos(move.get("span_to")) if move.get("span_to") else None

        if action not in {"place", "remove"}:
            return False, f"Unknown action: {action}"
        if position is None or position not in ALL_COORDS:
            return False, f"Invalid position: {move.get('position')}"
        move["position"] = position

        if action == "place":
            if block not in self.available_blocks:
                return False, f"Block '{block}' not in available blocks"
            stack = self.current_structure[position]
            if layer != len(stack):
                return False, (
                    f"Wrong layer at {position}: got {layer}, "
                    f"expected {len(stack)} (stack has {len(stack)} blocks)"
                )
            if len(stack) >= 3:
                return False, f"Stack full at {position} (3 blocks already)"

            if block.endswith("l"):
                if span_to == position:
                    return False, f"span_to cannot be same as position: {position}"
                if not span_to:
                    return False, f"Large block '{block}' needs span_to"
                if position in self.invisible_cells or span_to in self.invisible_cells:
                    return False, (
                        f"Illegal span into invisible cell: {position}<->{span_to}. "
                        f"Large blocks cannot include {sorted(self.invisible_cells)}."
                    )
                if span_to not in ALL_COORDS or span_to not in orthogonal_neighbors(position):
                    return False, f"span_to {span_to} is not adjacent to {position}"
                neighbor_stack = self.current_structure[span_to]
                if len(neighbor_stack) != len(stack):
                    return False, (
                        f"span_to {span_to} has {len(neighbor_stack)} blocks, "
                        f"{position} has {len(stack)} — both must be at same height"
                    )
                if len(neighbor_stack) >= 3:
                    return False, f"Neighbor {span_to} stack is already full"
                layer_spans = self.current_spans.get(layer, [])
                if any(position in (a, b) or span_to in (a, b) for a, b in layer_spans):
                    return False, (
                        f"{position} or {span_to} already occupied by another span at layer {layer}"
                    )
                move["span_to"] = span_to

        elif action == "remove":
            stack = self.current_structure[position]
            if not stack:
                return False, f"Cannot remove from {position} — stack is empty"
            if layer != len(stack) - 1:
                return False, (
                    f"Cannot remove layer {layer} at {position} — "
                    f"must remove top block first (layer {len(stack) - 1})"
                )
            top_block = stack[-1]
            if top_block.endswith("l"):
                layer_spans = self.current_spans.get(layer, [])
                partner = next(
                    (b if a == position else a for a, b in layer_spans if position in (a, b)),
                    None,
                )
                if partner is None:
                    return False, (
                        f"Large block at {position} layer {layer} has no recorded span partner"
                    )
                if span_to != partner:
                    return False, (
                        f"Incorrect span_to for large block removal at {position} layer {layer}: "
                        f"got {span_to}, expected {partner}"
                    )
                move["span_to"] = span_to

        return True, "ok"

    # ------------------------------------------------------------- execution
    def _apply_move(self, move: Dict[str, Any]) -> None:
        action = move["action"]
        position = move["position"]
        block = move.get("block")
        layer = move["layer"]
        span_to = move.get("span_to")

        if action == "place":
            if block.endswith("l"):
                self.current_structure[position].append(block)
                self.current_structure[span_to].append(block)
                self.current_spans.setdefault(layer, []).append((position, span_to))
            else:
                self.current_structure[position].append(block)
        else:
            if self.current_structure[position][-1].endswith("l"):
                self.current_structure[position].pop()
                self.current_structure[span_to].pop()
                self.current_spans[layer] = [
                    (a, b)
                    for a, b in self.current_spans.get(layer, [])
                    if not (a == position and b == span_to)
                    and not (a == span_to and b == position)
                ]
            else:
                self.current_structure[position].pop()

    def _side_placement(
        self, action: str, position: str, layer: int, eval_block: Optional[str]
    ) -> bool:
        checks = []
        for coords in WALLS.values():
            if position not in coords:
                continue
            target_stack = self.target_structure[position]
            try:
                if action == "place":
                    checks.append(target_stack[layer] == eval_block)
                else:
                    checks.append(target_stack[layer] != eval_block)
            except IndexError:
                checks.append(False if action == "place" else True)
        return any(checks) if checks else False

    def execute_move(self, move: Dict[str, Any]) -> Dict[str, Any]:
        """Validate, apply, and evaluate a move. Never mutates state on failure."""
        move = dict(move)
        move.setdefault("span_to", None)
        ok, reason = self.validate_move(move)
        if not ok:
            return {"ok": False, "error": reason, "move": move}

        removed_block = None
        if move["action"] == "remove":
            removed_block = self.current_structure[move["position"]][-1]

        self._apply_move(move)
        self.turn += 1

        position = move["position"]
        layer = move["layer"]
        if move["action"] == "place":
            structure_placement = self.target_structure[position][layer] == move["block"]
            eval_block = move["block"]
        else:
            structure_placement = self.target_structure[position][layer] != removed_block
            eval_block = removed_block

        side_placement = self._side_placement(move["action"], position, layer, eval_block)
        return {
            "ok": True,
            "error": None,
            "move": move,
            "removed_block": removed_block,
            "structure_placement": structure_placement,
            "side_placement": side_placement,
            "board_complete": self.board_complete(),
        }
