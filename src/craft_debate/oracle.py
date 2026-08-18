"""Oracle: enumerate verified moves that make forward progress toward the target.

Faithful port of ``agents/oracle.py::enumerate_correct_actions`` from the official
CRAFT repository.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .domain import ALL_COORDS
from .environment import GameState

FLAG_OK = "ok"
FLAG_MISSING_SPAN = "missing_span_info"
FLAG_BLOCKED_REMOVE = "blocked_correct_remove"
FLAG_SIM_FAILED = "sim_failed"
FLAG_LARGE_SKIP_ENDPOINT = "large_block_secondary_endpoint_skipped"


def _find_span_partner(
    pos: str, layer: int, spans: Dict[int, List[Tuple[str, str]]]
) -> Tuple[Optional[str], Optional[Tuple[str, str]]]:
    for a, b in spans.get(layer, []):
        if pos in (a, b):
            partner = b if a == pos else a
            return partner, (a, b)
    return None, None


def _unrunnable(move: Dict[str, Any], flag: str, source: str, detail: str) -> Dict[str, Any]:
    return {
        "move": move,
        "structure_placement": False,
        "side_placement": False,
        "overall_progress": 0.0,
        "flag": flag,
        "source": source,
        "detail": detail,
    }


def enumerate_correct_actions(state: GameState) -> List[Dict[str, Any]]:
    """Return every currently valid move that makes progress toward the target."""
    results: List[Dict[str, Any]] = []
    seen_spans = set()

    current = state.current_structure
    target = state.target_structure
    t_spans = state.target_spans
    c_spans = state.current_spans

    for pos in ALL_COORDS:
        current_stack = current.get(pos, [])
        target_stack = target.get(pos, [])
        current_depth = len(current_stack)
        target_depth = len(target_stack)

        # Case A: target wants more blocks here.
        if current_depth < target_depth:
            needed_block = target_stack[current_depth]
            move = {
                "action": "place",
                "block": needed_block,
                "position": pos,
                "layer": current_depth,
                "span_to": None,
            }
            if needed_block.endswith("l"):
                partner, _ = _find_span_partner(pos, current_depth, t_spans)
                if partner is None:
                    results.append(
                        _unrunnable(
                            move,
                            FLAG_MISSING_SPAN,
                            "target_place",
                            f"No span partner found for {pos} layer {current_depth}",
                        )
                    )
                    continue
                canonical = tuple(sorted([pos, partner]))
                if canonical in seen_spans:
                    continue
                seen_spans.add(canonical)
                move["span_to"] = partner

            sim = state.simulate_move(move)
            if sim["ok"] and sim["structure_placement"]:
                results.append(
                    {
                        "move": move,
                        "structure_placement": True,
                        "side_placement": sim["side_placement"],
                        "overall_progress": sim["overall_progress"],
                        "flag": FLAG_OK,
                        "source": "target_place",
                    }
                )
            else:
                results.append(
                    _unrunnable(move, FLAG_SIM_FAILED, "target_place", sim.get("error", ""))
                )

        # Case B: current board has blocks the target does not want.
        elif current_depth > target_depth:
            top_layer = current_depth - 1
            top_block = current_stack[top_layer]
            move = {
                "action": "remove",
                "block": top_block,
                "position": pos,
                "layer": top_layer,
                "span_to": None,
            }
            if top_block.endswith("l"):
                partner, _ = _find_span_partner(pos, top_layer, c_spans)
                if partner is None:
                    results.append(
                        _unrunnable(
                            move,
                            FLAG_MISSING_SPAN,
                            "excess_remove",
                            f"No span partner found for {pos} layer {top_layer}",
                        )
                    )
                    continue
                canonical = tuple(sorted([pos, partner]))
                if canonical in seen_spans:
                    continue
                seen_spans.add(canonical)
                move["span_to"] = partner

            sim = state.simulate_move(move)
            if sim["ok"] and sim["structure_placement"]:
                results.append(
                    {
                        "move": move,
                        "structure_placement": True,
                        "side_placement": sim["side_placement"],
                        "overall_progress": sim["overall_progress"],
                        "flag": FLAG_OK,
                        "source": "excess_remove",
                    }
                )
            else:
                results.append(
                    _unrunnable(move, FLAG_SIM_FAILED, "excess_remove", sim.get("error", ""))
                )

        # Case C: same depth — fix the first wrong layer.
        else:
            for layer_idx in range(current_depth):
                if current_stack[layer_idx] == target_stack[layer_idx]:
                    continue
                if layer_idx == current_depth - 1:
                    top_block = current_stack[layer_idx]
                    move = {
                        "action": "remove",
                        "block": top_block,
                        "position": pos,
                        "layer": layer_idx,
                        "span_to": None,
                    }
                    if top_block.endswith("l"):
                        partner, _ = _find_span_partner(pos, layer_idx, c_spans)
                        if partner is None:
                            results.append(
                                _unrunnable(
                                    move,
                                    FLAG_MISSING_SPAN,
                                    "wrong_block_remove",
                                    f"No span partner for wrong large block at {pos} layer {layer_idx}",
                                )
                            )
                            break
                        canonical = tuple(sorted([pos, partner]))
                        if canonical in seen_spans:
                            break
                        seen_spans.add(canonical)
                        move["span_to"] = partner

                    sim = state.simulate_move(move)
                    if sim["ok"] and sim["structure_placement"]:
                        results.append(
                            {
                                "move": move,
                                "structure_placement": True,
                                "side_placement": sim["side_placement"],
                                "overall_progress": sim["overall_progress"],
                                "flag": FLAG_OK,
                                "source": "wrong_block_remove",
                            }
                        )
                    else:
                        results.append(
                            _unrunnable(
                                move,
                                FLAG_SIM_FAILED,
                                "wrong_block_remove",
                                sim.get("error", ""),
                            )
                        )
                else:
                    # Wrong block is buried — expose it by removing the correct top block.
                    top_layer = current_depth - 1
                    top_block = current_stack[top_layer]
                    move = {
                        "action": "remove",
                        "block": top_block,
                        "position": pos,
                        "layer": top_layer,
                        "span_to": None,
                    }
                    if top_block.endswith("l"):
                        partner, _ = _find_span_partner(pos, top_layer, c_spans)
                        if partner is None:
                            results.append(
                                _unrunnable(
                                    move,
                                    FLAG_MISSING_SPAN,
                                    "expose_buried_wrong",
                                    f"No span partner for top block at {pos} layer {top_layer}",
                                )
                            )
                            break
                        canonical = tuple(sorted([pos, partner]))
                        if canonical in seen_spans:
                            break
                        seen_spans.add(canonical)
                        move["span_to"] = partner

                    sim = state.simulate_move(move)
                    if sim["ok"]:
                        results.append(
                            {
                                "move": move,
                                "structure_placement": True,
                                "side_placement": sim["side_placement"],
                                "overall_progress": sim["overall_progress"],
                                "flag": FLAG_OK,
                                "source": "expose_buried_wrong",
                            }
                        )
                    else:
                        results.append(
                            _unrunnable(
                                move,
                                FLAG_SIM_FAILED,
                                "expose_buried_wrong",
                                sim.get("error", ""),
                            )
                        )
                break

    return results


def sample_oracle_moves(state: GameState, n: int, rng) -> List[Dict[str, Any]]:
    ok_moves = [entry["move"] for entry in enumerate_correct_actions(state) if entry["flag"] == FLAG_OK]
    if len(ok_moves) > n:
        ok_moves = rng.sample(ok_moves, n)
    return ok_moves
