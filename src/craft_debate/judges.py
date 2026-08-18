"""Optional SG / MM / PS LLM judges from the paper's Appendix E."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

JUDGE_SYSTEM = "You are a careful evaluator. Return only valid JSON."


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _format_oracle_moves(moves: List[Dict[str, Any]]) -> str:
    lines = []
    for m in moves:
        span = f" spanning to {m['span_to']}" if m.get("span_to") else ""
        if m["action"] == "place":
            lines.append(f"  PLACE {m['block']} at {m['position']} layer {m['layer']}{span}")
        else:
            lines.append(f"  REMOVE from {m['position']} layer {m['layer']}{span}")
    return "\n".join(lines) if lines else "(none)"


def sg_judge_prompt(
    target_view: Dict[str, Any],
    board_state: Dict[str, Any],
    oracle_moves: List[Dict[str, Any]],
    internal_thinking: str,
) -> str:
    return f"""You are evaluating the spatial grounding quality of a director agent in a collaborative construction task.
The director has a private view of one wall of a 3D target structure and must reason about what blocks are missing before instructing a builder.

TARGET VIEW (what this director needs the structure to look like):
{_json_dumps(target_view)}

CURRENT BOARD STATE:
{_json_dumps(board_state)}

ORACLE CORRECT MOVES THIS TURN:
{_format_oracle_moves(oracle_moves)}

DIRECTOR INTERNAL REASONING:
{internal_thinking}

EVALUATION:
For each question below answer with "Yes", "No", or "Unclear" and provide a brief one-sentence justification.

Questions:
1. Does the internal reasoning correctly identify at least one block that is missing from this director's visible wall, based on the target view and current board state?
2. Does the internal reasoning avoid describing blocks or positions that are already correctly placed on the board?
3. Does the internal reasoning reference the correct layer for the missing block, accounting for what is already stacked at that position?
4. Does the internal reasoning identify at least one action that matches or closely corresponds to one of the oracle correct moves?
5. Is the physical action implied by the internal reasoning executable given the current board state, respecting stacking order?
6. Does the internal reasoning correctly interpret the size of the missing block (small versus large) based on the target view?
7. Does the internal reasoning stay within this director's visible wall cells rather than describing cells belonging to another director?

Return your response as a JSON object with keys SG1 through SG7, each containing an "answer" field ("Yes", "No", or "Unclear") and a "reason" field. Return only valid JSON with no additional text."""


def mm_judge_prompt(
    internal_thinking: str,
    public_message: str,
    other_messages: Dict[str, str],
    conv_window: str,
) -> str:
    other_str = "\n".join(f"{d}: {msg}" for d, msg in other_messages.items() if msg)
    return f"""You are evaluating the Theory-of-Mind quality of a director agent in a collaborative construction task.
The director must produce a public message calibrated to what the builder and other directors already know, not just what she can see.

DIRECTOR INTERNAL REASONING (background context only):
{internal_thinking}

DIRECTOR PUBLIC MESSAGE (what was broadcast to all agents):
{public_message}

OTHER DIRECTORS' MESSAGES THIS TURN:
{other_str}

RECENT CONVERSATION HISTORY (last few turns):
{conv_window}

EVALUATION:
For each question below answer with "Yes", "No", or "Unclear" and provide a brief one-sentence justification.

Questions:
1. Does the public message add information not already communicated by the other directors in this turn or the immediately preceding turn?
2. Does the public message avoid repeating an instruction already given and acted upon in a previous turn?
3. Does the public message reflect awareness of what the builder already knows from the conversation history?
4. Does the public message accurately translate the key finding from the internal reasoning into natural language without losing critical spatial information?
5. Does the public message focus on information uniquely visible from this director's wall rather than information another director could have provided equally well?
6. Is the public message specific enough for the builder to execute without needing further clarification, naming a block type, location, and action?
7. If another director gave a conflicting instruction this turn, does the public message acknowledge or attempt to resolve the conflict?
8. Does the public message show awareness of the boundary between what this director uniquely sees and what other directors can also see?

Return your response as a JSON object with keys MM1 through MM8, each containing an "answer" field ("Yes", "No", or "Unclear") and a "reason" field. Return only valid JSON with no additional text."""


def ps_judge_prompt(
    director_messages: Dict[str, str],
    oracle_moves: List[Dict[str, Any]],
    board_state: Dict[str, Any],
    builder_confirmation: str,
    condition: str,
) -> str:
    msgs_str = "\n".join(f"  {did}: {msg}" for did, msg in director_messages.items() if msg.strip())
    condition_note = (
        "NOTE: The builder successfully selected an oracle correct move this turn."
        if condition == "C1_followed"
        else "NOTE: The builder did NOT select an oracle correct move this turn."
    )
    return f"""You are evaluating whether the collective director messages in a collaborative 3D construction task were pragmatically sufficient to guide a builder agent toward a correct verified move.
Three directors each hold a private 2D projection of a target 3D structure and must communicate with a builder through natural language only. The builder does not have access to the target structure and must infer the correct action from director messages alone.

CURRENT BOARD STATE:
{_json_dumps(board_state)}

ORACLE CORRECT MOVES THIS TURN (verified by game engine as making forward progress toward target):
{_format_oracle_moves(oracle_moves)}

DIRECTOR MESSAGES THIS TURN:
{msgs_str}

BUILDER RESPONSE AND REASONING:
{builder_confirmation}

{condition_note}

EVALUATION — for each question answer "Yes", "No", or "Unclear" with a brief one-sentence justification.

PS1. Do the director messages collectively identify at least one specific location on the board that needs a block placed or removed?
PS2. Do the director messages collectively specify the correct block type — both color AND size (small vs large/domino) — for at least one of the oracle correct moves?
PS3. Would a rational builder reading only these director messages have sufficient information to select at least one oracle correct move without needing to perform independent spatial reasoning about the target structure?
PS4. Do the director messages use precise spatial anchors that uniquely identify the target location (explicit coordinates, unambiguous landmark references, or clear directional anchors), rather than vague relative language that could map to multiple grid positions?
PS5. Does the builder confirmation indicate it correctly understood the directors' collective intent for this turn, regardless of whether the move execution ultimately succeeded?
PS6. If the builder did not execute the correct move, was the failure primarily attributable to director underspecification (missing position, wrong block type, wrong size, ambiguous spatial language) rather than builder execution mechanics (wrong layer computation, missing span endpoint, stacking constraint violation)? Answer "N/A" if the builder successfully executed an oracle correct move this turn.

Return your response as a JSON object with keys PS1 through PS6, each containing an "answer" field ("Yes", "No", "Unclear", or "N/A") and a "reason" field (one sentence). Return only valid JSON with no additional text."""


def _parse_json_response(raw: str) -> Optional[Dict[str, Any]]:
    text = raw.strip()
    for candidate in (text, text[text.find("{") : text.rfind("}") + 1] if "{" in text else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _score_answers(parsed: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in parsed.items():
        if isinstance(value, dict) and "answer" in value:
            answer = str(value.get("answer", "")).strip().lower()
            numeric = {"yes": 1.0, "no": 0.0, "unclear": 0.5}.get(answer)
            result[key] = {
                "answer": answer or value.get("answer"),
                "reason": value.get("reason", ""),
                "score": numeric,
            }
    return result


async def _call(client: Any, prompt: str) -> Dict[str, Any]:
    response = await client.complete(JUDGE_SYSTEM, prompt)
    parsed = _parse_json_response(response["content"])
    return {
        "raw_response": response["content"],
        "parsed": parsed,
        "parse_error": None if parsed else "Judge response was not valid JSON",
    }


async def run_judges(
    client: Any,
    *,
    board_state: Dict[str, Any],
    oracle_moves: List[Dict[str, Any]],
    proposers: List[Dict[str, Any]],
    judge_confirmation: str,
    conversation_window: str,
    condition: str,
) -> Dict[str, Any]:
    """Run SG/MM per proposer and PS once. Returns a JSON-safe result tree."""
    sg_tasks = [
        _call(
            client,
            sg_judge_prompt(
                target_view=p["target_view"],
                board_state=board_state,
                oracle_moves=oracle_moves,
                internal_thinking=p["internal_thinking"],
            ),
        )
        for p in proposers
    ]
    messages_by_director = {
        p["director_id"]: p["public_message"] for p in proposers
    }
    mm_tasks = [
        _call(
            client,
            mm_judge_prompt(
                internal_thinking=p["internal_thinking"],
                public_message=p["public_message"],
                other_messages={
                    d: m for d, m in messages_by_director.items() if d != p["director_id"]
                },
                conv_window=conversation_window,
            ),
        )
        for p in proposers
    ]
    ps_task = _call(
        client,
        ps_judge_prompt(
            director_messages=messages_by_director,
            oracle_moves=oracle_moves,
            board_state=board_state,
            builder_confirmation=judge_confirmation,
            condition=condition,
        ),
    )

    sg_results, mm_results, ps_result = await asyncio.gather(
        asyncio.gather(*sg_tasks),
        asyncio.gather(*mm_tasks),
        ps_task,
    )

    sg_scored = [
        _score_answers(res["parsed"] or {}, "SG") | {"_raw": res} for res in sg_results
    ]
    mm_scored = [
        _score_answers(res["parsed"] or {}, "MM") | {"_raw": res} for res in mm_results
    ]
    ps_scored = _score_answers(ps_result["parsed"] or {}, "PS")
    ps_scored["_raw"] = ps_result

    def dimension_mean(rows: List[Dict[str, Any]]) -> Optional[float]:
        values = [
            v["score"]
            for row in rows
            for v in row.values()
            if isinstance(v, dict) and v.get("score") is not None
        ]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "SG": {
            "per_proposer": [
                {
                    "agent_id": p["agent_id"],
                    "questions": {k: v for k, v in scored.items() if not k.startswith("_")},
                    "raw_response": scored["_raw"]["raw_response"],
                    "parse_error": scored["_raw"]["parse_error"],
                }
                for p, scored in zip(proposers, sg_scored)
            ],
            "mean": dimension_mean(sg_scored),
        },
        "MM": {
            "per_proposer": [
                {
                    "agent_id": p["agent_id"],
                    "questions": {k: v for k, v in scored.items() if not k.startswith("_")},
                    "raw_response": scored["_raw"]["raw_response"],
                    "parse_error": scored["_raw"]["parse_error"],
                }
                for p, scored in zip(proposers, mm_scored)
            ],
            "mean": dimension_mean(mm_scored),
        },
        "PS": {
            "questions": {k: v for k, v in ps_scored.items() if not k.startswith("_")},
            "raw_response": ps_scored["_raw"]["raw_response"],
            "parse_error": ps_scored["_raw"]["parse_error"],
            "mean": dimension_mean([ps_scored]),
        },
    }
