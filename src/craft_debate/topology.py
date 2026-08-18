"""The Debate topology: 3 proposers -> 3 critics -> 1 judge per round.

The first three agents answer the question in parallel. Their answers are
concatenated and handed to the second layer of three agents, which debate and
critique them. The six outputs are then aggregated to the final agent, which
produces one synthesized answer. That answer is executed in the CRAFT
environment and its resulting task progress is the round's score. In the next
round the proposers see the question again plus the previous round's
synthesized answer.
"""

from __future__ import annotations

import asyncio
import copy
import json
import random
import re
from typing import Any, Dict, List, Optional

from .domain import get_director_views
from .environment import GameState
from .oracle import sample_oracle_moves
from .progress import calculate_progress
from .prompts import (
    CRITIC_FOCUS_LABELS,
    build_critic_prompt,
    build_judge_prompt,
    build_proposer_prompt,
)

PROPOSER_SYSTEM = (
    "You are a proposer in a 7-agent Debate about a grounded 3D block-construction task. "
    "Answer the question in the exact <think>/<message> format requested."
)
CRITIC_SYSTEM = (
    "You are a critic in a 7-agent Debate about a grounded 3D block-construction task. "
    "Debate the proposers' answers in the exact <critique>/<message> format requested."
)
JUDGE_SYSTEM = (
    "You are the final judge in a 7-agent Debate about a grounded 3D block-construction "
    "task. Synthesize exactly one builder move in the exact <move> format requested."
)


# ------------------------------------------------------------------------- parsing
def _extract_tag(text: str, tag: str) -> Optional[str]:
    match = re.search(
        rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    open_match = re.search(rf"<{tag}>\s*(.*)$", text, re.DOTALL | re.IGNORECASE)
    if open_match:
        return open_match.group(1).strip()
    return None


def parse_proposer_response(text: str) -> Dict[str, str]:
    thinking = _extract_tag(text, "think")
    message = _extract_tag(text, "message")
    return {
        "internal_thinking": thinking or "",
        "public_message": message or "No message provided",
        "raw_response": text,
    }


def parse_critic_response(text: str) -> Dict[str, str]:
    critique = _extract_tag(text, "critique")
    message = _extract_tag(text, "message")
    return {
        "critique": critique or "",
        "public_message": message or "No message provided",
        "raw_response": text,
    }


def parse_judge_response(text: str) -> Dict[str, Any]:
    """Parse a builder-style move line into a move dict or a clarification."""
    wrapped = _extract_tag(text, "move") or ""
    haystack = wrapped or text
    match = re.search(
        r"^\s*(PLACE|REMOVE|CLARIFY)\s*:(.*)$", haystack, re.MULTILINE | re.IGNORECASE
    )
    if not match:
        return {
            "action": "unparsed",
            "move": None,
            "clarification": None,
            "confirmation": "",
            "parse_error": "No PLACE/REMOVE/CLARIFY line found",
            "raw_response": text,
        }

    action = match.group(1).strip().lower()
    parts = [p.strip() for p in match.group(2).split(":")]
    parsed: Dict[str, Any] = {
        "action": action,
        "move": None,
        "clarification": None,
        "confirmation": "",
        "parse_error": None,
        "raw_response": text,
    }

    if action == "clarify":
        parsed["clarification"] = " ".join(parts)
        return parsed

    try:
        if action == "place":
            if len(parts) < 4:
                raise ValueError("PLACE needs block:position:layer[:span_to]:CONFIRM")
            block, position, layer = parts[0], parts[1], int(parts[2])
            idx = 3
            span_to = None
            if parts[idx].lower() != "confirm":
                span_to = parts[idx]
                idx += 1
            if parts[idx].lower() != "confirm":
                raise ValueError("Missing CONFIRM marker in PLACE line")
            confirmation = ":".join(parts[idx + 1 :]) if len(parts) > idx + 1 else ""
            parsed["move"] = {
                "action": "place",
                "block": block.lower(),
                "position": position,
                "layer": layer,
                "span_to": span_to,
            }
            parsed["confirmation"] = confirmation
        elif action == "remove":
            if len(parts) < 3:
                raise ValueError("REMOVE needs position:layer[:span_to]:CONFIRM")
            position, layer = parts[0], int(parts[1])
            idx = 2
            span_to = None
            if parts[idx].lower() != "confirm":
                span_to = parts[idx]
                idx += 1
            if parts[idx].lower() != "confirm":
                raise ValueError("Missing CONFIRM marker in REMOVE line")
            confirmation = ":".join(parts[idx + 1 :]) if len(parts) > idx + 1 else ""
            parsed["move"] = {
                "action": "remove",
                "block": None,
                "position": position,
                "layer": layer,
                "span_to": span_to,
            }
            parsed["confirmation"] = confirmation
    except (ValueError, IndexError) as exc:
        parsed["action"] = "unparsed"
        parsed["move"] = None
        parsed["parse_error"] = str(exc)
    return parsed


def _format_move(move: Dict[str, Any]) -> str:
    if move["action"] == "place":
        text = f"PLACE {move['block']} at {move['position']} layer {move['layer']}"
    else:
        text = f"REMOVE from {move['position']} layer {move['layer']}"
    if move.get("span_to"):
        text += f" spanning to {move['span_to']}"
    return text


# ----------------------------------------------------------------------- topology
class Debate:
    """One structure-run instance of the 7-agent Debate."""

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        structure_data: Dict[str, Any],
        structure_index: int,
        run_index: int,
        clients: Optional[Dict[str, Any]] = None,
        client: Any = None,
        judges_client: Optional[Any] = None,
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.structure_data = structure_data
        self.structure_index = structure_index
        self.run_index = run_index
        if clients is not None:
            self.clients = clients
        else:  # backward-compatible fallback: one model for everything
            self.clients = {
                "proposers": client,
                "critics": client,
                "judge": client,
                "judges": judges_client or client,
            }
        self.verbose = verbose

        debate_cfg = config["debate"]
        target = structure_data["structure"]
        spans = structure_data["spans"]
        self.env = GameState(
            target,
            spans,
            start_from_empty=debate_cfg.get("start_from_empty", True),
        )
        self.target_views = get_director_views(target, spans)
        self.max_rounds = int(debate_cfg["max_rounds"])
        self.terminate_early = bool(debate_cfg.get("terminate_early", False))
        self.oracle_cfg = debate_cfg.get("oracle", {"enabled": True, "n": 5})
        self.history_cfg = debate_cfg.get("history", {"max_messages": 50, "trim_to": 40})
        self.roles = debate_cfg["roles"]
        self.run_judges = bool(config.get("judges", {}).get("enabled", False))

        seed = int(config.get("seed", 42))
        self.rng = random.Random(seed * 1000 + structure_index * 10 + run_index)
        self.conversation: List[str] = []
        self.prev_judge_answer: Optional[str] = None
        self.rounds: List[Dict[str, Any]] = []
        self.baseline = calculate_progress(
            self.env.current_structure, self.env.target_structure
        )

    # ---------------------------------------------------------------- round
    async def run_round(self, round_number: int) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "round_number": round_number,
            "structure_before": copy.deepcopy(self.env.current_structure),
            "spans_before": {str(k): list(v) for k, v in self.env.current_spans.items()},
            "conversation_before": list(self.conversation),
        }
        conversation_text = "\n".join(self.conversation)

        # ---- layer 1: three proposers answer the question in parallel ----
        proposers = await asyncio.gather(
            *[self._run_proposer(role, conversation_text, round_number) for role in self.roles["proposers"]]
        )
        record["proposers"] = proposers
        proposer_messages = {
            f"{p['agent_id']} ({p['director_id']})": p["public_message"] for p in proposers
        }
        for p in proposers:
            self.conversation.append(f"{p['director_id']} ({p['agent_id']}): {p['public_message']}")

        # ---- layer 2: three critics debate the proposers in parallel ----
        critics = await asyncio.gather(
            *[
                self._run_critic(role, conversation_text, proposer_messages, round_number)
                for role in self.roles["critics"]
            ]
        )
        record["critics"] = critics
        critic_messages = {
            f"{c['agent_id']} ({CRITIC_FOCUS_LABELS.get(c['focus'], c['focus'])})": c["public_message"]
            for c in critics
        }
        for c in critics:
            focus = CRITIC_FOCUS_LABELS.get(c["focus"], c["focus"])
            self.conversation.append(f"{c['agent_id']} ({focus}): {c['public_message']}")

        # ---- oracle candidates (paper's oracle-assisted Builder) ----
        oracle_moves: Optional[List[Dict[str, Any]]] = None
        if self.oracle_cfg.get("enabled", True):
            oracle_rng = random.Random(self.structure_index * 1000 + round_number)
            oracle_moves = sample_oracle_moves(self.env, int(self.oracle_cfg.get("n", 5)), oracle_rng)
            record["oracle_moves"] = copy.deepcopy(oracle_moves)

        # ---- final layer: one judge synthesizes the answer ----
        judge_id = self.roles["judge"]["id"]
        judge_prompt = build_judge_prompt(
            agent_id=judge_id,
            board_state=self.env.current_structure,
            available_blocks=self.env.available_blocks,
            conversation=conversation_text,
            proposer_messages=proposer_messages,
            critic_messages=critic_messages,
            oracle_moves=oracle_moves,
            prev_judge_answer=self.prev_judge_answer,
            round_number=round_number,
        )
        judge_resp = await self.clients["judge"].complete(
            JUDGE_SYSTEM, judge_prompt, {"kind": "judge", "oracle_moves": oracle_moves}
        )
        parsed_judge = parse_judge_response(judge_resp["content"])
        parsed_judge["prompt"] = judge_prompt
        parsed_judge["usage"] = judge_resp.get("usage", {})
        parsed_judge["latency_seconds"] = judge_resp.get("latency_seconds")
        record["judge"] = parsed_judge

        # ---- execute the judge's answer and score it ----
        execution = self._execute_judge_answer(parsed_judge, judge_id)
        record["execution"] = execution

        score = calculate_progress(self.env.current_structure, self.env.target_structure)
        score["delta"] = round(
            score["overall_progress"]
            - (self.rounds[-1]["score"]["overall_progress"] if self.rounds else self.baseline["overall_progress"]),
            6,
        )
        record["score"] = score

        # ---- optional paper judges (SG / MM / PS) ----
        if self.run_judges and self.clients.get("judges") is not None:
            from .judges import run_judges

            followed = bool(
                execution.get("move")
                and oracle_moves
                and any(
                    execution["move"].get("action") == m["action"]
                    and execution["move"].get("position") == m["position"]
                    and execution["move"].get("layer") == m["layer"]
                    for m in oracle_moves
                )
            )
            record["judge_evals"] = await run_judges(
                self.clients["judges"],
                board_state=self.env.current_structure,
                oracle_moves=oracle_moves or [],
                proposers=[
                    {
                        "agent_id": p["agent_id"],
                        "director_id": p["director_id"],
                        "target_view": p["target_view"],
                        "internal_thinking": p["internal_thinking"],
                        "public_message": p["public_message"],
                    }
                    for p in proposers
                ],
                judge_confirmation=parsed_judge.get("confirmation", ""),
                conversation_window="\n".join(self.conversation[-6:]),
                condition="C1_followed" if followed else "C2_not_followed",
            )

        # ---- keep the synthesized answer visible to next round's proposers ----
        if parsed_judge["action"] == "clarify":
            self.prev_judge_answer = f"CLARIFY: {parsed_judge['clarification']}"
        elif execution["ok"] is False:
            self.prev_judge_answer = f"FAILED: {execution['error']}"
        elif execution.get("move"):
            summary = _format_move(execution["move"])
            self.prev_judge_answer = (
                f"{summary}. {parsed_judge.get('confirmation') or ''}".strip()
            )
        else:
            self.prev_judge_answer = f"Unparsed answer: {parsed_judge.get('parse_error')}"

        self._trim_history()
        if self.verbose:
            action = parsed_judge.get("action", "?")
            ok = execution.get("ok")
            print(
                f"  [Debate] round {round_number:>2}/{self.max_rounds} | judge={action:<8} "
                f"executed={ok} | overall_progress={score['overall_progress']:.4f} "
                f"(delta {score['delta']:+.4f})"
            )
        return record

    async def _run_proposer(
        self, role: Dict[str, str], conversation_text: str, round_number: int
    ) -> Dict[str, Any]:
        agent_id = role["id"]
        director_id = role["director"]
        archetype = role.get("archetype", "observant")
        prompt = build_proposer_prompt(
            agent_id=agent_id,
            director_id=director_id,
            archetype=archetype,
            target_view=self.target_views[director_id],
            board_state=self.env.current_structure,
            conversation=conversation_text,
            prev_judge_answer=self.prev_judge_answer,
            round_number=round_number,
            available_blocks=self.env.available_blocks,
        )
        response = await self.clients["proposers"].complete(
            PROPOSER_SYSTEM, prompt, {"kind": "proposer", "agent_id": agent_id}
        )
        parsed = parse_proposer_response(response["content"])
        return {
            "agent_id": agent_id,
            "director_id": director_id,
            "archetype": archetype,
            "target_view": copy.deepcopy(self.target_views[director_id]),
            "prompt": prompt,
            "internal_thinking": parsed["internal_thinking"],
            "public_message": parsed["public_message"],
            "raw_response": parsed["raw_response"],
            "usage": response.get("usage", {}),
            "latency_seconds": response.get("latency_seconds"),
        }

    async def _run_critic(
        self,
        role: Dict[str, str],
        conversation_text: str,
        proposer_messages: Dict[str, str],
        round_number: int,
    ) -> Dict[str, Any]:
        agent_id = role["id"]
        focus = role["focus"]
        prompt = build_critic_prompt(
            agent_id=agent_id,
            focus=focus,
            board_state=self.env.current_structure,
            conversation=conversation_text,
            proposer_messages=proposer_messages,
            prev_judge_answer=self.prev_judge_answer,
            round_number=round_number,
        )
        response = await self.clients["critics"].complete(
            CRITIC_SYSTEM, prompt, {"kind": "critic", "agent_id": agent_id}
        )
        parsed = parse_critic_response(response["content"])
        return {
            "agent_id": agent_id,
            "focus": focus,
            "prompt": prompt,
            "critique": parsed["critique"],
            "public_message": parsed["public_message"],
            "raw_response": parsed["raw_response"],
            "usage": response.get("usage", {}),
            "latency_seconds": response.get("latency_seconds"),
        }

    def _execute_judge_answer(
        self, parsed_judge: Dict[str, Any], judge_id: str
    ) -> Dict[str, Any]:
        if parsed_judge["action"] == "clarify":
            question = parsed_judge.get("clarification") or ""
            self.conversation.append(f"Judge ({judge_id}): CLARIFY — {question}")
            return {
                "action": "clarify",
                "move": None,
                "ok": None,
                "error": None,
                "clarification": question,
                "structure_after": copy.deepcopy(self.env.current_structure),
                "spans_after": {str(k): list(v) for k, v in self.env.current_spans.items()},
            }

        move = parsed_judge.get("move")
        confirmation = parsed_judge.get("confirmation") or ""
        if not move:
            self.conversation.append(
                f"Judge ({judge_id}): FAILED — could not parse an answer "
                f"({parsed_judge.get('parse_error')})"
            )
            return {
                "action": "unparsed",
                "move": None,
                "ok": False,
                "error": parsed_judge.get("parse_error"),
                "structure_after": copy.deepcopy(self.env.current_structure),
                "spans_after": {str(k): list(v) for k, v in self.env.current_spans.items()},
            }

        result = self.env.execute_move(move)
        if result["ok"]:
            board_json = json.dumps(self.env.current_structure)
            self.conversation.append(
                f"Judge ({judge_id}): {confirmation or _format_move(move)}. "
                f"Current board: {board_json}"
            )
        else:
            board_json = json.dumps(self.env.current_structure)
            self.conversation.append(
                f"Judge ({judge_id}): FAILED — {result['error']}. "
                f"Board unchanged: {board_json}"
            )
        return {
            "action": move["action"],
            "move": result["move"],
            "confirmation": confirmation,
            "ok": result["ok"],
            "error": result["error"],
            "removed_block": result.get("removed_block"),
            "structure_placement": result.get("structure_placement"),
            "side_placement": result.get("side_placement"),
            "board_complete": result.get("board_complete"),
            "structure_after": copy.deepcopy(self.env.current_structure),
            "spans_after": {str(k): list(v) for k, v in self.env.current_spans.items()},
        }

    def _trim_history(self) -> None:
        max_messages = int(self.history_cfg.get("max_messages", 50))
        trim_to = int(self.history_cfg.get("trim_to", 40))
        if len(self.conversation) > max_messages:
            self.conversation = self.conversation[-trim_to:]

    # -------------------------------------------------------------- results
    def game_record(self) -> Dict[str, Any]:
        final_score = (
            self.rounds[-1]["score"]["overall_progress"]
            if self.rounds
            else self.baseline["overall_progress"]
        )
        return {
            "structure_id": self.structure_data["id"],
            "structure_index": self.structure_index,
            "complexity": self.structure_data["complexity"],
            "metadata": self.structure_data.get("metadata", {}),
            "run_index": self.run_index,
            "target_structure": copy.deepcopy(self.structure_data["structure"]),
            "target_spans": {str(k): list(v) for k, v in self.structure_data["spans"].items()},
            "target_director_views": copy.deepcopy(self.target_views),
            "proposer_roles": list(self.roles["proposers"]),
            "critic_roles": list(self.roles["critics"]),
            "judge_role": self.roles["judge"],
            "baseline_progress": self.baseline["overall_progress"],
            "rounds": self.rounds,
            "final_progress": final_score,
            "final_structure": copy.deepcopy(self.env.current_structure),
            "final_spans": {str(k): list(v) for k, v in self.env.current_spans.items()},
            "completed": self.env.board_complete(),
            "rounds_completed": len(self.rounds),
        }

    async def run(self) -> Dict[str, Any]:
        for round_number in range(1, self.max_rounds + 1):
            record = await self.run_round(round_number)
            self.rounds.append(record)
            if self.terminate_early and record["execution"].get("board_complete"):
                break
        return self.game_record()
