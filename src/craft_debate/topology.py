"""Oracle-free CRAFT debate: 3 Directors -> 3 reconciliations -> 1 Builder."""

from __future__ import annotations

import asyncio
import copy
import re
import time
from typing import Any, Dict, List, Optional

from .domain import get_director_views
from .environment import (
    GameState,
    get_all_physically_legal_actions,
    validate_physical_action,
)
from .progress import calculate_progress
from .prompts import (
    build_builder_prompt,
    build_observation_prompt,
    build_reconciliation_prompt,
)

OBSERVATION_SYSTEM = (
    "You are a CRAFT Director making an independent partial-observation proposal. "
    "Respect the information boundary and exact tagged format."
)
RECONCILIATION_SYSTEM = (
    "You are the same CRAFT Director reconciling three communicated proposals using "
    "your own private evidence. Respect the information boundary and tagged format."
)
BUILDER_SYSTEM = (
    "You are the CRAFT Builder. Select exactly one ID from the supplied complete legal mask."
)


def _extract_tag(text: str, tag: str) -> Optional[str]:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _structured_fields(text: str, fields: List[str]) -> Dict[str, str]:
    return {field: _extract_tag(text, field) or "" for field in fields}


def parse_builder_response(text: str) -> Dict[str, Any]:
    wrapped = _extract_tag(text, "action_id")
    candidate = (wrapped or "").strip().upper()
    if not re.fullmatch(r"A\d{4,}", candidate):
        return {
            "action_id": None,
            "parse_error": "No valid <action_id>A####</action_id> found",
            "raw_response": text,
        }
    return {"action_id": candidate, "parse_error": None, "raw_response": text}


def parse_judge_response(text: str) -> Dict[str, Any]:
    """Legacy parser retained for the separate paper-protocol runner."""
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
    action = match.group(1).lower()
    parts = [part.strip() for part in match.group(2).split(":")]
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
            index = 3
            span_to = None
            if parts[index].lower() != "confirm":
                span_to = parts[index]
                index += 1
            if parts[index].lower() != "confirm":
                raise ValueError("Missing CONFIRM marker")
            parsed["move"] = {
                "action": "place",
                "block": block.lower(),
                "position": position,
                "layer": layer,
                "span_to": span_to,
            }
        else:
            if len(parts) < 3:
                raise ValueError("REMOVE needs position:layer[:span_to]:CONFIRM")
            position, layer = parts[0], int(parts[1])
            index = 2
            span_to = None
            if parts[index].lower() != "confirm":
                span_to = parts[index]
                index += 1
            if parts[index].lower() != "confirm":
                raise ValueError("Missing CONFIRM marker")
            parsed["move"] = {
                "action": "remove",
                "block": None,
                "position": position,
                "layer": layer,
                "span_to": span_to,
            }
        parsed["confirmation"] = ":".join(parts[index + 1 :])
    except (IndexError, ValueError) as exc:
        parsed["action"] = "unparsed"
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


class Debate:
    """One fixed 3+3+1 structure/run instance."""

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
            self.clients = {
                "phase1": clients.get("phase1") or clients.get("proposers"),
                "reconciliation": clients.get("reconciliation") or clients.get("critics"),
                "builder": clients.get("builder") or clients.get("judge"),
            }
        else:
            self.clients = {"phase1": client, "reconciliation": client, "builder": client}
        self.verbose = verbose

        debate_cfg = config["debate"]
        target = structure_data["structure"]
        spans = structure_data["spans"]
        self.env = GameState(
            target,
            spans,
            start_from_empty=debate_cfg.get("start_from_empty", True),
        )
        self.private_views = get_director_views(target, spans)
        self.max_rounds = int(debate_cfg["max_rounds"])
        self.terminate_early = bool(debate_cfg.get("terminate_early", False))
        self.history_cfg = debate_cfg.get("history", {"max_messages": 50, "trim_to": 40})
        roles = debate_cfg["roles"]
        self.directors = roles.get("directors") or roles.get("proposers") or []
        self.builder = roles.get("builder") or roles.get("judge") or {"id": "Builder"}
        if len(self.directors) != 3 or {r.get("director") for r in self.directors} != {"D1", "D2", "D3"}:
            raise ValueError("debate.roles.directors must define exactly D1, D2, and D3")

        self.public_history: List[str] = []
        self.previous_builder_result: Optional[str] = None
        self.rounds: List[Dict[str, Any]] = []
        self.baseline = calculate_progress(
            self.env.current_structure, self.env.target_structure
        )

    async def run_round(self, round_number: int) -> Dict[str, Any]:
        public_state = self.env.snapshot()
        history_text = "\n".join(self.public_history)
        record: Dict[str, Any] = {
            "round_number": round_number,
            "public_state_before": copy.deepcopy(public_state),
            "public_history_before": list(self.public_history),
        }

        phase_started = time.monotonic()
        observations = await asyncio.gather(
            *[
                self._run_observation(role, public_state, history_text)
                for role in self.directors
            ]
        )
        phase1_latency = round(time.monotonic() - phase_started, 3)
        record["phase1"] = observations
        phase1_messages = {
            item["director_id"]: {
                "observation": item["observation"],
                "proposed_action": item["proposed_action"],
                "reasoning": item["reasoning"],
                "confidence": item["confidence"],
            }
            for item in observations
        }

        phase_started = time.monotonic()
        reconciliations = await asyncio.gather(
            *[
                self._run_reconciliation(
                    role, public_state, phase1_messages, history_text
                )
                for role in self.directors
            ]
        )
        reconciliation_latency = round(time.monotonic() - phase_started, 3)
        record["reconciliation"] = reconciliations
        reconciliation_messages = {
            item["director_id"]: {
                key: item[key]
                for key in (
                    "agreement",
                    "contradictions",
                    "revision",
                    "complementary_evidence",
                    "recommended_action",
                    "reasoning",
                    "confidence",
                )
            }
            for item in reconciliations
        }

        legal_actions = get_all_physically_legal_actions(
            public_state, self.env.available_blocks
        )
        record["legal_action_mask"] = copy.deepcopy(legal_actions)
        builder_prompt = build_builder_prompt(
            builder_id=self.builder["id"],
            public_state=public_state,
            reconciliations=reconciliation_messages,
            legal_actions=legal_actions,
        )
        phase_started = time.monotonic()
        builder_response = await self.clients["builder"].complete(
            BUILDER_SYSTEM,
            builder_prompt,
            {"kind": "builder", "legal_actions": legal_actions},
        )
        builder_phase_latency = round(time.monotonic() - phase_started, 3)
        builder = parse_builder_response(builder_response["content"])
        selected = next(
            (action for action in legal_actions if action["id"] == builder["action_id"]),
            None,
        )
        builder.update(
            {
                "selected_action": copy.deepcopy(selected),
                "prompt": builder_prompt,
                "usage": builder_response.get("usage", {}),
                "latency_seconds": builder_response.get("latency_seconds"),
            }
        )
        record["builder"] = builder
        record["phase_latency_seconds"] = {
            "phase1": phase1_latency,
            "reconciliation": reconciliation_latency,
            "builder": builder_phase_latency,
        }

        physical_validation, execution, evaluation = self._validate_execute_evaluate(
            public_state, selected, builder
        )
        record["physical_validation"] = physical_validation
        record["execution"] = execution
        record["evaluation"] = evaluation
        # Kept as an output alias for existing plot/summary readers.
        record["score"] = evaluation["score"]

        self._append_public_result(builder, physical_validation, execution)
        self._trim_history()
        if self.verbose:
            print(
                f"  [Debate] round {round_number:>2}/{self.max_rounds} | "
                f"action_id={builder.get('action_id') or '-':<6} executed={execution['ok']} "
                f"| overall_progress={evaluation['score']['overall_progress']:.4f} "
                f"(delta {evaluation['score']['delta']:+.4f})"
            )
        return record

    async def _run_observation(
        self, role: Dict[str, str], public_state: Dict[str, Any], history: str
    ) -> Dict[str, Any]:
        director_id = role["director"]
        prompt = build_observation_prompt(
            director_id=director_id,
            archetype=role.get("archetype", "observant"),
            private_view=self.private_views[director_id],
            public_state=public_state,
            public_history=history,
            previous_builder_result=self.previous_builder_result,
        )
        response = await self.clients["phase1"].complete(
            OBSERVATION_SYSTEM, prompt, {"kind": "observation", "director_id": director_id}
        )
        parsed = _structured_fields(
            response["content"], ["observation", "proposed_action", "reasoning", "confidence"]
        )
        return {
            "agent_id": role.get("id", director_id),
            "director_id": director_id,
            "archetype": role.get("archetype", "observant"),
            **parsed,
            "prompt": prompt,
            "raw_response": response["content"],
            "usage": response.get("usage", {}),
            "latency_seconds": response.get("latency_seconds"),
        }

    async def _run_reconciliation(
        self,
        role: Dict[str, str],
        public_state: Dict[str, Any],
        phase1_messages: Dict[str, Dict[str, str]],
        history: str,
    ) -> Dict[str, Any]:
        director_id = role["director"]
        prompt = build_reconciliation_prompt(
            director_id=director_id,
            archetype=role.get("archetype", "observant"),
            private_view=self.private_views[director_id],
            public_state=public_state,
            phase1_messages=phase1_messages,
            public_history=history,
            previous_builder_result=self.previous_builder_result,
        )
        response = await self.clients["reconciliation"].complete(
            RECONCILIATION_SYSTEM,
            prompt,
            {"kind": "reconciliation", "director_id": director_id},
        )
        fields = [
            "agreement",
            "contradictions",
            "revision",
            "complementary_evidence",
            "recommended_action",
            "reasoning",
            "confidence",
        ]
        parsed = _structured_fields(response["content"], fields)
        return {
            "agent_id": role.get("id", director_id),
            "director_id": director_id,
            "archetype": role.get("archetype", "observant"),
            **parsed,
            "prompt": prompt,
            "raw_response": response["content"],
            "usage": response.get("usage", {}),
            "latency_seconds": response.get("latency_seconds"),
        }

    def _validate_execute_evaluate(
        self,
        public_state: Dict[str, Any],
        selected: Optional[Dict[str, Any]],
        builder: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        if selected is None:
            reason = builder.get("parse_error") or "Selected action ID is not in legal mask"
            validation = {
                "ok": False,
                "reason": reason,
                "validator": "deterministic_public_physics",
            }
            execution = {
                "ok": False,
                "error": reason,
                "move": None,
                "public_state_after": self.env.snapshot(),
            }
            target_metrics: Dict[str, Any] = {}
        else:
            ok, reason, normalized = validate_physical_action(
                public_state, selected, self.env.available_blocks
            )
            validation = {
                "ok": ok,
                "reason": reason,
                "validator": "deterministic_public_physics",
                "action_id": selected["id"],
                "normalized_action": normalized,
            }
            if ok:
                result = self.env.execute_move(normalized)
                execution = {
                    "ok": result["ok"],
                    "error": result["error"],
                    "move": result["move"],
                    "public_state_after": self.env.snapshot(),
                }
                target_metrics = {
                    "structure_placement": result.get("structure_placement"),
                    "side_placement": result.get("side_placement"),
                    "board_complete": result.get("board_complete"),
                }
            else:
                execution = {
                    "ok": False,
                    "error": reason,
                    "move": normalized,
                    "public_state_after": self.env.snapshot(),
                }
                target_metrics = {}

        score = calculate_progress(self.env.current_structure, self.env.target_structure)
        previous = (
            self.rounds[-1]["score"]["overall_progress"]
            if self.rounds
            else self.baseline["overall_progress"]
        )
        score["delta"] = round(score["overall_progress"] - previous, 6)
        return validation, execution, {"score": score, **target_metrics}

    def _append_public_result(
        self,
        builder: Dict[str, Any],
        validation: Dict[str, Any],
        execution: Dict[str, Any],
    ) -> None:
        if execution["ok"] and execution.get("move"):
            result = f"Builder executed {builder['action_id']}: {_format_move(execution['move'])}."
        else:
            result = f"Builder action failed validation/execution: {validation['reason']}."
        self.previous_builder_result = result
        self.public_history.append(result)

    def _trim_history(self) -> None:
        maximum = int(self.history_cfg.get("max_messages", 50))
        trim_to = int(self.history_cfg.get("trim_to", 40))
        if len(self.public_history) > maximum:
            self.public_history = self.public_history[-trim_to:]

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
            # Ground truth is retained only in the offline trajectory/evaluation record.
            "evaluation_target_structure": copy.deepcopy(self.structure_data["structure"]),
            "evaluation_target_spans": {
                str(k): list(v) for k, v in self.structure_data["spans"].items()
            },
            "director_roles": list(self.directors),
            "builder_role": self.builder,
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
            if self.terminate_early and record["evaluation"].get("board_complete"):
                break
        return self.game_record()
