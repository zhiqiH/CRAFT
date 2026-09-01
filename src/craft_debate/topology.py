"""Generative CRAFT debate: 3 Directors -> 3 reconciliations -> 1 Builder."""

from __future__ import annotations

import asyncio
import copy
import re
import time
from typing import Any, Dict, List, Optional

from .domain import get_director_views, norm_pos
from .environment import GameState, validate_physical_action
from .progress import calculate_progress
from .prompts import (
    build_builder_prompt,
    build_observation_prompt,
    build_reconciliation_prompt,
)

OBSERVATION_SYSTEM = (
    "You are a CRAFT Director making an independent wall observation. Keep your "
    "analysis private and communicate only a concise natural-language message."
)
RECONCILIATION_SYSTEM = (
    "You are the same CRAFT Director reconciling three public messages with your own "
    "wall evidence. Keep analysis private and communicate one final public message."
)
BUILDER_SYSTEM = (
    "You are the CRAFT Builder. Infer and generate one complete PLACE, REMOVE, or "
    "CLARIFY response from the Directors' public messages and the current board."
)

DIRECTOR_TEMPLATE_MESSAGES = {
    "your concise public instruction or cross-check.",
    "your final public instruction or clarification.",
}
COORD = r"\(\s*[0-2]\s*,\s*[0-2]\s*\)"
PLACE_MOVE_PATTERN = re.compile(
    rf"^PLACE\s*:\s*(?P<block>[gbryo][sl])\s*:\s*"
    rf"(?P<position>{COORD})\s*:\s*(?P<layer>[0-2])\s*:\s*"
    rf"(?:(?P<span_to>{COORD})\s*:\s*)?CONFIRM\s*:\s*(?P<confirmation>.+)$",
    re.IGNORECASE | re.DOTALL,
)
REMOVE_MOVE_PATTERN = re.compile(
    rf"^REMOVE\s*:\s*(?P<position>{COORD})\s*:\s*"
    rf"(?P<layer>[0-2])\s*:\s*(?:(?P<span_to>{COORD})\s*:\s*)?"
    rf"CONFIRM\s*:\s*(?P<confirmation>.+)$",
    re.IGNORECASE | re.DOTALL,
)
CLARIFY_MOVE_PATTERN = re.compile(
    r"^CLARIFY\s*:\s*(?P<clarification>.+)$", re.IGNORECASE | re.DOTALL
)


def _extract_tag(text: str, tag: str) -> Optional[str]:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_private_analysis(text: str) -> str:
    for tag in ("analysis", "think"):
        values = re.findall(
            rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE
        )
        if values:
            return values[0].strip()
        opening = re.search(rf"<{tag}>\s*(.*)$", text, re.DOTALL | re.IGNORECASE)
        if opening:
            return opening.group(1).strip().removesuffix("```").strip()
    return ""


def parse_director_response(text: str) -> Dict[str, Any]:
    """Extract one public message while keeping all Director reasoning private."""
    errors: List[str] = []
    warnings: List[str] = []
    opening_matches = list(re.finditer(r"<message>", text, re.IGNORECASE))
    closed_values = re.findall(
        r"<message>\s*(.*?)\s*</message>", text, re.DOTALL | re.IGNORECASE
    )

    public_message = ""
    if len(opening_matches) != 1:
        errors.append(
            f"Expected exactly one <message> element, found {len(opening_matches)}"
        )
    elif len(closed_values) == 1:
        public_message = closed_values[0].strip()
    elif not closed_values:
        public_message = text[opening_matches[0].end() :].strip()
        public_message = public_message.removesuffix("```").strip()
        warnings.append("Recovered an unclosed <message> element")
    else:
        errors.append(
            f"Expected exactly one closed <message> element, found {len(closed_values)}"
        )

    normalized_template = public_message.strip().strip("[]").strip().lower().rstrip(".")
    template_values = {value.rstrip(".") for value in DIRECTOR_TEMPLATE_MESSAGES}
    if not public_message:
        errors.append("<message> is empty")
    elif normalized_template in template_values:
        errors.append("<message> copied the response template instead of a message")
    elif re.search(r"</?[A-Za-z][^>]*>", public_message):
        errors.append("<message> contains a nested tag and was quarantined")

    exact_pattern = (
        r"\s*<(?P<tag>analysis|think)>.*?</(?P=tag)>\s*"
        r"<message>.*?</message>\s*"
    )
    valid = not errors
    parse_mode = (
        "exact"
        if valid and re.fullmatch(exact_pattern, text, re.DOTALL | re.IGNORECASE)
        else "recovered"
        if valid
        else "invalid"
    )
    return {
        "private_analysis": _extract_private_analysis(text),
        "public_message": public_message if valid else "",
        "protocol_valid": valid,
        "protocol_errors": errors,
        "protocol_warnings": warnings,
        "parse_mode": parse_mode,
    }


def _communication_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    """Return the complete and intentionally tiny cross-agent payload."""
    if not item["protocol_valid"]:
        return {
            "protocol_valid": False,
            "protocol_errors": list(item["protocol_errors"]),
        }
    return {"protocol_valid": True, "message": item["public_message"]}


def _parse_builder_move_line(line: str) -> Dict[str, Any]:
    """Parse one complete generative Builder line without correcting its meaning."""
    cleaned = line.strip()
    cleaned = re.sub(r"^```(?:[A-Za-z0-9_-]+)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip().strip("[]").strip()
    place = PLACE_MOVE_PATTERN.fullmatch(cleaned)
    if place:
        return {
            "action": "place",
            "move": {
                "action": "place",
                "block": place.group("block").lower(),
                "position": norm_pos(place.group("position")),
                "layer": int(place.group("layer")),
                "span_to": norm_pos(place.group("span_to"))
                if place.group("span_to")
                else None,
            },
            "clarification": None,
            "confirmation": place.group("confirmation").strip(),
            "parse_error": None,
        }

    remove = REMOVE_MOVE_PATTERN.fullmatch(cleaned)
    if remove:
        return {
            "action": "remove",
            "move": {
                "action": "remove",
                "block": None,
                "position": norm_pos(remove.group("position")),
                "layer": int(remove.group("layer")),
                "span_to": norm_pos(remove.group("span_to"))
                if remove.group("span_to")
                else None,
            },
            "clarification": None,
            "confirmation": remove.group("confirmation").strip(),
            "parse_error": None,
        }

    clarify = CLARIFY_MOVE_PATTERN.fullmatch(cleaned)
    if clarify and clarify.group("clarification").strip():
        return {
            "action": "clarify",
            "move": None,
            "clarification": clarify.group("clarification").strip(),
            "confirmation": "",
            "parse_error": None,
        }

    return {
        "action": "unparsed",
        "move": None,
        "clarification": None,
        "confirmation": "",
        "parse_error": "Move does not match PLACE/REMOVE/CLARIFY response grammar",
    }


def parse_builder_response(text: str) -> Dict[str, Any]:
    """Parse one explicit Builder move, recovering harmless wrapper deviations."""
    openings = list(re.finditer(r"<move>", text, re.IGNORECASE))
    closed_values = re.findall(
        r"<move>\s*(.*?)\s*</move>", text, re.DOTALL | re.IGNORECASE
    )
    payload: Optional[str] = None
    exact = False

    if len(openings) > 1 or len(closed_values) > 1:
        error = "Ambiguous Builder response contains multiple <move> elements"
    elif len(openings) == 1 and len(closed_values) == 1:
        payload = closed_values[0].strip()
        exact_pattern = (
            r"\s*<(?P<tag>analysis|think)>.*?</(?P=tag)>\s*"
            r"<move>.*?</move>\s*"
        )
        exact = bool(re.fullmatch(exact_pattern, text, re.DOTALL | re.IGNORECASE))
        error = None
    elif len(openings) == 1:
        payload = text[openings[0].end() :].strip().removesuffix("```").strip()
        error = None
    else:
        unclosed_private = any(
            len(re.findall(rf"<{tag}>", text, re.IGNORECASE))
            > len(re.findall(rf"</{tag}>", text, re.IGNORECASE))
            for tag in ("analysis", "think")
        )
        if unclosed_private:
            error = "No public move follows the unclosed private analysis"
        else:
            without_private = re.sub(
                r"<(analysis|think)>.*?</\1>",
                "",
                text,
                flags=re.DOTALL | re.IGNORECASE,
            )
            action_lines = [
                line.strip().strip("[]").strip()
                for line in without_private.splitlines()
                if re.match(
                    r"^\s*\[?\s*(PLACE|REMOVE|CLARIFY)\s*:",
                    line,
                    re.IGNORECASE,
                )
            ]
            if len(action_lines) == 1:
                payload = action_lines[0]
                error = None
            elif len(action_lines) > 1:
                error = "Ambiguous Builder response contains multiple action lines"
            else:
                error = "No <move> element or unique PLACE/REMOVE/CLARIFY line found"

    if error is None and payload is not None:
        if re.search(r"</?[A-Za-z][^>]*>", payload):
            error = "Builder move contains a nested tag and was quarantined"
            parsed = {
                "action": "unparsed",
                "move": None,
                "clarification": None,
                "confirmation": "",
            }
        else:
            action_markers = re.findall(
                r"\b(?:PLACE|REMOVE|CLARIFY)\s*:", payload, re.IGNORECASE
            )
        if error is None and len(action_markers) != 1:
            error = "Builder move must contain exactly one action line"
            parsed = {
                "action": "unparsed",
                "move": None,
                "clarification": None,
                "confirmation": "",
            }
        elif error is None:
            parsed = _parse_builder_move_line(payload)
            error = parsed.get("parse_error")
    else:
        parsed = {
            "action": "unparsed",
            "move": None,
            "clarification": None,
            "confirmation": "",
        }

    valid = error is None
    return {
        **parsed,
        "private_analysis": _extract_private_analysis(text),
        "parse_error": error,
        "parse_mode": "exact" if valid and exact else "recovered" if valid else "invalid",
        "protocol_valid": valid,
        "protocol_errors": [error] if error else [],
        "raw_response": text,
    }


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
        self.consecutive_no_execution = 0
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
            item["director_id"]: _communication_payload(item)
            for item in observations
        }
        record["phase1_public_messages"] = copy.deepcopy(phase1_messages)

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
            item["director_id"]: _communication_payload(item)
            for item in reconciliations
        }
        record["reconciliation_public_messages"] = copy.deepcopy(
            reconciliation_messages
        )

        builder_prompt = build_builder_prompt(
            builder_id=self.builder["id"],
            public_state=public_state,
            reconciliations=reconciliation_messages,
            available_blocks=self.env.available_blocks,
            previous_builder_result=self.previous_builder_result,
        )
        phase_started = time.monotonic()
        builder_response = await self.clients["builder"].complete(
            BUILDER_SYSTEM,
            builder_prompt,
            {"kind": "builder", "public_state": copy.deepcopy(public_state)},
        )
        builder_phase_latency = round(time.monotonic() - phase_started, 3)
        builder = parse_builder_response(builder_response["content"])
        builder.update(
            {
                "prompt": builder_prompt,
                "usage": builder_response.get("usage", {}),
                "latency_seconds": builder_response.get("latency_seconds"),
                "provider_reasoning": builder_response.get("reasoning_content", ""),
            }
        )
        record["builder"] = builder
        record["protocol_status"] = {
            "phase1_valid": sum(item["protocol_valid"] for item in observations),
            "phase1_total": len(observations),
            "reconciliation_valid": sum(
                item["protocol_valid"] for item in reconciliations
            ),
            "reconciliation_total": len(reconciliations),
            "builder_valid": builder["protocol_valid"],
            "builder_parse_mode": builder["parse_mode"],
            "builder_action": builder["action"],
        }
        record["phase_latency_seconds"] = {
            "phase1": phase1_latency,
            "reconciliation": reconciliation_latency,
            "builder": builder_phase_latency,
        }

        physical_validation, execution, evaluation = self._validate_execute_evaluate(
            public_state, builder
        )
        record["physical_validation"] = physical_validation
        record["execution"] = execution
        record["evaluation"] = evaluation
        # Kept as an output alias for existing plot/summary readers.
        record["score"] = evaluation["score"]

        if execution["ok"]:
            self.consecutive_no_execution = 0
        else:
            self.consecutive_no_execution += 1
        record["stability"] = {
            "consecutive_rounds_without_execution": self.consecutive_no_execution
        }

        self._append_public_messages(round_number, reconciliations)
        self._append_public_result(builder, physical_validation, execution)
        self._trim_history()
        record["public_history_after"] = list(self.public_history)
        if self.verbose:
            protocol = record["protocol_status"]
            action_label = builder["action"].upper() if builder["protocol_valid"] else "-"
            physics = physical_validation["ok"]
            physics_label = "n/a" if physics is None else str(bool(physics))
            print(
                f"  [Debate] round {round_number:>2}/{self.max_rounds} | "
                f"action={action_label:<7} parse={builder['parse_mode']:<9} "
                f"physics={physics_label:<5} executed={execution['ok']} "
                f"| overall_progress={evaluation['score']['overall_progress']:.4f} "
                f"(delta {evaluation['score']['delta']:+.4f}) | "
                f"protocol=p1:{protocol['phase1_valid']}/{protocol['phase1_total']} "
                f"rec:{protocol['reconciliation_valid']}/{protocol['reconciliation_total']} "
                f"no_exec_streak:{self.consecutive_no_execution}"
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
        )
        response = await self.clients["phase1"].complete(
            OBSERVATION_SYSTEM, prompt, {"kind": "observation", "director_id": director_id}
        )
        parsed = parse_director_response(response["content"])
        return {
            "agent_id": role.get("id", director_id),
            "director_id": director_id,
            "archetype": role.get("archetype", "observant"),
            **parsed,
            "prompt": prompt,
            "raw_response": response["content"],
            "usage": response.get("usage", {}),
            "latency_seconds": response.get("latency_seconds"),
            "provider_reasoning": response.get("reasoning_content", ""),
        }

    async def _run_reconciliation(
        self,
        role: Dict[str, str],
        public_state: Dict[str, Any],
        phase1_messages: Dict[str, Dict[str, Any]],
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
        )
        response = await self.clients["reconciliation"].complete(
            RECONCILIATION_SYSTEM,
            prompt,
            {"kind": "reconciliation", "director_id": director_id},
        )
        parsed = parse_director_response(response["content"])
        return {
            "agent_id": role.get("id", director_id),
            "director_id": director_id,
            "archetype": role.get("archetype", "observant"),
            **parsed,
            "prompt": prompt,
            "raw_response": response["content"],
            "usage": response.get("usage", {}),
            "latency_seconds": response.get("latency_seconds"),
            "provider_reasoning": response.get("reasoning_content", ""),
        }

    def _validate_execute_evaluate(
        self,
        public_state: Dict[str, Any],
        builder: Dict[str, Any],
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        move = builder.get("move")
        target_metrics: Dict[str, Any] = {}
        if not builder["protocol_valid"]:
            reason = builder.get("parse_error") or "Builder response could not be parsed"
            validation = {
                "ok": False,
                "reason": reason,
                "validator": "deterministic_public_physics",
            }
            execution = {
                "ok": False,
                "status": "parse_rejected",
                "error": reason,
                "move": None,
                "public_state_after": self.env.snapshot(),
            }
        elif builder["action"] == "clarify":
            reason = builder.get("clarification") or "Builder requested clarification"
            validation = {
                "ok": None,
                "reason": reason,
                "validator": "not_applicable",
            }
            execution = {
                "ok": False,
                "status": "clarify",
                "error": None,
                "move": None,
                "clarification": reason,
                "public_state_after": self.env.snapshot(),
            }
        else:
            ok, reason, normalized = validate_physical_action(
                public_state, move, self.env.available_blocks
            )
            validation = {
                "ok": ok,
                "reason": reason,
                "validator": "deterministic_public_physics",
                "normalized_action": normalized,
            }
            if ok:
                result = self.env.execute_move(normalized)
                execution = {
                    "ok": result["ok"],
                    "status": "executed" if result["ok"] else "execution_failed",
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
                    "status": "physics_rejected",
                    "error": reason,
                    "move": normalized,
                    "public_state_after": self.env.snapshot(),
                }

        score = calculate_progress(self.env.current_structure, self.env.target_structure)
        previous = (
            self.rounds[-1]["score"]["overall_progress"]
            if self.rounds
            else self.baseline["overall_progress"]
        )
        score["delta"] = round(score["overall_progress"] - previous, 6)
        return validation, execution, {"score": score, **target_metrics}

    def _append_public_messages(
        self, round_number: int, reconciliations: List[Dict[str, Any]]
    ) -> None:
        """Keep final public utterances as cross-round conversation memory."""
        for item in reconciliations:
            if item["protocol_valid"]:
                self.public_history.append(
                    f"Round {round_number} {item['director_id']}: {item['public_message']}"
                )

    def _append_public_result(
        self,
        builder: Dict[str, Any],
        validation: Dict[str, Any],
        execution: Dict[str, Any],
    ) -> None:
        if execution["ok"] and execution.get("move"):
            result = f"Builder executed: {_format_move(execution['move'])}."
        elif execution.get("status") == "clarify":
            result = f"Builder asked: {builder['clarification']}"
        elif execution.get("status") == "parse_rejected":
            result = f"Builder response was not executed: {builder['parse_error']}."
        else:
            result = f"Builder action was not executed: {validation['reason']}."
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
