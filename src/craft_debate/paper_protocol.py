"""Faithful CRAFT paper protocol: 3 Directors -> 1 oracle-assisted Builder.

Ported from the prompts printed in CRAFT (arXiv:2603.25268v2) Appendix D,
Figures 9-14, with the turn flow described in Section 5.2:

* 20 communicative turns per game
* each turn samples 1-3 unique Directors who speak sequentially
* Directors see the full current board, their private target view, and the
  conversation history of prior Director responses
* the Builder sees the current turn's Director discussion plus up to 5
  oracle-verified candidate moves and selects one (or CLARIFY)
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Dict, List, Optional

from .api import LLMClient  # noqa: F401  (kept importable for type clarity)
from .domain import ALL_COORDS, AVAILABLE_BLOCKS, PERSPECTIVE_DESCRIPTIONS, get_director_views
from .environment import GameState
from .oracle import sample_oracle_moves
from .progress import calculate_progress
from .prompts import ARCHETYPES, BLOCK_ENCODING_REFERENCE, COORDINATE_REFERENCE, SPATIAL_ORIENTATION

DIRECTOR_TYPES = ["assertive", "cautious", "observant", "skeptical", "synthesizer"]
PAPER_ORACLE_N = 5


def validate_paper_oracle_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Require the Oracle setting used by the paper protocol.

    Oracle availability and candidate count are experimental constants for the
    current RL studies, not tunable treatment variables.
    """
    oracle_cfg = config.get("oracle", {"enabled": True, "n": PAPER_ORACLE_N})
    enabled = oracle_cfg.get("enabled", True) is True
    candidate_count = oracle_cfg.get("n", PAPER_ORACLE_N)
    valid_count = (
        isinstance(candidate_count, int)
        and not isinstance(candidate_count, bool)
        and candidate_count == PAPER_ORACLE_N
    )
    if not enabled or not valid_count:
        raise ValueError(
            f"paper protocol requires oracle.enabled=true and oracle.n={PAPER_ORACLE_N}; "
            "Oracle is fixed while studying RL communication and turn control"
        )
    return oracle_cfg


def deterministic_archetype(structure_index: int, run_index: int, director_id: str) -> str:
    """Deterministic archetype per (structure_index, run_index, director_id).

    The paper (Appendix B, Table 2) assigns personalities deterministically;
    a stable hash is used here so results are reproducible across processes.
    """
    director_num = {"D1": 0, "D2": 1, "D3": 2}[director_id]
    digest = hashlib.md5(
        f"{structure_index}|{director_num}|{run_index}".encode("utf-8")
    ).hexdigest()
    return DIRECTOR_TYPES[int(digest, 16) % len(DIRECTOR_TYPES)]


def parse_director_response(text: str) -> Dict[str, str]:
    """Parse the paper's <analysis>/<message> format (accepts <think> too)."""

    def extract(tag: str) -> Optional[str]:
        match = re.search(
            rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL | re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        open_match = re.search(rf"<{tag}>\s*(.*)$", text, re.DOTALL | re.IGNORECASE)
        if open_match:
            return open_match.group(1).strip()
        return None

    analysis = extract("analysis")
    if analysis is None:
        analysis = extract("think")
    message = extract("message")
    return {
        "internal_thinking": analysis or "",
        "public_message": message or "No message provided",
        "raw_response": text,
    }


def parse_builder_response(text: str) -> Dict[str, Any]:
    """Parse the paper's Builder action format."""
    wrapped = re.search(r"<move>\s*(.*?)\s*</move>", text, re.DOTALL | re.IGNORECASE)
    haystack = wrapped.group(1) if wrapped else text
    match = re.search(
        r"^\s*(PLACE|REMOVE|CLARIFY)\s*:(.*)$",
        haystack,
        re.MULTILINE | re.IGNORECASE,
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
                raise ValueError("Missing CONFIRM marker in PLACE line")
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
                raise ValueError("Missing CONFIRM marker in REMOVE line")
            parsed["move"] = {
                "action": "remove",
                "block": None,
                "position": position,
                "layer": layer,
                "span_to": span_to,
            }
        parsed["confirmation"] = ":".join(parts[index + 1 :])
    except (ValueError, IndexError) as exc:
        parsed["action"] = "unparsed"
        parsed["move"] = None
        parsed["parse_error"] = str(exc)
    return parsed


def build_director_prompt(
    *,
    director_id: str,
    archetype: str,
    target_view: Dict[str, Any],
    board_state: Dict[str, List[str]],
    conversation_history: str,
) -> str:
    """Director prompt, verbatim from paper Figures 9-11."""
    personality = ARCHETYPES.get(archetype, ARCHETYPES["observant"])
    perspective = PERSPECTIVE_DESCRIPTIONS[director_id]
    example_target = {
        "row_0": [
            {"color": "blue", "size": 1},
            {"color": "orange", "size": 2},
            {"color": "orange", "size": 2},
        ],
        "row_1": [
            {"color": "yellow", "size": 1},
            {"color": "yellow", "size": 1},
            {"color": "orange", "size": 1},
        ],
        "row_2": [
            {"color": "yellow", "size": 1},
            {"color": "blue", "size": 1},
            {"color": "green", "size": 1},
        ],
    }
    empty_board = {coord: [] for coord in ALL_COORDS}

    return f"""You are Director {director_id} in a collaborative LEGO construction task. You are sitting around a physical board with a Builder and two other Directors. From where the builder sits, D1 is to their left, D2 is across from them, and D3 is to their right.

YOU ARE {archetype}.

YOUR PERSONALITY:
{personality}

VERY IMPORTANT: You must adopt this personality in both your internal reasoning and your public utterances.

YOUR PERSPECTIVE:
{perspective}

{SPATIAL_ORIENTATION}

HOW TO INTERPRET YOUR TARGET VIEW
- IMPORTANT: In the JSON, keys are named row_0/row_1/row_2, but they refer to LAYERS (vertical stack depth), not grid rows.
- row_0 = layer_0 (bottom layer / stack depth 0)
- row_1 = layer_1 (middle layer / stack depth 1)
- row_2 = layer_2 (top layer / stack depth 2)
- in each layer, blocks are listed from LEFT to RIGHT according to YOUR VIEW
- In your PUBLIC message, say "bottom layer / middle layer / top layer" (avoid saying "bottom row").
- color=none means that cell should be empty
- size of 1 = the block is a small block, size of 2 = the block is a large block and spans two adjacent cells
- if two adjacent cells in your target view have the same color and BOTH are size 2, this means that a SINGLE large block occupies both those cells

EXAMPLE ANALYSIS OF TARGET VIEW AND BOARD STATE
D2's target view:
{json.dumps(example_target, indent=2)}

Current board state:
{json.dumps(empty_board, indent=2)}

Correct D2 analysis:
[From my perspective, the current board state has all cells empty. My target view specifies that layer 0 should have a blue small block in my bottom left corner (0,0), and then a large orange block spanning the middle and right cells (0,1) and (0,2).

Going left to right, layer 1 should have two small yellow blocks at (0,0) and (0,1), and a small orange block at (0,2).

Finally, layer 2 should consist of a yellow small block at (0,0), a blue small block at (0,1), and a green small block at (0,2).

To start, I need the builder to place a large orange block spanning (0,1) and (0,2), which are the middle and right cells of my bottom layer. This is the first action to align with my target view.]

Correct D2 utterance based on this analysis:
[Put a large orange block across the middle and the right side of my bottom layer.]

YOUR JOB:
Help the builder complete the structure by giving clear and correct instructions based on your private view.

RULES FOR REASONING (use only in your thinking):
- Carefully compare your target view with the current board state.
- Identify missing blocks, incorrect blocks, or incorrectly placed blocks.
- Determine the correct color, size, and layer for each required block.
- Respect physical constraints: blocks must be placed on valid support and large blocks must span correctly.
- Plan instructions that move the current board closer to your target view.
- Do not assume access to other directors' views; reason only from your own perspective.

RULES FOR SPEAKING (in your public message):
- Give clear, concise, and actionable instructions to the builder.
- Use natural language descriptions (e.g., "left", "right", "middle") based on your perspective.
- Refer to layers as "bottom", "middle", or "top" (not row numbers).
- Specify block color, size, and placement clearly.
- Avoid mentioning coordinates or JSON-style representations.
- Do not include internal reasoning in your message.
- Focus on one step or a small number of steps that the builder can execute reliably.

EXAMPLE UTTERANCES:
[So, the second layer on top of the yellow will be orange, and then another orange, and then a yellow.]
[And then it'll go blue, yellow, green.]
[On top of green there goes a blue. And on top of red there goes a yellow.]

CURRENT BOARD STATE:
{json.dumps(board_state, indent=2)}

TARGET VIEW:
{json.dumps(target_view, indent=2)}

CONVERSATION HISTORY:
{conversation_history or "(no prior director messages yet)"}

RESPONSE FORMAT:
Return your response in the following format:

<analysis>
[Your internal reasoning here]
</analysis>
<message>
[Your instruction to the builder here]
</message>"""


DIRECTOR_SYSTEM = "You are Director {director_id} in a collaborative LEGO construction task."

BUILDER_SYSTEM = (
    "You are a Builder in a collaborative LEGO task. "
    "You have been given VERIFIED CANDIDATE MOVES — you MUST choose exactly one from the list. "
    "Respond in the specified PLACE/REMOVE/CLARIFY format. "
    "In your CONFIRM field, write 2-3 sentences: which director(s) you followed, "
    "whether others agreed or conflicted, and why you chose this candidate."
)

def format_oracle_moves(oracle_moves: List[Dict[str, Any]]) -> str:
    if not oracle_moves:
        return "(no verified moves available this turn — CLARIFY)"
    lines = []
    for m in oracle_moves:
        action = m["action"]
        block = m.get("block", "")
        pos = m["position"]
        layer = m["layer"]
        span = m.get("span_to")
        if action == "place":
            line = f"PLACE {block} at {pos} layer {layer}"
            if span:
                line += f" spanning to {span}"
        else:
            line = f"REMOVE from {pos} layer {layer}"
            if span:
                line += f" spanning to {span}"
        lines.append(line)
    return "\n".join(lines)


def build_builder_prompt(
    *,
    board_state: Dict[str, List[str]],
    available_blocks: List[str],
    director_discussion: str,
    oracle_moves: List[Dict[str, Any]],
) -> str:
    """Builder prompt, verbatim from paper Figures 12-14."""
    decision_context = f"""CANDIDATE MOVES (verified physically valid for this turn):
{format_oracle_moves(oracle_moves)}"""
    return _build_builder_prompt(
        board_state=board_state,
        available_blocks=available_blocks,
        director_discussion=director_discussion,
        decision_context=decision_context,
    )


def _build_builder_prompt(
    *,
    board_state: Dict[str, List[str]],
    available_blocks: List[str],
    director_discussion: str,
    decision_context: str,
) -> str:
    """Render the paper's Oracle-assisted Builder rules."""
    empty_board = {coord: [] for coord in ALL_COORDS}
    span_example = {
        "(0,0)": [],
        "(0,1)": [],
        "(0,2)": [],
        "(1,0)": [],
        "(1,1)": [],
        "(1,2)": [],
        "(2,0)": [],
        "(2,1)": ["gl"],
        "(2,2)": ["gl"],
    }
    count_example = {
        "(0,0)": ["os"],
        "(0,1)": [],
        "(0,2)": [],
        "(1,0)": [],
        "(1,1)": [],
        "(1,2)": ["bl"],
        "(2,0)": ["gl", "bl"],
        "(2,1)": ["gl", "bl"],
        "(2,2)": ["bl"],
    }

    return f"""You are a Builder in a collaborative LEGO construction task.

The three Directors (D1, D2, and D3) have to instruct you to build a single structure that is consistent with the private views of the structure they have.
Your job is to place, move, or remove blocks on the board to build the structure.
From a top-down view of the target structure, D1's private view is of the left wall of the structure, D2's view is of the top wall of the structure, and D3's view of the right wall of the structure.
From where the builder sits, D1 is to their left, D2 is across from them, and D3 is to their right.

{SPATIAL_ORIENTATION}

DIRECTOR PERSPECTIVE GUIDE:
D1: From left to right, sees cells (0,0), (1,0), (2,0) across all layers.
D2: From left to right, sees cells (0,0), (0,1), (0,2) across all layers.
D3: From left to right, sees cells (0,2), (1,2), (2,2) across all layers.

When interpreting the instructions from D1, D2, or D3, you MUST adopt the frame of reference of the speaker.
For instance, to D1, "my bottom left corner" is coordinate (0,0) at layer 0 and "my top right corner" is coordinate (2,0) at layer 2.
To D2, "my bottom left corner" is coordinate (0,0) at layer 0 and "my top right corner" is coordinate (0,2) at layer 2.
To D3, "my bottom left corner" is coordinate (0,2) at layer 0 and "my top right corner" is coordinate (2,2) at layer 2.

EXAMPLE FRAME OF REFERENCE ANALYSIS:

Given board state
{json.dumps(empty_board, indent=2)}

Given utterance
[D1: Could you please place a small orange block in my bottom left corner?]
Correct move
[PLACE:os:(0,0):0:CONFIRM:Placing small orange block at bottom-left of D1's side as requested]

Given utterance
[D2: Please remove the large orange block from my bottom left and middle cells.]
Correct move
[REMOVE:(0,0):0:(0,1):CONFIRM:Removing the large orange block from bottom-left+bottom-middle of D2's side as requested]

Given utterance
[D3: Let's begin by placing a large green block across the left and middle cells of my bottom layer.]
Correct move
[PLACE:gl:(0,2):0:(1,2):CONFIRM:Placing large green block across the left and middle cells of D3's bottom layer as requested]

Positions invisible to ALL directors: (1,1) and (2,1).
A large block that is visible to ANY of the directors CANNOT span EITHER (1,1) or (2,1).
— only inferred from what's missing in other views.

CURRENT BOARD STATE:
{json.dumps(board_state, indent=2)}

AVAILABLE BLOCKS:
{", ".join(available_blocks)}

BLOCK REFERENCE:
{BLOCK_ENCODING_REFERENCE}

COORDINATE REFERENCE:
{COORDINATE_REFERENCE}

{decision_context}

DIRECTOR DISCUSSION:
{director_discussion}

DECISION RULE:
If 2+ directors agree on a block or position, do that first.
If all three disagree, pick the most specific instruction.

STACKING RULES:
- "layer" means stack depth, NOT grid row.
- ALWAYS calculate layer from CURRENT BOARD STATE, never trust director-specified layers.
- Before ANY place: count blocks at target position from CURRENT BOARD STATE.
  You MUST place new blocks one layer above the number of blocks at that position.
- Before ANY remove: verify position is non-empty in CURRENT BOARD STATE.
  If empty, do NOT attempt removal — tell directors and suggest placing instead.

FRAME OF REFERENCE RULE:
IMPORTANT: When choosing where to place a block, you MUST adopt the frame of reference of the director whose instruction you are following.
REMINDER: "The left" of D1's view is coordinate (0,0) and "the right" is coordinate (2,0).
"The left" of D2's view is coordinate (0,0) and "the right" is coordinate (0,2).
"The left" of D3's view is coordinate (0,2) and "the right" is coordinate (2,2).
NEVER deviate from these frames of reference when executing instructions.

LARGE BLOCK RULE:
Large blocks span TWO adjacent cells — you MUST specify both endpoints.

To choose span_to:
- Identify the TWO director-relative cells explicitly referenced (e.g., "left+middle", "middle+right", "bottom left+bottom middle").
- Convert those two cells into global coordinates using the DIRECTOR PERSPECTIVE GUIDE.
- Ensure BOTH cells lie on the correct wall for that director.
- Set position to one endpoint and span_to to the other endpoint.
- Before outputting, verify:
  (a) position and span_to are orthogonal neighbors,
  (b) both endpoint stacks have the SAME height (so placement/removal is on the same layer),
  (c) neither endpoint is an invisible cell ((1,1) or (2,1)).

NEVER place OR remove a large block if span_to is None — it will always fail.
Format:
PLACE:block:position:layer:span_to:CONFIRM:reason
Example:
PLACE:gl:(0,0):0:(1,0):CONFIRM:Placing large green block across the left and middle cells of D1's bottom layer as requested

If a director says "green large in the corner", you must figure out which two adjacent cells it spans from the CURRENT BOARD STATE.
NEVER place OR remove a large block if span_to is None — it will always fail.
If you try to remove a large block, you MUST check the board state to see where spans contain the same block.

EXAMPLE SPAN ANALYSIS:
Given board state
{json.dumps(span_example, indent=2)}

Given raw move
[REMOVE:(2,2):0:CONFIRM:Removing the large green block from the bottom layer as requested by D3.]
Correct move
[REMOVE:(2,2):0:(2,1):CONFIRM:Removing the large green block from the bottom layer as requested by D3.]

WHEN MOVES FAIL:
- Explain WHY: e.g., "I can't remove any block from the middle cell on the bottom layer. There is no block there. Suggest placing [block] instead."
- Never silently retry the same failed move.

BEFORE PLACING:
Think step by step to make sure that you have interpreted the instructions, including block color and size, and the director's frame of reference, correctly. Do not place a block at the same place where you have previously removed a block of the same color. Count blocks at target position from CURRENT BOARD STATE.

EXAMPLE BLOCK COUNT AT TARGET POSITION:
Given board state:
{json.dumps(count_example, indent=2)}

Given raw move
[PLACE:gl:(2,2):0:(2,1):CONFIRM:Placing large green block across the left and middle cells of D3's bottom layer as requested.]
Correct move
[PLACE:gl:(2,2):2:(2,1):CONFIRM:Placing large green block across the left and middle cells of D3's bottom layer as requested.]

OUTPUT FORMAT — Choose ONE of these exact formats:

1. To place small block: PLACE:block_code:position:layer:CONFIRM:interpretation
Example: PLACE:bs:(0,0):0:CONFIRM:Placing blue small block at bottom-left of D1's side as requested

2. To place large block: PLACE:block_code:position:layer:span_to:CONFIRM:interpretation
Example: PLACE:gl:(0,0):0:(1,0):CONFIRM:Placing large green block across left and middle cells of D1's bottom layer

3. To remove small block: REMOVE:position:layer:CONFIRM:interpretation
Example: REMOVE:(1,2):0:CONFIRM:Removing the block from middle-right of D3's side as requested

4. To remove large block: REMOVE:position:layer:span_to:CONFIRM:interpretation
Example: REMOVE:(2,2):0:(2,1):CONFIRM:Removing large green block from D3's bottom layer as requested
NOTE: REMOVE never includes block code — do NOT write REMOVE:bl:(0,0):...

5. To clarify: CLARIFY:your specific question
Example: CLARIFY:Which blue block should I move - the one on top or bottom?

Always include CONFIRM section to show what you understood from their instructions."""


class PaperGame:
    """One structure-run instance of the paper's 3-Director + Builder protocol."""

    def __init__(
        self,
        *,
        config: Dict[str, Any],
        structure_data: Dict[str, Any],
        structure_index: int,
        run_index: int,
        director_client: Any,
        builder_client: Any,
        verbose: bool = True,
    ) -> None:
        self.config = config
        self.structure_data = structure_data
        self.structure_index = structure_index
        self.run_index = run_index
        self.director_client = director_client
        self.builder_client = builder_client
        self.verbose = verbose

        target = structure_data["structure"]
        spans = structure_data["spans"]
        self.env = GameState(target, spans, start_from_empty=True)
        self.target_views = get_director_views(target, spans)
        self.available_blocks = list(AVAILABLE_BLOCKS)
        self.max_turns = int(config["turns"])
        validate_paper_oracle_config(config)
        self.stop_on_complete = bool(config.get("stop_on_complete", False))

        seed = int(config.get("seed", 42))
        self.select_rng = random.Random(seed * 1000 + structure_index * 100 + run_index)
        self.archetypes = {
            did: deterministic_archetype(structure_index, run_index, did)
            for did in ("D1", "D2", "D3")
        }

        self.conversation: List[str] = []
        self.turns: List[Dict[str, Any]] = []
        self.baseline = calculate_progress(
            self.env.current_structure, self.env.target_structure
        )

    def _sample_speakers(self) -> List[str]:
        k = self.select_rng.randint(1, 3)
        return self.select_rng.sample(["D1", "D2", "D3"], k)

    async def run_turn(self, turn_number: int) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "turn_number": turn_number,
            "structure_before": {k: list(v) for k, v in self.env.current_structure.items()},
            "director_responses": {},
        }

        speakers = self._sample_speakers()
        record["director_order"] = speakers
        for did in speakers:
            conversation_text = "\n".join(self.conversation)
            prompt = build_director_prompt(
                director_id=did,
                archetype=self.archetypes[did],
                target_view=self.target_views[did],
                board_state=self.env.current_structure,
                conversation_history=conversation_text,
            )
            response = await self.director_client.complete(
                DIRECTOR_SYSTEM.format(director_id=did),
                prompt,
                {"kind": "director", "director_id": did},
            )
            parsed = parse_director_response(response["content"])
            entry = {
                "director_id": did,
                "archetype": self.archetypes[did],
                "prompt": prompt,
                "internal_thinking": parsed["internal_thinking"],
                "public_message": parsed["public_message"],
                "raw_response": parsed["raw_response"],
                "usage": response.get("usage", {}),
                "latency_seconds": response.get("latency_seconds"),
            }
            record["director_responses"][did] = entry
            self.conversation.append(f"{did}: {parsed['public_message']}")
            if self.verbose:
                print(f"    {did} ({self.archetypes[did]}): {parsed['public_message'][:90]}")

        discussion = "\n".join(
            f"{did}: {entry['public_message']}"
            for did, entry in record["director_responses"].items()
        )
        record["director_discussion"] = discussion

        oracle_rng = random.Random(self.structure_index * 1000 + turn_number)
        oracle_moves = sample_oracle_moves(self.env, PAPER_ORACLE_N, oracle_rng)
        record["oracle_moves"] = [
            {k: v for k, v in move.items()} for move in oracle_moves
        ]

        builder_prompt = build_builder_prompt(
            board_state=self.env.current_structure,
            available_blocks=self.available_blocks,
            director_discussion=discussion,
            oracle_moves=oracle_moves,
        )

        response = await self.builder_client.complete(
            BUILDER_SYSTEM,
            builder_prompt,
            {"kind": "judge", "oracle_moves": oracle_moves},
        )
        parsed_move = self._parse_builder_output(response["content"])
        record["builder_prompt"] = builder_prompt
        record["builder_response"] = {
            "raw": response["content"],
            "parsed": parsed_move,
            "usage": response.get("usage", {}),
            "latency_seconds": response.get("latency_seconds"),
        }

        execution = self._execute_move(parsed_move)
        record["execution"] = execution
        followed = None
        if oracle_moves:
            move = execution.get("move")
            followed = bool(
                move
                and any(
                    m.get("action") == move.get("action")
                    and m.get("position") == move.get("position")
                    and m.get("layer") == move.get("layer")
                    for m in oracle_moves
                )
            )
        record["builder_followed_oracle"] = followed

        score = calculate_progress(self.env.current_structure, self.env.target_structure)
        prev = (
            self.turns[-1]["score"]["overall_progress"]
            if self.turns
            else self.baseline["overall_progress"]
        )
        score["delta"] = round(score["overall_progress"] - prev, 6)
        record["score"] = score
        record["conversation_snapshot"] = list(self.conversation)

        if self.verbose:
            action = parsed_move.get("action", "?")
            ok = execution.get("ok")
            print(
                f"  [Paper] turn {turn_number:>2}/{self.max_turns} | speakers={','.join(speakers)} "
                f"| builder={action:<8} executed={ok} followed_oracle={followed} "
                f"| overall_progress={score['overall_progress']:.4f} (delta {score['delta']:+.4f})"
            )
        return record

    def _execute_move(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        if parsed["action"] == "clarify":
            return {
                "action": "clarify",
                "move": None,
                "ok": None,
                "error": None,
                "clarification": parsed.get("clarification"),
                "structure_after": {k: list(v) for k, v in self.env.current_structure.items()},
            }

        move = parsed.get("move")
        if not move:
            return {
                "action": "unparsed",
                "move": None,
                "ok": False,
                "error": parsed.get("parse_error"),
                "structure_after": {k: list(v) for k, v in self.env.current_structure.items()},
            }

        result = self.env.execute_move(move)
        return {
            "action": move["action"],
            "move": result["move"],
            "confirmation": parsed.get("confirmation", ""),
            "ok": result["ok"],
            "error": result["error"],
            "removed_block": result.get("removed_block"),
            "board_complete": result.get("board_complete"),
            "structure_after": {k: list(v) for k, v in self.env.current_structure.items()},
        }

    @staticmethod
    def _parse_builder_output(text: str) -> Dict[str, Any]:
        """Parse the builder line even when the model mimics the bracketed examples."""
        candidates = []
        for line in text.strip().splitlines():
            cleaned = line.strip().strip("[]").strip()
            if cleaned:
                candidates.append(cleaned)
        for cleaned in candidates:
            if re.match(r"^(PLACE|REMOVE|CLARIFY)\s*:", cleaned, re.IGNORECASE):
                return parse_builder_response(cleaned)
        return parse_builder_response(text.strip().strip("[]"))

    def game_record(self) -> Dict[str, Any]:
        final_score = (
            self.turns[-1]["score"]["overall_progress"]
            if self.turns
            else self.baseline["overall_progress"]
        )
        return {
            "structure_id": self.structure_data["id"],
            "structure_index": self.structure_index,
            "complexity": self.structure_data["complexity"],
            "metadata": self.structure_data.get("metadata", {}),
            "run_index": self.run_index,
            "archetypes": self.archetypes,
            "target_structure": {k: list(v) for k, v in self.structure_data["structure"].items()},
            "target_spans": {str(k): list(v) for k, v in self.structure_data["spans"].items()},
            "target_director_views": self.target_views,
            "baseline_progress": self.baseline["overall_progress"],
            "turns": self.turns,
            "final_progress": final_score,
            "final_structure": {k: list(v) for k, v in self.env.current_structure.items()},
            "completed": self.env.board_complete(),
            "turns_completed": len(self.turns),
        }

    async def run(self) -> Dict[str, Any]:
        for turn_number in range(1, self.max_turns + 1):
            record = await self.run_turn(turn_number)
            self.turns.append(record)
            if self.stop_on_complete and record["execution"].get("board_complete"):
                break
        return self.game_record()
