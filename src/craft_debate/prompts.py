"""Prompts for the generative 3 Directors -> 3 reconciliations -> Builder flow."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .domain import PERSPECTIVE_DESCRIPTIONS

ARCHETYPES: Dict[str, str] = {
    "assertive": (
        "You are confident and direct. You form hypotheses quickly from your data "
        "and share them, but you genuinely listen to other groups and update your "
        "thinking when their evidence is compelling. You sometimes move faster than "
        "the evidence warrants but you're not closed-minded."
    ),
    "cautious": (
        "You are methodical and prefer to verify before claiming. You ask clarifying "
        "questions and often synthesize what others have said before adding your own "
        "interpretation. You can make claims when evidence is strong enough — you're "
        "not paralyzed, just careful."
    ),
    "observant": (
        "You notice patterns and anomalies in your data that others might overlook. "
        "You tend to flag inconsistencies and ask 'does this match what you're seeing?' "
        "rather than broadcasting conclusions. You're collaborative by nature and "
        "often connect dots across groups."
    ),
    "skeptical": (
        "You question assumptions including your own. When someone makes a claim you "
        "probe it — not to be difficult but because you want the group to get it right. "
        "You're comfortable with uncertainty and say so openly."
    ),
    "synthesizer": (
        "You actively try to integrate what all groups are saying into a coherent "
        "picture. You summarize, reconcile contradictions, and push the group toward "
        "a shared understanding."
    ),
}

# Kept public because the separate paper-protocol module imports these references.
BLOCK_ENCODING_REFERENCE = """BLOCK ENCODING:
Colors: g=green, b=blue, r=red, y=yellow, o=orange
Sizes:  s=small (1 cell), l=large (2 adjacent cells)
Examples: gs=green small, gl=green large, bs=blue small, yl=yellow large, ol=orange large"""

COORDINATE_REFERENCE = """COORDINATE SYSTEM (3x3 grid):
(0,0) (0,1) (0,2)   <- top row    (i=0)
(1,0) (1,1) (1,2)   <- middle row (i=1)
(2,0) (2,1) (2,2)   <- bottom row (i=2)
j=0 = left column, j=1 = center column, j=2 = right column"""

SPATIAL_ORIENTATION = """SPATIAL ORIENTATION (use only in your thinking)
The coordinate grid from above:
  (0,0) (0,1) (0,2)   <- this is the "far" / "back" row
  (1,0) (1,1) (1,2)
  (2,0) (2,1) (2,2)   <- this is the "near" / "front" row
Large blocks span SIDEWAYS or FORWARD/BACK — never stacked vertically."""

DIRECTOR_PERSPECTIVE_GUIDE = """DIRECTOR FRAMES:
D1 sees (0,0), (1,0), (2,0) from left to right.
D2 sees (0,0), (0,1), (0,2) from left to right.
D3 sees (0,2), (1,2), (2,2) from left to right."""


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _format_public_messages(
    messages: Dict[str, Dict[str, Any]], *, label: str
) -> str:
    """Render an already-sanitized communication-edge payload."""
    lines = []
    for director_id in ("D1", "D2", "D3"):
        item = messages.get(director_id, {"protocol_valid": False})
        prefix = f"{label}_{director_id}"
        if item.get("protocol_valid"):
            message = json.dumps(item["message"], ensure_ascii=False)
            lines.append(f"{prefix}: {message}")
        else:
            errors = "; ".join(item.get("protocol_errors", [])) or "message unavailable"
            unavailable = json.dumps(f"[unavailable: {errors}]", ensure_ascii=False)
            lines.append(f"{prefix}: {unavailable}")
    return "\n".join(lines)


def build_observation_prompt(
    *,
    director_id: str,
    archetype: str,
    private_view: Dict[str, Any],
    public_state: Dict[str, Any],
    public_history: str,
) -> str:
    """Phase 1: one Director produces one natural-language public utterance."""
    return f"""You are Director {director_id} in Phase 1 of a collaborative block-building task.
{ARCHETYPES.get(archetype, ARCHETYPES['observant'])}

Use only YOUR PRIVATE WALL VIEW, CURRENT PUBLIC BOARD, and PUBLIC HISTORY below.
Do not assume what another Director sees. Your analysis is private; only the text
inside <message> is communicated to the other agents.

{PERSPECTIVE_DESCRIPTIONS[director_id]}
Private-view row_0, row_1, and row_2 mean the bottom, middle, and top vertical
layers. Entries within a layer run left-to-right in YOUR frame. color=none means
empty. Adjacent entries with the same color and size=2 describe one large block.

Compare the wall view with the public board and communicate the most useful next
step. In <message>:
- use natural language and your own left/middle/right frame;
- say bottom/middle/top layer and specify color and small/large size;
- for a large block, name both relative cells that it spans;
- focus on one placement or removal that the Builder can perform;
- if your wall already matches or evidence is insufficient, say so and ask for a
  specific cross-check instead of inventing certainty;
- do not use coordinates, JSON, block codes, or machine-action syntax.

Return exactly these two elements. Keep analysis under 120 words and message under
45 words.
<analysis>Your private comparison and reasoning.</analysis>
<message>Your concise public instruction or cross-check.</message>

CURRENT PUBLIC BOARD:
{_json_dumps(public_state)}

YOUR PRIVATE VIEW ({director_id} ONLY):
{_json_dumps(private_view)}

PUBLIC HISTORY:
{public_history or '(none)'}"""


def build_reconciliation_prompt(
    *,
    director_id: str,
    archetype: str,
    private_view: Dict[str, Any],
    public_state: Dict[str, Any],
    phase1_messages: Dict[str, Dict[str, Any]],
    public_history: str,
) -> str:
    """Phase 2: the same identity reconciles the three public utterances."""
    public_messages = _format_public_messages(phase1_messages, label="PUBLIC_PHASE1")
    return f"""You are Director {director_id} in Phase 2 of a collaborative block-building task.
You are the SAME Director identity as in Phase 1. {ARCHETYPES.get(archetype, ARCHETYPES['observant'])}

Use only YOUR PRIVATE WALL VIEW, CURRENT PUBLIC BOARD, PUBLIC HISTORY, and the three
public Phase-1 messages below. They are communicated claims, not anyone else's
private reasoning or raw view. An unavailable message contains no usable claim.

Reconcile agreement and contradictions. Check claims against your own wall evidence,
revise your Phase-1 position when warranted, and produce one final public utterance.
In <message>, use natural speaker-relative language (left/middle/right and
bottom/middle/top), give one concrete placement/removal instruction with color and
size, or ask one specific clarification when no defensible action follows. Do not use
coordinates, JSON, block codes, or machine-action syntax.

Return exactly these two elements. Keep analysis under 160 words and message under
60 words.
<analysis>Your private reconciliation and reasoning.</analysis>
<message>Your final public instruction or clarification.</message>

YOUR OWN FRAME:
{PERSPECTIVE_DESCRIPTIONS[director_id]}

PUBLIC SPEAKER FRAMES FOR INTERPRETING ALL THREE MESSAGES:
{DIRECTOR_PERSPECTIVE_GUIDE}

CURRENT PUBLIC BOARD:
{_json_dumps(public_state)}

YOUR PRIVATE VIEW ({director_id} ONLY):
{_json_dumps(private_view)}

CURRENT-ROUND PUBLIC PHASE-1 MESSAGES:
{public_messages}

PUBLIC HISTORY:
{public_history or '(none)'}"""


def build_builder_prompt(
    *,
    builder_id: str,
    public_state: Dict[str, Any],
    reconciliations: Dict[str, Dict[str, Any]],
    available_blocks: List[str],
    previous_builder_result: Optional[str],
) -> str:
    """Builder prompt for open generation followed by deterministic validation."""
    public_messages = _format_public_messages(reconciliations, label="FINAL_PUBLIC")
    return f"""You are {builder_id}, the Builder in a collaborative block-building task.

Infer the next physical action from the current public board and the three Directors'
final public messages. An unavailable message contains no usable claim. Generate the
action yourself; do not expect a menu or an identifier.

SPEAKER FRAMES:
{DIRECTOR_PERSPECTIVE_GUIDE}
Always convert left/middle/right using the frame of the Director who said it.

BOARD AND PHYSICS:
- A stack array is ordered bottom to top. Its length is the next placement layer.
- A small block occupies one cell.
- A large block occupies two orthogonally adjacent cells at equal stack height and
  requires both endpoints. It cannot include (1,1) or (2,1).
- Place only at the next layer of every occupied endpoint, with a maximum of 3 layers.
- Remove only a top block. Removing a large block requires its exact partner shown in
  the board's spans data.
- Recalculate layer and span from the current board even when a message is imprecise.
- Never silently repeat an action described as failed in the last Builder result.

DECISION:
Prefer instructions supported by multiple Directors. When messages conflict, use the
most specific instruction that is consistent with the public board and speaker frame.
If a single safe interpretation cannot be made, return one specific clarification
question. Produce only one action per round.

CURRENT PUBLIC BOARD:
{_json_dumps(public_state)}

AVAILABLE BLOCK CODES:
{', '.join(available_blocks)}

{BLOCK_ENCODING_REFERENCE}

{COORDINATE_REFERENCE}

FINAL DIRECTOR MESSAGES:
{public_messages}

LAST PUBLIC BUILDER RESULT:
{previous_builder_result or '(none)'}

RESPONSE CONTRACT:
Return exactly these two XML elements and no others. Keep analysis under 180 words.
<analysis>Your private interpretation and physical checks.</analysis>
<move>one complete action line</move>

The action line must match one grammar:
- PLACE:block_code:position:layer:CONFIRM:short interpretation
- PLACE:block_code:position:layer:span_to:CONFIRM:short interpretation
- REMOVE:position:layer:CONFIRM:short interpretation
- REMOVE:position:layer:span_to:CONFIRM:short interpretation
- CLARIFY:specific question for the Directors

Replace every lowercase field name with a concrete value. Keep the uppercase words
and colon separators exactly as shown."""
