"""Prompts for the oracle-free 3 Directors -> 3 reconciliations -> Builder flow."""

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

DIRECTOR_ACTION_GRAMMAR = """CANONICAL ACTION GRAMMAR:
Use exactly one of these forms in an action element:
- PLACE BLOCK_CODE AT POSITION LAYER L
- PLACE BLOCK_CODE AT POSITION LAYER L SPAN_TO POSITION
- REMOVE AT POSITION LAYER L
- REMOVE AT POSITION LAYER L SPAN_TO POSITION
Replace every uppercase parameter token with a concrete value. Keywords PLACE, AT,
LAYER, SPAN_TO, and REMOVE stay uppercase. BLOCK_CODE is one of gs/gl/bs/bl/rs/rl/
ys/yl/os/ol; POSITION is a coordinate such as those shown on the public board; L is
0, 1, or 2. SPAN_TO is required for a large block and forbidden for a small block."""


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def build_observation_prompt(
    *,
    director_id: str,
    archetype: str,
    private_view: Dict[str, Any],
    public_state: Dict[str, Any],
    public_history: str,
    previous_builder_result: Optional[str],
) -> str:
    """Phase 1: one Director sees only its private view and public context."""
    previous = previous_builder_result or "(none)"
    return f"""You are Director {director_id} in Phase 1, independent observation.
{ARCHETYPES.get(archetype, ARCHETYPES['observant'])}

Information boundary: use only YOUR PRIVATE VIEW, CURRENT PUBLIC BOARD, and PUBLIC
HISTORY below. You do not know the hidden 3D target, other Directors' raw views,
any oracle action, target-conditioned ranking, or ground-truth progress/score.

{PERSPECTIVE_DESCRIPTIONS[director_id]}
{SPATIAL_ORIENTATION}
{BLOCK_ENCODING_REFERENCE}
{DIRECTOR_ACTION_GRAMMAR}
Private-view row_0/row_1/row_2 mean bottom/middle/top LAYERS. Within each layer,
entries run left-to-right in your own frame. color=none means empty; size=2 means
a large block when the adjacent matching entry is also size=2.

Compare your private view with the public board. Recommend exactly one primitive
PLACE or REMOVE action. If evidence is insufficient, still give the best concrete
proposal and lower confidence. Do not claim access to another view.

Return exactly four XML elements, in this order, with no text outside them. The
element names are `observation`, `proposed_action`, `reasoning`, and `confidence`.
Write your own value inside every element; never repeat these field descriptions.
`proposed_action` must contain only one action in CANONICAL ACTION GRAMMAR.
`confidence` must contain only a decimal number from 0 to 1.

CURRENT PUBLIC BOARD:
{_json_dumps(public_state)}

YOUR PRIVATE VIEW ({director_id} ONLY):
{_json_dumps(private_view)}

PREVIOUS BUILDER RESULT:
{previous}

ALLOWED PUBLIC HISTORY:
{public_history or '(none)'}"""


def build_reconciliation_prompt(
    *,
    director_id: str,
    archetype: str,
    private_view: Dict[str, Any],
    public_state: Dict[str, Any],
    phase1_messages: Dict[str, Dict[str, Any]],
    public_history: str,
    previous_builder_result: Optional[str],
) -> str:
    """Phase 2: the same Director reconciles messages using its own private view."""
    return f"""You are Director {director_id} in Phase 2, cross-view reconciliation.
You are the SAME Director identity as in Phase 1. {ARCHETYPES.get(archetype, ARCHETYPES['observant'])}

Information boundary: you may use YOUR PRIVATE VIEW, CURRENT PUBLIC BOARD, and all
three serialized PHASE-1 MESSAGES. The messages are other Directors' communicated
claims, not their raw private views. You do not know the hidden 3D target, raw views
of other Directors, oracle actions, rankings, or ground-truth progress/score.

Identify agreement and contradictions, correct your earlier claim if needed, and
judge when another Director likely has complementary evidence. Finish with exactly
one revised primitive action recommendation and calibrated confidence.

Some serialized Phase-1 messages may have `protocol_valid=false`; ignore their
semantic content and do not infer what they intended.

Return exactly seven XML elements, in this order, with no text outside them. The
element names are `agreement`, `contradictions`, `revision`,
`complementary_evidence`, `recommended_action`, `reasoning`, and `confidence`.
Write your own value inside every element; never repeat these field descriptions.
`recommended_action` must contain only one action in CANONICAL ACTION GRAMMAR.
`confidence` must contain only a decimal number from 0 to 1.

{PERSPECTIVE_DESCRIPTIONS[director_id]}
{SPATIAL_ORIENTATION}
{BLOCK_ENCODING_REFERENCE}
{DIRECTOR_ACTION_GRAMMAR}

CURRENT PUBLIC BOARD:
{_json_dumps(public_state)}

YOUR PRIVATE VIEW ({director_id} ONLY):
{_json_dumps(private_view)}

ALL PHASE-1 MESSAGES:
{_json_dumps(phase1_messages)}

PREVIOUS BUILDER RESULT:
{previous_builder_result or '(none)'}

ALLOWED PUBLIC HISTORY:
{public_history or '(none)'}"""


def build_builder_prompt(
    *,
    builder_id: str,
    public_state: Dict[str, Any],
    reconciliations: Dict[str, Dict[str, Any]],
    legal_actions: List[Dict[str, Any]],
) -> str:
    """Builder prompt containing exactly the allowed three input classes."""
    return f"""You are {builder_id}, the Builder. Choose exactly one action ID.

You receive only: (1) the current public board, (2) the three Directors'
reconciliation outputs, and (3) the complete physically legal action mask.
You have no hidden target, raw private views, Phase-1 transcripts, oracle candidates,
target-conditioned rankings, or ground-truth progress/score.

Interpret Director-relative descriptions using:
{DIRECTOR_PERSPECTIVE_GUIDE}

Every action in LEGAL ACTION MASK is physically legal but may be wrong for the hidden
target. Select the action best supported by the reconciliations. Do not invent or
rewrite an action. A reconciliation with `protocol_valid=false` has been quarantined;
do not infer its missing semantic content. Return exactly one XML element named
`action_id`, containing one ID copied from the current mask, with no other text. No
sample action ID is provided because the ID must come from the current mask.

CURRENT PUBLIC BOARD:
{_json_dumps(public_state)}

PHASE-2 RECONCILIATIONS:
{_json_dumps(reconciliations)}

COMPLETE LEGAL ACTION MASK:
{_json_dumps(legal_actions)}"""
