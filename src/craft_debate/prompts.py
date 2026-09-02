"""Shared prompt references used by the paper-protocol reproduction."""

from __future__ import annotations

from typing import Dict


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
