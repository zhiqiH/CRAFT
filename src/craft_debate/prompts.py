"""Prompts for the 7 Debate agents, adapted from the official CRAFT prompts."""

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

CRITIC_FOCUS_LABELS = {
    "spatial_grounding": "spatial grounding",
    "mind_modeling": "mind modeling",
    "pragmatic_sufficiency": "pragmatic sufficiency",
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

DIRECTOR_PERSPECTIVE_GUIDE = """DIRECTOR PERSPECTIVE GUIDE:
D1: From left to right, sees cells (0,0), (1,0), (2,0) across all layers.
D2: From left to right, sees cells (0,0), (0,1), (0,2) across all layers.
D3: From left to right, sees cells (0,2), (1,2), (2,2) across all layers."""

FRAME_OF_REFERENCE_RULE = """FRAME OF REFERENCE RULE:
When interpreting instructions from D1, D2, or D3 you MUST adopt the frame of
reference of the speaker. To D1, "my bottom left corner" is (0,0) layer 0 and
"my top right corner" is (2,0) layer 2. To D2, "my bottom left corner" is (0,0)
layer 0 and "my top right corner" is (0,2) layer 2. To D3, "my bottom left
corner" is (0,2) layer 0 and "my top right corner" is (2,2) layer 2.
Positions invisible to ALL directors: (1,1) and (2,1). A large block visible to
ANY director cannot span either of those cells."""

STACKING_RULES = """STACKING RULES:
- "layer" means stack depth, NOT grid row.
- ALWAYS calculate layer from CURRENT BOARD STATE, never trust agent-specified layers.
- Before ANY place: count blocks at the target position from CURRENT BOARD STATE.
  You MUST place new blocks one layer above the number of blocks at that position.
- Before ANY remove: verify the position is non-empty in CURRENT BOARD STATE.
  If empty, do NOT attempt removal — tell the team and suggest placing instead.
- You may only remove the TOP block of a stack."""

LARGE_BLOCK_RULE = """LARGE BLOCK RULE:
Large blocks span TWO adjacent cells — you MUST specify both endpoints via span_to.
To choose span_to: identify the two director-relative cells explicitly referenced,
convert them to global coordinates with the DIRECTOR PERSPECTIVE GUIDE, and make
sure both lie on the correct wall. Before outputting verify:
  (a) position and span_to are orthogonal neighbours,
  (b) both endpoint stacks have the SAME height,
  (c) neither endpoint is an invisible cell ((1,1) or (2,1)).
NEVER place or remove a large block with span_to missing — it will always fail."""


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def build_proposer_prompt(
    *,
    agent_id: str,
    director_id: str,
    archetype: str,
    target_view: Dict[str, Any],
    board_state: Dict[str, List[str]],
    conversation: str,
    prev_judge_answer: Optional[str],
    round_number: int,
    available_blocks: List[str],
) -> str:
    personality = ARCHETYPES.get(archetype, ARCHETYPES["observant"])
    prev_section = ""
    if prev_judge_answer:
        prev_section = f"""
### PREVIOUS ROUND'S SYNTHESIZED ANSWER
The Judge's final answer last round was:
{prev_judge_answer}

Account for it: if it already covers your wall, do not repeat it; if it is wrong
for your wall, explicitly correct it with evidence from your target view."""

    return f"""You are Agent {agent_id}, the {archetype} proposer acting as Director {director_id} in a 7-agent Debate about a collaborative LEGO construction task.
You sit around a physical board with a Builder (the final Judge agent), two other proposers (the other Directors), and three critic agents.
From where the builder sits, D1 is to their left, D2 is across from them, and D3 is to their right.

YOU ARE {archetype.upper()}

### YOUR PERSONALITY
{personality}

VERY IMPORTANT: ADOPT THIS PERSONALITY IN YOUR INTERNAL REASONING AND PUBLIC UTTERANCES.

### YOUR PERSPECTIVE
{PERSPECTIVE_DESCRIPTIONS[director_id]}

{SPATIAL_ORIENTATION}

### HOW TO INTERPRET YOUR TARGET VIEW
- In the JSON, keys row_0/row_1/row_2 are LAYERS (vertical stack depth), NOT grid rows.
- row_0 = layer_0 (bottom layer / stack depth 0), row_1 = layer_1 (middle), row_2 = layer_2 (top).
- In each layer, blocks are listed from LEFT to RIGHT according to YOUR VIEW.
- In your PUBLIC message say "bottom layer / middle layer / top layer" (avoid "bottom row").
- color=none means that cell should be empty.
- size 1 = a small block; size 2 = a large block spanning two adjacent cells.
- Two adjacent cells in your target view with the same color and BOTH size 2 mean a SINGLE large block occupies both cells.

### EXAMPLE ANALYSIS OF TARGET VIEW AND BOARD STATE
D2's target view:
{{
  "row_0": [{{"color": "blue", "size": 1}}, {{"color": "orange", "size": 2}}, {{"color": "orange", "size": 2}}],
  "row_1": [{{"color": "yellow", "size": 1}}, {{"color": "yellow", "size": 1}}, {{"color": "orange", "size": 1}}],
  "row_2": [{{"color": "yellow", "size": 1}}, {{"color": "blue", "size": 1}}, {{"color": "green", "size": 1}}]
}}
Current board state: all cells empty.
Correct D2 analysis:
[From my perspective, the current board state has all cells empty. My target view
specifies that layer 0 should have a blue small block in my bottom left corner (0,0),
and then a large orange block spanning the middle and right cells (0,1) and (0,2).
Going left to right, layer 1 should have two small yellow blocks at (0,0) and (0,1),
and a small orange block at (0,2). Finally, layer 2 should consist of a yellow small
block at (0,0), a blue small block at (0,1), and a green small block at (0,2).
To start, the builder should place a large orange block spanning (0,1) and (0,2),
the middle and right cells of my bottom layer.]
Correct D2 utterance:
[Put a large orange block across the middle and the right side of my bottom layer.]

### YOUR JOB
- Look at your target view and compare it to the current board state.
- First, look for blocks on the board that are ALREADY consistent with your private
  target view — DO NOT TALK ABOUT THESE.
- Then figure out what the builder needs to do to make the board look correct from
  your perspective — placing missing blocks or removing wrongly placed ones.
- Talk naturally to your team, like a real person, using a wide diversity of phrasings.

### RULES FOR REASONING (private, in think tags)
- Think step by step. Use BOTH coordinates and layer numbers to work out what's missing.
- Examine the current game board closely — never ask the builder to put a block on a
  layer that has no support underneath it.
- Check whether the other directors already covered what you need.
- If your view is already complete, say so briefly.
- If someone else instructs the builder to do something that would destroy part of
  your wall, say so.
- Before removing a block, check the current board to make sure a block actually
  exists at that position.

### RULES FOR SPEAKING (public message)
- One combined message, max 35 words, that does TWO things in one or two sentences:
  1) briefly describe what ONLY YOU can see from your side (what's there, missing, wrong);
  2) give ONE specific instruction based on that observation.
- Your personality must shine through.
- Use natural spatial language: "on top of the green one", "the corner near me",
  "next to the blue block", "bottom left", "stack another one there".
- Never say coordinate numbers or layer numbers out loud.
- Never use block codes like 'gs' or 'ol' — say "small green" or "large orange".
- Speak from YOUR OWN frame of reference. For D1, "my bottom left corner" is (0,0)
  layer 0 and "my top right corner" is (2,0) layer 2. For D2, "my bottom left corner"
  is (0,0) layer 0 and "my top right corner" is (0,2) layer 2. For D3, "my bottom left
  corner" is (0,2) layer 0 and "my top right corner" is (2,2) layer 2.
- If the Judge asked a clarification question in the previous round, answer it directly
  at the start of your message before giving your instruction.
- If the Judge said a move failed, acknowledge it and suggest a correction.
{prev_section}

### RESPONSE FORMAT — EXACTLY TWO TAGS
<think>
[Your private reasoning — use coordinates freely here to work out what's needed]
</think>
<message>
[Natural human speech only — no coordinates, no block codes, no layer numbers]
</message>

### CURRENT BOARD STATE (full — what is actually built right now)
{_json_dumps(board_state)}

### YOUR TARGET VIEW (what YOU need the structure to look like from your side)
{_json_dumps(target_view)}

### AVAILABLE BLOCKS
{", ".join(available_blocks)}

### CONVERSATION SO FAR
{conversation or "(no messages yet)"}"""


def build_critic_prompt(
    *,
    agent_id: str,
    focus: str,
    board_state: Dict[str, List[str]],
    conversation: str,
    proposer_messages: Dict[str, str],
    prev_judge_answer: Optional[str],
    round_number: int,
) -> str:
    focus_label = CRITIC_FOCUS_LABELS.get(focus, focus)
    focus_instructions = {
        "spatial_grounding": """You are the SPATIAL GROUNDING critic. Attack the three answers for grounding errors against the CURRENT BOARD STATE:
- a block cannot be removed from an empty cell;
- only the TOP block of a stack can be removed;
- a new block must go one layer above the current stack height (a placement on an unsupported layer is illegal);
- large blocks span two adjacent cells at the same height and can never touch the invisible cells (1,1) or (2,1);
- the claimed color/size must be consistent with what that proposer's wall can actually see;
- check whether any proposed action would damage a wall that is already correct.
State clearly which answers are safe, which are impossible, and why. Then give ONE corrected instruction.""",
        "mind_modeling": """You are the MIND MODELING critic. Attack the answers for epistemic calibration:
- redundancy: has this information already been communicated and/or already acted on in earlier rounds?
- conflicts between the three proposers — which conflict is real and which is only apparent?
- does each answer reflect awareness of what the Judge already knows from the conversation history?
- does each answer use the proposer's uniquely visible wall, or information another proposer could provide equally well?
- does it respect the previous round's synthesized answer (build on it, or correct it with evidence)?
Then give ONE de-duplicated, conflict-resolved instruction.""",
        "pragmatic_sufficiency": """You are the PRAGMATIC SUFFICIENCY critic. Judge the three answers TOGETHER as a collective:
- do they collectively identify one specific board location that needs a place or remove?
- do they collectively pin down the block type — BOTH color AND size — for that move?
- would a rational Judge, reading only these three answers, be able to pick one correct move without independent spatial reasoning about the target?
- are the spatial anchors precise and unambiguous rather than vague relative language?
Pick the single highest-value, most actionable next move implied by the group and state it as ONE concrete instruction.""",
    }
    prev_section = ""
    if prev_judge_answer:
        prev_section = (
            "\n### PREVIOUS ROUND'S SYNTHESIZED ANSWER\n"
            f"The Judge's final answer last round was:\n{prev_judge_answer}\n"
            "Check whether the proposers are building on it or silently repeating it."
        )
    messages_block = "\n".join(f"{agent}: {message}" for agent, message in proposer_messages.items())

    return f"""You are Agent {agent_id}, the {focus_label} critic in a 7-agent Debate about a collaborative LEGO construction task.
Three proposer agents (P1/D1, P2/D2, P3/D3) just answered the question in parallel. Your job is to debate and critique their answers so the Judge (the final agent) can pick a single, physically valid next move.
You may use coordinates and block codes inside your critique, but your public message must stay in natural spatial language (no coordinates, no block codes).

{focus_instructions.get(focus, focus_instructions["spatial_grounding"])}
{prev_section}

### RESPONSE FORMAT — EXACTLY TWO TAGS
<critique>
[Bullet-point critique; coordinates and block codes allowed here]
</critique>
<message>
[ONE corrected natural-language instruction for the Judge; no coordinates, no block codes]
</message>

### CURRENT BOARD STATE (full)
{_json_dumps(board_state)}

### PROPOSER ANSWERS THIS ROUND
{messages_block}

### CONVERSATION SO FAR
{conversation or "(no messages yet)"}"""


def build_judge_prompt(
    *,
    agent_id: str,
    board_state: Dict[str, List[str]],
    available_blocks: List[str],
    conversation: str,
    proposer_messages: Dict[str, str],
    critic_messages: Dict[str, str],
    oracle_moves: Optional[List[Dict[str, Any]]],
    prev_judge_answer: Optional[str],
    round_number: int,
) -> str:
    oracle_section = ""
    if oracle_moves:
        lines = []
        for m in oracle_moves:
            action = m["action"]
            block = m.get("block", "")
            pos = m["position"]
            layer = m["layer"]
            span = m.get("span_to")
            if action == "place":
                lines.append(
                    f"  PLACE {block} at {pos} layer {layer}"
                    + (f" spanning to {span}" if span else "")
                )
            elif action == "remove":
                lines.append(
                    f"  REMOVE from {pos} layer {layer}"
                    + (f" spanning to {span}" if span else "")
                )
        oracle_section = f"""
CANDIDATE MOVES (verified physically valid for this turn):
{chr(10).join(lines)}

From this list, select the move that at least one proposer/critic is asking for.
If no candidate clearly matches what any agent is describing, CLARIFY."""

    proposer_block = "\n".join(
        f"{agent}: {message}" for agent, message in proposer_messages.items()
    )
    critic_block = "\n".join(
        f"{agent}: {message}" for agent, message in critic_messages.items()
    )
    prev_section = ""
    if prev_judge_answer:
        prev_section = (
            "\n### PREVIOUS ROUND'S SYNTHESIZED ANSWER (your own answer last round)\n"
            f"{prev_judge_answer}"
        )

    return f"""You are Agent {agent_id}, the JUDGE in a 7-agent Debate about a collaborative LEGO construction task.
Three proposers (Directors D1, D2, D3) and three critics just debated the next move in parallel layers. You are the Builder: synthesize ONE final answer and execute exactly one move, or ask a clarification question.

{SPATIAL_ORIENTATION}

{DIRECTOR_PERSPECTIVE_GUIDE}

{FRAME_OF_REFERENCE_RULE}

CURRENT BOARD STATE:
{_json_dumps(board_state)}

AVAILABLE BLOCKS: {", ".join(available_blocks)}

{BLOCK_ENCODING_REFERENCE}

{COORDINATE_REFERENCE}
{oracle_section}

PROPOSER ANSWERS (this round):
{proposer_block}

CRITIC PANEL (this round):
{critic_block}

CONVERSATION SO FAR:
{conversation or "(no messages yet)"}
{prev_section}

DECISION RULE: If 2+ agents agree on a block or position, do that first. If the
proposers disagree, follow the critic recommendation most consistent with the
CURRENT BOARD STATE.

{STACKING_RULES}

{LARGE_BLOCK_RULE}

WHEN MOVES FAIL:
- Explain WHY: e.g. "I can't remove any block from the middle cell on the bottom
  layer. There is no block there. Suggest placing [block] instead."
- Never silently retry the same failed move.

BEFORE PLACING: think step by step to make sure you interpreted the block color,
size, and the speaker's frame of reference correctly. Count blocks at the target
position from CURRENT BOARD STATE.

OUTPUT FORMAT — Choose ONE of these exact formats, wrapped in a single <move> tag:

1. Place small block: PLACE:block_code:position:layer:CONFIRM:interpretation
   Example: PLACE:bs:(0,0):0:CONFIRM:Placing blue small block at bottom-left of D1's side as requested
2. Place large block: PLACE:block_code:position:layer:span_to:CONFIRM:interpretation
   Example: PLACE:gl:(0,0):0:(1,0):CONFIRM:Placing large green block across left and middle cells of D1's bottom layer
3. Remove small block: REMOVE:position:layer:CONFIRM:interpretation
   Example: REMOVE:(1,2):0:CONFIRM:Removing the block from middle-right of D3's side as requested
4. Remove large block: REMOVE:position:layer:span_to:CONFIRM:interpretation
   Example: REMOVE:(2,2):0:(2,1):CONFIRM:Removing large green block from D3's bottom layer as requested
   NOTE: REMOVE never includes a block code — do NOT write REMOVE:bl:(0,0):...
5. Clarify: CLARIFY:your specific question

Always include the CONFIRM section to show what you understood from the debate.
Return the move line inside <move> ... </move> tags and nothing else after it.

<move>PLACE:gs:(0,0):0:CONFIRM:Placing small green block at bottom-left of D1's side as requested</move>"""
