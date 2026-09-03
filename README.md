# CRAFT Paper Reproduction

This repository reproduces the gameplay protocol from **CRAFT: Grounded
Multi-Agent Coordination Under Partial Information** (arXiv:2603.25268v2).

Each game follows the paper setup:

```text
1-3 randomly selected Directors speak sequentially
        ↓
Builder receives the current discussion and up to 5 oracle-verified moves
        ↓
Builder places/removes one block or requests clarification
        ↓
The environment executes the action and records task progress
```

The implementation keeps the paper's three partial Director views, personality
archetypes, 20-turn interaction, fixed Builder role, oracle-assisted candidate
moves, stacking rules, and progress metrics.

## Files

```text
benchmark/craft_structures_20.json   paper evaluation structures
benchmark/craft-80.json              additional generated structures
benchmark/generate_benchmark.py      structure generator
config/paper_config.json             experiment configuration
scripts/run_paper.py                 paper-protocol runner
scripts/check_setup.py               local setup check
src/craft_debate/paper_protocol.py   Director/Builder prompts and turn flow
src/craft_debate/oracle.py           verified move enumeration
src/craft_debate/environment.py      board rules and action execution
src/craft_debate/progress.py         CRAFT task metrics
```

The source package retains its historical Python import name so the benchmark
generator remains unchanged.

## Setup

```bash
python3 -m pip install -e .
python3 scripts/check_setup.py
```

API keys can be supplied through environment variables or files under
`.secret/`. The default configuration uses a local Ollama Director and an OpenAI
Builder.

## Run the paper protocol

Run the configured structures and repetitions:

```bash
python3 scripts/run_paper.py
```

Useful overrides:

```bash
python3 scripts/run_paper.py --structures 0,1,2 --runs 1,2,3
python3 scripts/run_paper.py --turns 20 --name paper-reproduction
python3 scripts/run_paper.py --mock --structures 0 --runs 1 --turns 2
```

The runner writes a full trajectory to `trajectories/`, a compact summary to
`results/`, and a score plot alongside the summary.

## Run the controlled no-oracle ablation

The no-oracle condition keeps the benchmark, Director/Builder models,
temperatures, turn budget, speaker sampling, board state, and current-turn
Director discussion identical to the paper condition. Its only intervention is
to remove target-derived candidate moves from the Builder's system and user
prompts. The Builder must generate its next action from the current board and
the Directors' public messages.

Run the checked-in matched configuration:

```bash
python3 scripts/run_paper.py --config config/no_oracle_config.json
```

Or override any existing config at the command line:

```bash
python3 scripts/run_paper.py --no-oracle --name no-oracle-ablation
```

No-oracle turns record `oracle_exposed_to_builder: false`, contain no
`oracle_moves`, and send no Oracle metadata to the Builder. The physics engine
still validates the Builder's autonomous action, but it does not suggest or
repair actions.

## Generate benchmarks

The default generator behavior is unchanged:

```bash
python3 benchmark/generate_benchmark.py --count 80
```

It also retains the optional hollow mode for future target-identifiable
experiments without oracle assistance. Hollow structures force the two cells
invisible to all Directors, `(1,1)` and `(2,1)`, to be empty:

```bash
python3 benchmark/generate_benchmark.py \
  --count 20 \
  --empty-hidden-cells \
  --out benchmark/craft-20-hollow.json
```

No hollow dataset is checked into the current paper-reproduction state; generate
one only when it is needed.
