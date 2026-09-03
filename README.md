# CRAFT Adaptive Communication and Horizon Control

This repository studies adaptive coordination in **CRAFT: Grounded Multi-Agent
Coordination Under Partial Information** (arXiv:2603.25268v2).

The paper protocol is the fixed experimental environment:

```text
Director scheduling policy selects 1-3 unique Directors sequentially
        ↓
Builder receives their discussion and up to 5 Oracle-verified moves
        ↓
Builder selects one move; the environment executes it
        ↓
Turn policy either continues or stops the episode
```

Oracle is no longer an experimental variable. The active code requires
`oracle.enabled=true` and `oracle.n=5`, matching the paper. The research
variables are first **communication inside a construction turn** and then
**episode length across construction turns**.

## Research roadmap

### Stage 0 — fixed paper baseline

The current implementation reproduces the fixed environment used by both RL
stages:

- three Directors with partial private views;
- 1-3 unique Directors sampled at random and queried sequentially per turn;
- an Oracle-assisted Builder with up to five verified candidate moves;
- a fixed 20-turn horizon and the paper's board, stacking, and progress rules.

This baseline remains unchanged while evaluating learned controllers.

### Stage 1 — inner RL for Director communication

First, keep the episode horizon fixed at 20 and learn only the communication
schedule within each construction turn.

At the start of a turn and after every Director message, the inner policy chooses
one action:

```text
ASK_D1 | ASK_D2 | ASK_D3 | HANDOFF_TO_BUILDER
```

Each Director may speak at most once per turn, at least one Director must speak,
and `HANDOFF_TO_BUILDER` ends communication for that turn. The Builder then acts
under the unchanged paper protocol, including Oracle n=5.

The policy observation should contain only information available at decision
time: current board, public conversation, current-turn messages, queried-Director
mask/order, and turn index. It must not contain private target views, ground-truth
progress, or Oracle candidate moves. Those privileged signals may be used to
compute a training reward or critic target, but not as policy input.

A starting reward is:

```text
r_inner = progress_delta
          - lambda_call * number_of_director_calls
          - lambda_token * director_tokens
          - lambda_invalid * invalid_builder_action
```

Evaluate the following communication conditions under the same fixed 20-turn
horizon:

| Method | Director scheduling | Turn horizon |
| --- | --- | --- |
| Original | Random 1-3 | Fixed 20 |
| Fixed-1 | Exactly 1 | Fixed 20 |
| Fixed-2 | Exactly 2 | Fixed 20 |
| Fixed-3 | Exactly 3 | Fixed 20 |
| Heuristic-Comm | Adaptive heuristic | Fixed 20 |
| RL-Comm | Learned inner policy | Fixed 20 |

For Fixed-1 and Fixed-2, balance Director identities and speaking orders so the
comparison measures communication budget rather than a favored viewpoint.

### Stage 2 — outer RL for episode length

Only after selecting and freezing the Stage-1 communication policy, learn when to
end the episode. After each Builder execution, the outer policy chooses:

```text
CONTINUE | STOP
```

Keep 20 turns as a hard safety cap. The outer policy observes the public
trajectory and execution outcomes, but not private targets, ground-truth progress,
or Oracle candidates. A starting objective is terminal task quality minus a
per-turn cost:

```text
R_outer = final_progress - lambda_turn * turns_used
```

Compare Fixed-20, heuristic stopping, and RL stopping with exactly the same frozen
inner scheduler. A random-scheduler replication can be reported separately for
comparison with the original paper protocol. This prevents changes in
communication and stopping from being credited to the wrong controller.

Jointly training both policies is a later extension, after the isolated inner and
outer studies are understood.

## Experimental controls and reporting

Hold the benchmark split, Director and Builder models, temperatures, seeds,
physical rules, completion threshold, and Oracle n=5 constant across methods.
Report both task quality and resource use:

- final progress and completion rate;
- Director calls and tokens per construction turn;
- total construction turns, latency, and invalid-action rate;
- progress per Director call and progress per token;
- quality-cost Pareto curves, with confidence intervals across structures and
  seeds.

The historical Oracle-count sweep is preserved under
[`archive/oracle_sweep_20260902`](archive/oracle_sweep_20260902/README.md). It is
archival evidence, not part of the active treatment matrix.

## Current status

The fixed Oracle n=5 paper baseline is implemented. Stage-1 inner RL and Stage-2
outer RL are the next implementation milestones; neither controller is presented
as complete in the current code.

## Repository map

```text
benchmark/craft_structures_20.json   paper evaluation structures
benchmark/craft-80.json              additional generated structures
benchmark/generate_benchmark.py      structure generator
config/paper_config.json             fixed Oracle n=5 configuration
scripts/run_paper.py                 paper-baseline runner
scripts/check_setup.py               local setup check
src/craft_debate/paper_protocol.py   Director/Builder prompts and turn flow
src/craft_debate/oracle.py           verified move enumeration
src/craft_debate/environment.py      board rules and action execution
src/craft_debate/progress.py         CRAFT task metrics
archive/oracle_sweep_20260902/       frozen Oracle-count experiments
```

The source package retains its historical Python import name so the benchmark
generator remains unchanged.

## Setup and baseline run

```bash
python3 -m pip install -e .
python3 scripts/check_setup.py
python3 scripts/run_paper.py
```

API keys can be supplied through environment variables or files under
`.secret/`. The default configuration uses a local Ollama Director and an OpenAI
Builder.

Useful baseline overrides:

```bash
python3 scripts/run_paper.py --structures 0,1,2 --runs 1,2,3
python3 scripts/run_paper.py --turns 20 --name paper-baseline
python3 scripts/run_paper.py --mock --structures 0 --runs 1 --turns 2
```

The runner writes a full trajectory to `trajectories/`, a compact summary to
`results/`, and a score plot alongside the summary. Any config that disables the
Oracle or changes its candidate count is rejected before an experiment starts.

## Generate benchmarks

```bash
python3 benchmark/generate_benchmark.py --count 80
```

The generator also retains its optional hollow mode for future controlled tasks:

```bash
python3 benchmark/generate_benchmark.py \
  --count 20 \
  --empty-hidden-cells \
  --out benchmark/craft-20-hollow.json
```
