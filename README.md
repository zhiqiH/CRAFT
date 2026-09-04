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

## Complete experiment matrix

All methods keep the benchmark, models, temperatures, physical rules, completion
metric, and Oracle n=5 unchanged. Adaptive horizons always use 20 turns as a
hard upper bound.

| Method | Director scheduling | Turn horizon | Status |
| --- | --- | --- | --- |
| Original | Random 1-3 | Fixed 20 | Implemented |
| Fixed-1 | Exactly 1 | Fixed 20 | Implemented |
| Fixed-2 | Exactly 2 | Fixed 20 | Implemented |
| Fixed-3 | Exactly 3 | Fixed 20 | Implemented |
| Heuristic-Comm | Adaptive heuristic | Fixed 20 | Planned |
| Heuristic-Stop | Random 1-3 | Adaptive heuristic, max 20 | Planned |
| RL-Comm | Learned Director policy | Fixed 20 | Planned |
| RL-Turn | Random 1-3 | Learned stop policy, max 20 | Planned |
| Joint RL | Learned Director policy | Learned stop policy, max 20 | Later extension |

The implementation order is: establish the four fixed-horizon baselines, study
Director adaptation while holding the horizon fixed, study stopping while
holding the Original Director scheduler fixed, and only then train both
controllers jointly. This isolates the effect of each intervention.

## Research roadmap

### Stage 0 — fixed-horizon baselines

The current implementation provides the four baseline conditions used by the
later adaptive experiments:

- three Directors with partial private views;
- Original samples 1-3 unique Directors and queries them sequentially;
- Fixed-1/2/3 query exactly 1/2/3 Directors per turn;
- an Oracle-assisted Builder with up to five verified candidate moves;
- a fixed 20-turn horizon and the paper's board, stacking, and progress rules.

Fixed-1 cycles through D1/D2/D3. Fixed-2 cycles through all six ordered pairs,
and Fixed-3 cycles through all six speaking permutations. Across runs 1, 2, and
3 at 20 turns, Director identities and speaking orders are exactly balanced.

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

Compare Heuristic-Comm and RL-Comm against all four Stage-0 baselines under the
same fixed 20-turn horizon.

### Stage 2 — outer RL for episode length

Only after the Stage-1 communication experiments are established, learn when to
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

For the isolated Heuristic-Stop and RL-Turn rows, retain the Original random 1-3
Director scheduler. Compare both methods with Original so any improvement is
attributable to the stopping controller rather than a communication change.

### Stage 3 — joint RL

Jointly train Director scheduling and stopping only after the isolated inner and
outer studies are understood. The horizon remains capped at 20.

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

Original and the balanced Fixed-1/2/3 baselines are implemented. Heuristic-Comm,
RL-Comm, Heuristic-Stop, RL-Turn, and Joint RL remain planned and are not
presented as completed experiments.

## Repository map

```text
benchmark/craft_structures_20.json   paper evaluation structures
benchmark/craft-80.json              additional generated structures
benchmark/generate_benchmark.py      structure generator
config/paper_config.json             fixed Oracle n=5 configuration
scripts/run_paper.py                 paper-baseline runner
scripts/run_baselines.py             matched four-baseline runner and comparison
scripts/check_setup.py               local setup check
src/craft_debate/paper_protocol.py   Director/Builder prompts and turn flow
src/craft_debate/oracle.py           verified move enumeration
src/craft_debate/environment.py      board rules and action execution
src/craft_debate/progress.py         CRAFT task metrics
archive/oracle_sweep_20260902/       frozen Oracle-count experiments
```

The source package retains its historical Python import name so the benchmark
generator remains unchanged.

## Remote setup

```bash
python3 -m pip install -e .
python3 scripts/check_setup.py
```

API keys can be supplied through environment variables or files under
`.secret/`. The default configuration uses a local Ollama Director and an OpenAI
Builder. On the remote machine, make sure Ollama is running and the configured
Director model is installed before starting an experiment.

Run a no-cost pipeline check first; mock output is for validation only and must
not be reported as an experimental result:

```bash
python3 scripts/run_baselines.py \
  --mock \
  --structures 0 \
  --runs 1 \
  --turns 2 \
  --quiet \
  --name-prefix baseline-smoke
```

## Run the four baselines remotely

The following command runs all four matched conditions. If `--structures` or
`--runs` is omitted, the values in `config/paper_config.json` are used.

```bash
python3 scripts/run_baselines.py \
  --config config/paper_config.json \
  --turns 20 \
  --quiet \
  --name-prefix director-baselines
```

To override the configured evaluation subset, pass the exact same structure and
run lists to the suite once; it forwards them unchanged to all four methods:

```bash
python3 scripts/run_baselines.py \
  --structures 0,1,2 \
  --runs 1,2,3 \
  --turns 20 \
  --quiet \
  --name-prefix director-baselines-s012-r123
```

To run or resume one condition separately:

```bash
python3 scripts/run_paper.py --director-schedule original --turns 20 --name baseline-original
python3 scripts/run_paper.py --director-schedule fixed-1 --turns 20 --name baseline-fixed-1
python3 scripts/run_paper.py --director-schedule fixed-2 --turns 20 --name baseline-fixed-2
python3 scripts/run_paper.py --director-schedule fixed-3 --turns 20 --name baseline-fixed-3
```

Use the same `--structures` and `--runs` values for every separate condition.
The suite is preferred because it enforces that match automatically.

## Baseline outputs

Each condition writes a full trajectory to `trajectories/`, a compact summary
to `results/`, and a score curve beside the summary. The four-condition runner
also writes `*-comparison.json` and `*-comparison.png` with:

- mean final progress and SEM;
- completion rate;
- Director calls per turn and total Director/Builder tokens;
- invalid-action rate;
- paths to every condition's summary and trajectory.

Any configuration that disables the Oracle, changes its candidate count from 5,
or names an unsupported Director schedule is rejected before the experiment
starts.

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
