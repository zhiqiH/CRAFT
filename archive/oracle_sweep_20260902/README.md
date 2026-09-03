# Archived Oracle-count sweep — 2026-09-02

This directory is a frozen copy of the historical Oracle candidate-count sweep.
Oracle count is no longer an active research variable; the current protocol is
fixed at the paper setting, n=5.

## Shared experimental setting

- benchmark structures: indices 0, 1, and 2;
- repetitions: runs 1, 2, and 3 (9 games per condition);
- horizon: fixed 20 turns;
- Directors: `qwen2.5:7b-instruct-q5_K_M`, temperature 0.0;
- Builder: `gpt-4o-mini`, temperature 0.1;
- varied factor: maximum Oracle candidate count only.

## Recorded results

| Oracle n | Games | Completed | Mean final progress |
| ---: | ---: | ---: | ---: |
| 1 | 9 | 0 | 0.566339 |
| 3 | 9 | 0 | 0.559351 |
| 5 | 9 | 0 | 0.548428 |
| 7 | 9 | 0 | 0.566339 |
| 9 | 9 | 0 | 0.501742 |

`trajectories/` contains the full turn-level records. `results/` contains the
compact summaries and plots. These files were copied byte-for-byte from the
corresponding active output directories so the historical evidence remains
available while the active code and README move to RL control.

The observed ordering is not monotonic in Oracle count and none of the nine-game
conditions completed a structure. Treat this sweep as exploratory evidence, not
as a strong estimate of an Oracle-count effect.
