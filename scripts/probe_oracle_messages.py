#!/usr/bin/env python3
"""Tiny paired message ablation; imports the existing protocol without changing it.

Uses saved Fixed-1 states/messages, makes only fresh Builder calls, and executes
each response on its own state copy. This is a one-step diagnostic, not a rollout.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import itertools
import json
import random
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results" / "oracle-message-probe"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from craft_debate.domain import AVAILABLE_BLOCKS, norm_pos  # noqa: E402
from craft_debate.environment import GameState  # noqa: E402
from craft_debate.oracle import sample_oracle_moves  # noqa: E402
from craft_debate.paper_protocol import (  # noqa: E402
    BUILDER_SYSTEM,
    PAPER_ORACLE_N,
    PaperGame,
    build_builder_prompt,
    validate_paper_oracle_config,
)
from craft_debate.progress import calculate_progress  # noqa: E402
from run_paper import make_client  # noqa: E402

CONDITIONS = ("normal", "no_message", "unrelated")
SCHEMA_VERSION = 1


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def integer_list(value: str) -> List[int]:
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integers") from exc
    if not values or len(values) != len(set(values)) or min(values) < 0:
        raise argparse.ArgumentTypeError("Use distinct nonnegative integers")
    return values


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def canonical_move(move: Dict[str, Any]) -> Any:
    """Compare full actions, treating reversed large-block endpoints as equivalent."""
    if not move:
        return None
    position = norm_pos(move.get("position")) or str(move.get("position"))
    span = move.get("span_to")
    positions = (position,)
    if span:
        positions = tuple(sorted((position, norm_pos(span) or str(span))))
    return (
        move.get("action"),
        move.get("block") if move.get("action") == "place" else None,
        positions,
        move.get("layer"),
    )


def state_for(case: Dict[str, Any]) -> GameState:
    state = GameState(case["target_structure"], case["target_spans"])
    state.current_structure = copy.deepcopy(case["structure_before"])
    state.current_spans = {
        int(layer): [tuple(pair) for pair in pairs]
        for layer, pairs in case["spans_before"].items()
    }
    return state


def evaluate_move(case: Dict[str, Any], move: Dict[str, Any]) -> Dict[str, Any]:
    """A fresh branch per evaluation, including the mechanical reference."""
    state = state_for(case)
    execution = state.execute_move(move) if move else {
        "ok": None, "error": None, "move": None
    }
    after = calculate_progress(state.current_structure, state.target_structure)
    actual = execution.get("move")
    matches = [
        i for i, candidate in enumerate(case["oracle_moves"])
        if actual and canonical_move(actual) == canonical_move(candidate)
    ]
    delta = after["overall_progress"] - case["progress_before"]
    return {
        "execution": execution,
        "exact_oracle_match": bool(matches),
        "executed_oracle": bool(matches) and execution["ok"] is True,
        "oracle_index": matches[0] if matches else None,
        "progress_after": after["overall_progress"],
        "progress_delta": delta,
        "positive_progress": delta > 1e-10,
        "negative_progress": delta < -1e-10,
        "structure_after": copy.deepcopy(state.current_structure),
    }


def ensure_reachable(game: Dict[str, Any]) -> None:
    """Require an actual construction witness before spending any API calls."""
    state = GameState(game["target_structure"], game["target_spans"])
    for turn in range(1, 31):
        if state.board_complete():
            return
        moves = sample_oracle_moves(state, PAPER_ORACLE_N, random.Random(turn))
        if not moves:
            break
        if not state.execute_move(moves[0])["ok"]:
            break
    raise ValueError(
        f"Structure {game['structure_index']} has no verified construction witness "
        "under the current engine. Choose reachable structures (defaults: 7,8)."
    )


def choose_donor(games: List[Dict[str, Any]], case: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Select by identity and turn distance only, never by performance or content."""
    donors = []
    for game in games:
        if game["structure_index"] == case["structure_index"]:
            continue
        for turn in game["turns"]:
            responses = turn.get("director_responses", {})
            if len(responses) != 1 or case["director_id"] not in responses:
                continue
            message = responses[case["director_id"]]["public_message"].strip()
            if not message or message == "No message provided":
                continue
            donors.append({
                "structure_index": game["structure_index"],
                "run_index": game["run_index"],
                "turn_number": turn["turn_number"],
                "director_id": case["director_id"],
                "public_message": message,
            })
    if not donors:
        raise ValueError(f"No other-structure message for {case['director_id']}")
    distance = min(abs(d["turn_number"] - case["turn_number"]) for d in donors)
    nearest = [d for d in donors if abs(d["turn_number"] - case["turn_number"]) == distance]
    nearest.sort(key=lambda d: (d["structure_index"], d["run_index"], d["turn_number"]))
    return random.Random(f"{seed}:{case['case_id']}:donor").choice(nearest)


def select_cases(source: Dict[str, Any], structures: List[int], runs: List[int],
                 turns: List[int], seed: int) -> List[Dict[str, Any]]:
    games = source["games"]
    index = {(g["structure_index"], g["run_index"]): g for g in games}
    if len(index) != len(games):
        raise ValueError("Duplicate structure/run records in source trajectory")
    cases = []
    for structure, run in itertools.product(structures, runs):
        if (structure, run) not in index:
            raise ValueError(f"Missing source game: structure={structure}, run={run}")
        game = index[(structure, run)]
        if game.get("director_schedule") != "fixed-1":
            raise ValueError("This probe requires a saved Fixed-1 trajectory")
        ensure_reachable(game)
        state = GameState(game["target_structure"], game["target_spans"])
        found = set()
        for turn in game["turns"]:
            number = turn["turn_number"]
            if state.current_structure != turn["structure_before"]:
                raise ValueError(f"Replay disagrees with source before s{structure}/r{run}/t{number}")
            if number in turns:
                responses = turn.get("director_responses", {})
                if len(responses) != 1:
                    raise ValueError("Selected turns must contain exactly one Director message")
                did, response = next(iter(responses.items()))
                message = response["public_message"].strip()
                if not message or message == "No message provided":
                    raise ValueError("Selected normal condition has no usable public message")
                oracle = turn["oracle_moves"]
                if not 2 <= len(oracle) <= PAPER_ORACLE_N:
                    raise ValueError(
                        f"s{structure}/r{run}/t{number} has {len(oracle)} candidates; "
                        "choose a turn with 2-5 candidates to test an actual choice."
                    )
                regenerated = sample_oracle_moves(
                    state, PAPER_ORACLE_N, random.Random(structure * 1000 + number)
                )
                if oracle != regenerated:
                    raise ValueError("Recorded Oracle list differs from the current protocol")
                case = {
                    "case_id": f"s{structure}-r{run}-t{number}",
                    "structure_index": structure, "run_index": run, "turn_number": number,
                    "director_id": did, "normal_message": message,
                    "target_structure": game["target_structure"],
                    "target_spans": game["target_spans"],
                    "structure_before": copy.deepcopy(state.current_structure),
                    "spans_before": copy.deepcopy(state.current_spans),
                    "progress_before": calculate_progress(
                        state.current_structure, state.target_structure
                    )["overall_progress"],
                    "oracle_moves": copy.deepcopy(oracle),
                }
                donor = choose_donor(games, case, seed)
                case["donor"] = donor
                case["discussions"] = {
                    "normal": f"{did}: {message}",
                    "no_message": "(no director messages this turn)",
                    "unrelated": f"{did}: {donor['public_message']}",
                }
                candidates = [evaluate_move(case, move) for move in oracle]
                if not all(row["execution"]["ok"] for row in candidates):
                    raise ValueError("An Oracle candidate failed in the restored state")
                case["uniform_oracle_reference"] = {
                    "kind": "exact uniform expectation; mechanical executor, no LLM",
                    "mean_progress_delta": statistics.mean(row["progress_delta"] for row in candidates),
                    "positive_progress_rate": statistics.mean(row["positive_progress"] for row in candidates),
                    "candidate_progress_deltas": [row["progress_delta"] for row in candidates],
                }
                cases.append(case)
                found.add(number)
            saved_execution = turn["execution"]
            move = saved_execution.get("move")
            if move:
                replayed = state.execute_move(move)
                if replayed["ok"] != saved_execution["ok"]:
                    raise ValueError("Replay execution status differs from the saved trajectory")
            if state.current_structure != saved_execution["structure_after"]:
                raise ValueError("Replay board differs from the saved trajectory")
            if number >= max(turns):
                break
        if found != set(turns):
            raise ValueError(f"Missing requested turns: {sorted(set(turns) - found)}")
    return cases


def case_prompt(case: Dict[str, Any], condition: str) -> str:
    return build_builder_prompt(
        board_state=case["structure_before"], available_blocks=list(AVAILABLE_BLOCKS),
        director_discussion=case["discussions"][condition], oracle_moves=case["oracle_moves"],
    )


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {}
    normal = {(r["case_id"], r["repeat"]): r for r in rows if r["condition"] == "normal"}
    for condition in CONDITIONS:
        selected = [r for r in rows if r["condition"] == condition]
        if not selected:
            continue
        mean = lambda key: statistics.mean(r[key] for r in selected)
        paired = [(r, normal[(r["case_id"], r["repeat"])]) for r in selected
                  if (r["case_id"], r["repeat"]) in normal]
        result[condition] = {
            "n": len(selected), "executed_oracle_rate": mean("executed_oracle"),
            "positive_progress_rate": mean("positive_progress"),
            "mean_progress_delta": mean("progress_delta"),
            "invalid_action_rate": mean("invalid_action"), "clarify_rate": mean("clarify"),
            "total_tokens": sum(r["response"].get("usage", {}).get("total_tokens") or 0 for r in selected),
            "paired_with_normal_n": len(paired),
            "paired_mean_delta_minus_normal": statistics.mean(
                r["progress_delta"] - n["progress_delta"] for r, n in paired
            ) if paired else None,
            "same_executed_action_as_normal_rate": statistics.mean(
                r["execution"]["ok"] is True and n["execution"]["ok"] is True
                and canonical_move(r["execution"].get("move")) == canonical_move(n["execution"].get("move"))
                for r, n in paired
            ) if paired else None,
        }
    return result


def save_checkpoint(path: Path, result: Dict[str, Any]) -> None:
    result["summary"] = summarize(result["responses"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def print_summary(result: Dict[str, Any]) -> None:
    print("\nMOCK: pipeline validation only" if result["mock"] else "\nOne-step paired diagnostic (not episode scores)")
    print(f"{'condition':<13} {'n':>3} {'oracle OK':>10} {'positive':>10} {'mean delta':>12} {'invalid':>9} {'clarify':>9}")
    for condition, row in result["summary"].items():
        print(f"{condition:<13} {row['n']:>3} {row['executed_oracle_rate']:>10.1%} "
              f"{row['positive_progress_rate']:>10.1%} {row['mean_progress_delta']:>+12.5f} "
              f"{row['invalid_action_rate']:>9.1%} {row['clarify_rate']:>9.1%}")
    reference = statistics.mean(c["uniform_oracle_reference"]["mean_progress_delta"] for c in result["cases"])
    print(f"Mechanical uniform-Oracle mean delta: {reference:+.5f} (zero API calls)")


async def run_probe(args: argparse.Namespace) -> None:
    trajectory_path = resolve(args.trajectory)
    source_bytes = trajectory_path.read_bytes()
    source = json.loads(source_bytes)
    if source.get("experiment", {}).get("mock"):
        raise ValueError("Use a real saved trajectory; --mock only replaces fresh Builder calls")
    config = json.loads(resolve(args.config).read_text(encoding="utf-8")) if args.config else copy.deepcopy(source["experiment"]["config"])
    validate_paper_oracle_config(config)
    cases = select_cases(source, args.structures, args.runs, args.turns, args.seed)
    jobs = []
    orders = list(itertools.permutations(CONDITIONS))
    random.Random(args.seed).shuffle(orders)
    for repeat in range(1, args.repeats + 1):
        for i, case in enumerate(cases):
            order = orders[((repeat - 1) * len(cases) + i) % len(orders)]
            for condition in order:
                jobs.append({"case_id": case["case_id"], "condition": condition, "repeat": repeat})
    print(f"Cases: {len(cases)} | repeats: {args.repeats} | fresh Builder requests: {len(jobs)} | Director requests: 0")
    print(f"Builder: {config['builder']['model']} | temperature: {config['builder'].get('temperature', 0)}")
    for case in cases:
        print(f"  {case['case_id']}: {len(case['oracle_moves'])} candidates, {case['director_id']}, "
              f"donor=s{case['donor']['structure_index']}/r{case['donor']['run_index']}/t{case['donor']['turn_number']}")
    if args.dry_run:
        print("DRY RUN: source replay verified; no API calls, credentials, or output files used.")
        return

    # Resume refuses changes to input data, prompts, evaluation code, or settings.
    code_hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in [Path(__file__).resolve(), ROOT / "scripts/run_paper.py",
                                *sorted((ROOT / "src/craft_debate").glob("*.py"))]}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "config": config, "mock": args.mock, "seed": args.seed,
        "cases": cases, "jobs": jobs, "code_sha256": code_hashes,
        "builder_system": BUILDER_SYSTEM,
    }
    fingerprint = digest(manifest)
    output = OUTPUT_ROOT / args.name
    checkpoint = output / "results.json"
    if output.exists():
        if not args.resume:
            raise ValueError(f"Output exists: {output}. Use --resume or a new --name.")
        result = json.loads(checkpoint.read_text(encoding="utf-8"))
        if result.get("fingerprint") != fingerprint:
            raise ValueError("Resume settings/source/code differ. Restore them or use a new --name.")
    else:
        if args.resume:
            raise ValueError("Nothing to resume; run without --resume first")
        result = {
            "fingerprint": fingerprint, "created_at": datetime.now(timezone.utc).isoformat(),
            "name": args.name, "mock": args.mock, "status": "running",
            "kind": "matched one-step message diagnostic; not a full-episode baseline",
            "source_trajectory": str(trajectory_path), "manifest": manifest,
            "cases": cases, "planned_builder_requests": len(jobs), "responses": [],
        }
    completed = {(r["case_id"], r["condition"], r["repeat"]) for r in result["responses"]}
    pending = [j for j in jobs if (j["case_id"], j["condition"], j["repeat"]) not in completed]
    if pending:
        # Reuse the existing client. No Director client or model is initialized.
        client = make_client(config["builder"], args.mock, config.get("api", {}))
        if not output.exists():
            output.mkdir(parents=True)
        result["status"] = "running"
        result.pop("last_error", None)
        save_checkpoint(checkpoint, result)
        by_id = {case["case_id"]: case for case in cases}
        try:
            for job in pending:
                case = by_id[job["case_id"]]
                prompt = case_prompt(case, job["condition"])
                response = await client.complete(
                    BUILDER_SYSTEM, prompt, {"kind": "judge", "oracle_moves": copy.deepcopy(case["oracle_moves"])}
                )
                parsed = PaperGame._parse_builder_output(response["content"])
                row = evaluate_move(case, parsed.get("move"))
                if parsed["action"] == "unparsed":
                    row["execution"].update(ok=False, error=parsed.get("parse_error"))
                row.update(job)
                row.update({
                    "response": response, "parsed": parsed, "builder_prompt": prompt,
                    "clarify": parsed["action"] == "clarify",
                    "invalid_action": row["execution"]["ok"] is False,
                })
                result["responses"].append(row)
                save_checkpoint(checkpoint, result)
                print(f"[{len(result['responses'])}/{len(jobs)}] {job['case_id']} {job['condition']:<10} "
                      f"oracle={row['executed_oracle']} delta={row['progress_delta']:+.5f}", flush=True)
        except Exception as exc:
            result["status"] = "interrupted"
            result["last_error"] = str(exc)
            save_checkpoint(checkpoint, result)
            raise
    result["status"] = "complete"
    save_checkpoint(checkpoint, result)
    print_summary(result)
    print(f"Results: {checkpoint}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", default="trajectories/director-baselines-fixed-1.json")
    parser.add_argument("--config", help="Optional config override; default: saved trajectory config")
    parser.add_argument("--structures", type=integer_list, default=[7, 8], help="Zero-based indices; default: 7,8")
    parser.add_argument("--runs", type=integer_list, default=[1])
    parser.add_argument("--turns", type=integer_list, default=[1, 6, 11], help="One-based source turns; default: 1,6,11")
    parser.add_argument("--repeats", type=int, default=1, help="Fresh Builder repeats for each condition (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Donor and request-order seed; not an API decoding seed")
    parser.add_argument("--name", default=f"probe-{datetime.now():%Y%m%d-%H%M%S}")
    parser.add_argument("--dry-run", action="store_true", help="Validate cases and print request count; no API calls or output")
    parser.add_argument("--mock", action="store_true", help="Offline plumbing check; NOT evidence about message dependence")
    parser.add_argument("--resume", action="store_true", help="Resume saved successful calls with identical options and code")
    args = parser.parse_args()
    if args.repeats < 1 or min(args.runs) < 1 or min(args.turns) < 1:
        parser.error("repeats, runs and turns must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        parser.error("--name must be a simple filename, without directory separators")
    return args


def main() -> int:
    try:
        asyncio.run(run_probe(parse_args()))
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
