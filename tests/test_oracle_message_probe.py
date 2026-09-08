"""Checks that the diagnostic preserves pairing and measures real action errors."""

import asyncio
import copy
import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("oracle_message_probe", ROOT / "scripts/probe_oracle_messages.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def fixture_case():
    target = {coord: [] for coord in probe.GameState({}, {}).current_structure}
    target["(0,0)"] = ["gl", "rs"]
    target["(0,1)"] = ["gl", "bs"]
    target["(2,2)"] = ["ys"]
    state = probe.GameState(target, {0: [("(0,0)", "(0,1)")]})
    state.execute_move({"action": "place", "block": "gl", "position": "(0,0)", "span_to": "(0,1)", "layer": 0})
    oracle = [{"action": "place", "block": "rs", "position": "(0,0)", "layer": 1, "span_to": None},
              {"action": "place", "block": "ys", "position": "(2,2)", "layer": 0, "span_to": None}]
    return {
        "case_id": "s7-r1-t6", "structure_index": 7, "run_index": 1, "turn_number": 6,
        "director_id": "D1", "normal_message": "unique normal utterance",
        "target_structure": target, "target_spans": state.target_spans,
        "structure_before": state.current_structure, "spans_before": state.current_spans,
        "progress_before": probe.calculate_progress(state.current_structure, target)["overall_progress"],
        "oracle_moves": oracle,
        "discussions": {"normal": "D1: unique normal utterance", "no_message": "(no director messages this turn)", "unrelated": "D1: unique unrelated utterance"},
        "donor": {"structure_index": 8, "run_index": 1, "turn_number": 6},
        "uniform_oracle_reference": {"mean_progress_delta": 0.1},
    }


class ProbeTests(unittest.TestCase):
    def test_candidate_match_checks_color_and_span(self):
        move = {"action": "place", "block": "gl", "position": "(0,0)", "span_to": "(0,1)", "layer": 0}
        reversed_move = dict(move, position="(0,1)", span_to="(0,0)")
        self.assertEqual(probe.canonical_move(move), probe.canonical_move(reversed_move))
        self.assertNotEqual(probe.canonical_move(move), probe.canonical_move(dict(move, block="rl")))
        self.assertNotEqual(probe.canonical_move(move), probe.canonical_move(dict(move, span_to="(1,0)")))

    def test_branches_are_independent_and_wrong_color_is_not_oracle(self):
        case = fixture_case()
        before = copy.deepcopy(case)
        correct = probe.evaluate_move(case, case["oracle_moves"][0])
        incorrect = probe.evaluate_move(case, dict(case["oracle_moves"][0], block="bs"))
        repeated = probe.evaluate_move(case, case["oracle_moves"][0])
        self.assertEqual(before, case)
        self.assertEqual(correct, repeated)
        self.assertTrue(correct["executed_oracle"])
        self.assertTrue(incorrect["execution"]["ok"])
        self.assertFalse(incorrect["executed_oracle"])

    def test_restored_large_span_is_removable(self):
        case = fixture_case()
        move = {"action": "remove", "position": "(0,1)", "span_to": "(0,0)", "layer": 0}
        row = probe.evaluate_move(case, move)
        self.assertTrue(row["execution"]["ok"])
        self.assertEqual([], row["structure_after"]["(0,0)"])
        self.assertEqual([], row["structure_after"]["(0,1)"])

    def test_only_discussion_changes_between_prompts(self):
        case = fixture_case()
        prompts = [probe.case_prompt(case, condition).replace(case["discussions"][condition], "DISCUSSION") for condition in probe.CONDITIONS]
        self.assertEqual(prompts[0], prompts[1])
        self.assertEqual(prompts[0], prompts[2])

    def test_unreachable_target_rejected(self):
        game = {"structure_index": 0, "target_structure": {"(1,1)": ["gl"], "(1,2)": ["gl"]},
                "target_spans": {0: [("(1,1)", "(1,2)")]}}
        with self.assertRaisesRegex(ValueError, "construction witness"):
            probe.ensure_reachable(game)

    def test_failed_call_resumes_without_repeating_successful_calls(self):
        case = fixture_case()
        calls = []
        class FailingClient:
            async def complete(self, system, prompt, meta):
                calls.append(prompt)
                if len(calls) == 2:
                    raise RuntimeError("test interruption")
                return {"content": "PLACE:rs:(0,0):1:CONFIRM:test", "usage": {"total_tokens": 1}}

        class CompletingClient:
            async def complete(self, system, prompt, meta):
                calls.append(prompt)
                return {"content": "PLACE:rs:(0,0):1:CONFIRM:test", "usage": {"total_tokens": 1}}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(json.dumps({"experiment": {"mock": False, "config": {
                "oracle": {"enabled": True, "n": 5}, "builder": {"model": "mock", "temperature": 0.1}}}}), encoding="utf-8")
            args = Namespace(trajectory=str(source), config=None, structures=[7], runs=[1], turns=[6],
                             seed=42, repeats=1, name="resume-test", mock=True, dry_run=False, resume=False)
            with patch.object(probe, "select_cases", return_value=[case]), patch.object(probe, "OUTPUT_ROOT", root / "outputs"):
                with patch.object(probe, "make_client", return_value=FailingClient()):
                    with self.assertRaisesRegex(RuntimeError, "test interruption"):
                        asyncio.run(probe.run_probe(args))
                checkpoint = root / "outputs/resume-test/results.json"
                saved = json.loads(checkpoint.read_text())
                self.assertEqual(1, len(saved["responses"]))
                self.assertEqual("interrupted", saved["status"])
                first_prompt = saved["responses"][0]["builder_prompt"]
                args.resume = True
                with patch.object(probe, "make_client", return_value=CompletingClient()):
                    asyncio.run(probe.run_probe(args))
                saved = json.loads(checkpoint.read_text())
                self.assertEqual(3, len(saved["responses"]))
                self.assertEqual("complete", saved["status"])
                self.assertEqual(1, calls.count(first_prompt))
                # Completed resumes require no credentials or client initialization.
                with patch.object(probe, "make_client", side_effect=AssertionError("must not call")):
                    asyncio.run(probe.run_probe(args))
                args.seed = 43
                with self.assertRaisesRegex(ValueError, "Resume settings"):
                    asyncio.run(probe.run_probe(args))


if __name__ == "__main__":
    unittest.main()
