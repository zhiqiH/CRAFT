import asyncio
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from craft_debate.api import MockLLM  # noqa: E402
from craft_debate.benchmark import load_structures  # noqa: E402
from craft_debate.domain import ALL_COORDS  # noqa: E402
from craft_debate.environment import (  # noqa: E402
    GameState,
    get_all_physically_legal_actions,
    validate_physical_action,
)
from craft_debate.io import build_summary  # noqa: E402
from craft_debate.topology import (  # noqa: E402
    Debate,
    parse_builder_response,
    parse_director_response,
)


class RecordingMock(MockLLM):
    def __init__(self) -> None:
        super().__init__("recording-mock")
        self.calls = []

    async def complete(self, system, user, meta=None):
        self.calls.append({"system": system, "user": user, "meta": copy.deepcopy(meta or {})})
        return await super().complete(system, user, meta)


class TemplateEchoMock(RecordingMock):
    async def complete(self, system, user, meta=None):
        self.calls.append({"system": system, "user": user, "meta": copy.deepcopy(meta or {})})
        return {
            "content": (
                "<observation>Relevant private evidence, concise but spatially precise.</observation>\n"
                "Actual observation outside the element.\n"
                "<proposed_action>One concrete PLACE or REMOVE recommendation.</proposed_action>\n"
                "PLACE gs AT (0,0) LAYER 0\n"
                "<reasoning>Why that action follows from your evidence and public physics.</reasoning>\n"
                "Actual reasoning outside the element.\n"
                "<confidence>A number from 0 to 1.</confidence>\n"
                "0.9"
            ),
            "usage": {},
            "latency_seconds": 0.0,
        }


class FixedBuilderMock(RecordingMock):
    def __init__(self, content):
        super().__init__()
        self.content = content

    async def complete(self, system, user, meta=None):
        self.calls.append({"system": system, "user": user, "meta": copy.deepcopy(meta or {})})
        return {
            "content": self.content,
            "usage": {},
            "latency_seconds": 0.0,
        }


class LegalActionMaskTests(unittest.TestCase):
    def setUp(self):
        self.empty = {"structure": {coord: [] for coord in ALL_COORDS}, "spans": {}}

    def test_empty_board_mask_is_complete_and_legal(self):
        actions = get_all_physically_legal_actions(self.empty)
        self.assertEqual(75, len(actions))
        self.assertEqual(45, sum(a["block"].endswith("s") for a in actions))
        self.assertEqual(30, sum(a["block"].endswith("l") for a in actions))
        self.assertEqual(len(actions), len({a["id"] for a in actions}))
        self.assertTrue(all(validate_physical_action(self.empty, a)[0] for a in actions))

    def test_mask_depends_on_public_board_not_target(self):
        target_a = {coord: [] for coord in ALL_COORDS}
        target_b = {coord: ["rs"] for coord in ALL_COORDS}
        state_a = GameState(target_a)
        state_b = GameState(target_b)
        state_a.current_structure["(0,0)"] = ["gs"]
        state_b.current_structure["(0,0)"] = ["gs"]
        self.assertEqual(
            get_all_physically_legal_actions(state_a.snapshot()),
            get_all_physically_legal_actions(state_b.snapshot()),
        )

    def test_large_block_has_one_canonical_remove_action(self):
        state = GameState({coord: [] for coord in ALL_COORDS})
        result = state.execute_move(
            {
                "action": "place",
                "block": "gl",
                "position": "(0,0)",
                "layer": 0,
                "span_to": "(0,1)",
            }
        )
        self.assertTrue(result["ok"])
        removals = [
            action
            for action in get_all_physically_legal_actions(state.snapshot())
            if action["action"] == "remove"
        ]
        self.assertEqual(1, len(removals))
        self.assertTrue(validate_physical_action(state.snapshot(), removals[0])[0])


class ProtocolParsingTests(unittest.TestCase):
    phase1_fields = ["observation", "proposed_action", "reasoning", "confidence"]

    def test_valid_director_message_is_normalized(self):
        response = (
            "<observation>The bottom-left cell needs a green small block.</observation>\n"
            "<proposed_action>PLACE gs AT (0,0) LAYER 0</proposed_action>\n"
            "<reasoning>My private view and the public board agree.</reasoning>\n"
            "<confidence>0.8</confidence>"
        )
        parsed = parse_director_response(
            response,
            fields=self.phase1_fields,
            action_field="proposed_action",
        )
        self.assertTrue(parsed["protocol_valid"])
        self.assertEqual(
            {
                "action": "place",
                "block": "gs",
                "position": "(0,0)",
                "layer": 0,
                "span_to": None,
            },
            parsed["normalized_action"],
        )
        self.assertEqual(0.8, parsed["confidence_value"])

    def test_template_echo_and_outside_answer_are_quarantined(self):
        response = (
            "<observation>Relevant private evidence, concise but spatially precise.</observation>\n"
            "Actual observation outside the element.\n"
            "<proposed_action>One concrete PLACE or REMOVE recommendation.</proposed_action>\n"
            "PLACE gs AT (0,0) LAYER 0\n"
            "<reasoning>Why that action follows from your evidence and public physics.</reasoning>\n"
            "Actual reasoning outside the element.\n"
            "<confidence>A number from 0 to 1.</confidence>\n"
            "0.9"
        )
        parsed = parse_director_response(
            response,
            fields=self.phase1_fields,
            action_field="proposed_action",
        )
        self.assertFalse(parsed["protocol_valid"])
        self.assertIsNone(parsed["normalized_action"])
        self.assertIsNone(parsed["confidence_value"])
        self.assertTrue(
            any("copied a protocol description" in error for error in parsed["protocol_errors"])
        )
        self.assertIn(
            "Response contains text outside the required elements",
            parsed["protocol_errors"],
        )

    def test_builder_accepts_one_unique_id_and_rejects_ambiguity(self):
        exact = parse_builder_response("<action_id>A0012</action_id>")
        self.assertEqual("A0012", exact["action_id"])
        self.assertEqual("exact", exact["parse_mode"])

        recovered = parse_builder_response(
            "I choose <action_id>A0012</action_id> because it is supported."
        )
        self.assertEqual("A0012", recovered["action_id"])
        self.assertEqual("recovered", recovered["parse_mode"])

        repeated = parse_builder_response("A0012 is my choice; final answer: A0012")
        self.assertEqual("A0012", repeated["action_id"])
        self.assertEqual("recovered", repeated["parse_mode"])

        tagged_final = parse_builder_response(
            "I considered A0012, then chose <action_id>A0013</action_id>."
        )
        self.assertEqual("A0013", tagged_final["action_id"])
        self.assertEqual("recovered", tagged_final["parse_mode"])

        ambiguous = parse_builder_response("I considered A0012 but selected A0013")
        self.assertIsNone(ambiguous["action_id"])
        self.assertEqual("invalid", ambiguous["parse_mode"])
        self.assertIsNotNone(ambiguous["parse_error"])

        missing = parse_builder_response("I cannot choose an action.")
        self.assertIsNone(missing["action_id"])
        self.assertEqual("invalid", missing["parse_mode"])

    def test_summary_exposes_protocol_validity_rates(self):
        experiment = {
            "games": [
                {
                    "structure_id": "s1",
                    "structure_index": 0,
                    "run_index": 1,
                    "complexity": "easy",
                    "baseline_progress": 0.0,
                    "final_progress": 0.1,
                    "completed": False,
                    "rounds": [
                        {
                            "score": {
                                "overall_progress": 0.1,
                                "completion_percentage": 0.1,
                                "iou_score": 0.1,
                                "position_accuracy": 0.1,
                            },
                            "protocol_status": {
                                "phase1_valid": 2,
                                "phase1_total": 3,
                                "reconciliation_valid": 3,
                                "reconciliation_total": 3,
                                "builder_valid": True,
                            },
                        }
                    ],
                }
            ]
        }
        quality = build_summary(experiment)["aggregate"]["protocol_quality"]
        self.assertEqual(0.666667, quality["phase1_valid_rate"])
        self.assertEqual(1.0, quality["reconciliation_valid_rate"])
        self.assertEqual(1.0, quality["builder_valid_rate"])


class DebateBoundaryTests(unittest.TestCase):
    def run_one_round_with_builder(self, builder):
        config = json.loads((ROOT / "config" / "debate_config.json").read_text())
        config["debate"]["max_rounds"] = 1
        structure = load_structures(ROOT / "benchmark" / "craft_structures_20.json")[0]
        phase1 = RecordingMock()
        reconciliation = RecordingMock()
        debate = Debate(
            config=config,
            structure_data=structure,
            structure_index=0,
            run_index=1,
            clients={
                "phase1": phase1,
                "reconciliation": reconciliation,
                "builder": builder,
            },
            verbose=False,
        )
        game = asyncio.run(debate.run())
        return game, phase1, reconciliation

    def test_fixed_seven_calls_and_prompt_boundaries(self):
        config = json.loads((ROOT / "config" / "debate_config.json").read_text())
        config["debate"]["max_rounds"] = 1
        structure = load_structures(ROOT / "benchmark" / "craft_structures_20.json")[0]
        phase1 = RecordingMock()
        reconciliation = RecordingMock()
        builder = RecordingMock()
        debate = Debate(
            config=config,
            structure_data=structure,
            structure_index=0,
            run_index=1,
            clients={
                "phase1": phase1,
                "reconciliation": reconciliation,
                "builder": builder,
            },
            verbose=False,
        )
        debate.private_views = {
            "D1": {"private_marker": "PRIVATE_D1"},
            "D2": {"private_marker": "PRIVATE_D2"},
            "D3": {"private_marker": "PRIVATE_D3"},
        }

        game = asyncio.run(debate.run())
        self.assertEqual(
            (3, 3, 1),
            (len(phase1.calls), len(reconciliation.calls), len(builder.calls)),
        )
        for call in phase1.calls:
            own = call["meta"]["director_id"]
            self.assertIn(f"PRIVATE_{own}", call["user"])
            for other in {"D1", "D2", "D3"} - {own}:
                self.assertNotIn(f"PRIVATE_{other}", call["user"])
        for call in reconciliation.calls:
            own = call["meta"]["director_id"]
            self.assertIn(f"PRIVATE_{own}", call["user"])
            for other in {"D1", "D2", "D3"} - {own}:
                self.assertNotIn(f"PRIVATE_{other}", call["user"])
        self.assertNotIn("PRIVATE_D1", builder.calls[0]["user"])
        self.assertNotIn("PRIVATE_D2", builder.calls[0]["user"])
        self.assertNotIn("PRIVATE_D3", builder.calls[0]["user"])
        self.assertEqual({"kind", "legal_actions"}, set(builder.calls[0]["meta"]))

        round_record = game["rounds"][0]
        self.assertEqual(3, len(round_record["phase1"]))
        self.assertEqual(3, len(round_record["reconciliation"]))
        self.assertTrue(round_record["legal_action_mask"])
        self.assertTrue(round_record["physical_validation"]["ok"])
        self.assertEqual(
            {"phase1", "reconciliation", "builder"},
            set(round_record["phase_latency_seconds"]),
        )
        for call_record in round_record["phase1"] + round_record["reconciliation"]:
            self.assertIn("usage", call_record)
            self.assertIn("latency_seconds", call_record)
            self.assertTrue(call_record["protocol_valid"])
        self.assertIn("usage", round_record["builder"])
        self.assertEqual(
            {
                "phase1_valid": 3,
                "phase1_total": 3,
                "reconciliation_valid": 3,
                "reconciliation_total": 3,
                "builder_valid": True,
            },
            round_record["protocol_status"],
        )

    def test_invalid_phase1_message_is_quarantined_without_changing_topology(self):
        config = json.loads((ROOT / "config" / "debate_config.json").read_text())
        config["debate"]["max_rounds"] = 1
        structure = load_structures(ROOT / "benchmark" / "craft_structures_20.json")[0]
        phase1 = TemplateEchoMock()
        reconciliation = RecordingMock()
        builder = RecordingMock()
        debate = Debate(
            config=config,
            structure_data=structure,
            structure_index=0,
            run_index=1,
            clients={
                "phase1": phase1,
                "reconciliation": reconciliation,
                "builder": builder,
            },
            verbose=False,
        )

        game = asyncio.run(debate.run())

        self.assertEqual(
            (3, 3, 1),
            (len(phase1.calls), len(reconciliation.calls), len(builder.calls)),
        )
        self.assertEqual(0, game["rounds"][0]["protocol_status"]["phase1_valid"])
        for call in reconciliation.calls:
            self.assertIn('"protocol_valid": false', call["user"])
            self.assertNotIn("Actual observation outside the element", call["user"])
            self.assertNotIn("PLACE gs AT (0,0) LAYER 0\n", call["user"])

    def test_recovered_builder_id_executes_without_changing_topology(self):
        builder = FixedBuilderMock(
            "I choose <action_id>A0001</action_id> because it has the strongest support."
        )
        game, phase1, reconciliation = self.run_one_round_with_builder(builder)
        round_record = game["rounds"][0]

        self.assertEqual(
            (3, 3, 1),
            (len(phase1.calls), len(reconciliation.calls), len(builder.calls)),
        )
        self.assertEqual("recovered", round_record["builder"]["parse_mode"])
        self.assertTrue(round_record["builder"]["protocol_valid"])
        self.assertTrue(round_record["execution"]["ok"])

    def test_out_of_mask_builder_id_is_still_rejected(self):
        builder = FixedBuilderMock("<action_id>A9999</action_id>")
        game, _, _ = self.run_one_round_with_builder(builder)
        round_record = game["rounds"][0]

        self.assertEqual("A9999", round_record["builder"]["action_id"])
        self.assertFalse(round_record["builder"]["protocol_valid"])
        self.assertFalse(round_record["execution"]["ok"])


if __name__ == "__main__":
    unittest.main()
