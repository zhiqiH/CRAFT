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
from craft_debate.topology import Debate  # noqa: E402


class RecordingMock(MockLLM):
    def __init__(self) -> None:
        super().__init__("recording-mock")
        self.calls = []

    async def complete(self, system, user, meta=None):
        self.calls.append({"system": system, "user": user, "meta": copy.deepcopy(meta or {})})
        return await super().complete(system, user, meta)


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


class DebateBoundaryTests(unittest.TestCase):
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
        self.assertEqual((3, 3, 1), (len(phase1.calls), len(reconciliation.calls), len(builder.calls)))
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
        self.assertIn("usage", round_record["builder"])


if __name__ == "__main__":
    unittest.main()
