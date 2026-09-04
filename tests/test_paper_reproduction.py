import asyncio
from collections import Counter
import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from craft_debate.api import MockLLM  # noqa: E402
from craft_debate.paper_protocol import (  # noqa: E402
    DIRECTOR_ORDERS,
    PAPER_ORACLE_N,
    PaperGame,
    parse_builder_response,
    validate_director_schedule,
    validate_paper_oracle_config,
)


class PaperProtocolTests(unittest.TestCase):
    def test_builder_action_parser(self):
        parsed = parse_builder_response(
            "<move>PLACE:gs:(0,0):0:CONFIRM:following D1</move>"
        )
        self.assertEqual("place", parsed["action"])
        self.assertEqual("gs", parsed["move"]["block"])
        self.assertEqual("(0,0)", parsed["move"]["position"])
        self.assertEqual(0, parsed["move"]["layer"])

    def test_mock_paper_game_runs(self):
        config = json.loads(
            (PROJECT_ROOT / "config" / "paper_config.json").read_text(encoding="utf-8")
        )
        config["turns"] = 2
        structures = json.loads(
            (PROJECT_ROOT / "benchmark" / "craft_structures_20.json").read_text(
                encoding="utf-8"
            )
        )
        game = PaperGame(
            config=config,
            structure_data=structures[0],
            structure_index=0,
            run_index=1,
            director_client=MockLLM("mock-director"),
            builder_client=MockLLM("mock-builder"),
            verbose=False,
        )
        record = asyncio.run(game.run())
        self.assertEqual(2, record["turns_completed"])
        self.assertTrue(all("oracle_moves" in turn for turn in record["turns"]))
        self.assertTrue(all(turn["execution"]["ok"] for turn in record["turns"]))

    def test_paper_oracle_is_fixed_at_five(self):
        config = json.loads(
            (PROJECT_ROOT / "config" / "paper_config.json").read_text(encoding="utf-8")
        )
        oracle_cfg = validate_paper_oracle_config(config)
        self.assertTrue(oracle_cfg["enabled"])
        self.assertEqual(PAPER_ORACLE_N, oracle_cfg["n"])

    def test_non_paper_oracle_settings_are_rejected(self):
        for oracle_cfg in (
            {"enabled": False, "n": PAPER_ORACLE_N},
            {"enabled": True, "n": 3},
            {"enabled": True, "n": "5"},
        ):
            with self.subTest(oracle=oracle_cfg):
                with self.assertRaisesRegex(ValueError, "oracle"):
                    validate_paper_oracle_config({"oracle": oracle_cfg})

    def test_unknown_director_schedule_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "director_schedule"):
            validate_director_schedule({"director_schedule": "fixed-4"})

    def test_fixed_schedules_are_balanced_across_three_runs(self):
        base_config = json.loads(
            (PROJECT_ROOT / "config" / "paper_config.json").read_text(encoding="utf-8")
        )
        structure = json.loads(
            (PROJECT_ROOT / "benchmark" / "craft_structures_20.json").read_text(
                encoding="utf-8"
            )
        )[0]

        for schedule, expected_calls in (
            ("fixed-1", 1),
            ("fixed-2", 2),
            ("fixed-3", 3),
        ):
            with self.subTest(schedule=schedule):
                config = dict(base_config)
                config["director_schedule"] = schedule
                identity_counts = Counter()
                order_counts = Counter()
                for run_index in (1, 2, 3):
                    game = PaperGame(
                        config=config,
                        structure_data=structure,
                        structure_index=0,
                        run_index=run_index,
                        director_client=MockLLM("mock-director"),
                        builder_client=MockLLM("mock-builder"),
                        verbose=False,
                    )
                    for turn_number in range(1, 21):
                        order = game._select_speakers(turn_number)
                        self.assertEqual(expected_calls, len(order))
                        self.assertEqual(len(order), len(set(order)))
                        identity_counts.update(order)
                        order_counts[tuple(order)] += 1

                self.assertEqual(1, len(set(identity_counts.values())))
                self.assertEqual(set(DIRECTOR_ORDERS[schedule]), set(order_counts))
                self.assertEqual(1, len(set(order_counts.values())))


if __name__ == "__main__":
    unittest.main()
