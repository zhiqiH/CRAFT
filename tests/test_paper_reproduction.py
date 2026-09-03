import asyncio
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from craft_debate.api import MockLLM  # noqa: E402
from craft_debate.paper_protocol import (  # noqa: E402
    PaperGame,
    build_no_oracle_builder_prompt,
    parse_builder_response,
)


class RecordingNoOracleBuilder:
    def __init__(self):
        self.calls = []

    async def complete(self, system, user, meta=None):
        self.calls.append({"system": system, "user": user, "meta": meta})
        return {
            "content": "<move>PLACE:gs:(0,0):0:CONFIRM:following D1's public message</move>",
            "usage": {},
            "latency_seconds": 0.0,
        }


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

    def test_no_oracle_config_changes_only_the_exposure_flag(self):
        paper_config = json.loads(
            (PROJECT_ROOT / "config" / "paper_config.json").read_text(encoding="utf-8")
        )
        no_oracle_config = json.loads(
            (PROJECT_ROOT / "config" / "no_oracle_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertTrue(paper_config["oracle"]["enabled"])
        self.assertFalse(no_oracle_config["oracle"]["enabled"])
        paper_config["oracle"]["enabled"] = False
        self.assertEqual(paper_config, no_oracle_config)

    def test_no_oracle_prompt_contains_only_public_decision_inputs(self):
        prompt = build_no_oracle_builder_prompt(
            board_state={f"({i},{j})": [] for i in range(3) for j in range(3)},
            available_blocks=["gs", "gl"],
            director_discussion="D1: Place a small green block in my bottom-left corner.",
        )
        self.assertIn("CURRENT BOARD STATE", prompt)
        self.assertIn("DIRECTOR DISCUSSION", prompt)
        self.assertIn("D1: Place a small green block", prompt)
        self.assertNotIn("CANDIDATE MOVES", prompt)
        self.assertNotIn("oracle", prompt.lower())

    def test_no_oracle_game_never_calls_or_exposes_oracle(self):
        config = json.loads(
            (PROJECT_ROOT / "config" / "no_oracle_config.json").read_text(
                encoding="utf-8"
            )
        )
        config["turns"] = 1
        structures = json.loads(
            (PROJECT_ROOT / "benchmark" / "craft_structures_20.json").read_text(
                encoding="utf-8"
            )
        )
        builder = RecordingNoOracleBuilder()
        game = PaperGame(
            config=config,
            structure_data=structures[0],
            structure_index=0,
            run_index=1,
            director_client=MockLLM("mock-director"),
            builder_client=builder,
            verbose=False,
        )

        with patch(
            "craft_debate.paper_protocol.sample_oracle_moves",
            side_effect=AssertionError("oracle called in no-oracle mode"),
        ):
            record = asyncio.run(game.run())

        turn = record["turns"][0]
        self.assertNotIn("oracle_moves", turn)
        self.assertFalse(turn["oracle_exposed_to_builder"])
        self.assertIsNone(turn["builder_followed_oracle"])
        self.assertTrue(turn["execution"]["ok"])
        self.assertEqual({"kind": "builder_autonomous"}, builder.calls[0]["meta"])
        self.assertNotIn("oracle", builder.calls[0]["system"].lower())
        self.assertNotIn("oracle", builder.calls[0]["user"].lower())
        self.assertNotIn("oracle", json.dumps(builder.calls[0]["meta"]).lower())


if __name__ == "__main__":
    unittest.main()
