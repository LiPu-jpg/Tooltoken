import json
import tempfile
import unittest
from pathlib import Path

from latent_register.prepare_toolgen_pilot import (
    prepare_toolgen_pilot,
    target_tokens,
)


def conversation(user, assistant):
    return {
        "conversations": [
            {"role": "user", "content": user, "loss": False},
            {"role": "assistant", "content": assistant, "loss": True},
        ]
    }


class PrepareToolGenPilotTest(unittest.TestCase):
    def test_target_tokens_supports_trajectory_message_keys(self):
        record = {
            "conversations": [
                {"from": "assistant", "value": "<<A&&one>>"},
                {"from": "user", "value": "documentation"},
                {"from": "assistant", "value": "{\"x\":1}"},
                {"from": "assistant", "value": "<<B&&two>>"},
            ]
        }
        self.assertEqual(target_tokens(record), {"<<A&&one>>", "<<B&&two>>"})

    def test_builds_seeded_subset_without_partial_trajectories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registration = root / "registration.json"
            retrieval = root / "retrieval.json"
            trajectories = root / "trajectories.json"
            tokens = ["<<A&&one>>", "<<B&&two>>", "<<C&&three>>"]
            registration.write_text(
                json.dumps(
                    [
                        conversation(
                            f"Tool Name: {name}. Tool Description: {name}. "
                            f"Api Name: {name} Api Description: {name}.",
                            token,
                        )
                        for name, token in zip(["A", "B", "C"], tokens)
                    ]
                ),
                encoding="utf-8",
            )
            retrieval.write_text(
                json.dumps([conversation(f"query {index}", token) for index, token in enumerate(tokens)]),
                encoding="utf-8",
            )
            trajectories.write_text(
                json.dumps(
                    [
                        {
                            "id": "single-a",
                            "conversations": [{"from": "assistant", "value": tokens[0]}],
                        },
                        {
                            "id": "mixed",
                            "conversations": [
                                {"from": "assistant", "value": tokens[0]},
                                {"from": "assistant", "value": tokens[1]},
                            ],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            output = root / "output"
            manifest = prepare_toolgen_pilot(
                registration=registration,
                retrieval=retrieval,
                trajectories=trajectories,
                output_dir=output,
                tool_count=1,
                seed=17,
            )
            selected = {
                json.loads(line)["token"]
                for line in (output / "selected_tools.jsonl").read_text().splitlines()
            }
            self.assertEqual(len(selected), 1)
            retained_trajectories = json.loads(
                (output / "toolgen_atomic_G123_dfs.json").read_text()
            )
            for record in retained_trajectories:
                self.assertTrue(target_tokens(record) <= selected)
            self.assertEqual(
                manifest["counts"]["trajectories"].get(
                    "excluded_mixed_selected_and_unselected", 0
                ),
                int(tokens[0] in selected) ^ int(tokens[1] in selected),
            )
            self.assertTrue(manifest["audits"]["selected_tokens_unique"])


if __name__ == "__main__":
    unittest.main()
