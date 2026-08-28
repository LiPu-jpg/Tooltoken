import json
import tempfile
import unittest
from pathlib import Path

from latent_register.summarize import summarize


class SummaryTests(unittest.TestCase):
    def test_reports_paired_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed, raw, registered in ((1, 0.2, 0.3), (2, 0.4, 0.6)):
                seed_dir = root / f"seed-{seed}"
                seed_dir.mkdir()
                metrics_raw = {"top1": raw, "top5": raw, "mrr": raw, "mean_rank": raw}
                metrics_registered = {
                    "top1": registered,
                    "top5": registered,
                    "mrr": registered,
                    "mean_rank": registered,
                }
                payload = {
                    "config": {"seed": seed},
                    "split": {"eval_tools": 10, "tool_overlap": 0},
                    "raw_frozen_cosine": metrics_raw,
                    "registered_output_rows": metrics_registered,
                    "unseen_slot_audit": {
                        "max_score_delta": 0.0,
                        "prediction_disagreements": 0,
                    },
                }
                (seed_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")

            result = summarize(root)
            self.assertAlmostEqual(result["metrics"]["top1"]["paired_delta_mean"], 0.15)
            self.assertEqual(result["slot_audit"]["prediction_disagreements"], 0)


if __name__ == "__main__":
    unittest.main()
