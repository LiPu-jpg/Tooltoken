import copy
import unittest

from latent_register.compare_bidirectional import (
    compare_bidirectional,
    validate_training_log,
)


def result_payload() -> dict:
    return {
        "split": {"train_tools": 8, "eval_tools": 2, "tool_overlap": 0},
        "physical_token_audit": {
            "heldout_ids_seen_during_training": 0,
            "heldout_input_row_max_change": 0.0,
            "heldout_output_row_max_change": 0.0,
            "all_backbone_parameters_frozen": True,
            "dynamic_output_full_head_max_delta": 0.0,
        },
        "selection": {"generated_physical_rows": {"top1": 0.2, "top5": 0.6}},
        "input_readback_teacher_forced": {"registered": {"nll": 0.8}},
        "schema_readback_teacher_forced": {"registered": {"nll": 0.9}},
        "trajectory_readback_teacher_forced": {"nll": 2.0},
        "input_readback_generation": {
            "registered": {
                "exact_arguments": 0.1,
                "exact_schema_keys": 0.2,
                "key_precision": 0.4,
                "key_recall": 0.5,
                "shared_key_value_accuracy": 0.8,
            }
        },
        "constrained_registered_pipeline": {"end_to_end_exact_arguments": 0.03},
        "autoregressive_registered_trajectory": {
            "json_valid": 0.7,
            "readback_exact_arguments_given_correct_selection": 0.1,
            "end_to_end_exact_arguments": 0.02,
        },
        "config": {
            "seed": 17,
            "input_memory_source": "pooled",
            "trajectory_loss_weight": 0.0,
            "output_dir": "/baseline",
        },
    }


class BidirectionalComparisonTests(unittest.TestCase):
    def test_reports_strict_single_variable_comparison(self) -> None:
        baseline = result_payload()
        candidate = copy.deepcopy(baseline)
        candidate["config"]["trajectory_loss_weight"] = 1.0
        candidate["config"]["output_dir"] = "/candidate"
        candidate["autoregressive_registered_trajectory"][
            "end_to_end_exact_arguments"
        ] = 0.06

        result = compare_bidirectional(
            baseline, candidate, {"trajectory_loss_weight"}
        )

        self.assertTrue(result["audits_passed"])
        self.assertAlmostEqual(
            result["metrics"]["trajectory_end_to_end_exact"]["delta"], 0.04
        )
        self.assertFalse(result["gates"]["trajectory_loss_advancement"]["passed"])

    def test_reports_passing_trajectory_and_semantic_prefix_gates(self) -> None:
        baseline = result_payload()
        candidate = copy.deepcopy(baseline)
        candidate["config"]["trajectory_loss_weight"] = 1.0
        candidate["trajectory_readback_teacher_forced"]["nll"] = 1.5
        candidate["autoregressive_registered_trajectory"].update(
            {
                "json_valid": 0.95,
                "readback_exact_arguments_given_correct_selection": 0.35,
            }
        )
        candidate["input_readback_generation"]["registered"].update(
            {
                "exact_schema_keys": 0.32,
                "key_recall": 0.62,
            }
        )

        result = compare_bidirectional(
            baseline, candidate, {"trajectory_loss_weight"}
        )

        self.assertTrue(result["gates"]["trajectory_loss_advancement"]["passed"])
        self.assertTrue(result["gates"]["semantic_prefix_readiness"]["passed"])

    def test_rejects_unexpected_config_change(self) -> None:
        baseline = result_payload()
        candidate = copy.deepcopy(baseline)
        candidate["config"].update(
            {"trajectory_loss_weight": 1.0, "input_memory_slots": 16}
        )

        with self.assertRaisesRegex(ValueError, "Unexpected config differences"):
            compare_bidirectional(baseline, candidate, {"trajectory_loss_weight"})

    def test_rejects_failed_heldout_audit(self) -> None:
        baseline = result_payload()
        candidate = copy.deepcopy(baseline)
        candidate["config"]["trajectory_loss_weight"] = 1.0
        candidate["physical_token_audit"]["heldout_ids_seen_during_training"] = 1

        with self.assertRaisesRegex(ValueError, "failed physical-token audits"):
            compare_bidirectional(baseline, candidate, {"trajectory_loss_weight"})

    def test_reports_passing_token_resampler_gate(self) -> None:
        baseline = result_payload()
        candidate = copy.deepcopy(baseline)
        candidate["config"]["input_memory_source"] = "token_resampler"
        candidate["input_readback_teacher_forced"]["registered"]["nll"] = 0.7
        candidate["schema_readback_teacher_forced"]["registered"]["nll"] = 0.8
        candidate["input_readback_generation"]["registered"].update(
            {
                "exact_arguments": 0.08,
                "exact_schema_keys": 0.23,
                "key_recall": 0.55,
            }
        )
        candidate["autoregressive_registered_trajectory"].update(
            {
                "json_valid": 0.9,
                "readback_exact_arguments_given_correct_selection": 0.2,
            }
        )

        result = compare_bidirectional(
            baseline, candidate, {"input_memory_source"}
        )

        self.assertTrue(result["gates"]["token_resampler_advancement"]["passed"])

    def test_rejects_reversed_token_resampler_comparison(self) -> None:
        baseline = result_payload()
        baseline["config"]["input_memory_source"] = "token_resampler"
        candidate = copy.deepcopy(baseline)
        candidate["config"]["input_memory_source"] = "pooled"

        with self.assertRaisesRegex(ValueError, "from pooled to token_resampler"):
            compare_bidirectional(baseline, candidate, {"input_memory_source"})

    def test_reports_passing_memory_capacity_gate(self) -> None:
        baseline = result_payload()
        baseline["config"]["input_memory_slots"] = 8
        candidate = copy.deepcopy(baseline)
        candidate["config"]["input_memory_slots"] = 32
        candidate["input_readback_teacher_forced"]["registered"]["nll"] = 0.7
        candidate["schema_readback_teacher_forced"]["registered"]["nll"] = 0.8
        candidate["input_readback_generation"]["registered"].update(
            {
                "exact_arguments": 0.08,
                "exact_schema_keys": 0.25,
                "key_recall": 0.60,
            }
        )
        candidate["autoregressive_registered_trajectory"].update(
            {
                "json_valid": 0.9,
                "readback_exact_arguments_given_correct_selection": 0.15,
            }
        )

        result = compare_bidirectional(baseline, candidate, {"input_memory_slots"})

        self.assertTrue(result["gates"]["memory_capacity_advancement"]["passed"])

    def test_rejects_wrong_memory_capacity_direction(self) -> None:
        baseline = result_payload()
        baseline["config"]["input_memory_slots"] = 16
        candidate = copy.deepcopy(baseline)
        candidate["config"]["input_memory_slots"] = 32

        with self.assertRaisesRegex(ValueError, "from 8 to 32"):
            compare_bidirectional(baseline, candidate, {"input_memory_slots"})

    def test_reports_passing_readout_interface_gate(self) -> None:
        baseline = result_payload()
        candidate = copy.deepcopy(baseline)
        candidate["config"]["input_memory_interface"] = "readout_cross_attention"
        candidate["input_interface_audit"] = {
            "readout_adapter_nonzero_gradient_steps": 3,
            "readout_adapter_max_gradient_norm": 1.2,
            "readout_output_max_change": 0.05,
        }
        candidate["input_readback_teacher_forced"]["registered"]["nll"] = 0.7
        candidate["schema_readback_teacher_forced"]["registered"]["nll"] = 0.8
        candidate["input_readback_generation"]["registered"].update(
            {
                "exact_arguments": 0.08,
                "exact_schema_keys": 0.25,
                "key_recall": 0.60,
            }
        )
        candidate["autoregressive_registered_trajectory"].update(
            {
                "json_valid": 0.9,
                "readback_exact_arguments_given_correct_selection": 0.2,
            }
        )

        result = compare_bidirectional(
            baseline, candidate, {"input_memory_interface"}
        )

        self.assertTrue(result["gates"]["readout_interface_advancement"]["passed"])

    def test_rejects_readout_interface_without_adapter_change(self) -> None:
        baseline = result_payload()
        candidate = copy.deepcopy(baseline)
        candidate["config"]["input_memory_interface"] = "readout_cross_attention"
        candidate["input_interface_audit"] = {
            "readout_adapter_nonzero_gradient_steps": 3,
            "readout_adapter_max_gradient_norm": 1.2,
            "readout_output_max_change": 0.0,
        }

        with self.assertRaisesRegex(ValueError, "failed interface audit"):
            compare_bidirectional(baseline, candidate, {"input_memory_interface"})

    def test_rejects_reversed_readout_interface_comparison(self) -> None:
        baseline = result_payload()
        baseline["config"]["input_memory_interface"] = "readout_cross_attention"
        candidate = copy.deepcopy(baseline)
        candidate["config"]["input_memory_interface"] = "prompt_slots"

        with self.assertRaisesRegex(ValueError, "from prompt_slots"):
            compare_bidirectional(baseline, candidate, {"input_memory_interface"})

    def test_audits_gated_layerwise_interface(self) -> None:
        baseline = result_payload()
        candidate = copy.deepcopy(baseline)
        candidate["config"].update(
            {
                "input_memory_interface": "gated_layerwise_cross_attention",
                "layerwise_memory_layers": "auto:4",
                "layerwise_memory_max_gate": 0.25,
            }
        )
        candidate["input_interface_audit"] = {
            "readout_adapter_nonzero_gradient_steps": 10,
            "readout_adapter_max_gradient_norm": 1.2,
            "readout_output_max_change": None,
            "layerwise_initial_gates": {"6": 0.0, "13": 0.0},
            "layerwise_final_gates": {"6": 0.03, "13": -0.04},
            "layerwise_max_abs_gate": 0.04,
        }

        result = compare_bidirectional(
            baseline,
            candidate,
            {
                "input_memory_interface",
                "layerwise_memory_layers",
                "layerwise_memory_max_gate",
            },
        )

        self.assertIn("layerwise_interface_advancement", result["gates"])
        self.assertEqual(
            result["input_interface_audit"]["layerwise_initial_gates"]["6"],
            0.0,
        )

    def test_audits_matched_layerwise_interface_as_single_change(self) -> None:
        baseline = result_payload()
        baseline["config"].update(
            {
                "input_memory_interface": "prompt_slots",
                "layerwise_memory_layers": "auto:4",
                "layerwise_memory_max_gate": 0.25,
            }
        )
        candidate = copy.deepcopy(baseline)
        candidate["config"][
            "input_memory_interface"
        ] = "prompt_slots_plus_layerwise"
        candidate["input_interface_audit"] = {
            "readout_adapter_nonzero_gradient_steps": 10,
            "readout_adapter_max_gradient_norm": 1.2,
            "readout_output_max_change": None,
            "layerwise_initial_gates": {"6": 0.0},
            "layerwise_final_gates": {"6": 0.03},
            "layerwise_max_abs_gate": 0.03,
        }

        result = compare_bidirectional(
            baseline, candidate, {"input_memory_interface"}
        )

        self.assertEqual(
            set(result["config_differences"]), {"input_memory_interface"}
        )
        self.assertIn("layerwise_interface_advancement", result["gates"])

    def test_audits_trajectory_only_layerwise_interface(self) -> None:
        baseline = result_payload()
        baseline["config"].update(
            {
                "input_memory_interface": "prompt_slots",
                "layerwise_memory_layers": "auto:4",
                "layerwise_memory_max_gate": 0.25,
            }
        )
        candidate = copy.deepcopy(baseline)
        candidate["config"][
            "input_memory_interface"
        ] = "prompt_slots_plus_trajectory_layerwise"
        candidate["input_interface_audit"] = {
            "readout_adapter_nonzero_gradient_steps": 10,
            "readout_adapter_max_gradient_norm": 1.2,
            "readout_output_max_change": None,
            "layerwise_initial_gates": {"6": 0.0},
            "layerwise_final_gates": {"6": 0.03},
            "layerwise_max_abs_gate": 0.03,
        }

        result = compare_bidirectional(
            baseline, candidate, {"input_memory_interface"}
        )

        self.assertIn("layerwise_interface_advancement", result["gates"])

    def test_validates_warmup_and_joint_trajectory_log(self) -> None:
        text = "\n".join(
            [
                "input_epoch=1/3 phase=schema_warmup loss=0.8 execution=0.0 "
                "schema=0.8 distill=0.0 trajectory=0.0",
                "input_epoch=2/3 phase=joint loss=2.0 execution=0.5 schema=0.4 "
                "distill=0.1 trajectory=1.0",
                "input_epoch=3/3 phase=joint loss=1.5 execution=0.4 schema=0.3 "
                "distill=0.1 trajectory=0.7",
            ]
        )

        result = validate_training_log(
            text,
            1,
            1.0,
            schema_loss_weight=1.0,
            distill_loss_weight=1.0,
            expected_joint_epochs=2,
        )

        self.assertTrue(result["joint_trajectory_loss_active"])
        self.assertEqual(result["first_joint_trajectory_loss"], 1.0)
        self.assertEqual(result["last_joint_trajectory_loss"], 0.7)

    def test_rejects_inactive_joint_trajectory_loss(self) -> None:
        text = (
            "input_epoch=1/1 phase=joint loss=1.0 execution=0.5 "
            "schema=0.5 distill=0.0 trajectory=0.0"
        )
        with self.assertRaisesRegex(ValueError, "Trajectory loss is inactive"):
            validate_training_log(text, 0, 1.0)


if __name__ == "__main__":
    unittest.main()
