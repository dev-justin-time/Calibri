import json
import os
import tempfile
import unittest
from types import SimpleNamespace

from calibrix.image_adapters import ComfyUIAdapter, ScriptedImageAdapter
from calibrix.image_scorers import OfflineImageQuality
from calibrix.image_validation import ValidationConfig, _gate, validate_real_kernel
from calibrix.scorers import seed_prompts


class TestComfyUIRealPath(unittest.TestCase):
    def test_node_mode_serializes_real_custom_node_not_metadata(self):
        adapter = ComfyUIAdapter(
            checkpoint="flux-dev.safetensors",
            modulation_mode="calibrix_node",
            n_blocks=19,
        )
        specs = [
            SimpleNamespace(
                component="attn",
                kernel=SimpleNamespace(weight=1.1, position=0.5, floor=0.9,
                                       focus=1e6, ripple=0.0, phase=0.0),
                gains=lambda: [1.1] * 19,
            ),
            SimpleNamespace(
                component="mlp",
                kernel=SimpleNamespace(weight=1.0, position=0.5, floor=0.95,
                                       focus=1e6, ripple=0.0, phase=0.0),
                gains=lambda: [1.0] * 19,
            ),
        ]
        adapter.apply_modulation(specs)
        workflow = adapter._workflow("a product photo", 42, adapter._last_gains)
        self.assertEqual(workflow["8"]["class_type"], "CalibrixKernelScale")
        self.assertEqual(workflow["5"]["inputs"]["model"], ["8", 0])
        self.assertIn("attn:", workflow["8"]["inputs"]["kernel_spec"])
        self.assertNotIn("ConditioningCombine", json.dumps(workflow))

    def test_rejects_old_metadata_only_modes(self):
        with self.assertRaises(ValueError):
            ComfyUIAdapter(modulation_mode="conditioning").spec_sites()

    def test_gate_cannot_verify_without_node_probe(self):
        report = {
            "prompt_counts": {"holdout": 4},
            "scores": {
                "train": {"baseline": {"ImageQuality": 0.5}, "kernel": {"ImageQuality": 0.6}},
                "holdout": {"baseline": {"ImageQuality": 0.5}, "kernel": {"ImageQuality": 0.6}},
            },
        }
        gate = _gate(report, ValidationConfig(min_primary_delta=0.01), {
            "backend": "comfyui",
            "node_available": False,
            "workflow_has_node": True,
        })
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["status"], "measured")
        self.assertTrue(any("node" in reason.lower() for reason in gate["reasons"]))

    def test_validation_dry_run_writes_paired_artifacts(self):
        prompts = seed_prompts([f"product photo {i} on a white background" for i in range(8)])
        baseline = ScriptedImageAdapter(n_layers=4, seed=7)
        kernel = ScriptedImageAdapter(n_layers=4, seed=7)
        kernel.apply_to_vector([1.05, 0.5, 1e6, 0.95, 0.0, 0.0,
                                1.0, 0.5, 1e6, 1.0, 0.0, 0.0])
        with tempfile.TemporaryDirectory() as out:
            report = validate_real_kernel(
                baseline, kernel, "attn:1.05@0.5:0.95:1e6|mlp:1@0.5:1:1e6",
                prompts, [OfflineImageQuality(prompts)], out,
                config=ValidationConfig(primary_metric="OfflineImageQuality"),
                backend_probe={"backend": "comfyui", "node_available": True,
                               "workflow_has_node": True},
            )
            self.assertIn(report["evidence"]["status"], ("measured", "verified"))
            self.assertTrue(os.path.isfile(os.path.join(out, "validation.json")))
            self.assertEqual(len(report["artifacts"]["baseline"]), 8)
            self.assertEqual(len(report["artifacts"]["kernel"]), 8)

    def test_gate_requires_real_holdout_improvement(self):
        report = {
            "prompt_counts": {"holdout": 4},
            "scores": {
                "train": {"baseline": {"ImageQuality": 0.5}, "kernel": {"ImageQuality": 0.7}},
                "holdout": {"baseline": {"ImageQuality": 0.5}, "kernel": {"ImageQuality": 0.501}},
            },
        }
        gate = _gate(report, ValidationConfig(min_primary_delta=0.01), {
            "backend": "comfyui", "node_available": True,
            "workflow_has_node": True,
        })
        self.assertFalse(gate["passed"])
        self.assertIn("primary holdout delta", " ".join(gate["reasons"]))


if __name__ == "__main__":
    unittest.main()
