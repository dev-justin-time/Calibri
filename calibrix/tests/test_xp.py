# SPDX-License-Identifier: MIT
"""Tests for the experimental stacked-approach units (XP-1..XP-7).

These pin the *quantitative* claims that docs/studio_xp_logic.md reports:
if a stacked approach stops delivering its expected effect, the suite
fails — the doc can never silently go stale.
"""

import unittest

from calibrix.studio import xp


class TestXPUnits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = xp.run_experiments()
        cls.results = {name: u["result"] for name, u in cls.batch["units"].items()}

    def test_schema_and_all_units_pass(self):
        self.assertEqual(self.batch["schema"], "calibrix.xp-results/1.0")
        self.assertEqual(len(self.batch["units"]), 7)
        for name, unit in self.batch["units"].items():
            self.assertIsNone(unit["error"], f"{name} errored: {unit['error']}")
            self.assertEqual(unit["status"], "PASS",
                             f"{name} did not pass: {unit['result']}")

    def test_xp1_converges_and_measures_quantization_gap(self):
        r = self.results["run_xp1"]
        # near the true optimum (-0.4) of the dequantized surface
        self.assertGreaterEqual(r["fitness_dequantized"], -0.5)
        # the gap is measured, not assumed; keep it small but honest
        self.assertLessEqual(abs(r["quantization_gap_pct"]), 15.0)

    def test_xp2_knee_on_grid_and_front_structure(self):
        r = self.results["run_xp2"]
        self.assertIn(r["knee_nfe"], r["nfe_grid"])
        self.assertGreaterEqual(r["max_quality"], r["knee_quality"])

    def test_xp3_ucb_reaches_target_cheaper_than_random(self):
        r = self.results["run_xp3"]
        self.assertTrue(r["ucb_hit_target"])
        self.assertLessEqual(r["ucb_spend_usd"],
                             r["random_spend_usd_at_same_ceiling"])

    def test_xp4_robust_pick_lies_on_first_front(self):
        r = self.results["run_xp4"]
        self.assertIn(r["knee_pick"], r["first_front"])
        self.assertLessEqual(r["robustness_deltas"][r["knee_pick"]], 0.2)

    def test_xp5_quality_floor_respected_by_packed_gates(self):
        r = self.results["run_xp5"]
        self.assertTrue(r["floor_respected"])
        self.assertIn(r["selected_mode"], ("int8", "fp8_e4m3", "float32"))
        # dequantized gates stay faithful to the float kernel
        self.assertLessEqual(r["relative_gain_error"], 0.05)

    def test_xp6_recovery_and_robustness_retention(self):
        r = self.results["run_xp6"]
        self.assertGreaterEqual(r["recovery_pct"], 60.0)
        self.assertGreaterEqual(r["robustness_retention_pct"], 80.0)

    def test_xp7_warm_start_matches_cold_at_equal_evals(self):
        r = self.results["run_xp7"]
        self.assertTrue(r["warm_faster"])
        self.assertTrue(r["warm_converges_earlier"])
        self.assertEqual(r["warm"]["evaluations"], r["cold"]["evaluations"])
        self.assertGreaterEqual(r["warm"]["final_fitness"],
                                r["cold"]["final_fitness"])


if __name__ == "__main__":
    unittest.main()
