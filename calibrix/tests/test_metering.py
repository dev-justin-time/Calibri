import json
import unittest

from calibrix.metering import (
    MODEL_PRESETS,
    QuoteConfig,
    format_quote_text,
    plan_budget,
    quote_from_preset,
    quote_job,
)


class TestQuoteJob(unittest.TestCase):
    def test_basic_quote_totals(self):
        q = quote_job(
            n_sites=20, n_components=2, n_trials=12, prompts_per_eval=64,
            steps_per_gen=15.0, popsize=8, price_per_1k_steps=1.0,
            enforce_min_power=False,
        )
        # expected: 12*8+2+1 = 99 evals * 64 prompts * 15 steps = 95040 steps
        self.assertEqual(q["evals"], 99)
        self.assertEqual(q["steps"], 95040)
        compute = 95040 / 1000 * 1.0
        margin = compute * 0.30
        prep = 1.5 * 120
        expected_total = compute + margin + prep
        self.assertAlmostEqual(q["subtotal"], round(expected_total, 2))
        self.assertAlmostEqual(q["total"], round(expected_total, 2))
        self.assertFalse(q["minimum_fee_applied"])
        # 240 params / popsize 8 means the request is underpowered, but with
        # enforce_min_power=False it is priced as requested and flagged only
        self.assertTrue(q["requested"]["underpowered"])
        self.assertEqual(q["requested"]["quoted_trials"], 12)

    def test_minimum_fee_applied(self):
        q = quote_job(
            n_sites=4, n_components=1, n_trials=2, prompts_per_eval=8,
            steps_per_gen=15.0, popsize=4, price_per_1k_steps=0.0,
        )
        # zero compute -> subtotal is prep only (180) < 250 default minimum
        # (24 params x popsize 4 still corrects 2 trials to 60, but compute stays 0)
        self.assertTrue(q["requested"]["underpowered"])
        self.assertTrue(q["minimum_fee_applied"])
        self.assertEqual(q["total"], 250.0)
        items = [li["item"] for li in q["line_items"]]
        self.assertNotIn("Rush surcharge", items)

    def test_rush_pricing(self):
        base = quote_job(
            n_sites=20, n_components=2, n_trials=12, prompts_per_eval=64,
            steps_per_gen=15.0, popsize=8, price_per_1k_steps=1.0,
            enforce_min_power=False,
        )
        rush = quote_job(
            n_sites=20, n_components=2, n_trials=12, prompts_per_eval=64,
            steps_per_gen=15.0, popsize=8, price_per_1k_steps=1.0,
            rush=True, enforce_min_power=False,
        )
        self.assertGreater(rush["total"], base["total"])
        items = [li["item"] for li in rush["line_items"]]
        self.assertIn("Rush surcharge", items)
        # rush multiplier applies to compute+margin+prep
        compute = 95040 / 1000 * 1.0
        expected = (compute * 1.3 + 180) * 1.5
        self.assertAlmostEqual(rush["total"], round(expected, 2), places=1)

    def test_underpowered_quote_is_corrected(self):
        # 60 sites x 2 comps = 720 params; 12 trials x 8 pop is far below the
        # >= 10 generations per parameter heuristic
        raw_plan = plan_budget(n_sites=60, n_components=2, n_trials=12,
                               prompts_per_eval=64, steps_per_gen=30.0, popsize=8)
        self.assertTrue(raw_plan["underpowered"])

        q = quote_job(n_sites=60, n_components=2, n_trials=12,
                      prompts_per_eval=64, steps_per_gen=30.0, popsize=8,
                      price_per_1k_steps=1.0)
        self.assertTrue(q["requested"]["underpowered"])
        self.assertEqual(q["requested"]["n_trials"], 12)
        self.assertGreater(q["requested"]["quoted_trials"], 12)
        self.assertEqual(q["evals"], q["requested"]["quoted_trials"] * 8 + 3)

        # with enforce_min_power=False the raw request is priced as-is
        q_raw = quote_job(n_sites=60, n_components=2, n_trials=12,
                          prompts_per_eval=64, steps_per_gen=30.0, popsize=8,
                          price_per_1k_steps=1.0, enforce_min_power=False)
        self.assertEqual(q_raw["requested"]["quoted_trials"], 12)
        # underpowered describes the request itself, even when not enforced
        self.assertTrue(q_raw["requested"]["underpowered"])
        self.assertLess(q_raw["total"], q["total"])

    def test_custom_quote_config(self):
        qc = QuoteConfig(margin=0.0, prep_hours=0.0, minimum_fee=0.0)
        q = quote_job(n_sites=20, n_components=2, n_trials=12,
                      prompts_per_eval=64, steps_per_gen=15.0, popsize=8,
                      price_per_1k_steps=1.0, quote=qc, enforce_min_power=False)
        self.assertAlmostEqual(q["total"], 95.04, places=2)

    def test_large_popsize_avoids_correction(self):
        # 240 params need 300 trials at popsize 8, but only 12 at popsize 200
        q = quote_job(n_sites=20, n_components=2, n_trials=12,
                      prompts_per_eval=64, steps_per_gen=15.0, popsize=200,
                      price_per_1k_steps=1.0)
        self.assertFalse(q["requested"]["underpowered"])
        self.assertEqual(q["requested"]["quoted_trials"], 12)


class TestQuoteFromPreset(unittest.TestCase):
    def test_all_presets_quote(self):
        for name in MODEL_PRESETS:
            q = quote_from_preset(name)
            self.assertIn("total", q)
            self.assertGreater(q["total"], 0)
            self.assertEqual(q["model"]["kernel_params"],
                             MODEL_PRESETS[name]["n_sites"] * MODEL_PRESETS[name]["n_components"] * 6)

    def test_unknown_preset_raises(self):
        with self.assertRaises(KeyError):
            quote_from_preset("not-a-model")

    def test_preset_matches_manual(self):
        p = MODEL_PRESETS["flux-dev-gates"]
        manual = quote_job(n_sites=p["n_sites"], n_components=p["n_components"],
                           n_trials=12, prompts_per_eval=64, steps_per_gen=p["steps"],
                           popsize=8, price_per_1k_steps=p["price_per_1k_steps"])
        from_preset = quote_from_preset("flux-dev-gates")
        self.assertEqual(manual["total"], from_preset["total"])


class TestFormatQuoteText(unittest.TestCase):
    def test_text_rendering(self):
        q = quote_job(n_sites=20, n_components=2, n_trials=12,
                      prompts_per_eval=64, steps_per_gen=15.0, popsize=8,
                      price_per_1k_steps=1.0, rush=True)
        text = format_quote_text(q, preset="sd35-medium-gates")
        self.assertIn("Calibrix hosted calibration quote - sd35-medium-gates", text)
        self.assertIn("GPU compute", text)
        self.assertIn("Rush surcharge", text)
        self.assertIn("TOTAL", text)
        self.assertIn("USD", text)

    def test_minimum_fee_rendering(self):
        q = quote_job(n_sites=2, n_components=1, n_trials=2, prompts_per_eval=8,
                      steps_per_gen=15.0, popsize=4, price_per_1k_steps=0.0)
        text = format_quote_text(q)
        self.assertIn("Minimum job fee", text)


if __name__ == "__main__":
    unittest.main()
