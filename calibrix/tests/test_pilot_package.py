import json
import os
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PILOT = os.path.join(ROOT, "usecases", "4_kernel_marketplace", "pilot")


class TestProductPhotoPilotPackage(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(PILOT, "package.json"), encoding="utf-8") as handle:
            self.package = json.load(handle)
        with open(os.path.join(PILOT, "acceptance_criteria.json"), encoding="utf-8") as handle:
            self.acceptance = json.load(handle)
        with open(os.path.join(PILOT, "pricing.json"), encoding="utf-8") as handle:
            self.pricing = json.load(handle)

    def test_prompt_pack_matches_package_split(self):
        path = os.path.join(PILOT, "baseline_prompts.jsonl")
        with open(path, encoding="utf-8") as handle:
            prompts = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual(len(prompts), self.package["delivery"]["prompt_count"])
        self.assertEqual(sum(p["split"] == "train" for p in prompts), 16)
        self.assertEqual(sum(p["split"] == "holdout" for p in prompts), 8)
        self.assertEqual(len({p["id"] for p in prompts}), len(prompts))
        self.assertTrue(all(p["prompt"] and p["required"] for p in prompts))

    def test_pricing_and_acceptance_are_aligned(self):
        self.assertEqual(self.pricing["offer"]["pilot_price"], 295)
        self.assertEqual(self.acceptance["scope"]["prompt_pack"], "baseline_prompts.jsonl")
        self.assertEqual(self.acceptance["gates"]["machine_quality"]["primary_metric"],
                         "OfflineImageQuality or an agreed real image reward")
        self.assertFalse(self.acceptance["gates"]["business"]["customer_roi_proven"])
        self.assertIn("failure_policy", self.acceptance)

    def test_landing_page_is_honest_and_has_cta(self):
        with open(os.path.join(PILOT, "landing.html"), encoding="utf-8") as handle:
            page = handle.read()
        self.assertIn("$295", page)
        self.assertIn("Request the pilot", page)
        self.assertIn("same prompts and seeds", page)
        self.assertIn("does not claim conversion lift", page)
        self.assertIn("buyer_questionnaire.md", page)


if __name__ == "__main__":
    unittest.main()
