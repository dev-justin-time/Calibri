import json
import tempfile
import unittest
from types import SimpleNamespace

from calibrix.doberwatch import ANONYMOUS_SEED, BITE, Doberwatch, Source
from calibrix.report import (build_report, build_verification_section,
                             write_report)


class TestReportEvidence(unittest.TestCase):
    def _result(self):
        return {
            "baseline_train": {"Quality": 0.5},
            "baseline_holdout": {"Quality": 0.5},
            "n_params": 12,
            "n_evals": 2,
            "elapsed_seconds": 0.1,
            "overfit": {"flagged": False, "details": []},
            "best": SimpleNamespace(
                trial=1,
                fitness=0.8,
                objectives={"Quality": 0.8},
                holdout={"Quality": 0.75},
                steps_used=10,
                vector=[1.0],
            ),
            "trials": [],
        }

    def test_scripted_adapter_is_simulation_not_customer_proof(self):
        report = build_report(self._result(), "ScriptedAdapter (offline)", ["Quality"])
        self.assertEqual(report["evidence"]["status"], "simulation")
        self.assertFalse(report["evidence"]["customer_roi_proven"])
        self.assertIn("not evidence", report["evidence"]["note"])

    def test_explicit_verified_status_is_preserved_but_roi_stays_unproven(self):
        report = build_report(
            self._result(),
            "CalibriFluxAdapter",
            ["Quality"],
            evidence={
                "status": "verified",
                "quality_gate": "PASS",
                "note": "Reviewed real-model holdout package.",
            },
        )
        self.assertEqual(report["evidence"]["status"], "verified")
        self.assertEqual(report["evidence"]["quality_gate"], "PASS")
        self.assertFalse(report["evidence"]["customer_roi_proven"])


class TestVerificationReport(unittest.TestCase):
    """The dashboard must show the grade *and* every fact-check verdict."""

    CRITERIA = ("acknowledges_charge", "cites_policy", "states_remedy",
                "gives_timeline", "offers_escalation_path",
                "no_unsupported_denial")

    def _result(self):
        return {
            "baseline_train": {"Quality": 0.5},
            "baseline_holdout": {"Quality": 0.5},
            "n_params": 12,
            "n_evals": 2,
            "elapsed_seconds": 0.1,
            "overfit": {"flagged": False, "details": []},
            "best": SimpleNamespace(trial=1, fitness=0.8,
                                    objectives={"Quality": 0.8},
                                    holdout={"Quality": 0.75},
                                    steps_used=10, vector=[1.0]),
            "trials": [],
        }

    def _submission(self):
        """A corroborated billing reply: grade + fact-check in one payload."""
        dw = Doberwatch(load_private=False)
        response = "The duplicate charge is reversed within 5 business days."
        sources = [
            Source("policy-doc", response, kind="authority"),
            Source("policy-page",
                   "Duplicates are reversed within 5 business days.",
                   kind="document"),
        ]
        submission = dw.submit(
            "duplicate charge refund", response, "billing_dispute",
            criterion_scores={name: 1.0 for name in self.CRITERIA},
            evidence={"cites_policy": ["policy_citation: billing-policy"]},
            sources=sources, reference_response=response)
        return dw, submission

    def test_no_section_without_a_grade(self):
        report = build_report(self._result(), "CalibriFluxAdapter", ["Quality"])
        self.assertIsNone(report["verification"])
        self.assertIsNone(build_verification_section())

    def test_submit_payload_unwraps_from_either_slot(self):
        _dw, submission = self._submission()
        for section in (build_verification_section(submission),
                        build_verification_section(verification=submission)):
            self.assertEqual(section["verdict"], submission["grade"]["verdict"])
            self.assertEqual(section["rubric_id"],
                             "billing_dispute.support_reply.v1")
            self.assertEqual(section["claim_count"], 1)
            self.assertTrue(section["claims"][0]["material"])
            self.assertEqual(section["contradiction_count"], 0)

    def test_gates_and_criteria_are_carried_into_the_section(self):
        dw, _ = self._submission()
        out = dw.submit(
            ANONYMOUS_SEED[0]["request"], ANONYMOUS_SEED[0]["response"],
            "billing_dispute",
            criterion_scores={name: 1.0 for name in self.CRITERIA},
            evidence={"cites_policy": ["policy_citation: billing-policy"]},
            reference_response="")
        section = build_verification_section(out)
        self.assertTrue(section["corroboration_gate_applied"])
        self.assertEqual(section["missing_corroboration"], ["cites_policy"])
        cites = next(c for c in section["criteria"]
                     if c["criterion"] == "cites_policy")
        self.assertTrue(cites["requires_corroboration"])
        self.assertAlmostEqual(cites["credited"], 0.5, places=4)

    def test_dashboard_renders_grade_and_fact_check_verdicts(self):
        _dw, submission = self._submission()
        claim = submission["verification"]["claims"][0]["claim"]
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_report(self._result(), "doberwatch", ["Quality"],
                                 out_dir=tmp, verification=submission)
            with open(paths["json"], encoding="utf-8") as handle:
                report = json.load(handle)
            with open(paths["html"], encoding="utf-8") as handle:
                page = handle.read()
        self.assertEqual(report["verification"]["claims"][0]["claim"], claim)
        self.assertIn("Per-claim verification", page)
        self.assertIn("Fact-check verdicts per claim", page)
        self.assertIn("Criteria \u2014 judged vs credited", page)
        self.assertIn(claim, page)
        self.assertIn('class="pill ok"', page)

    def test_a_contradicted_claim_renders_as_a_bad_pill(self):
        dw = Doberwatch(load_private=False)
        response = "The provider gives a 2 minute eviction notice."
        out = dw.submit(
            "spot instance notice", response, "billing_dispute",
            criterion_scores={name: 1.0 for name in self.CRITERIA},
            evidence={"cites_policy": ["policy_citation: billing-policy"]},
            sources=[Source("provider-docs",
                            "The provider gives a 30 second eviction notice.",
                            kind="authority")],
            reference_response=response)
        section = build_verification_section(out)
        self.assertEqual(section["contradiction_count"], 1)
        self.assertEqual(section["claims"][0]["status"], "contradicted")
        # A refuted material claim is worse than an unverified one: the veto
        # must show up in the dashboard, not just in the JSON.
        self.assertTrue(section["contradiction_veto"])
        self.assertNotEqual(section["verdict"], "PASS")
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_report(self._result(), "doberwatch", ["Quality"],
                                 out_dir=tmp, verification=out)
            with open(paths["html"], encoding="utf-8") as handle:
                page = handle.read()
        self.assertIn('class="pill bad"', page)
        self.assertIn("provider-docs", page)
        self.assertIn("contradiction veto", page)

    def test_claim_text_is_html_escaped(self):
        section = build_verification_section(
            {"verdict": BITE, "score": 0.1, "criteria": [], "domain": "d"},
            {"material_grade": 0.0, "claims": [{
                "claim": "<script>alert(1)</script>",
                "status": "unverified", "grade": 0.0, "material": True,
                "support": [], "contradict": [], "reasons": ["<b>no</b>"],
            }], "claim_count": 1, "material_count": 1, "status_counts": {}},
        )
        # The section keeps the raw text (JSON is data, not markup)...
        self.assertEqual(section["claims"][0]["claim"],
                         "<script>alert(1)</script>")
        # ...but the rendered dashboard escapes it.
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_report(self._result(), "doberwatch", ["Quality"],
                                 out_dir=tmp, grade={
                                     "verdict": BITE, "score": 0.1,
                                     "criteria": [], "domain": "d"},
                                 verification={
                                     "material_grade": 0.0, "claim_count": 1,
                                     "material_count": 1, "claims": [{
                                         "claim": "<script>alert(1)</script>",
                                         "status": "unverified", "grade": 0.0,
                                         "material": True, "support": [],
                                         "contradict": [], "reasons": [],
                                     }]})
            with open(paths["html"], encoding="utf-8") as handle:
                page = handle.read()
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)


if __name__ == "__main__":
    unittest.main()
