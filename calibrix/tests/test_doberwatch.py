# SPDX-License-Identifier: MIT
"""Tests for the Doberwatch Finance watchdog.

These pin the guarantees the watchdog is sold on: a grade is evidence-backed
and drift-aware, the cache never shares without consent and always audits,
complaints follow the Growl/Bark/Bite ladder, checkpoint decisions fall out of
the documented cost formulas, and the ten closest answers are offered before
any model call.
"""

import os
import tempfile
import unittest

from calibrix.doberwatch import (
    ANONYMOUS_SEED,
    BARK,
    BITE,
    GRACE_PERIOD_S,
    GRADED_RESPONSES,
    GROWL,
    PASS,
    PRIVATE_SEED,
    CheckpointCostModel,
    CheckpointWatchdog,
    ConsentPolicy,
    Criterion,
    Doberwatch,
    FailureModel,
    GrowlBarkBite,
    ResponseCache,
    Rubric,
    Source,
    assess_fair_value,
    breakeven_checkpoint_interval_s,
    check_interval_claims,
    checkpoint_plan,
    checkpoint_write_usd,
    compare_claim_to_source,
    compare_offers,
    corroboration_for,
    cross_model_agreement,
    escalation_path,
    expected_eviction_loss_usd,
    expected_waste_fraction,
    grade_response,
    hash_embedding,
    implied_overhead_s,
    extract_numbers,
    has_negation,
    is_material,
    monthly_waste_usd,
    normalize_number,
    numeric_conflict,
    split_claims,
    sources_for,
    refund_strategy,
    tokenize,
    top_k,
    verdict_for,
    verify_claim,
    verify_response,
    young_daly_interval_s,
)
from calibrix.doberwatch.similarity import cosine, similarity


def _rubric() -> Rubric:
    return Rubric(
        rubric_id="test.support.v1", domain="test", task="support_reply",
        criteria=(
            Criterion("proven", 0.5, requires_evidence=True,
                      evidence_kind="citation"),
            Criterion("helpful", 0.5),
        ),
        pass_threshold=0.70, growl_threshold=0.55, bite_threshold=0.35,
        expected_evidence=("citation",),
    )


class TestSimilarity(unittest.TestCase):
    def test_tokenize_drops_stopwords_and_short_tokens(self):
        self.assertEqual(tokenize("The quick brown fox a I x"), ["quick", "brown", "fox"])

    def test_identical_text_is_fully_similar(self):
        self.assertAlmostEqual(similarity("duplicate charge billing", "duplicate charge billing"), 1.0)

    def test_embedding_is_unit_norm(self):
        vec = hash_embedding("refund denied please help")
        self.assertAlmostEqual(float((vec ** 2).sum()), 1.0, places=9)

    def test_empty_text_is_zero_risk(self):
        self.assertEqual(cosine(hash_embedding(""), hash_embedding("anything")), 0.0)

    def test_top_k_is_ordered_and_deterministic(self):
        candidates = [("duplicate charge removed", "a"),
                      ("unrelated quantum zebra", "b"),
                      ("charge duplicated again", "c")]
        first = top_k("duplicate charge", candidates, k=2)
        second = top_k("duplicate charge", candidates, k=2)
        self.assertEqual([p for _s, p in first], [p for _s, p in second])
        self.assertEqual(first[0][1], "a")
        self.assertGreaterEqual(first[0][0], first[1][0])


class TestGrading(unittest.TestCase):
    def test_weights_normalize(self):
        rubric = _rubric()
        self.assertAlmostEqual(sum(rubric.normalized_weights().values()), 1.0)

    def test_pass_requires_evidence_and_quality(self):
        grade = grade_response(_rubric(), {"proven": 1.0, "helpful": 1.0},
                               evidence={"proven": ["citation: clause-2"]})
        self.assertEqual(grade.verdict, PASS)
        self.assertFalse(grade.complaint)
        self.assertAlmostEqual(grade.score, 1.0, places=6)

    def test_missing_evidence_caps_the_criterion(self):
        grade = grade_response(_rubric(), {"proven": 1.0, "helpful": 1.0})
        proven = next(r for r in grade.criteria if r["criterion"] == "proven")
        self.assertAlmostEqual(proven["credited"], 0.5)
        self.assertEqual(grade.missing_evidence, ["proven"])

    def test_evidence_gate_prevents_a_clean_pass(self):
        grade = grade_response(_rubric(), {"proven": 1.0, "helpful": 1.0})
        self.assertTrue(grade.evidence_gate_applied)
        self.assertEqual(grade.verdict, GROWL)  # capped at growl_threshold
        self.assertTrue(grade.complaint)

    def test_verdict_bands(self):
        rubric = _rubric()
        self.assertEqual(verdict_for(0.95, rubric), PASS)
        self.assertEqual(verdict_for(0.60, rubric), GROWL)
        self.assertEqual(verdict_for(0.40, rubric), BARK)
        self.assertEqual(verdict_for(0.10, rubric), BITE)

    def test_anti_drift_penalises_unrelated_answers(self):
        on_topic = grade_response(
            _rubric(), {"proven": 0.9, "helpful": 0.9},
            evidence={"proven": ["citation"]},
            response="duplicate charge refunded within five business days",
            reference="duplicate charge refunded within five business days")
        drifted = grade_response(
            _rubric(), {"proven": 0.9, "helpful": 0.9},
            evidence={"proven": ["citation"]},
            response="zebra quantum xylophone",
            reference="duplicate charge refunded within five business days")
        self.assertEqual(on_topic.drift_penalty, 0.0)   # identical to reference
        self.assertEqual(drifted.drift_penalty, 0.30)   # max penalty at drift ~0
        self.assertGreater(drifted.drift_penalty, on_topic.drift_penalty)
        self.assertLess(drifted.score, on_topic.score)


class TestCacheConsent(unittest.TestCase):
    def test_deny_policy_refuses_to_cache(self):
        cache = ResponseCache(policy=ConsentPolicy(mode="deny"))
        with self.assertRaises(PermissionError):
            cache.record("req", "resp", "billing_dispute", 0.9, PASS)

    def test_local_only_stores_but_never_shares(self):
        cache = ResponseCache()  # default: local_only
        entry = cache.record("req", "resp", "billing_dispute", 0.9, PASS, consent=True)
        self.assertEqual(entry.source, "local")
        share = cache.share_anonymous(entry.entry_id, consent=True)
        self.assertFalse(share["shared"])

    def test_sharing_requires_policy_and_entry_consent(self):
        policy = ConsentPolicy(mode="anonymous_share")
        cache = ResponseCache(policy=policy)
        consented = cache.record("req", "resp", "billing_dispute", 0.9, PASS,
                                 consent=True)
        not_consented = cache.record("req2", "resp2", "billing_dispute", 0.9, PASS)
        self.assertTrue(cache.share_anonymous(consented.entry_id, True)["shared"])
        self.assertFalse(cache.share_anonymous(not_consented.entry_id, True)["shared"])
        self.assertFalse(cache.share_anonymous(consented.entry_id, False)["shared"])

    def test_pii_is_masked_before_storage(self):
        cache = ResponseCache()
        entry = cache.record("email me at bob@corp.com about card",
                             "reply to bob@corp.com", "billing_dispute", 0.9, PASS)
        self.assertIn("[email]", entry.request)
        self.assertIn("email", entry.pii_masked)

    def test_audit_chain_detects_tampering(self):
        cache = ResponseCache()
        cache.record("req", "resp", "billing_dispute", 0.9, PASS)
        self.assertTrue(cache.audit_verify()["valid"])
        cache.audit_log()[0]["event"] = "tampered"
        self.assertFalse(cache.audit_verify()["valid"])

    def test_persistence_keeps_entries_and_chain(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "cache.json")
            cache = ResponseCache(policy=ConsentPolicy(mode="local_only"), path=path)
            cache.record("req", "resp", "billing_dispute", 0.9, PASS)
            reloaded = ResponseCache(path=path)
            self.assertEqual(len(reloaded), 1)
            self.assertTrue(reloaded.audit_verify()["valid"])
            self.assertEqual(reloaded.policy.mode, "local_only")

    def test_retrieve_is_audited_and_domain_filtered(self):
        cache = ResponseCache()
        cache.record("duplicate charge on my invoice", "reversed", "billing_dispute", 0.9, PASS)
        cache.record("collector keeps calling", "validate", "debt_collection", 0.9, PASS)
        hits = cache.retrieve("duplicate charge", domain="billing_dispute", k=10)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].entry.domain, "billing_dispute")
        events = [e["event"] for e in cache.audit_log()]
        self.assertIn("cache.read", events)
        self.assertIn("cache.record", events)


class TestComplaintEscalation(unittest.TestCase):
    def _failing(self):
        return grade_response(_rubric(), {"proven": 0.1, "helpful": 0.1})

    def test_passing_response_cannot_be_filed(self):
        passing = grade_response(_rubric(), {"proven": 1.0, "helpful": 1.0},
                                 evidence={"proven": ["citation"]})
        with self.assertRaises(ValueError):
            GrowlBarkBite().file("test", passing)

    def test_severity_follows_the_grade_verdict(self):
        workflow = GrowlBarkBite()
        complaint = workflow.file("test", self._failing())
        self.assertEqual(complaint.severity, BITE)
        self.assertEqual(complaint.status, "open")
        self.assertEqual(len(workflow.open_complaints()), 1)

    def test_escalate_then_resolve(self):
        workflow = GrowlBarkBite()
        complaint = workflow.file("test", self._failing())
        escalated = workflow.escalate(complaint.complaint_id, "no reply in 14 days")
        self.assertEqual(escalated.stage, 2)
        self.assertEqual(escalated.status, "escalated")
        resolved = workflow.resolve(complaint.complaint_id, "refunded")
        self.assertEqual(resolved.status, "resolved")
        self.assertEqual(workflow.open_complaints(), [])
        self.assertEqual(workflow.counts()["total"], 1)

    def test_audit_callback_receives_events(self):
        seen = []
        workflow = GrowlBarkBite(audit=lambda event, payload: seen.append(event))
        workflow.file("test", self._failing())
        self.assertIn("complaint.filed", seen)


class TestCheckpointEconomics(unittest.TestCase):
    def test_write_cost_formula(self):
        self.assertAlmostEqual(checkpoint_write_usd(1_000_000_000, 0.02), 0.02)
        self.assertEqual(checkpoint_write_usd(0), 0.0)
        with self.assertRaises(ValueError):
            checkpoint_write_usd(-1)

    def test_expected_loss_scales_with_hours_and_probability(self):
        self.assertAlmostEqual(expected_eviction_loss_usd(3600, 2.0), 2.0)
        self.assertAlmostEqual(expected_eviction_loss_usd(3600, 2.0, 0.5), 1.0)
        with self.assertRaises(ValueError):
            expected_eviction_loss_usd(10, 2.0, 1.5)

    def test_breakeven_interval_is_where_loss_equals_write(self):
        write = checkpoint_write_usd(1_000_000_000, 0.02)   # 0.02
        interval = breakeven_checkpoint_interval_s(1_000_000_000, 0.02, 2.0)
        self.assertAlmostEqual(interval, 36.0)
        self.assertAlmostEqual(expected_eviction_loss_usd(interval, 2.0), write)

    def test_watchdog_decides_from_loss_versus_write(self):
        wd = CheckpointWatchdog(usd_per_gb=0.02, usd_per_hour=2.0)
        early = wd.should_checkpoint(30, 1_000_000_000)
        due = wd.should_checkpoint(36, 1_000_000_000)
        self.assertFalse(early["checkpoint"])
        self.assertLess(early["net_usd"], 0)
        self.assertTrue(due["checkpoint"])
        self.assertEqual(due["reason"], "expected_loss_exceeds_write")

    def test_eviction_signal_overrides_cadence(self):
        wd = CheckpointWatchdog(eviction_prob=0.0, min_interval_s=600)
        self.assertTrue(wd.should_checkpoint(1, 1000, eviction_signal=True)["checkpoint"])
        self.assertFalse(wd.should_checkpoint(1, 1000)["checkpoint"])

    def test_budget_breaker_trips_and_report(self):
        wd = CheckpointWatchdog(usd_per_gb=0.02, budget_usd=0.03)
        wd.note_checkpoint(1_000_000_000)
        self.assertFalse(wd.tripped)
        wd.note_checkpoint(1_000_000_000)
        self.assertTrue(wd.tripped)
        report = wd.report()
        self.assertEqual(report["checkpoints"], 2)
        self.assertAlmostEqual(report["spend_usd"], 0.04)
        self.assertAlmostEqual(report["remaining_usd"], 0.0)


class TestValueAndRefund(unittest.TestCase):
    def test_fair_value_at_the_pass_bar(self):
        value = assess_fair_value(price_paid_usd=100.0, score=0.70, pass_threshold=0.70)
        self.assertAlmostEqual(value.fair_value_usd, 100.0)
        self.assertAlmostEqual(value.value_ratio, 1.0)
        self.assertEqual(value.verdict, "fair")

    def test_half_quality_is_half_value(self):
        value = assess_fair_value(price_paid_usd=100.0, score=0.35, pass_threshold=0.70)
        self.assertAlmostEqual(value.fair_value_usd, 50.0)
        self.assertAlmostEqual(value.overpaid_usd, 50.0)
        self.assertEqual(value.verdict, "grossly_overpriced")

    def test_compare_offers_ranks_best_value_first(self):
        ranked = compare_offers([
            {"offer_id": "cheap-good", "price_usd": 40.0, "score": 0.70},
            {"offer_id": "dear-bad", "price_usd": 200.0, "score": 0.20},
        ])
        self.assertEqual(ranked[0]["offer_id"], "cheap-good")
        self.assertLess(ranked[0]["value_ratio"], ranked[1]["value_ratio"])

    def test_refund_strategy_ladder(self):
        growl = refund_strategy(GROWL, 100.0)
        self.assertAlmostEqual(growl["ask_usd"], 15.0)
        self.assertEqual(growl["escalation_rung"], 1)
        bite = refund_strategy(BITE, 100.0)
        self.assertAlmostEqual(bite["ask_usd"], 100.0)
        self.assertEqual(bite["escalation_rung"], 4)
        self.assertEqual(len(bite["escalation_path"]), 4)
        self.assertEqual(bite["escalation_path"][0]["actor"], "merchant_frontline")
        self.assertEqual(len(escalation_path(2)), 2)
        with self.assertRaises(ValueError):
            refund_strategy("NOPE", 10.0)


class TestVerification(unittest.TestCase):
    def test_numeric_conflict_detects_disagreeing_numbers(self):
        self.assertTrue(numeric_conflict("grace period is 2 minutes",
                                         "grace period is 30 seconds"))
        self.assertFalse(numeric_conflict("grace period is 2 minutes",
                                          "a 2 minute grace period"))
        self.assertFalse(numeric_conflict("grace period applies", "it is 2 minutes"))

    def test_materiality_selects_facts_worth_grading(self):
        self.assertTrue(is_material("The deadline is 14 days."))
        self.assertTrue(is_material("Please cite policy section 4."))
        self.assertTrue(is_material("The fee is $12."))
        self.assertFalse(is_material("Thank you for your patience."))

    def test_compare_claim_to_source_classifies(self):
        self.assertEqual(compare_claim_to_source("checkpoint every 3 hours",
                                                 "checkpoint every 3 hours"),
                         "support")
        self.assertEqual(compare_claim_to_source("grace period is 2 minutes",
                                                 "grace period is 30 seconds"),
                         "contradict")
        self.assertEqual(compare_claim_to_source("grace period is 2 minutes",
                                                 "the weather is nice today"),
                         "neutral")

    def test_corroboration_needs_two_independent_sources(self):
        claim = "The grace period is 2 minutes for AWS spot instances."
        one = [Source("a", claim, kind="authority")]
        two = one + [Source("b", claim, kind="document")]
        single = verify_claim(claim, one)
        both = verify_claim(claim, two)
        self.assertEqual(single.status, "single_source")
        self.assertLessEqual(single.grade, 0.5)
        self.assertEqual(both.status, "corroborated")
        self.assertAlmostEqual(both.grade, 0.8, places=6)  # mean trust 0.9/0.7

    def test_contradiction_zeroes_the_grade(self):
        result = verify_claim("The grace period is 2 minutes.",
                              [Source("doc", "The grace period is 30 seconds.",
                                      kind="authority")])
        self.assertEqual(result.status, "contradicted")
        self.assertEqual(result.grade, 0.0)

    def test_derived_source_is_not_an_independent_witness(self):
        claim = "The grace period is 2 minutes."
        sources = [Source("a", claim, kind="authority"),
                   Source("b", claim, kind="document", independent=False,
                          derived_from="a")]
        self.assertEqual(verify_claim(claim, sources).status, "single_source")

    def test_verify_response_grades_only_material_claims(self):
        response = "The deadline is 14 days. Thank you for your patience."
        good = [Source("d1", "The deadline is 14 days.", kind="authority"),
                Source("d2", "Response deadline is 14 days.", kind="document")]
        result = verify_response(response, good)
        self.assertEqual(result["material_count"], 1)
        self.assertAlmostEqual(result["material_grade"], 0.8, places=6)
        bad = [Source("d1", "The deadline is 30 days.", kind="authority")]
        self.assertEqual(verify_response(response, bad)["material_grade"], 0.0)
        self.assertTrue(verify_response(response, bad)["contradictions"])

    def test_cross_model_agreement_flags_the_outlier(self):
        result = cross_model_agreement({
            "m1": "cancel my subscription and refund the last payment",
            "m2": "cancel the subscription and refund the last payment",
            "m3": "zebra quantum xylophone",
        })
        self.assertIn("m3", result["divergent"])
        self.assertLess(result["consensus"]["m3"], result["consensus"]["m1"])
        self.assertAlmostEqual(result["consensus"]["m1"],
                               result["consensus"]["m2"], places=6)


class TestCheckpointModel(unittest.TestCase):
    def test_young_daly_formula(self):
        self.assertAlmostEqual(young_daly_interval_s(100.0, 10000.0),
                               (2 * 100.0 * 10000.0) ** 0.5)
        self.assertEqual(young_daly_interval_s(0.0, 10000.0), float("inf"))

    def test_mtbf_and_optimal_interval(self):
        model = CheckpointCostModel(n_gpus=4, overhead_s=129.6)
        self.assertAlmostEqual(model.mtbf_s(), 450000.0)   # 125 h
        self.assertAlmostEqual(model.optimal_interval_s(), 10800.0, places=3)  # 3 h
        self.assertAlmostEqual(model.waste_fraction(),
                               expected_waste_fraction(129.6, 450000.0))

    def test_implied_overhead_inverts_the_formula(self):
        cost = implied_overhead_s(10800.0, 4)
        self.assertAlmostEqual(cost, 129.6, places=4)
        self.assertAlmostEqual(
            young_daly_interval_s(cost, FailureModel(4).mtbf_seconds()),
            10800.0, places=4)

    def test_claimed_frequency_table_fails_young_daly_scaling(self):
        findings = check_interval_claims()
        self.assertEqual(findings["verdict"],
                         "INCONSISTENT-WITH-YOUNG/DALY-SQRT-SCALING")
        self.assertFalse(findings["single_cost_consistent"])
        self.assertGreater(findings["implied_overhead_spread"], 10.0)
        rows = {r["n_gpus"]: r for r in findings["rows"]}
        # only the 4-GPU row reproduces from its own implied checkpoint cost
        self.assertAlmostEqual(rows[4]["ratio_claimed_over_predicted"], 1.0, places=6)
        self.assertLess(rows[16]["ratio_claimed_over_predicted"], 0.5)
        self.assertLess(rows[64]["ratio_claimed_over_predicted"], 0.5)

    def test_monthly_overhead_arithmetic(self):
        # 5% of $24/hr over 720 h is ~$864, not the ~$24k a pasted claim asserted
        self.assertAlmostEqual(monthly_waste_usd(24.0, 0.05), 864.0)

    def test_spot_plan_respects_the_grace_window(self):
        model = CheckpointCostModel(n_gpus=64, overhead_s=5.0)
        plan = checkpoint_plan(model, environment="spot", provider="gcp",
                               usd_per_hour=30.0)
        self.assertEqual(plan["grace_period_s"], GRACE_PERIOD_S["gcp"])
        self.assertLessEqual(plan["interval_s"], 30.0)
        self.assertTrue(plan["fits_grace"])

    def test_plan_flags_a_checkpoint_too_slow_for_the_grace_window(self):
        plan = checkpoint_plan(CheckpointCostModel(n_gpus=8, overhead_s=200.0),
                               environment="spot", provider="aws")
        self.assertFalse(plan["fits_grace"])
        self.assertTrue(any("grace window" in action for action in plan["actions"]))

    def test_over_tuned_flag_follows_the_five_percent_rule(self):
        plan = checkpoint_plan(CheckpointCostModel(n_gpus=8, overhead_s=2000.0),
                               environment="local_nvme", usd_per_hour=24.0)
        self.assertTrue(plan["over_tuned"])
        self.assertGreater(plan["overhead_pct"], 0.05)
        self.assertGreater(plan["optimal_hourly_waste_usd"], 0.0)


class TestDoberwatchCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dw = Doberwatch()

    def test_seed_corpus_is_split_and_all_pass(self):
        self.assertEqual(len(GRADED_RESPONSES), len(ANONYMOUS_SEED) + len(PRIVATE_SEED))
        self.assertEqual(len(PRIVATE_SEED), 3)
        self.assertTrue(all(e["shareable"] for e in ANONYMOUS_SEED))
        self.assertFalse(any(e["shareable"] for e in PRIVATE_SEED))
        for record in ANONYMOUS_SEED:
            grade = self.dw._grade_seed_record(record)
            self.assertEqual(grade.verdict, PASS,
                             f"{record['entry_id']} did not pass its rubric")

    def test_advise_serves_an_exact_passing_match_without_a_model(self):
        request = ANONYMOUS_SEED[0]["request"]
        advice = self.dw.advise(request, ANONYMOUS_SEED[0]["domain"])
        self.assertEqual(advice["decision"], "exact_match")
        self.assertFalse(advice["model_call_needed"])
        self.assertIsNotNone(advice["recommended_response"])
        self.assertLessEqual(len(advice["top_candidates"]), 10)
        self.assertTrue(advice["ask_if_more_information_needed"])

    def test_advise_offers_ten_closest_before_model_calls(self):
        advice = self.dw.advise("duplicate charge refund", "billing_dispute")
        self.assertLessEqual(len(advice["top_candidates"]), 10)
        scores = [c["score"] for c in advice["top_candidates"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_qualifying_questions_always_asked(self):
        for decision_request, domain in (("duplicate charge refund", "billing_dispute"),
                                          ("zebra quantum", "billing_dispute")):
            advice = self.dw.advise(decision_request, domain)
            self.assertTrue(advice["qualifying_questions"])
            self.assertTrue(advice["next_step"])

    def test_unknown_request_needs_a_model_call(self):
        advice = self.dw.advise("zebra quantum", "billing_dispute")
        self.assertEqual(advice["decision"], "no_match")
        self.assertTrue(advice["model_call_needed"])

    def test_submit_good_response_records_and_passes(self):
        request = ANONYMOUS_SEED[0]["request"]
        good = ANONYMOUS_SEED[0]["response"]
        out = self.dw.submit(
            request, good, "billing_dispute",
            criterion_scores={"acknowledges_charge": 1.0, "cites_policy": 1.0,
                              "states_remedy": 1.0, "gives_timeline": 1.0,
                              "offers_escalation_path": 1.0,
                              "no_unsupported_denial": 1.0},
            evidence={"cites_policy": ["policy_citation: billing-policy"]},
            corroboration={"cites_policy": 0.9},
            reference_response=good)
        self.assertEqual(out["grade"]["verdict"], PASS)
        self.assertIsNone(out["complaint"])
        self.assertEqual(out["cached_source"], "local")
        self.assertTrue(out["audit_valid"])

    def test_submit_unverified_response_cannot_pass(self):
        """Fact-check every response: no verification, no PASS."""
        out = self.dw.submit(
            ANONYMOUS_SEED[0]["request"], ANONYMOUS_SEED[0]["response"],
            "billing_dispute",
            criterion_scores={"acknowledges_charge": 1.0, "cites_policy": 1.0,
                              "states_remedy": 1.0, "gives_timeline": 1.0,
                              "offers_escalation_path": 1.0,
                              "no_unsupported_denial": 1.0},
            evidence={"cites_policy": ["policy_citation: billing-policy"]},
            reference_response="")
        self.assertTrue(out["grade"]["corroboration_gate_applied"])
        self.assertEqual(out["grade"]["missing_corroboration"], ["cites_policy"])
        self.assertEqual(out["grade"]["verdict"], GROWL)
        self.assertIsNotNone(out["complaint"])

    def test_submit_with_sources_corroborates_before_grading(self):
        response = "The duplicate charge was reversed within 5 business days."
        sources = [
            Source("policy-1", response, kind="authority"),
            Source("policy-2", "Duplicates are reversed within 5 business days.",
                   kind="document"),
        ]
        out = self.dw.submit(
            "duplicate charge refund", response, "billing_dispute",
            criterion_scores={"acknowledges_charge": 1.0, "cites_policy": 1.0,
                              "states_remedy": 1.0, "gives_timeline": 1.0,
                              "offers_escalation_path": 1.0,
                              "no_unsupported_denial": 1.0},
            evidence={"cites_policy": ["policy_citation: billing-policy"]},
            sources=sources, reference_response=response)
        self.assertIsNotNone(out["verification"])
        self.assertEqual(out["verification"]["status_counts"], {"corroborated": 1})
        self.assertEqual(out["grade"]["verdict"], PASS)
        self.assertFalse(out["grade"]["corroboration_gate_applied"])

    def test_a_contradicted_material_claim_cannot_pass(self):
        """Zeroing one criterion is not enough when the answer is simply false."""
        response = "The provider gives a 2 minute eviction notice."
        sources = [Source("provider-docs",
                          "The provider gives a 30 second eviction notice.",
                          kind="authority")]
        out = self.dw.submit(
            "spot instance notice", response, "billing_dispute",
            criterion_scores={"acknowledges_charge": 1.0, "cites_policy": 1.0,
                              "states_remedy": 1.0, "gives_timeline": 1.0,
                              "offers_escalation_path": 1.0,
                              "no_unsupported_denial": 1.0},
            evidence={"cites_policy": ["policy_citation: billing-policy"]},
            sources=sources, reference_response=response)
        self.assertTrue(out["verification"]["contradictions"])
        self.assertTrue(out["grade"]["contradiction_veto"])
        self.assertEqual(out["grade"]["verdict"], GROWL)
        self.assertIsNotNone(out["complaint"])

    def test_grade_matters_and_cross_model(self):
        response = "The grace period is 2 minutes for AWS spot instances."
        sources = [Source("aws-doc", response, kind="authority"),
                   Source("aws-blog",
                          "Grace period is 2 minutes for AWS spot instances.",
                          kind="document")]
        graded = self.dw.grade_matters(response, sources, "billing_dispute")
        self.assertTrue(graded["material_claims"])
        self.assertTrue(graded["passes"])
        self.assertAlmostEqual(graded["material_grade"], 0.8, places=4)
        agreement = self.dw.cross_model_check({
            "m1": "cancel my subscription and refund the last payment",
            "m2": "please cancel my subscription and refund the last charge",
            "m3": "quantum zebra",
        })
        self.assertIn("m3", agreement["divergent"])
        self.assertLess(agreement["consensus"]["m3"], agreement["consensus"]["m1"])

    def test_plan_checkpoints_and_fact_check_are_wired(self):
        plan = self.dw.plan_checkpoints(8, overhead_s=20.0, environment="spot",
                                        provider="aws", usd_per_hour=24.0)
        self.assertTrue(plan["fits_grace"])
        self.assertLessEqual(plan["interval_s"], GRACE_PERIOD_S["aws"])
        findings = self.dw.fact_check_checkpoint_claims()
        self.assertEqual(findings["verdict"],
                         "INCONSISTENT-WITH-YOUNG/DALY-SQRT-SCALING")
        self.assertTrue(self.dw.audit()["valid"])

    def test_submit_bad_response_files_complaint_and_refund_plan(self):
        out = self.dw.submit(
            "duplicate charge refund", "not our problem", "billing_dispute",
            criterion_scores={name: 0.05 for name in
                              ("acknowledges_charge", "cites_policy", "states_remedy",
                               "gives_timeline", "offers_escalation_path",
                               "no_unsupported_denial")},
            price_paid_usd=80.0, reference_response="")
        self.assertEqual(out["grade"]["verdict"], BITE)
        self.assertIsNotNone(out["complaint"])
        self.assertEqual(out["complaint"]["severity"], BITE)
        self.assertAlmostEqual(out["refund_plan"]["ask_usd"], 80.0)
        self.assertGreaterEqual(len(out["escalation_path"]), 1)

    def test_checkpoint_and_audit_wiring(self):
        decision = self.dw.checkpoint_decision(3600, 1_000_000_000)
        self.assertTrue(decision["checkpoint"])
        recorded = self.dw.note_checkpoint(1_000_000_000)
        self.assertEqual(recorded["checkpoints"], 1)
        audit = self.dw.audit()
        self.assertTrue(audit["valid"])
        self.assertGreater(audit["events"], 0)
        self.assertIn("by_source", audit["cache"])

    def test_export_is_portable_and_holds_seed(self):
        payload = self.dw.export()
        self.assertIn("seed_anonymous", payload)
        self.assertIn("cache.load_reference", payload)


class TestSeedCorroboration(unittest.TestCase):
    """The corpus must be *verified*, not approved by review.

    Every entry ships the witnesses behind its factual claims, and the loader
    fact-checks the response against them exactly as it would a live answer.
    """

    def setUp(self):
        self.dw = Doberwatch(load_private=True)

    def test_every_seed_record_ships_witnesses(self):
        for record in GRADED_RESPONSES:
            sources = sources_for(record)
            where = record["entry_id"]
            self.assertGreaterEqual(len(sources), 2, where)
            ids = [s.source_id for s in sources]
            self.assertEqual(len(ids), len(set(ids)), f"{where}: duplicate ids")
            # Two independent witnesses per claim is what a full grade needs,
            # so a record must ship at least one non-document witness too.
            self.assertIn("authority", {s.kind for s in sources}, where)
            for source in sources:
                self.assertTrue(source.source_id, where)
                self.assertTrue(source.text, where)
                self.assertTrue(source.independent, where)
                self.assertIn(source.kind, ("authority", "citation", "document"))

    def test_every_material_claim_is_corroborated(self):
        for record in GRADED_RESPONSES:
            verification = verify_response(record["response"],
                                           sources_for(record))
            where = record["entry_id"]
            self.assertFalse(verification["contradictions"], where)
            self.assertGreaterEqual(verification["material_grade"], 0.7, where)
            self.assertTrue(verification["all_corroborated"], where)
            for claim in verification["claims"]:
                if claim["material"]:
                    self.assertEqual(claim["status"], "corroborated",
                                     f"{where}: {claim['claim'][:60]}")
                    self.assertGreaterEqual(claim["independent_support"], 2)

    def test_loader_records_a_verification_for_every_entry(self):
        self.assertEqual(len(self.dw.seed_verification), len(GRADED_RESPONSES))
        for entry_id, verification in self.dw.seed_verification.items():
            self.assertIsNotNone(verification, entry_id)
            self.assertGreaterEqual(verification["material_grade"], 0.7)

    def test_corroboration_is_earned_not_granted(self):
        """Strip the witnesses and the entry can no longer reach PASS."""
        record = dict(ANONYMOUS_SEED[0])
        with_sources = self.dw._grade_seed_record(record)
        without = self.dw._grade_seed_record({**record, "sources": []})
        self.assertEqual(with_sources.verdict, PASS)
        self.assertFalse(with_sources.corroboration_gate_applied)
        self.assertTrue(without.corroboration_gate_applied)
        self.assertEqual(without.verdict, GROWL)
        self.assertTrue(without.complaint)

    def test_a_disagreeing_witness_breaks_the_entry(self):
        """Change one number in one witness and the answer stops passing."""
        record = dict(ANONYMOUS_SEED[0])
        sources = [dict(s) for s in record["sources"]]
        sources[0] = {**sources[0],
                      "text": sources[0]["text"].replace("5-7", "30")}
        poisoned = {**record, "sources": sources}
        verification = self.dw._verify_seed_record(poisoned)
        self.assertTrue(verification["any_refuted"])
        grade = self.dw._grade_seed_record(poisoned, verification)
        self.assertTrue(grade.contradiction_veto)
        self.assertNotEqual(grade.verdict, PASS)

    def test_a_contested_claim_also_blocks_pass(self):
        """One witness denying a fact is enough, even if another agrees."""
        record = dict(ANONYMOUS_SEED[0])
        sources = [dict(s) for s in record["sources"]]
        # Only the policy witness is tampered with: the case log still agrees,
        # so the claim is contested rather than flatly contradicted.
        sources[0] = {**sources[0],
                      "text": sources[0]["text"].replace("5-7", "30")}
        verification = self.dw._verify_seed_record({**record, "sources": sources})
        self.assertIn("contested", verification["status_counts"])
        self.assertIn("contested",
                      [c["status"] for c in verification["refuted"]])
        grade = self.dw._grade_seed_record(record, verification)
        self.assertTrue(grade.contradiction_veto)
        self.assertNotEqual(grade.verdict, PASS)

    def test_seed_corroboration_is_computed_not_assumed(self):
        """No entry's corroboration is the old blanket 1.0."""
        for record in GRADED_RESPONSES:
            rubric = self.dw.rubric_for(record["domain"], record.get("rubric_id"))
            verification = self.dw._verify_seed_record(record)
            corroboration = corroboration_for(rubric, verification)
            self.assertTrue(corroboration, record["entry_id"])
            for name, value in corroboration.items():
                self.assertEqual(value, verification["material_grade"], name)
                self.assertLess(value, 1.0, record["entry_id"])

    def test_witnesses_are_independently_worded_not_cribbed(self):
        """Every material claim must be reworded rather than copy-pasted."""
        for record in GRADED_RESPONSES:
            response = record["response"].lower()
            for source in sources_for(record):
                self.assertNotIn(source.text.lower(), response,
                                 f"{record['entry_id']}: source {source.source_id!r} is a verbatim substring")
            verification = verify_response(record["response"], sources_for(record))
            for claim in verification["claims"]:
                if claim["material"] and claim["support"]:
                    # Need real rewording on at least one witness: not a near-crib.
                    non_cribs = [s for s in claim["support"]
                                 if claim["claim"][:40].lower() not in
                                 next(t for t in sources_for(record)
                                      if t.source_id == s).text.lower()]
                    self.assertTrue(non_cribs,
                                    f"{record['entry_id']}: every witness cribs the claim's first 40 chars")

    def test_advise_reports_provenance_for_the_answer_it_serves(self):
        advice = self.dw.advise(ANONYMOUS_SEED[0]["request"], "billing_dispute")
        self.assertEqual(advice["decision"], "exact_match")
        self.assertFalse(advice["model_call_needed"])
        provenance = advice["recommended_verification"]
        self.assertIsNotNone(provenance)
        self.assertEqual(provenance["claim_count"], len(provenance["claims"]))
        self.assertFalse(provenance["contradictions"])


class TestVerificationPrecision(unittest.TestCase):
    """Fact-checking is only as good as its text handling.

    Each of these was a live defect: numbers that compared equal when they
    differed, negations that were invisible, and claims split mid-number.
    """

    def test_trailing_zeros_are_not_stripped_from_integers(self):
        self.assertEqual(normalize_number("30"), "30")
        self.assertEqual(normalize_number("100"), "100")
        self.assertEqual(normalize_number("5.0"), "5")
        self.assertEqual(normalize_number("5.50"), "5.5")
        self.assertEqual(normalize_number("1,000"), "1000")

    def test_30_days_contradicts_3_days(self):
        self.assertEqual(extract_numbers("within 30 days"), frozenset({"30"}))
        self.assertTrue(numeric_conflict("posts within 30 days",
                                         "posts within 3 days"))
        self.assertEqual(compare_claim_to_source("The refund posts within 30 days",
                                                 "The refund posts within 3 days"),
                         "contradict")
        self.assertEqual(compare_claim_to_source("The refund posts within 30 days",
                                                 "The refund posts within 30 days"),
                         "support")

    def test_contracted_negations_are_detected(self):
        for text in ("the fee isn't refundable", "the fee isnt refundable",
                     "we don't refund that", "it hasn't posted",
                     "the item wasn't corrected", "the charge cannot stand"):
            self.assertTrue(has_negation(text), text)
        self.assertFalse(has_negation("the fee is refundable"))

    def test_polarity_mismatch_on_a_plain_word_still_works(self):
        self.assertTrue(has_negation("the late mark is unsupported"))
        self.assertTrue(has_negation("the fee does not apply"))

    def test_a_decimal_clause_stays_one_claim(self):
        claims = split_claims("The fee is set out in section 2.1 of the agreement. "
                              "It recurs, so escalate.")
        self.assertEqual(len(claims), 2)
        self.assertIn("2.1", claims[0])
        self.assertEqual(extract_numbers(claims[0]), frozenset({"2.1"}))


if __name__ == "__main__":
    unittest.main()
