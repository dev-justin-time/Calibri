# SPDX-License-Identifier: MIT
"""Tests for the multi-agent layer.

The agent layer shipped without tests, which is exactly how a wildcard
subscriber took the whole pipeline down: `Auditor` subscribes to "*" and the
bus indexed into the subscriber list instead of iterating it. These pin the
bus contract, the watchdog tool surface, the complaint clerk's case flow, and
the claim that the bus traffic is actually audited.
"""

import shutil
import tempfile
import unittest

from calibrix.agents import (Bus, ComplaintClerk, Orchestrator, ToolSurface,
                            run_pipeline)
from calibrix.agents.roles import Auditor

REQUEST = "duplicate charge refund"
RESPONSE = "The duplicate charge is reversed within 5 business days."
CRITERIA = ("acknowledges_charge", "cites_policy", "states_remedy",
            "gives_timeline", "offers_escalation_path",
            "no_unsupported_denial")
EVIDENCE = {"cites_policy": ["policy_citation: billing-policy"]}
DICT_SOURCES = [
    {"source_id": "policy-doc", "text": RESPONSE, "kind": "authority"},
    {"source_id": "policy-page",
     "text": "Duplicates are reversed within 5 business days.",
     "kind": "document"},
]


class AgentCase(unittest.TestCase):
    """A scratch workspace per test — the surface writes real files."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="calibrix_agents_")
        self.tools = ToolSurface(work_dir=self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _good_payload(self, **overrides):
        payload = {
            "request": REQUEST, "domain": "billing_dispute",
            "response": RESPONSE,
            "criterion_scores": {name: 1.0 for name in CRITERIA},
            "evidence": EVIDENCE, "sources": DICT_SOURCES,
            "reference_response": RESPONSE,
        }
        payload.update(overrides)
        return payload


class TestBus(AgentCase):
    def test_wildcard_observer_sees_every_message(self):
        """Regression: '*' subscribers used to crash dispatch outright."""
        bus = Bus()
        seen = []
        bus.subscribe("a.b", lambda msg, b: seen.append("topic:" + msg.topic))
        bus.subscribe("*", lambda msg, b: seen.append("watch:" + msg.topic))
        bus.publish("a.b", {"x": 1})
        bus.drain()
        self.assertEqual(seen, ["topic:a.b", "watch:a.b"])

    def test_dispatch_order_does_not_depend_on_subscription_order(self):
        """Topic handlers run before wildcard observers, always."""
        first, second = [], []
        a, b = Bus(), Bus()
        a.subscribe("a.b", lambda m, _b: first.append("topic"))
        a.subscribe("*", lambda m, _b: first.append("watch"))
        b.subscribe("*", lambda m, _b: second.append("watch"))
        b.subscribe("a.b", lambda m, _b: second.append("topic"))
        a.publish("a.b", {})
        b.publish("a.b", {})
        a.drain()
        b.drain()
        self.assertEqual(first, second)
        self.assertEqual(first, ["topic", "watch"])

    def test_publish_returns_the_message_and_logs_it(self):
        bus = Bus()
        bus.subscribe("t", lambda msg, b: b.publish("u", {"from": "t"}))
        first = bus.publish("t", {"n": 1})
        rounds = bus.drain()
        self.assertEqual(first.payload, {"n": 1})
        self.assertEqual([m.topic for m in bus.log], ["t", "u"])
        self.assertEqual(rounds, 2)
        self.assertEqual(bus.pending, 0)

    def test_round_cap_truncates_a_runaway_loop(self):
        """A runaway agent loop must show up as a truncated run, not a hang."""
        bus = Bus()
        bus.subscribe("loop", lambda msg, b: b.publish("loop", {}))
        bus.publish("loop", {})
        rounds = bus.drain(max_rounds=4)
        self.assertEqual(rounds, 4)
        self.assertEqual(bus.pending, 1)  # capped mid-wave: visible, not hidden

    def test_payload_is_copied_so_callers_cannot_mutate_it(self):
        bus = Bus()
        payload = {"n": 1}
        msg = bus.publish("t", payload)
        payload["n"] = 99
        self.assertEqual(msg.payload, {"n": 1})


class TestAuditTraffic(AgentCase):
    def test_auditor_seals_every_routed_message_in_order(self):
        bus = Bus()
        auditor = Auditor(bus, self.tools)
        bus.publish("a", {"n": 1})
        bus.publish("b", {"n": 2})
        bus.drain()
        self.assertEqual(auditor.handled, 2)
        self.assertEqual(len(self.tools.chain), 2)
        self.assertTrue(self.tools.audit_verify()["valid"])
        report = self.tools.audit_bus(bus.log)
        self.assertTrue(report["order_matches"])
        self.assertEqual(report["routed"], report["sealed"])
        self.assertEqual(report["unsealed"], [])

    def test_audit_bus_reports_traffic_that_was_never_sealed(self):
        """The check must be able to fail, or it is decoration."""
        bus = Bus()
        bus.publish("a", {})
        report = self.tools.audit_bus(bus.log)  # no Auditor registered
        self.assertFalse(report["order_matches"])
        self.assertEqual(report["unsealed"], ["a"])
        self.assertEqual(report["first_divergence"], 0)

    def test_a_tampered_chain_is_detected(self):
        self.tools.audit_append("auditor", "a", {"n": 1})
        self.tools.audit_append("auditor", "b", {"n": 2})
        self.tools.chain[0]["payload"] = {"n": 999}
        self.assertFalse(self.tools.audit_verify()["valid"])


class TestWatchdogSurface(AgentCase):
    def test_advise_offers_candidates_and_always_asks_questions(self):
        advice = self.tools.doberwatch_advise(REQUEST, "billing_dispute")
        self.assertLessEqual(len(advice["top_candidates"]), 10)
        self.assertTrue(advice["qualifying_questions"])
        self.assertIn(advice["decision"],
                      ("exact_match", "exact_match_unverified",
                       "similar_available", "no_match"))

    def test_submit_accepts_dict_sources_and_fact_checks(self):
        outcome = self.tools.doberwatch_submit(
            REQUEST, RESPONSE, "billing_dispute",
            {name: 1.0 for name in CRITERIA},
            evidence=EVIDENCE, sources=DICT_SOURCES,
            reference_response=RESPONSE)
        self.assertIsNotNone(outcome["verification"])
        self.assertEqual(outcome["verification"]["status_counts"],
                         {"corroborated": 1})
        self.assertEqual(outcome["grade"]["verdict"], "PASS")

    def test_dict_sources_can_contradict_and_veto_a_pass(self):
        outcome = self.tools.doberwatch_submit(
            REQUEST, "The fix takes 2 minutes.", "billing_dispute",
            {name: 1.0 for name in CRITERIA}, evidence=EVIDENCE,
            sources=[{"source_id": "docs", "text": "The fix takes 30 seconds.",
                      "kind": "authority"}],
            reference_response="")
        self.assertTrue(outcome["grade"]["contradiction_veto"])
        self.assertNotEqual(outcome["grade"]["verdict"], "PASS")

    def test_every_surface_call_is_sealed(self):
        self.tools.doberwatch_advise(REQUEST, "billing_dispute")
        self.tools.doberwatch_submit(
            REQUEST, RESPONSE, "billing_dispute",
            {name: 1.0 for name in CRITERIA}, evidence=EVIDENCE,
            sources=DICT_SOURCES, reference_response=RESPONSE)
        topics = [e["topic"] for e in self.tools.chain]
        self.assertEqual(topics, ["doberwatch.advise", "doberwatch.submit"])
        self.assertTrue(self.tools.audit_verify()["valid"])

    def test_doberwatch_audit_reports_both_chains(self):
        self.tools.doberwatch_advise(REQUEST, "billing_dispute")
        audit = self.tools.doberwatch_audit()
        self.assertTrue(audit["valid"])
        self.assertTrue(audit["surface_chain"]["valid"])
        self.assertGreater(audit["events"], 0)
        self.assertIn("by_source", audit["cache"])


class TestComplaintClerk(AgentCase):
    def _run_case(self, payload):
        bus = Bus()
        clerk = ComplaintClerk(bus, self.tools)
        Auditor(bus, self.tools)
        bus.publish("complaint.request", payload, sender="test")
        bus.drain()
        return bus, clerk

    def test_advises_before_grading_and_asks_first(self):
        bus, clerk = self._run_case(self._good_payload())
        topics = [m.topic for m in bus.log]
        self.assertEqual(topics, ["complaint.request", "complaint.advice",
                                  "complaint.resolved"])
        advice = next(m for m in bus.log if m.topic == "complaint.advice")
        self.assertTrue(advice.payload["qualifying_questions"])
        self.assertTrue(advice.payload["ask_if_more_information_needed"])
        self.assertEqual(clerk.handled, 1)

    def test_a_good_response_resolves_without_a_complaint(self):
        bus, _ = self._run_case(self._good_payload())
        resolved = next(m for m in bus.log if m.topic == "complaint.resolved")
        self.assertEqual(resolved.payload["verdict"], "PASS")
        self.assertIsNone(resolved.payload["complaint"])
        self.assertEqual(resolved.payload["escalation_path"], [])
        self.assertNotIn("complaint.escalated", [m.topic for m in bus.log])

    def test_a_bad_response_escalates_with_a_refund_ask(self):
        bus, _ = self._run_case(self._good_payload(
            response="not our problem",
            criterion_scores={name: 0.05 for name in CRITERIA},
            sources=None, reference_response="", price_paid_usd=80.0))
        topics = [m.topic for m in bus.log]
        self.assertIn("complaint.escalated", topics)
        resolved = next(m for m in bus.log if m.topic == "complaint.resolved")
        escalated = next(m for m in bus.log if m.topic == "complaint.escalated")
        self.assertNotEqual(resolved.payload["verdict"], "PASS")
        self.assertIsNotNone(resolved.payload["complaint"])
        self.assertEqual(escalated.payload["case_id"],
                         resolved.payload["case_id"])
        self.assertGreater(escalated.payload["ask_usd"], 0)
        self.assertTrue(escalated.payload["escalation_path"])

    def test_no_response_yet_asks_for_information_instead_of_grading(self):
        payload = self._good_payload()
        del payload["response"]
        bus, _ = self._run_case(payload)
        topics = [m.topic for m in bus.log]
        self.assertEqual(topics, ["complaint.request", "complaint.advice",
                                  "complaint.pending"])
        pending = next(m for m in bus.log if m.topic == "complaint.pending")
        self.assertTrue(pending.payload["qualifying_questions"])
        self.assertIn("no response", pending.payload["reason"])

    def test_a_payload_without_a_request_is_rejected(self):
        bus, _ = self._run_case(self._good_payload(request=""))
        rejected = next(m for m in bus.log if m.topic == "complaint.rejected")
        self.assertEqual(rejected.payload["stage"], "advise")
        self.assertNotIn("complaint.resolved", [m.topic for m in bus.log])

    def test_an_unknown_domain_is_rejected_not_raised(self):
        """A handler that raises would take the whole mission down."""
        bus, _ = self._run_case(self._good_payload(domain="not_a_domain"))
        topics = [m.topic for m in bus.log]
        self.assertNotIn("complaint.resolved", topics)
        rejected = next(m for m in bus.log if m.topic == "complaint.rejected")
        self.assertEqual(rejected.payload["stage"], "grade")
        self.assertIn("no rubric", rejected.payload["reason"])
        self.assertTrue(self.tools.audit_verify()["valid"])
        self.assertTrue(self.tools.audit_bus(bus.log)["order_matches"])

    def test_refuted_fact_survives_the_bus_as_a_veto(self):
        bus, _ = self._run_case(self._good_payload(
            response="The fix takes 2 minutes.",
            sources=[{"source_id": "docs", "text": "The fix takes 30 seconds.",
                      "kind": "authority"}],
            reference_response=""))
        resolved = next(m for m in bus.log if m.topic == "complaint.resolved")
        self.assertEqual(
            resolved.payload["verification"]["contradictions"][0]["status"],
            "contradicted")
        self.assertNotEqual(resolved.payload["verdict"], "PASS")


class TestOrchestrator(AgentCase):
    def test_pipeline_sells_and_seals_every_message(self):
        orch = Orchestrator(work_dir=self.tmp)
        out = orch.run(n_trials=2, popsize=2)
        self.assertIn(out["status"], ("SOLD", "READY-NOT-SOLD"))
        self.assertTrue(out["audit"]["valid"])
        self.assertEqual(out["audit_entries"], out["messages"])
        self.assertTrue(out["bus_audit"]["order_matches"])
        self.assertTrue(out["bus_audit"]["chain_valid"])
        self.assertEqual(out["bus_audit"]["unsealed"], [])
        self.assertIsNone(out["complaint_case"])

    def test_pipeline_runs_a_watchdog_case_on_the_same_bus(self):
        orch = Orchestrator(work_dir=self.tmp)
        out = orch.run(n_trials=2, popsize=2, complaint=self._case_payload())
        case = out["complaint_case"]
        self.assertIsNotNone(case)
        self.assertEqual(case["status"], "RESOLVED")
        self.assertEqual(case["escalated"], 1)
        self.assertNotEqual(case["last"]["verdict"], "PASS")
        self.assertGreater(case["last"]["refund_plan"]["ask_usd"], 0)
        # One chain covers the mission and the case, and nothing is missed.
        # Sealed == routed; the chain is longer only because direct surface
        # calls (advise/submit) seal themselves too.
        self.assertTrue(out["bus_audit"]["order_matches"])
        self.assertEqual(out["bus_audit"]["sealed"], out["messages"])
        self.assertEqual(out["audit_entries"], out["messages"] + 2)
        topics = [m.topic for m in orch.bus.log]
        self.assertIn("complaint.advice", topics)
        self.assertIn("complaint.escalated", topics)

    def test_a_malformed_case_does_not_take_the_mission_down(self):
        orch = Orchestrator(work_dir=self.tmp)
        out = orch.run(n_trials=2, popsize=2,
                       complaint={"request": "", "domain": "billing_dispute"})
        self.assertIn(out["status"], ("SOLD", "READY-NOT-SOLD"))
        case = out["complaint_case"]
        self.assertEqual(case["status"], "REJECTED")
        self.assertEqual(case["rejected"], 1)
        self.assertEqual(case["rejections"][0]["stage"], "advise")
        self.assertIsNone(case["last"])
        self.assertTrue(out["bus_audit"]["order_matches"])

    def test_run_pipeline_helper_accepts_a_complaint(self):
        out = run_pipeline(work_dir=self.tmp, n_trials=2, popsize=2,
                           complaint=self._case_payload())
        self.assertEqual(out["complaint_case"]["status"], "RESOLVED")

    def _case_payload(self):
        return {
            "request": REQUEST, "domain": "billing_dispute",
            "response": "not our problem",
            "criterion_scores": {name: 0.05 for name in CRITERIA},
            "reference_response": "", "price_paid_usd": 80.0,
        }


if __name__ == "__main__":
    unittest.main()
