# SPDX-License-Identifier: MIT
# Doberwatch Finance — the consumer watchdog core.
#
# This is the composition layer. It wires the pieces into one flow:
#
#   qualify()  -> always ask what the request is missing
#   advise()   -> offer the ten closest cached/reference answers BEFORE any
#                 model call, then ask whether more information is needed for
#                 an exact or similar request (and whether the cached answer
#                 can be served directly)
#   submit()   -> grade the response the user actually got, cache it, and, if
#                 it failed the rubric, file a Growl/Bark/Bite complaint with
#                 a refund ask, an escalation route, and a fair-value number
#
# Everything is deterministic and offline; one audit chain (owned by the
# cache) seals every step.

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from . import seed
from .cache import ConsentPolicy, ResponseCache, RetrievalHit
from .checkpoint import (DEFAULT_FAILURE_RATE_PER_GPU_HOUR, CheckpointCostModel,
                         check_interval_claims, checkpoint_plan)
from .grading import Grade, Rubric, grade_response
from .value import ValueAssessment, assess_fair_value, refund_strategy
from .verification import (Source, corroboration_for, cross_model_agreement,
                           verify_response)
from .watchdog import CheckpointWatchdog, Complaint, GrowlBarkBite

DOBERWATCH_SCHEMA = "calibrix.doberwatch/1.0"

EXACT_SIMILARITY = 0.92
SIMILAR_FLOOR = 0.30


class Doberwatch:
    """Consumer watchdog: retrieve, qualify, grade, complain — all audited."""

    def __init__(self, policy: Optional[ConsentPolicy] = None,
                 cache: Optional[ResponseCache] = None,
                 cache_path: Optional[str] = None,
                 load_anonymous: bool = True,
                 load_private: bool = True,
                 checkpoint_watchdog: Optional[CheckpointWatchdog] = None) -> None:
        self.cache = cache if cache is not None else ResponseCache(policy=policy,
                                                                   path=cache_path)
        self.workflow = GrowlBarkBite(audit=self.cache.audit)
        self.checkpoints = checkpoint_watchdog or CheckpointWatchdog()
        self.rubrics: Dict[str, Rubric] = dict(seed.RUBRICS)
        # entry_id -> the fact-check the loader ran for that seed answer, so the
        # provenance travels with the corpus instead of being implied by it.
        self.seed_verification: Dict[str, Optional[Dict[str, Any]]] = {}
        if load_anonymous:
            self._load_seed(seed.ANONYMOUS_SEED, "seed_anonymous")
        if load_private:
            self._load_seed(seed.PRIVATE_SEED, "seed_private")

    # ------------------------------------------------------------ rubric
    def rubric_for(self, domain: str, rubric_id: Optional[str] = None) -> Rubric:
        if rubric_id is not None:
            try:
                return self.rubrics[rubric_id]
            except KeyError as exc:
                raise KeyError(f"unknown rubric {rubric_id!r}") from exc
        rubric_id = seed.DEFAULT_RUBRIC_BY_DOMAIN.get(domain)
        if rubric_id is None:
            raise KeyError(f"no rubric for domain {domain!r}")
        return self.rubrics[rubric_id]

    # ------------------------------------------------------------ seeding
    def _verify_seed_record(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fact-check one corpus answer against the witnesses shipped with it.

        ``None`` when the record ships none, which is the honest answer: the
        rubric's corroboration gate then applies and the entry cannot PASS.
        """
        sources = seed.sources_for(record)
        if not sources:
            return None
        return verify_response(record["response"], sources)

    def _grade_seed_record(self, record: Dict[str, Any],
                           verification: Optional[Dict[str, Any]] = None) -> Grade:
        """Grade one corpus answer, corroboration *measured* from its sources.

        A seed entry's evidence references still stand for the review that
        produced it, but corroboration is not granted by that review: the
        response is fact-checked against the witnesses shipped with it, and
        the measured material-claim grade caps every criterion that demands
        corroboration. Pass ``verification`` to reuse a check already run.
        """
        rubric = self.rubric_for(record["domain"], record.get("rubric_id"))
        refs = list(record.get("evidence") or [])
        evidence = {c.name: refs for c in rubric.criteria if c.requires_evidence}
        if verification is None:
            verification = self._verify_seed_record(record)
        corroboration = (corroboration_for(rubric, verification)
                         if verification is not None else None)
        return grade_response(
            rubric, record["criteria_scores"], evidence=evidence,
            corroboration=corroboration,
            contradicted=bool(verification and verification["any_refuted"]))

    def _load_seed(self, records: Sequence[Dict[str, Any]], source: str) -> int:
        prepared: List[Dict[str, Any]] = []
        for record in records:
            verification = self._verify_seed_record(record)
            grade = self._grade_seed_record(record, verification)
            self.seed_verification[record["entry_id"]] = verification
            prepared.append({
                "entry_id": record["entry_id"],
                "domain": record["domain"],
                "task": record.get("task", "support_reply"),
                "request": record["request"],
                "response": record["response"],
                "grade": grade.score,
                "verdict": grade.verdict,
                "rubric_id": grade.rubric_id,
                "evidence": list(record.get("evidence") or []),
                "evidence_status": record.get("evidence_status", "reference_only"),
            })
        return self.cache.load_reference(prepared, source=source)

    # -------------------------------------------------------- qualifying
    def qualifying_questions(self, domain: str) -> List[str]:
        """The questions asked before any model call (never empty)."""
        return seed.qualifying_questions(domain)

    # ------------------------------------------------------------- advise
    def advise(self, request: str, domain: str, k: int = 10,
               min_similarity: float = 0.0,
               exact_threshold: float = EXACT_SIMILARITY,
               similar_floor: float = SIMILAR_FLOOR) -> Dict[str, Any]:
        """Offer the closest answers *before* a model call, then ask."""
        hits = self.cache.retrieve(request, domain=domain, k=k,
                                   min_similarity=min_similarity)
        exact = [h for h in hits if h.score >= exact_threshold]
        similar = [h for h in hits if h.score >= similar_floor]

        served: Optional[RetrievalHit] = None
        if exact and exact[0].entry.verdict == "PASS":
            served = exact[0]
            decision = "exact_match"
        elif exact:
            decision = "exact_match_unverified"
        elif similar:
            decision = "similar_available"
        else:
            decision = "no_match"

        model_call_needed = served is None
        next_step = {
            "exact_match": "A passing cached answer matches this exact request; confirm it "
                           "resolves the issue or ask for more detail before calling a model.",
            "exact_match_unverified": "An exact request match exists but did not pass its "
                                      "rubric; a model call is warranted.",
            "similar_available": "Close answers exist; confirm the amounts, dates, and "
                                 "context before calling a model.",
            "no_match": "No close cached answer; answer the qualifying questions, then "
                        "call a model.",
        }[decision]

        result = {
            "schema": DOBERWATCH_SCHEMA,
            "domain": domain,
            "decision": decision,
            "model_call_needed": model_call_needed,
            "candidate_count": len(hits),
            "top_candidates": [hit.to_dict() for hit in hits],
            "recommended_response": served.entry.response if served else None,
            "recommended_grade": round(served.entry.grade, 6) if served else None,
            "recommended_verification": (self.seed_verification.get(
                served.entry.entry_id) if served else None),
            "qualifying_questions": self.qualifying_questions(domain),
            "ask_if_more_information_needed": bool(hits),
            "next_step": next_step,
        }
        self.cache.audit("doberwatch.advise", {
            "domain": domain, "decision": decision,
            "candidates": len(hits), "model_call_needed": model_call_needed,
        })
        return result

    def top_ten(self, request: str, domain: Optional[str] = None) -> List[RetrievalHit]:
        """The ten closest stored responses (audited read)."""
        return self.cache.top_ten(request, domain=domain)

    # ------------------------------------------------------------- submit
    def submit(self, request: str, response: str, domain: str,
               criterion_scores: Dict[str, float],
               rubric_id: Optional[str] = None,
               evidence: Optional[Dict[str, Sequence[str]]] = None,
               sources: Optional[Sequence[Source]] = None,
               corroboration: Optional[Mapping[str, float]] = None,
               consent: bool = False,
               price_paid_usd: float = 0.0,
               reference_price_usd: Optional[float] = None,
               reference_response: Optional[str] = None) -> Dict[str, Any]:
        """Grade a received response, cache it, and complain if it is bad.

        Pass ``sources`` to fact-check the response against other sources or
        models before it can pass: the rubric's corroboration criteria are
        capped by the measured material-claim grade, and an entirely
        unverified response is barred from PASS.
        """
        rubric = self.rubric_for(domain, rubric_id)

        if reference_response is None:
            hits = self.cache.retrieve(request, domain=domain, k=1,
                                       min_similarity=SIMILAR_FLOOR)
            reference_response = hits[0].entry.response if hits else None

        verification: Optional[Dict[str, Any]] = None
        contradicted = False
        if corroboration is None and sources:
            verification = verify_response(response, sources)
            corroboration = corroboration_for(rubric, verification)
            contradicted = bool(verification["any_refuted"])

        grade = grade_response(rubric, criterion_scores, evidence=evidence,
                               corroboration=corroboration,
                               contradicted=contradicted,
                               response=response, reference=reference_response)
        entry = self.cache.record(
            request=request, response=response, domain=domain,
            grade=grade.score, verdict=grade.verdict, task=rubric.task,
            rubric_id=rubric.rubric_id, evidence=_flat_evidence(evidence),
            consent=consent)

        value: Optional[ValueAssessment] = None
        if price_paid_usd > 0:
            value = assess_fair_value(price_paid_usd, grade.score,
                                      rubric.pass_threshold, reference_price_usd)

        complaint: Optional[Complaint] = None
        refund: Optional[Dict[str, Any]] = None
        if grade.complaint:
            complaint = self.workflow.file(
                domain, grade,
                reason=f"{grade.verdict}: {rubric.rubric_id} scored "
                       f"{grade.score:.2f} < {rubric.pass_threshold:.2f}")
            refund = refund_strategy(grade.verdict,
                                     price_paid_usd=max(0.0, price_paid_usd),
                                     value=value, domain=domain)

        return {
            "schema": DOBERWATCH_SCHEMA,
            "domain": domain,
            "grade": grade.to_dict(),
            "verification": verification,
            "value": value.to_dict() if value else None,
            "cache_entry_id": entry.entry_id,
            "cached_source": entry.source,
            "complaint": complaint.to_dict() if complaint else None,
            "refund_plan": refund,
            "escalation_path": refund["escalation_path"] if refund else [],
            "audit_valid": self.cache.audit_verify()["valid"],
        }

    # ---------------------------------------------------------- fact-check
    def verify(self, response: str, sources: Sequence[Source],
               domain: Optional[str] = None,
               claims: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Fact-check a response against independent sources and audit it."""
        result = verify_response(response, sources, claims=claims)
        self.cache.audit("doberwatch.verify", {
            "domain": domain,
            "claims": result["claim_count"],
            "material_grade": round(result["material_grade"], 6),
            "contradictions": len(result["contradictions"]),
        })
        return result

    def grade_matters(self, response: str, sources: Sequence[Source],
                      domain: str) -> Dict[str, Any]:
        """Grade every material claim and decide whether the response passes."""
        verification = self.verify(response, sources, domain=domain)
        material = [c for c in verification["claims"] if c["material"]]
        return {
            "schema": DOBERWATCH_SCHEMA,
            "domain": domain,
            "material_claims": material,
            "unverified_claims": [c["claim"] for c in material if c["grade"] < 0.5],
            "contradictions": verification["contradictions"],
            "material_grade": verification["material_grade"],
            "passes": verification["material_grade"] >= 0.7,
            "verification": verification,
        }

    def cross_model_check(self, responses: Mapping[str, str]) -> Dict[str, Any]:
        """Do independent models agree? Outliers are flagged, not trusted."""
        result = cross_model_agreement(responses)
        self.cache.audit("doberwatch.cross_model", {
            "models": result["models"],
            "agreement": round(result["agreement"], 6),
            "divergent": result["divergent"],
        })
        return result

    # --------------------------------------------------------- checkpoints
    def plan_checkpoints(self, n_gpus: int, overhead_s: float,
                         restore_s: float = 0.0,
                         environment: str = "local_nvme",
                         provider: Optional[str] = None,
                         usd_per_hour: float = 0.0,
                         failure_rate_per_gpu_hour: float = DEFAULT_FAILURE_RATE_PER_GPU_HOUR
                         ) -> Dict[str, Any]:
        """Young/Daly interval + environment/grace-window plan (audited)."""
        model = CheckpointCostModel(n_gpus=n_gpus, overhead_s=overhead_s,
                                    restore_s=restore_s,
                                    failure_rate_per_gpu_hour=failure_rate_per_gpu_hour)
        plan = checkpoint_plan(model, environment=environment, provider=provider,
                               usd_per_hour=usd_per_hour)
        self.cache.audit("doberwatch.checkpoint_plan", {
            "environment": environment, "provider": provider,
            "interval_s": plan["interval_s"], "over_tuned": plan["over_tuned"],
            "fits_grace": plan["fits_grace"],
        })
        return plan

    def fact_check_checkpoint_claims(self) -> Dict[str, Any]:
        """Test a claimed checkpoint-frequency table against the formula."""
        return check_interval_claims()

    def checkpoint_decision(self, seconds_since_checkpoint: float, n_bytes: float,
                            eviction_signal: bool = False) -> Dict[str, Any]:
        decision = self.checkpoints.should_checkpoint(
            seconds_since_checkpoint, n_bytes, eviction_signal=eviction_signal)
        self.cache.audit("doberwatch.checkpoint_decision", decision)
        return decision

    def note_checkpoint(self, n_bytes: float) -> Dict[str, Any]:
        record = self.checkpoints.note_checkpoint(n_bytes)
        self.cache.audit("doberwatch.checkpoint_written", record)
        return record

    # ----------------------------------------------------------- evidence
    def audit(self) -> Dict[str, Any]:
        """Verify the shared chain and report what it has sealed."""
        verification = self.cache.audit_verify()
        return {**verification, "events": len(self.cache.audit_log()),
                "cache": self.cache.stats(),
                "complaints": self.workflow.counts()}

    def export(self) -> str:
        """The user's portable copy of everything Doberwatch holds."""
        return self.cache.export_json()


def _flat_evidence(evidence: Optional[Dict[str, Sequence[str]]]) -> List[str]:
    if not evidence:
        return []
    flat: List[str] = []
    for refs in evidence.values():
        flat.extend(str(r) for r in refs if r)
    return flat


__all__ = ["DOBERWATCH_SCHEMA", "EXACT_SIMILARITY", "SIMILAR_FLOOR", "Doberwatch"]
