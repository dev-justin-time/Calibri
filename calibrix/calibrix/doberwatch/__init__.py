# SPDX-License-Identifier: MIT
"""Doberwatch Finance — a consumer watchdog for AI/service responses.

The core idea: a response is only "bad" relative to a rubric fixed *before*
the answer arrives, and the data that proves it belongs to the user. Nothing
is believed until it is corroborated by independent sources.

  cache         — consent-gated, PII-masked, audit-chained store (user owns it)
  grading       — expert rubrics, evidence-backed grades, corroboration gate,
                  anti-drift penalty
  verification  — cross-source / cross-model fact-checking (numeric + polarity
                  conflict detection, corroboration grading)
  watchdog      — Growl / Bark / Bite complaints + checkpoint-cost economics
  checkpoint    — Young/Daly optimal checkpoint frequency, environment and
                  preemption-grace plans, frequency-table fact check
  value         — fair-value comparison, refund asks, escalation ladder
  seed          — the hardcoded graded reference corpus, by domain

Everything is deterministic, offline, and unit-tested; nothing here calls a
model or a network. ``Doberwatch`` composes the pieces into the flow:
qualify -> advise (ten closest answers before any model call) -> fact-check ->
grade -> complain / refund / escalate.
"""

from .cache import (CACHE_SCHEMA, CONSENT_MODES, CacheEntry, ConsentPolicy,
                    ResponseCache, RetrievalHit)
from .checkpoint import (CHECKPOINT_SCHEMA, DEFAULT_FAILURE_RATE_PER_GPU_HOUR,
                         ENVIRONMENTS, GRACE_PERIOD_S, CheckpointCostModel,
                         FailureModel, check_interval_claims, checkpoint_plan,
                         expected_waste_fraction, implied_overhead_s,
                         monthly_waste_usd, young_daly_interval_s)
from .core import DOBERWATCH_SCHEMA, Doberwatch, EXACT_SIMILARITY, SIMILAR_FLOOR
from .grading import (BARK, BITE, EVIDENCE_CAP, GROWL, PASS, Criterion, Grade,
                      Rubric, corroborates, grade_response, verdict_for)
from .seed import (ANONYMOUS_SEED, DOMAINS, GRADED_RESPONSES, PRIVATE_SEED,
                   RUBRICS, qualifying_questions, rubric_for_domain, sources_for)
from .similarity import cosine, hash_embedding, similarity, tokenize, top_k
from .value import (ESCALATION_LADDER, REFUND_TABLE, ValueAssessment,
                    assess_fair_value, compare_offers,
                    delivered_quality_fraction, escalation_path,
                    refund_strategy)
from .verification import (MIN_SOURCES, SINGLE_SOURCE_CAP, VERIFICATION_SCHEMA,
                           ClaimVerification, Source, corroboration_for,
                           compare_claim_to_source, cross_model_agreement,
                           extract_numbers, has_negation, is_material,
                           materiality_score, normalize_number, numeric_conflict,
                           split_claims, verify_claim, verify_response)
from .watchdog import (SEVERITIES, CheckpointWatchdog, Complaint, GrowlBarkBite,
                       breakeven_checkpoint_interval_s, checkpoint_write_usd,
                       expected_eviction_loss_usd)

__all__ = [
    "CACHE_SCHEMA", "CONSENT_MODES", "CacheEntry", "ConsentPolicy",
    "ResponseCache", "RetrievalHit",
    "CHECKPOINT_SCHEMA", "DEFAULT_FAILURE_RATE_PER_GPU_HOUR", "ENVIRONMENTS",
    "GRACE_PERIOD_S", "CheckpointCostModel", "FailureModel",
    "check_interval_claims", "checkpoint_plan", "expected_waste_fraction",
    "implied_overhead_s", "monthly_waste_usd", "young_daly_interval_s",
    "DOBERWATCH_SCHEMA", "Doberwatch", "EXACT_SIMILARITY", "SIMILAR_FLOOR",
    "BARK", "BITE", "EVIDENCE_CAP", "GROWL", "PASS", "Criterion", "Grade",
    "Rubric", "corroborates", "grade_response", "verdict_for",
    "ANONYMOUS_SEED", "PRIVATE_SEED", "GRADED_RESPONSES", "RUBRICS", "DOMAINS",
    "qualifying_questions", "rubric_for_domain", "sources_for",
    "cosine", "hash_embedding", "similarity", "tokenize", "top_k",
    "ESCALATION_LADDER", "REFUND_TABLE", "ValueAssessment", "assess_fair_value",
    "compare_offers", "delivered_quality_fraction", "escalation_path",
    "refund_strategy",
    "MIN_SOURCES", "SINGLE_SOURCE_CAP", "VERIFICATION_SCHEMA",
    "ClaimVerification", "Source", "corroboration_for",
    "compare_claim_to_source", "cross_model_agreement", "extract_numbers",
    "has_negation", "is_material", "materiality_score", "normalize_number",
    "numeric_conflict", "split_claims", "verify_claim", "verify_response",
    "SEVERITIES", "CheckpointWatchdog", "Complaint", "GrowlBarkBite",
    "breakeven_checkpoint_interval_s", "checkpoint_write_usd",
    "expected_eviction_loss_usd",
]
