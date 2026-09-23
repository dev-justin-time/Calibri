# Doberwatch Finance — consumer watchdog

A watchdog for AI/service responses: it complains when a response is bad, it
fact-checks the response before believing it, and it can prove the complaint.
Everything is deterministic and offline — no model, no network, one audit chain.

```
qualify -> advise (ten closest answers BEFORE any model call)
        -> fact-check every material claim against independent sources
        -> grade (evidence + corroboration + anti-drift)
        -> complain / refund / escalate
```

## Modules (`calibrix/calibrix/doberwatch/`)

| Module | Responsibility |
|--------|----------------|
| `similarity.py` | Deterministic hashing bag-of-words + cosine. Retrieval and drift scoring; no model. |
| `verification.py` | Cross-source and cross-model fact-checking: support/contradict/neutral, numeric + polarity conflict, corroboration grading. |
| `grading.py` | Expert `Rubric`s, evidence gate, **corroboration gate**, Growl/Bark/Bite bands, anti-drift penalty. |
| `cache.py` | Consent-gated, PII-masked, audit-chained store. The user owns the data; sharing is opt-in twice. |
| `watchdog.py` | Growl/Bark/Bite complaint ladder + checkpoint-cost economics. |
| `checkpoint.py` | Young/Daly optimal frequency, environment/grace-window plans, frequency-table fact check. |
| `value.py` | Fair-value assessment, offer comparison, refund asks, escalation ladder. |
| `seed.py` | The hardcoded graded reference corpus (by domain) + rubrics + qualifying questions. |
| `core.py` | `Doberwatch` composes the flow; the cache owns the single audit chain. |

Agent-side entry points live in `calibrix/agents/`: `ToolSurface`'s watchdog
surface and the `ComplaintClerk` role (see *Agent surface*, below).

Reused, not reinvented: audit chain = `governance.AuditChain` (D7/F083), PII
masking = `governance.mask_pii` (F088), budget breaker = `billing.CostBreaker`
(D5/F059).

## Rules the code enforces

**Complain only when the response is bad.** A rubric is fixed *before* the
answer: `>= 0.70` PASS, `>= 0.55` GROWL, `>= 0.35` BARK, below BITE (minor
rework -> partial refund -> full refund + chargeback/regulator).

**Grades are evidence-backed.** A criterion that requires evidence caps at
`EVIDENCE_CAP = 0.5` without it; a rubric with `expected_evidence` cannot PASS
without evidence. Anti-drift discounts answers that ignore the request.

**Nothing passes unverified.** A criterion flagged `requires_corroboration`
cannot be credited above its measured corroboration, and a rubric with such
criteria cannot PASS at all on an unverified response (the *corroboration
gate*). A **contradicted** material claim triggers the same cap (the
*contradiction veto*): zeroing the one criterion a refuted fact happens to map
to is not enough when the rest of the answer still reads well, because the
answer as a whole is then false. Verification is fully deterministic:

```
support     = close (>= 0.45) AND same polarity AND no numeric conflict
contradict  = close (>= 0.40) AND (polarity mismatch OR numeric conflict)
grade       = support_strength x min(1, independent_sources / 2)
              (x0.5 if contested, 0 if contradicted/unverified)
```

Two independent authorities at default trust grade a claim 0.8; one source
caps at 0.5; a contradicting source zeroes it. `cross_model_agreement` applies
the same idea to several models and flags the divergent one.

**Fact-check precision.** Three text-handling defects were found and fixed,
because corroboration built on them would have been theatre:

| Defect | Effect | Fix |
|--------|--------|-----|
| `normalize_number` stripped trailing zeros | `"30" -> "3"`, so `"within 30 days"` and `"within 3 days"` compared **equal** and a wrong number read as *support* | strip zeros only after a decimal point |
| `has_negation` matched raw tokens | `"isn't"`/`"can't"`/`"hasn't"` never matched the apostrophe-free set, so contracted negations were invisible and polarity was misread | normalize apostrophes before lookup, and cover the missing contractions |
| `split_claims` split on every full stop | `"section 2.1 of the agreement"` became `"section 2"` + `"1 of the agreement"`, so a numbered clause could never match a source spelling it correctly | a decimal point no longer ends a sentence |

A **contested** claim (one witness refutes it, another supports it) now also
triggers the veto, not just a flatly contradicted one: a material fact some
independent source denies must not pass because a second source agreed.
`verify_response` reports `refuted` / `any_refuted` for exactly this.

**The user owns the data.** Default policy `local_only`; PII masked at write
time; JSON-exportable; publishing one entry needs policy + recorded consent +
current confirmation. Every write/read/share/refusal/verify/complaint is sealed
into the hash chain and re-verified on reload.

## Checkpoint frequency — the better method

The first-order optimum is **Young/Daly** (Young 1974; Daly 2006):

```
tau* = sqrt(2 * C * mu)          C = checkpoint cost, mu = MTBF
mu   = 1 / (N * f)               N GPUs at failure rate f
=> tau* = sqrt(2 * C / (N * f))  and waste at optimum = sqrt(C / (2 * mu))
```

So the optimum shrinks as `1/sqrt(N)` — not `1/N`, and certainly not `1/N²`.
`checkpoint_plan` turns the formula into an interval plus operations: environment
caps (`cpu_memory`, `local_nvme`, `network_storage`, `spot`), the provider grace
window (AWS 120 s, GCP 30 s), the 5%-overhead rule, and the write-local-then-
upload + SIGTERM actions.

## Fact-check of the pasted checkpoint material

Verified 2026-09 against primary sources (links below). Reproduce with
`Doberwatch().fact_check_checkpoint_claims()` and `tests.test_doberwatch`.

| Claim | Verdict |
|-------|---------|
| "c* = N·f / (2·r·o)" | **Corrected.** Young/Daly is `tau* = sqrt(2·C·µ)`; the pasted form has the wrong dimensions (optimum must *grow* with MTBF and *shrink* with sqrt(cost)). |
| 4 GPUs -> ~3 h | **Consistent** with C ~= 130 s of checkpoint cost (derived: `10800² / (2·450000) = 129.6 s`). |
| 16 GPUs -> ~11 min | **Inconsistent.** Same formula gives ~90 min. |
| 64 GPUs -> ~3 min | **Inconsistent.** Same formula gives ~45 min. |
| The table comes from one cost model | **No.** The implied checkpoint cost spans ~225x (129.6 s / 1.94 s / 0.58 s). The rows drop ~16x for a 4x cluster, which is `1/N²`, not the sqrt law. |
| Gemini SOSP'23: 13x faster recovery via CPU-memory checkpoints | **Verified** (Wang et al., SOSP'23: "more than 13x", "checkpointing to CPU memory of the host"). |
| Gemini: "no training throughput loss" | **Partly verified** — the paper says "without significantly impacting training"; not literally zero. |
| Gemini: "enables per-iteration checkpointing" | **Not supported by that paper** (it targets the optimal interval; per-iteration is the LowDiff/JIT claim). |
| LowDiff reuses compressed gradients as differential checkpoints | **Verified** (arXiv:2509.04084). |
| LowDiff: up to once-per-iteration; reduces training time up to 89.2% | **Verified as stated** ("training time", "up to"); the pasted "89% reduction in *wasted* time" overstates the measured wording. |
| LowDiff: "<3.5% overhead" | **Unverified** — not in the abstract, no corroborating source found. |
| JIT checkpointing = checkpoint between forward and backward pass | **Contradicted** — Gupta et al. (EuroSys'24) checkpoint *when failures happen* (on-demand), not in a fixed step window. |
| AWS spot: 2-minute interruption notice | **Verified** (AWS docs). |
| GCP preemption: 30 seconds | **Verified** (provider comparisons). |
| 8xH100 ~ $24/hr | **Plausible** (market ~$2.40-$4/GPU-hr). |
| "5% overhead is ~$24K/month" | **Contradicted.** 5% x $24/hr x 720 h = **$864/month** (~28x). |
| "prevents a 2% annual failure (~$96K loss)" | **Unsupported/inconsistent.** 2% of 24/7 annual spend (~$207k) is ~$4.1k. |

Primary sources: Young/Daly — Benoit et al., *ACM TOPC* 2022 (`W_YD = sqrt(2 µ C)`)
and the original 1974/2006 papers; Gemini — Wang et al., SOSP'23
(`zhuangwang93.github.io/docs/Gemini_SOSP23.pdf`); LowDiff —
`arxiv.org/abs/2509.04084`; JIT — Gupta et al., EuroSys'24
(`dl.acm.org/doi/10.1145/3627703.3650085`); AWS spot notices — AWS EC2 User Guide;
GCP 30 s — cloud-provider comparisons.

## The hardcoded corpus

`seed.py` ships 16 rubric-graded reference answers across 6 consumer-finance
domains, split into an **anonymous** (13, shareable) and a **private** (3,
owner-only) scope. Grades are *computed* from the rubric at load time, so the
library cannot drift from its own rubric.

**Corroboration is earned, not granted.** Each entry carries the independent
witnesses behind its factual claims (`sources`), and at load time the response
is fact-checked against them with the same machinery a live answer goes
through. The measured material-claim grade caps every `requires_corroboration`
criterion; nothing is pre-approved. Reproduce with:

```python
from calibrix.doberwatch import GRADED_RESPONSES, sources_for, verify_response
truths = [verify_response(r["response"], sources_for(r)) for r in GRADED_RESPONSES]
min(t["material_grade"] for t in truths)   # 0.8 — every claim corroborated
any(t["any_refuted"] for t in truths)      # False — no witness disagrees
```

Shipped state: **88 witnesses** (44 `authority`, 42 `document`, 2 `citation`),
4-8 per entry, each adversary-worded (no witness cribs the reply's wording —
checked by `test_witnesses_are_independently_worded_not_cribbed`; terms of art
shared where a real document would share them, and a pure synonym paraphrase
like “repays”/“refund” scores ~0.24 rather than the 0.45 bar, so sharing them
is expected). Inflections are normalized (``reversed/reversal/reverse`` → one
token) so morphology is not mistaken for disagreement.

Every entry at a measured `material_grade` of 0.80-0.83 and
`corroboration_coverage` of 0.80-0.83 — the old blanket `1.0` is gone, which
is why seed scores sit at 0.87-0.96 rather than a flat 1.0. Each claim gets
one `authority` (the governing rule or primary record) and one `document`
(the case log), worded independently, with numbers and negation posture
matching the claim it backs.

Two guards prove this is not decoration, and both are tested:
`test_corroboration_is_earned_not_granted` strips the witnesses off an entry
and the corroboration gate then blocks PASS; `test_a_disagreeing_witness_
breaks_the_entry` changes one number in one witness and the entry stops
passing (via the contradiction veto).

**Honesty bound:** every entry is `evidence_status: reference_only` — a
rubric-reviewed reference answer, not a production-measured outcome. The
witnesses make the *factual claims* checkable against the documents a
consumer can actually obtain; they do not make the corpus production-measured,
and nothing here claims customer ROI. The escalation text is dispute
arithmetic from the evidence, not legal advice.

## Reporting surface

`report.py` renders the grade and every fact-check verdict into the same
self-contained HTML dashboard (`report.html`), from the same JSON it writes
(`report.json`) — so the dashboard shows exactly the data the API returned.
Pass either half or the whole payload; nothing else changes:

```python
from calibrix.report import write_report
out = Doberwatch().submit(request, response, domain, criterion_scores=...,
                           evidence=..., sources=[Source("aws-docs", ...)])
write_report(result, "doberwatch", ["Quality"], out_dir, verification=out)
```

`build_verification_section(grade, verification)` normalizes
`Grade.to_dict()` + `verify_response()` (or a whole `Doberwatch.submit()`
payload, from either argument) into a serializable section carrying the
verdict, both gates, the veto, per-criterion judged-vs-credited rows, and
per-claim status/grade/support/contradiction/reasons. The dashboard renders a
pill per verdict (`corroborated` green; `single_source`/`contested`/`GROWL`/
`BARK` amber; `contradicted`/`unverified`/`BITE` red) — an unknown verdict
stays amber rather than being silently promoted to green.

Both the table cells and the embedded JSON payload are escaped. The payload
escape matters: `json.dumps` leaves `<` alone, so a cached response or claim
containing `</script>` would otherwise close the inline script block and become
live markup.

## Agent surface (wired)

`ToolSurface` (`calibrix/agents/tools.py`) exposes the watchdog as a fourth
surface, rooted in the surface's work dir so the cache, the complaint ladder
and the watchdog's own chain persist across missions:

| Method | Does |
|--------|------|
| `doberwatch_advise(request, domain, k=10)` | The <=10 closest answers *before* any model call, plus the qualifying questions. |
| `doberwatch_submit(request, response, domain, criterion_scores, **kw)` | Grade, cache, fact-check and complain. ``sources`` may be dicts. |
| `doberwatch_verify(response, sources, domain)` | Fact-check without grading. |
| `doberwatch_audit()` | The watchdog's chain (verified) beside the surface chain. |
| `audit_bus(messages)` | Cross-checks sealed topics against routed topics. |

`ComplaintClerk` (role `complaint_clerk`) owns `complaint.request` and runs
both halves of one case file:

```
complaint.request -> complaint.advice    <=10 closest, then ask the questions
                  -> complaint.pending   no response yet: more info required
                  -> complaint.resolved  verdict, gates, veto, fact-check
                  -> complaint.escalated only when the grade is a complaint
                                         (severity, refund ask, ladder)
```

Run it on the same bus as the kernel mission:

```python
from calibrix.agents import run_pipeline
out = run_pipeline(work_dir, complaint={
    "request": "duplicate charge refund", "domain": "billing_dispute",
    "response": "not our problem", "criterion_scores": {...},
    "price_paid_usd": 80.0,
})
out["complaint_case"]   # status RESOLVED, verdict, refund ask, ladder
out["bus_audit"]        # sealed == routed, in order
```

**Auditing the traffic.** `Auditor` subscribes to `"*"` and seals every
payload into the surface hash chain, so the claim "the bus traffic is audited"
is *checkable*: `audit_bus()` reports `sealed`, `unsealed` and
`first_divergence`, and `tests/test_agents.py` pins both directions — the
passing case and the case where traffic was never sealed. Direct tool calls
seal themselves too, so the chain is longer than the message count by exactly
the number of direct watchdog calls.

Two defects were found and fixed while wiring this, both by the tests rather
than by inspection:

* `Bus._deliver` indexed the *subscriber list* with `.get` whenever a topic
  was `"*"`, raising `AttributeError` and killing the pipeline on the first
  message — i.e. the Auditor could never seal anything. Dispatch now runs
  topic handlers then wildcard observers, in that fixed order.
* `ledger._is_test_class` inspected only a class's *immediate* base, so a
  shared `class AgentCase(unittest.TestCase)` base hid every subclass and the
  ledger published an undercount of its own suite (256 vs 277). Bases are now
  resolved within the module.

The agent layer previously had **no tests at all**, which is how both shipped;
`tests/test_agents.py` (24 tests) now covers the bus contract, the watchdog
surface, the clerk's case flow and the pipeline.

## Verify

```bash
cd calibrix
python -m unittest tests.test_doberwatch -v     # 76 tests
python -m unittest tests.test_report -v         # 8 tests
python -m unittest tests.test_agents -v         # 24 tests
python -m unittest discover -s tests            # full suite (294 tests)
python -m calibrix.studio.ledger scan           # SPDX + cleanroom gate
```
