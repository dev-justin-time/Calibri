# SPDX-License-Identifier: MIT
"""Tool surfaces the agents drive.

Four surfaces, deliberately mirroring the deployment topologies:

  * ``offline``  — calibration, proof, profile fitting. No network, no GPU.
  * ``comfy``    — ComfyUI artifact verification/export + Rust accel parity.
  * ``online``   — the marketplace (listings, orders, fulfillment, licenses).
  * ``watchdog`` — Doberwatch: advise before a model call, grade what came
    back, fact-check it, and file the complaint when it fails.

Every method is thin: it composes existing, individually-tested platform
primitives and returns plain dicts so agents (and tests) never poke at
private state.
"""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..adapters import Prompt, ScriptedAdapter
from ..doberwatch import Doberwatch, Source
from ..engine import SearchConfig, SearchEngine
from ..heretic_bridge import fit_kernel_profile, probe as bridge_probe
from ..kernel import parse_kernel_spec
from ..marketplace.licenses import checksum_spec, verify_license
from ..marketplace.store import JsonStore, Listing, Order, OrderStatus
from ..accel import parity_check, rust_available
from ..studio.proof import build_dossier, gap_alarm
from ..studio.runtime import export_comfyui_workflow
from ..scorers import KeywordRate, LengthDrift


def _as_sources(raw: Any) -> Optional[List[Source]]:
    """Accept ``Source`` objects or plain ``{source_id, text, kind}`` dicts.

    Bus payloads must stay JSON-serializable, so a role can hand over dicts
    and the surface converts them; unknown shapes are dropped rather than
    guessed at.
    """
    if not raw:
        return None
    sources: List[Source] = []
    for item in raw:
        if isinstance(item, Source):
            sources.append(item)
        elif isinstance(item, dict):
            sources.append(Source(
                source_id=str(item.get("source_id", "")),
                text=str(item.get("text", "")),
                kind=str(item.get("kind", "document")),
                trust=item.get("trust"),
                independent=bool(item.get("independent", True)),
                url=str(item.get("url", "")),
                derived_from=str(item.get("derived_from", "")),
            ))
    return sources or None


class ToolSurface:
    """All agent-callable platform operations, grouped by surface."""

    def __init__(self, work_dir: str = "agent_workspace",
                 secret: Optional[str] = None) -> None:
        self.work = Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.secret = secret
        self.bridge_status = bridge_probe()
        self._doberwatch: Optional[Doberwatch] = None
        # Audit chain lives on the surface (not an agent) so ANY role's
        # actions are sealable and the chain survives agent replacement.
        self.chain: List[Dict[str, Any]] = []
        self.head: str = "0" * 64

    # ------------------------------------------------------------------
    # AUDIT primitive (hash-linked append-only log; used by the Auditor)
    # ------------------------------------------------------------------
    def audit_append(self, actor: str, topic: str,
                     payload: Any) -> Dict[str, Any]:
        entry = {"seq": len(self.chain), "ts": time.time(), "actor": actor,
                 "topic": topic, "payload": payload, "prev": self.head}
        digest = hashlib.sha256(
            json.dumps(entry, sort_keys=True, default=str).encode()).hexdigest()
        entry["digest"] = digest
        self.chain.append(entry)
        self.head = digest
        return entry

    def audit_verify(self) -> Dict[str, Any]:
        for i, e in enumerate(self.chain):
            check = {k: v for k, v in e.items() if k != "digest"}
            expect = hashlib.sha256(
                json.dumps(check, sort_keys=True, default=str).encode()).hexdigest()
            if expect != e["digest"] or (i and e["prev"] != self.chain[i - 1]["digest"]):
                return {"valid": False, "broken_at": i}
        return {"valid": True, "length": len(self.chain)}

    def audit_bus(self, messages: Sequence[Any]) -> Dict[str, Any]:
        """Cross-check the sealed chain against the routed traffic.

        ``Auditor`` subscribes to ``"*"`` and appends every message, so "the
        bus traffic is audited" is checkable rather than asserted: the sealed
        topics must match the routed topics, in order.
        """
        sealed = [e.get("topic") for e in self.chain
                  if e.get("actor") == "auditor"]
        routed = [getattr(m, "topic", None) for m in list(messages)]
        # Divergence is measured along the *routed* traffic: a message the
        # auditor never saw is the first divergence, not a no-op comparison.
        divergence = next((i for i, topic in enumerate(routed)
                           if i >= len(sealed) or sealed[i] != topic), None)
        if divergence is None and len(sealed) > len(routed):
            divergence = len(routed)
        return {
            "chain_valid": self.audit_verify()["valid"],
            "routed": len(routed),
            "sealed": len(sealed),
            "unsealed": routed[len(sealed):] if len(sealed) < len(routed) else [],
            "order_matches": divergence is None and len(sealed) == len(routed),
            "first_divergence": divergence,
        }

    # ------------------------------------------------------------------
    # OFFLINE surface
    # ------------------------------------------------------------------
    def calibrate(self, mission: Dict[str, Any]) -> Dict[str, Any]:
        """Run one hosted calibration search. mission: {model, n_layers,
        n_trials, popsize, seed} -> summary incl. best vector + fitness."""
        n_layers = int(mission.get("n_layers", 12))
        seed = int(mission.get("seed", 0))
        adapter = ScriptedAdapter(n_layers=n_layers, seed=seed)
        bank = list(mission.get("prompts") or [
            "a red apple on a wooden table",
            "product photo of a ceramic mug",
            "poster art of a city skyline at dusk",
            "a golden retriever running on a beach",
            "macro shot of a dewy leaf",
            "an old lighthouse in a storm",
        ])
        prompts = [Prompt(system="You are a helpful assistant.", user=u)
                   for u in bank]
        scorers = [KeywordRate(bank), LengthDrift(bank)]
        cfg = SearchConfig(
            n_trials=int(mission.get("n_trials", 6)),
            popsize=int(mission.get("popsize", 6)),
            optimizer="simple", seed=seed,
        )
        result = SearchEngine(adapter, scorers, prompts, cfg).run()
        best = result.get("best")
        return {
            "trials": len(result.get("trials") or []),
            "best_fitness": float(best.fitness) if best else None,
            "best_vector": list(best.vector) if best else None,
            "pareto_size": len(result.get("pareto") or []),
            "overfit": result.get("overfit"),
            "elapsed_seconds": result.get("elapsed_seconds"),
        }

    def check_overfit(self, calib_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Validator gate: the engine's own alarm + gap alarm re-check."""
        alarm = calib_summary.get("overfit") or {}
        return {"flagged": bool(alarm.get("flagged")),
                "gap": alarm.get("gap"),
                "rule": "engine holdout alarm (studio.proof.gap_alarm semantics)"}

    def fit_profile(self, vector: List[float]) -> Dict[str, Any]:
        """Heretic-bridge kernel-profile fit (MIT side, offline)."""
        strengths = [min(max(abs(v - 1.0), 0.0), 1.0) for v in vector]
        fit = fit_kernel_profile(strengths)
        return {"rmse": fit["rmse"], "spec_string": fit["spec_string"],
                "n_layers": fit["n_layers"]}

    def build_attestation(self, payload: Dict[str, Any],
                          name: str = "dossier") -> Dict[str, Any]:
        path = self.work / f"{name}.json"
        d = build_dossier(payload, path=str(path))
        return {"path": str(path), "kernel_seal": d.get("kernel_seal", "")}

    # ------------------------------------------------------------------
    # COMFY surface
    # ------------------------------------------------------------------
    def verify_spec(self, spec: str, n_sites: int = 12) -> Dict[str, Any]:
        """Spec must parse + Rust parity must hold (when the core ships)."""
        channels = parse_kernel_spec(spec)
        rep = parity_check(n_sites=n_sites)
        return {"channels": [c for c, _ in channels],
                "rust_available": rust_available(),
                "parity": rep["parity"],
                "max_abs_diff": rep["max_abs_diff"]}

    def export_workflow(self, spec: str, checkpoint: str,
                        name: str = "kernel_workflow") -> Dict[str, Any]:
        wf = export_comfyui_workflow(spec, checkpoint=checkpoint,
                                     workflow_name=name)
        path = self.work / f"{name}.json"
        path.write_text(json.dumps(wf, indent=2), encoding="utf-8")
        node_types = {n.get("type") for n in wf.get("nodes", [])}
        return {"path": str(path), "node_types": sorted(node_types),
                "has_calibrix_node": "CalibrixKernelScale" in node_types}

    # ------------------------------------------------------------------
    # ONLINE surface
    # ------------------------------------------------------------------
    def _store(self, name: str = "store") -> JsonStore:
        return JsonStore(path=str(self.work / f"{name}.json"),
                         secret=self.secret)

    def list_kernel(self, spec: str, title: str, model: str,
                    price_cents: int, metrics: Optional[Dict[str, float]] = None,
                    store_name: str = "store") -> Dict[str, Any]:
        store = self._store(store_name)
        listing = Listing(
            listing_id="lst_" + str(int(time.time() * 1000))[-9:],
            title=title, model=model, kernel_spec=spec,
            price_cents=int(price_cents),
            description=f"calibrated kernel for {model}",
            spec_checksum=checksum_spec(spec),
            scoring=dict(metrics or {}),
        )
        saved = store.add_listing(listing)
        return {"listing_id": saved.listing_id, "checksum": saved.spec_checksum}

    def sell_kernel(self, listing_id: str, buyer_email: str,
                    store_name: str = "store") -> Dict[str, Any]:
        """Full order cycle: create -> mock-pay -> fulfill -> verify license."""
        store = self._store(store_name)
        listing = store.get_listing(listing_id)
        if listing is None:
            return {"sold": False, "reason": "listing not found"}
        order = store.create_order(listing, buyer_email, provider="mock")
        order = store.mark_paid(order)
        order = store.fulfill_order(order)
        check = verify_license(order.license_key, spec=listing.kernel_spec,
                               secret=self.secret,
                               require_entitlement="comfyui")
        return {"order_id": order.order_id, "status": order.status,
                "sold": order.status == OrderStatus.FULFILLED and check["valid"],
                "license_valid": check["valid"],
                "license_reason": check.get("reason")}

    def revenue(self, store_name: str = "store") -> Dict[str, Any]:
        store = self._store(store_name)
        return {"orders": len(store._data.get("orders", {})),
                "listings": len(store._data.get("listings", {}))}

    # ------------------------------------------------------------------
    # DOBERWATCH surface (advise -> grade -> fact-check -> complain)
    # ------------------------------------------------------------------
    @property
    def doberwatch(self) -> Doberwatch:
        """The watchdog, rooted in this surface's work dir so its cache, its
        complaint ladder and its audit chain persist across missions."""
        if self._doberwatch is None:
            self._doberwatch = Doberwatch(
                cache_path=str(self.work / "doberwatch_cache.json"))
        return self._doberwatch

    def _seal(self, topic: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Seal a watchdog event into the surface chain as well, so a direct
        tool call leaves the same trace a routed message would."""
        return self.audit_append("doberwatch", topic, payload)

    def doberwatch_advise(self, request: str, domain: str,
                          k: int = 10) -> Dict[str, Any]:
        """The closest answers BEFORE any model call, plus the qualifying
        questions that must be asked first."""
        advice = self.doberwatch.advise(request, domain, k=k)
        self._seal("doberwatch.advise", {
            "domain": domain, "decision": advice["decision"],
            "model_call_needed": advice["model_call_needed"]})
        return advice

    def doberwatch_submit(self, request: str, response: str, domain: str,
                          criterion_scores: Dict[str, float],
                          **kwargs: Any) -> Dict[str, Any]:
        """Grade a received response, cache it, and complain if it fails.

        ``kwargs`` are forwarded to ``Doberwatch.submit`` (``sources``,
        ``evidence``, ``consent``, ``price_paid_usd``, ...); ``sources`` may
        be dicts so the whole payload stays JSON-serializable.
        """
        kwargs["sources"] = _as_sources(kwargs.get("sources"))
        outcome = self.doberwatch.submit(request, response, domain,
                                         criterion_scores, **kwargs)
        grade = outcome["grade"]
        self._seal("doberwatch.submit", {
            "domain": domain, "rubric_id": grade["rubric_id"],
            "verdict": grade["verdict"], "score": grade["score"],
            "complaint": bool(outcome["complaint"]),
            "contradiction_veto": bool(grade.get("contradiction_veto"))})
        return outcome

    def doberwatch_verify(self, response: str, sources: Any,
                          domain: Optional[str] = None) -> Dict[str, Any]:
        """Fact-check a response against independent sources/models."""
        result = self.doberwatch.verify(response, _as_sources(sources) or [],
                                        domain=domain)
        self._seal("doberwatch.verify", {
            "claims": result["claim_count"],
            "material_grade": round(result["material_grade"], 6),
            "contradictions": len(result["contradictions"])})
        return result

    def doberwatch_audit(self) -> Dict[str, Any]:
        """The watchdog's own chain (verified) beside the surface chain."""
        return {**self.doberwatch.audit(), "surface_chain": self.audit_verify()}
