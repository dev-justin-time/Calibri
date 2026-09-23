# SPDX-License-Identifier: MIT
# Doberwatch Finance — privacy-safe, audited response cache.
#
# Two rules drive every line here:
#
#   1. THE DATA IS THE USER'S. Responses are cached on the owner's own disk
#      or cloud (``path``), exportable as plain JSON, and *never* shared
#      unless the policy allows anonymous sharing AND the entry was recorded
#      with explicit per-entry consent. The default policy stores locally and
#      shares nothing.
#   2. EVERY WRITE, READ, AND SHARE IS AUDITED. The append-only hash chain is
#      the studio's proven AuditChain (D7/F083) — reused, not reinvented — so
#      a tampered or reordered history is detectable, and the history is
#      persisted so the proof survives a restart.
#
# Retrieval is a *local* search over the user's cache plus the hardcoded
# reference corpus; it is what lets Doberwatch offer the ten closest answers
# before any model call.

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from ..studio.governance import AuditChain, mask_pii
from .similarity import DEFAULT_DIM, cosine, hash_embedding

CACHE_SCHEMA = "calibrix.doberwatch.cache/1.0"

# deny              -> do not cache the user's responses at all
# local_only        -> cache on the owner's device/cloud, never share (DEFAULT)
# anonymous_share   -> cache, and allow sharing PII-masked entries that were
#                      individually consented to
CONSENT_MODES = ("deny", "local_only", "anonymous_share")


@dataclass
class ConsentPolicy:
    """The owner's caching consent, expressed as data (never a hidden default)."""

    mode: str = "local_only"
    allow_pii: bool = False
    retention_days: Optional[int] = 90

    def __post_init__(self) -> None:
        if self.mode not in CONSENT_MODES:
            raise ValueError(f"mode must be one of {CONSENT_MODES}, got {self.mode!r}")
        if self.retention_days is not None and self.retention_days <= 0:
            raise ValueError("retention_days must be positive or None")

    def may_cache(self) -> bool:
        return self.mode in ("local_only", "anonymous_share")

    def may_share(self, entry_consent: bool) -> bool:
        """Sharing needs the policy *and* the specific entry's consent."""
        return self.mode == "anonymous_share" and bool(entry_consent)

    def to_dict(self) -> Dict[str, Any]:
        return {"mode": self.mode, "allow_pii": self.allow_pii,
                "retention_days": self.retention_days}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsentPolicy":
        return cls(mode=str(data.get("mode", "local_only")),
                   allow_pii=bool(data.get("allow_pii", False)),
                   retention_days=data.get("retention_days", 90))


@dataclass
class CacheEntry:
    """One stored (request, response) pair with its grade and provenance."""

    entry_id: str
    domain: str
    request: str
    response: str
    grade: float
    verdict: str
    task: str = "support_reply"
    rubric_id: str = ""
    evidence: List[str] = field(default_factory=list)
    source: str = "local"          # local | seed_anonymous | seed_private
    consent: bool = False          # owner consented to anonymous sharing
    pii_masked: List[str] = field(default_factory=list)
    evidence_status: str = "reference_only"
    created_at: float = field(default_factory=time.time)
    digest: str = ""

    def canonical(self) -> str:
        payload = {k: v for k, v in self.to_dict().items() if k != "digest"}
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def seal(self) -> str:
        self.digest = hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()
        return self.digest

    def verify(self) -> bool:
        return bool(self.digest) and self.digest == hashlib.sha256(
            self.canonical().encode("utf-8")).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "domain": self.domain,
            "task": self.task,
            "request": self.request,
            "response": self.response,
            "grade": round(float(self.grade), 6),
            "verdict": self.verdict,
            "rubric_id": self.rubric_id,
            "evidence": list(self.evidence),
            "source": self.source,
            "consent": self.consent,
            "pii_masked": list(self.pii_masked),
            "evidence_status": self.evidence_status,
            "created_at": self.created_at,
            "digest": self.digest,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


@dataclass
class RetrievalHit:
    """A cached response offered as a close match for a new request."""

    score: float
    entry: CacheEntry

    def to_dict(self) -> Dict[str, Any]:
        return {"score": round(self.score, 6), **self.entry.to_dict()}


class ResponseCache:
    """User-owned, consent-gated, audit-chained store of graded responses."""

    def __init__(self, policy: Optional[ConsentPolicy] = None,
                 path: Optional[str] = None, owner: str = "user") -> None:
        self.owner = owner
        self.path = path
        self._policy_explicit = policy is not None
        self.policy = policy if policy is not None else ConsentPolicy()
        self.chain = AuditChain(actor=owner)
        self._entries: Dict[str, CacheEntry] = {}
        if path and os.path.exists(path):
            self._load()

    # ---------------------------------------------------------------- audit
    def _audit(self, event: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.chain.append(event, payload)

    def audit(self, event: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Public seal point: other Doberwatch components append here so the
        whole lifecycle (cache, complaints, checkpoints) shares one chain."""
        return self._audit(event, payload or {})

    def audit_verify(self) -> Dict[str, Any]:
        return self.chain.verify()

    def audit_log(self) -> List[Dict[str, Any]]:
        return self.chain.export()

    # ---------------------------------------------------------------- writes
    def record(self, request: str, response: str, domain: str, grade: float,
               verdict: str, task: str = "support_reply", rubric_id: str = "",
               evidence: Optional[Sequence[str]] = None, consent: bool = False,
               evidence_status: str = "local") -> CacheEntry:
        """Store a graded response for the owner, PII-masked and audited.

        Raises PermissionError when the policy is ``deny`` — the owner said no.
        """
        if not self.policy.may_cache():
            self._audit("cache.refused", {"domain": domain, "reason": "policy=deny"})
            raise PermissionError("caching denied by consent policy")

        masked_categories: List[str] = []
        if self.policy.allow_pii:
            stored_request, stored_response = request, response
        else:
            stored_request, found_req = mask_pii(request)
            stored_response, found_resp = mask_pii(response)
            masked_categories = sorted(set(found_req) | set(found_resp))

        entry = CacheEntry(
            entry_id="dw_" + uuid.uuid4().hex[:12],
            domain=domain,
            task=task,
            request=stored_request,
            response=stored_response,
            grade=float(grade),
            verdict=verdict,
            rubric_id=rubric_id,
            evidence=list(evidence or []),
            source="local",
            consent=bool(consent),
            pii_masked=masked_categories,
            evidence_status=evidence_status,
        )
        entry.seal()
        self._entries[entry.entry_id] = entry
        self._audit("cache.record", {
            "entry_id": entry.entry_id, "domain": domain,
            "verdict": verdict, "grade": round(float(grade), 6),
            "consent": bool(consent), "pii_masked": masked_categories,
            "digest": entry.digest,
        })
        self._flush()
        return entry

    def load_reference(self, records: Sequence[Dict[str, Any]],
                       source: str = "seed_anonymous") -> int:
        """Load hardcoded reference answers (already anonymized; not masked).

        Reference data is separate from the owner's captured responses: it is
        the frozen, graded corpus retrieved before model calls.
        """
        loaded = 0
        for record in records:
            entry = CacheEntry.from_dict({**record, "source": source,
                                          "consent": source == "seed_anonymous"})
            entry.seal()
            self._entries[entry.entry_id] = entry
            loaded += 1
        self._audit("cache.load_reference", {"source": source, "count": loaded})
        self._flush()
        return loaded

    # ----------------------------------------------------------------- reads
    def get(self, entry_id: str) -> Optional[CacheEntry]:
        return self._entries.get(entry_id)

    def retrieve(self, request: str, domain: Optional[str] = None,
                 k: int = 10, min_similarity: float = 0.0,
                 dim: int = DEFAULT_DIM) -> List[RetrievalHit]:
        """The ``k`` closest stored responses; always audited as a read."""
        query = hash_embedding(request, dim)
        hits: List[RetrievalHit] = []
        for entry in self._entries.values():
            if domain is not None and entry.domain != domain:
                continue
            score = cosine(query, hash_embedding(entry.request, dim))
            if score >= min_similarity:
                hits.append(RetrievalHit(score=score, entry=entry))
        hits.sort(key=lambda h: (-h.score, h.entry.entry_id))
        hits = hits[:max(0, k)]
        self._audit("cache.read", {
            "query_digest": hashlib.sha256(request.encode("utf-8")).hexdigest()[:16],
            "domain": domain,
            "returned": [h.entry.entry_id for h in hits],
        })
        return hits

    def top_ten(self, request: str, domain: Optional[str] = None,
                dim: int = DEFAULT_DIM) -> List[RetrievalHit]:
        return self.retrieve(request, domain=domain, k=10, dim=dim)

    # --------------------------------------------------------------- sharing
    def share_anonymous(self, entry_id: str, consent: bool) -> Dict[str, Any]:
        """Export one entry to the anonymous corpus.

        Refused unless the policy enables anonymous sharing, per-entry consent
        was recorded, and the entry is PII-clean (or the owner opted into PII).
        A refusal is not an error; it is the guarantee working.
        """
        entry = self._entries.get(entry_id)
        if entry is None:
            return {"shared": False, "reason": "unknown entry"}
        # Sharing needs BOTH the entry's recorded consent and a current,
        # explicit confirmation — a stale flag alone never publishes anything.
        if not self.policy.may_share(bool(consent) and entry.consent):
            self._audit("cache.share_refused", {
                "entry_id": entry_id, "policy": self.policy.mode,
                "entry_consent": entry.consent, "request_consent": bool(consent)})
            return {"shared": False, "reason": "policy or per-entry consent missing"}
        # PII was masked at write time (unless the owner opted into keeping
        # it), so the anonymous payload is already safe to publish.
        anonymous = {
            "anonymous_id": hashlib.sha256(
                (entry.domain + entry.request).encode("utf-8")).hexdigest()[:12],
            "domain": entry.domain,
            "task": entry.task,
            "request": entry.request,
            "response": entry.response,
            "grade": round(float(entry.grade), 6),
            "verdict": entry.verdict,
            "rubric_id": entry.rubric_id,
            "evidence_status": entry.evidence_status,
            "pii_masked": list(entry.pii_masked),
        }
        digest = hashlib.sha256(
            json.dumps(anonymous, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self._audit("cache.share", {"entry_id": entry_id,
                                    "anonymous_id": anonymous["anonymous_id"],
                                    "digest": digest})
        self._flush()
        return {"shared": True, "anonymous": anonymous, "digest": digest}

    # ----------------------------------------------------------- persistence
    def _flush(self) -> None:
        if not self.path:
            return
        payload = {
            "schema": CACHE_SCHEMA,
            "owner": self.owner,
            "policy": self.policy.to_dict(),
            "entries": [e.to_dict() for e in self._entries.values()],
            "audit": self.chain.export(),
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(tmp, self.path)

    def _load(self) -> None:
        with open(self.path, "r", encoding="utf-8") as handle:
            content = handle.read().strip()
        if not content:
            return
        data = json.loads(content)
        for record in data.get("entries", []):
            entry = CacheEntry.from_dict(record)
            self._entries[entry.entry_id] = entry
        # Adopt the stored consent only when the caller did not state one, so
        # reloading never silently widens or narrows the owner's consent.
        if not self._policy_explicit and data.get("policy"):
            self.policy = ConsentPolicy.from_dict(data["policy"])
        if data.get("audit"):
            self.chain = AuditChain.from_export(data["audit"], actor=self.owner)

    def export_json(self) -> str:
        """The owner's full, portable copy (entries + audit chain)."""
        return json.dumps({
            "schema": CACHE_SCHEMA,
            "owner": self.owner,
            "policy": self.policy.to_dict(),
            "entries": [e.to_dict() for e in self._entries.values()],
            "audit": self.chain.export(),
        }, indent=2, sort_keys=True)

    def stats(self) -> Dict[str, Any]:
        by_source: Dict[str, int] = {}
        by_domain: Dict[str, int] = {}
        for entry in self._entries.values():
            by_source[entry.source] = by_source.get(entry.source, 0) + 1
            by_domain[entry.domain] = by_domain.get(entry.domain, 0) + 1
        return {
            "schema": CACHE_SCHEMA,
            "entries": len(self._entries),
            "by_source": by_source,
            "by_domain": by_domain,
            "policy": self.policy.to_dict(),
            "audit_events": len(self.chain.export()),
        }

    def __len__(self) -> int:
        return len(self._entries)


__all__ = ["CACHE_SCHEMA", "CONSENT_MODES", "ConsentPolicy", "CacheEntry",
           "RetrievalHit", "ResponseCache"]
