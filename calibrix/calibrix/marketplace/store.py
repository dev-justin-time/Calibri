# SPDX-License-Identifier: MIT
# Calibrix marketplace store: JSON-backed listings + orders.
#
# Deliberately boring: one JSON file, atomic writes, no database. Enough to
# run a real small marketplace; swap for Postgres later behind the same
# interface. Every order carries the fulfillment payload (license key) so the
# ledger is the source of truth.

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .licenses import issue_license


EVIDENCE_STATUSES = ("simulation", "reference_only", "measured", "verified")


@dataclass
class Listing:
    listing_id: str
    title: str
    model: str                      # e.g. "FLUX.1-dev"
    kernel_spec: str                # compact spec consumed by the ComfyUI node
    price_cents: int
    description: str = ""
    category: str = "product-photo"
    tags: List[str] = field(default_factory=list)
    compatible_models: List[str] = field(default_factory=list)
    spec_checksum: str = ""         # filled on save via checksum_spec
    scoring: Dict[str, float] = field(default_factory=dict)   # report metrics
    # Evidence is intentionally explicit: a listing must not imply that an
    # offline demo or paper result is a customer-validated outcome.
    evidence_status: str = "simulation"  # simulation | reference_only | measured | verified
    evidence_note: str = ""
    report_path: str = ""
    holdout_score: Optional[float] = None
    baseline_score: Optional[float] = None
    quality_gate: str = "REVIEW"
    active: bool = True
    created_at: int = field(default_factory=lambda: int(time.time()))


@dataclass
class Order:
    order_id: str
    listing_id: str
    buyer_email: str
    status: str                     # OrderStatus value
    amount_cents: int
    provider: str                   # "mock" | "stripe"
    provider_ref: str = ""          # checkout session id
    license_key: str = ""           # filled on fulfillment
    license_id: str = ""
    created_at: int = field(default_factory=lambda: int(time.time()))
    paid_at: Optional[int] = None


class OrderStatus:
    PENDING = "pending"
    PAID = "paid"
    FULFILLED = "fulfilled"
    REFUNDED = "refunded"


class MarketplaceStore:
    """Interface: listings + orders with fulfillment."""

    def add_listing(self, listing: Listing) -> Listing: ...
    def get_listing(self, listing_id: str) -> Optional[Listing]: ...
    def list_listings(self, active_only: bool = True) -> List[Listing]: ...
    def create_order(self, listing: Listing, buyer_email: str,
                     provider: str) -> Order: ...
    def get_order(self, order_id: str) -> Optional[Order]: ...
    def get_order_by_session(self, session_id: str) -> Optional[Order]: ...
    def fulfill_order(self, order: Order, ttl_days: Optional[int] = None) -> Order:
        """Issue + persist the license for the order's listing kernel."""
        ...


class JsonStore(MarketplaceStore):
    def __init__(self, path: str = "marketplace_data.json",
                 secret: Optional[str] = None) -> None:
        self.path = path
        self.secret = secret
        self._data: Dict[str, Any] = {"listings": {}, "orders": {}}
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content:  # tolerate empty/zero-byte store files
                self._data = json.loads(content)
            self._data.setdefault("listings", {})
            self._data.setdefault("orders", {})

    # -- persistence ---------------------------------------------------
    def _flush(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, self.path)

    @staticmethod
    def _decode_listing(d: Dict[str, Any]) -> Listing:
        return Listing(**{k: d[k] for k in Listing.__dataclass_fields__ if k in d})

    @staticmethod
    def _decode_order(d: Dict[str, Any]) -> Order:
        return Order(**{k: d[k] for k in Order.__dataclass_fields__ if k in d})

    # -- listings -------------------------------------------------------
    def add_listing(self, listing: Listing) -> Listing:
        from .licenses import checksum_spec

        if listing.evidence_status not in EVIDENCE_STATUSES:
            raise ValueError(
                f"evidence_status must be one of {EVIDENCE_STATUSES}, "
                f"got {listing.evidence_status!r}"
            )
        if listing.evidence_status == "verified":
            missing = []
            if listing.quality_gate != "PASS":
                missing.append("quality_gate=PASS")
            if not listing.report_path:
                missing.append("report_path")
            if listing.holdout_score is None or listing.baseline_score is None:
                missing.append("holdout_score and baseline_score")
            if missing:
                raise ValueError(
                    "verified listings require reviewed evidence: " + ", ".join(missing)
                )
        listing.spec_checksum = checksum_spec(listing.kernel_spec)
        self._data["listings"][listing.listing_id] = asdict(listing)
        self._flush()
        return listing

    def promote_verified_listing(self, listing_id: str, evidence: Dict[str, Any]) -> Listing:
        """Promote a listing only from a matching, passing validation artifact."""
        listing = self.get_listing(listing_id)
        if listing is None:
            raise KeyError(f"unknown listing {listing_id!r}")
        if evidence.get("status") != "verified" or evidence.get("quality_gate") != "PASS":
            raise ValueError("only a passing verified validation can be promoted")
        if evidence.get("kernel_checksum") != listing.spec_checksum:
            raise ValueError("validation artifact does not match listing kernel checksum")
        required = ("report_path", "holdout_score", "baseline_score")
        missing = [key for key in required if evidence.get(key) in (None, "")]
        if missing:
            raise ValueError("verified promotion missing: " + ", ".join(missing))
        listing.evidence_status = "verified"
        listing.evidence_note = str(evidence.get("note", "Reviewed real-model holdout validation."))
        listing.report_path = str(evidence["report_path"])
        listing.holdout_score = float(evidence["holdout_score"])
        listing.baseline_score = float(evidence["baseline_score"])
        listing.quality_gate = "PASS"
        listing.scoring = {str(k): float(v) for k, v in (evidence.get("scoring") or {}).items()}
        return self.add_listing(listing)

    def get_listing(self, listing_id: str) -> Optional[Listing]:
        d = self._data["listings"].get(listing_id)
        return self._decode_listing(d) if d else None

    def list_listings(self, active_only: bool = True) -> List[Listing]:
        out = [self._decode_listing(d) for d in self._data["listings"].values()]
        return [l for l in out if l.active] if active_only else out

    # -- orders ----------------------------------------------------------
    def create_order(self, listing: Listing, buyer_email: str,
                     provider: str) -> Order:
        order = Order(
            order_id="ord_" + uuid.uuid4().hex[:12],
            listing_id=listing.listing_id,
            buyer_email=buyer_email,
            status=OrderStatus.PENDING,
            amount_cents=listing.price_cents,
            provider=provider,
        )
        self._data["orders"][order.order_id] = asdict(order)
        self._flush()
        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        d = self._data["orders"].get(order_id)
        return self._decode_order(d) if d else None

    def get_order_by_session(self, session_id: str) -> Optional[Order]:
        for d in self._data["orders"].values():
            if d.get("provider_ref") == session_id:
                return self._decode_order(d)
        return None

    def _save_order(self, order: Order) -> Order:
        self._data["orders"][order.order_id] = asdict(order)
        self._flush()
        return order

    def attach_session(self, order: Order, session_id: str) -> Order:
        order.provider_ref = session_id
        return self._save_order(order)

    def mark_paid(self, order: Order) -> Order:
        order.status = OrderStatus.PAID
        order.paid_at = int(time.time())
        return self._save_order(order)

    def fulfill_order(self, order: Order, ttl_days: Optional[int] = None) -> Order:
        if order.status == OrderStatus.FULFILLED:
            return order
        listing = self.get_listing(order.listing_id)
        if listing is None:
            raise ValueError(f"order {order.order_id} references missing listing")
        key, lic = issue_license(
            spec=listing.kernel_spec,
            buyer=order.buyer_email,
            product=listing.listing_id,
            entitlements=["comfyui", "commercial"],
            ttl_days=ttl_days,
            secret=self.secret,
        )
        order.license_key = key
        order.license_id = lic.license_id
        order.status = OrderStatus.FULFILLED
        return self._save_order(order)


# Backwards-friendly alias
Store = JsonStore
