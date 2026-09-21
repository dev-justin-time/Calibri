# SPDX-License-Identifier: MIT
# Calibrix marketplace licenses.
#
# A license binds a kernel artifact (identified by content hash of its spec
# string) to a buyer and entitlements. The key format is:
#
#   CBX1.<payload_b64url>.<hmac_b64url>
#
#   payload  = compact JSON: {"v":1,"lid":"...","sub":"<checksum>","nts":[..],
#              "ent":[...],"iat":<unix>,"exp":<unix|null>}
#   hmac     = HMAC-SHA256(secret, payload_bytes), url-safe base64, no padding
#
# Design goals:
#   * The ComfyUI node must verify licenses OFFLINE (no network, no deps):
#     the node embeds/reads the marketplace public secret fingerprint and can
#     check integrity + expiry + artifact binding with stdlib only.
#   * Keys are only as secret as the signing key; this is tamper-EVIDENCE and
#     licensing hygiene, not DRM. The kernel spec itself is delivered with it.
#   * `sub` is the SHA-256 of the kernel spec string, so a key is useless for
#     any other kernel artifact.

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

KEY_PREFIX = "CBX1"


def _b64url(data: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    import base64

    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def get_signing_secret() -> str:
    """Signing secret from env (set MARKETPLACE_SIGNING_SECRET in prod)."""
    secret = os.environ.get("MARKETPLACE_SIGNING_SECRET", "")
    if secret:
        return secret
    # Dev fallback: deterministic, obviously not for production use.
    return "calibrix-dev-secret-do-not-use-in-production"


def checksum_spec(spec: str) -> str:
    """SHA-256 of the exact kernel spec string (hex). Artifact identity."""
    return hashlib.sha256(spec.encode("utf-8")).hexdigest()


@dataclass
class License:
    license_id: str
    subject: str                     # checksum of the licensed kernel spec
    entitlements: List[str]          # e.g. ["comfyui", "commercial"]
    issued_at: int
    buyer: str = ""
    product: str = ""                # listing id
    expires_at: Optional[int] = None  # unix seconds; None = perpetual

    def to_payload(self) -> Dict[str, Any]:
        return {
            "v": 1,
            "lid": self.license_id,
            "sub": self.subject,
            "ent": list(self.entitlements),
            "iat": self.issued_at,
            "exp": self.expires_at,
            "buyer": self.buyer,
            "product": self.product,
        }

    @classmethod
    def from_payload(cls, p: Dict[str, Any]) -> "License":
        return cls(
            license_id=p["lid"],
            subject=p["sub"],
            entitlements=list(p.get("ent", [])),
            issued_at=int(p.get("iat", 0)),
            expires_at=p.get("exp"),
            buyer=p.get("buyer", ""),
            product=p.get("product", ""),
        )


def issue_license(
    spec: str,
    buyer: str,
    product: str,
    entitlements: Optional[List[str]] = None,
    ttl_days: Optional[int] = None,
    license_id: Optional[str] = None,
    secret: Optional[str] = None,
) -> tuple[str, License]:
    """Sign a license for `spec`. Returns (key_string, License)."""
    lic = License(
        license_id=license_id or ("cbx_" + _b64url(os.urandom(9)).lower()),
        subject=checksum_spec(spec),
        entitlements=entitlements or ["comfyui"],
        issued_at=int(time.time()),
        buyer=buyer,
        product=product,
        expires_at=(int(time.time()) + int(ttl_days) * 86400) if ttl_days else None,
    )
    payload = json.dumps(lic.to_payload(), separators=(",", ":"), sort_keys=True)
    sig = hmac.new((secret or get_signing_secret()).encode("utf-8"),
                   payload.encode("utf-8"), hashlib.sha256).digest()
    key = f"{KEY_PREFIX}.{_b64url(payload.encode('utf-8'))}.{_b64url(sig)}"
    return key, lic


def verify_license(
    key: str,
    spec: Optional[str] = None,
    secret: Optional[str] = None,
    now: Optional[int] = None,
    require_entitlement: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify a license key; optionally bind it to a kernel spec.

    Returns {"valid": bool, "reason": str|None, "license": License|None}.
    Offline-safe: needs only the signing secret (the node uses an env copy).
    """
    try:
        parts = key.strip().split(".")
        if len(parts) != 3 or parts[0] != KEY_PREFIX:
            return {"valid": False, "reason": "malformed key", "license": None}
        payload_bytes = _b64url_decode(parts[1])
        sig = _b64url_decode(parts[2])
        expected = hmac.new((secret or get_signing_secret()).encode("utf-8"),
                            payload_bytes, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return {"valid": False, "reason": "bad signature", "license": None}
        lic = License.from_payload(json.loads(payload_bytes.decode("utf-8")))
        if spec is not None and not hmac.compare_digest(lic.subject, checksum_spec(spec)):
            return {"valid": False, "reason": "license does not cover this kernel",
                    "license": lic}
        now_ts = int(now if now is not None else time.time())
        if lic.expires_at is not None and now_ts > lic.expires_at:
            return {"valid": False, "reason": "license expired", "license": lic}
        if require_entitlement and require_entitlement not in lic.entitlements:
            return {"valid": False, "reason": f"missing entitlement: {require_entitlement}",
                    "license": lic}
        return {"valid": True, "reason": None, "license": lic}
    except Exception as e:  # noqa: BLE001 - verification must never raise
        return {"valid": False, "reason": f"error: {e}", "license": None}
