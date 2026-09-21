# Calibrix marketplace payments: Stripe Checkout + offline mock.
#
# The real provider uses stripe-python (optional extra `stripe`). The mock
# mirrors its interface so the whole marketplace can be developed, tested,
# and demoed with zero network access and zero keys. `make_provider` picks
# based on availability of the key + package.

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional


class PaymentError(RuntimeError):
    pass


class BaseProvider:
    name: str = "base"

    def create_checkout(self, listing, order, success_url: str,
                        cancel_url: str) -> Dict[str, Any]:
        """Return {"session_id": str, "url": str}."""
        raise NotImplementedError

    def verify_webhook(self, payload: bytes, signature: str) -> Dict[str, Any]:
        """Return normalized event {"type": str, "session_id": str}."""
        raise NotImplementedError


class StripeProvider(BaseProvider):
    """Stripe Checkout Sessions + webhook signature verification."""

    name = "stripe"

    def __init__(self, api_key: str, webhook_secret: str) -> None:
        try:
            import stripe  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise PaymentError(
                "stripe package not installed; run `pip install calibrix[stripe]`"
            ) from e
        self._stripe = stripe
        stripe.api_key = api_key
        self.webhook_secret = webhook_secret

    def create_checkout(self, listing, order, success_url: str,
                        cancel_url: str) -> Dict[str, Any]:
        stripe = self._stripe
        session = stripe.checkout.Session.create(
            mode="payment",
            customer_email=order.buyer_email,
            client_reference_id=order.order_id,
            line_items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {
                        "name": listing.title,
                        "description": (listing.description or "")[:300],
                    },
                    "unit_amount": listing.price_cents,
                },
                "quantity": 1,
            }],
            metadata={"listing_id": listing.listing_id,
                      "order_id": order.order_id},
            success_url=success_url,
            cancel_url=cancel_url,
        )
        return {"session_id": session.id, "url": session.url}

    def verify_webhook(self, payload: bytes, signature: str) -> Dict[str, Any]:
        stripe = self._stripe
        event = stripe.Webhook.construct_event(
            payload, signature, self.webhook_secret
        )
        etype = event["type"]
        if etype == "checkout.session.completed":
            return {"type": etype,
                    "session_id": event["data"]["object"]["id"]}
        return {"type": etype, "session_id": None}


class MockStripeProvider(BaseProvider):
    """Offline provider: deterministic fake sessions + signed webhooks.

    Webhook signatures use the same HMAC scheme as the license keys so tests
    exercise real verification code paths.
    """

    name = "mock"

    def __init__(self, webhook_secret: str = "mock-webhook-secret") -> None:
        self.webhook_secret = webhook_secret

    def create_checkout(self, listing, order, success_url: str,
                        cancel_url: str) -> Dict[str, Any]:
        sid = "cs_mock_" + uuid.uuid4().hex[:12]
        return {"session_id": sid, "url": f"{success_url}#mock-checkout:{sid}"}

    def verify_webhook(self, payload: bytes, signature: str) -> Dict[str, Any]:
        import hashlib
        import hmac as _hmac

        expected = _hmac.new(self.webhook_secret.encode("utf-8"),
                             payload, hashlib.sha256).hexdigest()
        if not _hmac.compare_digest(expected, signature or ""):
            raise PaymentError("invalid webhook signature")
        event = __import__("json").loads(payload.decode("utf-8"))
        return {"type": event.get("type", ""),
                "session_id": (event.get("data", {}).get("object", {}) or {}).get("id")}


def make_provider(secret: Optional[str] = None) -> BaseProvider:
    """Choose Stripe when configured, else the offline mock."""
    api_key = os.environ.get("STRIPE_SECRET_KEY", "")
    wh_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", secret or "")
    if api_key:
        try:
            return StripeProvider(api_key, wh_secret or api_key)
        except PaymentError:
            pass  # fall through to mock when stripe is unavailable
    return MockStripeProvider()


def build_mock_event(session_id: str, etype: str = "checkout.session.completed",
                     webhook_secret: str = "mock-webhook-secret") -> tuple[bytes, str]:
    """Helper for tests/CLIs: construct a signed mock webhook payload."""
    payload = __import__("json").dumps(
        {"type": etype, "data": {"object": {"id": session_id}}},
        separators=(",", ":"),
    ).encode("utf-8")
    import hashlib
    import hmac as _hmac

    sig = _hmac.new(webhook_secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return payload, sig
