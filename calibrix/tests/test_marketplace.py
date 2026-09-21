import json
import unittest
from unittest import mock

from calibrix.marketplace.licenses import (
    checksum_spec,
    get_signing_secret,
    issue_license,
    verify_license,
)
from calibrix.marketplace.store import JsonStore, Listing, OrderStatus
from calibrix.marketplace.stripe_provider import (
    MockStripeProvider,
    PaymentError,
    build_mock_event,
    make_provider,
)
from calibrix.marketplace.server import MarketplaceServer

SPEC = "attn:1.14@0.42:0.86:0.35|mlp:1.06@0.55:0.92:0.28"


class TestLicenses(unittest.TestCase):
    def test_issue_and_verify_roundtrip(self):
        key, lic = issue_license(SPEC, buyer="a@b.co", product="flux-x",
                                 entitlements=["comfyui", "commercial"])
        res = verify_license(key, spec=SPEC)
        self.assertTrue(res["valid"])
        self.assertEqual(res["license"].license_id, lic.license_id)
        self.assertEqual(res["license"].buyer, "a@b.co")
        self.assertEqual(res["license"].subject, checksum_spec(SPEC))

    def test_wrong_spec_rejected(self):
        key, _ = issue_license(SPEC, buyer="a@b.co", product="p")
        res = verify_license(key, spec="attn:0.5@0.5:1.0:1e6")
        self.assertFalse(res["valid"])
        self.assertEqual(res["reason"], "license does not cover this kernel")

    def test_tampered_key_rejected(self):
        key, _ = issue_license(SPEC, buyer="a@b.co", product="p")
        head, payload, sig = key.split(".")
        # flip payload content (re-encode with different buyer)
        import base64

        forged = json.dumps({"v": 1, "lid": "stolen", "sub": "x", "ent": [],
                             "iat": 0, "exp": None}).encode()
        forged_key = head + "." + base64.urlsafe_b64encode(forged).decode().rstrip("=") + "." + sig
        res = verify_license(forged_key)
        self.assertFalse(res["valid"])
        self.assertEqual(res["reason"], "bad signature")

    def test_expired_license_rejected(self):
        key, _ = issue_license(SPEC, buyer="a@b.co", product="p", ttl_days=1)
        far_future = int(__import__("time").time()) + 10 * 86400
        res = verify_license(key, now=far_future)
        self.assertFalse(res["valid"])
        self.assertEqual(res["reason"], "license expired")

    def test_entitlement_required(self):
        key, _ = issue_license(SPEC, buyer="a@b.co", product="p",
                               entitlements=["comfyui"])
        res = verify_license(key, require_entitlement="commercial")
        self.assertFalse(res["valid"])
        self.assertIn("missing entitlement", res["reason"])

    def test_dev_secret_fallback(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertTrue(get_signing_secret().startswith("calibrix-dev-secret"))


class TestStore(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.store = JsonStore(self.tmp.name)
        self.listing = self.store.add_listing(Listing(
            listing_id="test-kernel",
            title="Test Kernel",
            model="FLUX.1-dev",
            kernel_spec=SPEC,
            price_cents=900,
            description="d",
            scoring={"pickscore": 0.2},
        ))

    def tearDown(self):
        import os

        os.unlink(self.tmp.name)

    def test_listing_checksum_filled(self):
        self.assertEqual(self.listing.spec_checksum, checksum_spec(SPEC))
        again = self.store.get_listing("test-kernel")
        self.assertEqual(again.spec_checksum, checksum_spec(SPEC))

    def test_order_lifecycle(self):
        order = self.store.create_order(self.listing, "b@c.co", "mock")
        self.assertEqual(order.status, OrderStatus.PENDING)
        order = self.store.fulfill_order(order)
        self.assertEqual(order.status, OrderStatus.FULFILLED)
        self.assertTrue(order.license_key.startswith("CBX1."))
        # license covers the listing's kernel
        res = verify_license(order.license_key, spec=SPEC)
        self.assertTrue(res["valid"])
        # persisted
        reloaded = self.store.get_order(order.order_id)
        self.assertEqual(reloaded.status, OrderStatus.FULFILLED)
        # idempotent
        same = self.store.fulfill_order(reloaded)
        self.assertEqual(same.license_key, order.license_key)

    def test_lookup_by_session(self):
        order = self.store.create_order(self.listing, "b@c.co", "mock")
        self.assertIsNone(self.store.get_order_by_session("nope"))
        self.store.attach_session(order, "cs_mock_abc")
        found = self.store.get_order_by_session("cs_mock_abc")
        self.assertEqual(found.order_id, order.order_id)


class TestMockProvider(unittest.TestCase):
    def test_signed_webhook_verification(self):
        provider = MockStripeProvider()
        payload, sig = build_mock_event("cs_mock_x")
        event = provider.verify_webhook(payload, sig)
        self.assertEqual(event["type"], "checkout.session.completed")
        self.assertEqual(event["session_id"], "cs_mock_x")

    def test_bad_signature_raises(self):
        provider = MockStripeProvider()
        payload, _ = build_mock_event("cs_mock_x")
        with self.assertRaises(PaymentError):
            provider.verify_webhook(payload, "deadbeef")

    def test_make_provider_falls_back_to_mock(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsInstance(make_provider(), MockStripeProvider)


class TestServer(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        self.store = JsonStore(self.tmp.name)
        self.listing = self.store.add_listing(Listing(
            listing_id="srv-kernel",
            title="Server Test",
            model="Qwen-Image",
            kernel_spec=SPEC,
            price_cents=500,
        ))
        self.provider = MockStripeProvider()
        self.server = MarketplaceServer(self.store, self.provider)

    def tearDown(self):
        import os

        os.unlink(self.tmp.name)

    def test_webhook_fulfills_order(self):
        order = self.store.create_order(self.listing, "b@c.co", "mock")
        session = self.provider.create_checkout(
            self.listing, order, "http://x/success", "http://x/")
        self.store.attach_session(order, session["session_id"])
        payload, sig = build_mock_event(session["session_id"])
        status, body = self.server.handle_webhook(payload, sig)
        self.assertEqual(status, 200)
        out = json.loads(body)
        self.assertEqual(out["status"], "ok")
        fulfilled = self.store.get_order(order.order_id)
        self.assertEqual(fulfilled.status, OrderStatus.FULFILLED)
        self.assertTrue(verify_license(fulfilled.license_key, spec=SPEC)["valid"])

    def test_webhook_rejects_bad_signature(self):
        status, _ = self.server.handle_webhook(b"{}", "nope")
        self.assertEqual(status, 400)

    def test_webhook_idempotent(self):
        order = self.store.create_order(self.listing, "b@c.co", "mock")
        session = self.provider.create_checkout(
            self.listing, order, "http://x/success", "http://x/")
        self.store.attach_session(order, session["session_id"])
        payload, sig = build_mock_event(session["session_id"])
        self.server.handle_webhook(payload, sig)
        status1, body1 = self.server.handle_webhook(payload, sig)
        status2, body2 = self.server.handle_webhook(payload, sig)
        self.assertEqual(status1, 200)
        self.assertEqual(status2, 200)
        self.assertEqual(json.loads(body1)["license_id"],
                         json.loads(body2)["license_id"])

    def test_storefront_renders(self):
        html = self.server.render_storefront()
        self.assertIn("Server Test", html)
        self.assertIn("$5.00", html)

    def test_checkout_validates_email(self):
        status, _, body = self.server.handle_checkout(
            {"listing_id": "srv-kernel", "email": "not-an-email"})
        self.assertEqual(status, 400)
        status, _, body = self.server.handle_checkout(
            {"listing_id": "missing", "email": "a@b.co"})
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()
