# Marketplace CLI: `calibrix-marketplace <command>`.
#
# Offline-first: `seed` + `serve` work with the mock provider and zero keys;
# setting STRIPE_SECRET_KEY + STRIPE_WEBHOOK_SECRET switches to real Stripe.

from __future__ import annotations

import argparse
import json
import sys
from typing import List

from .licenses import verify_license
from .server import serve
from .store import JsonStore, Listing, OrderStatus
from .stripe_provider import build_mock_event, make_provider


def _demo_listings() -> List[Listing]:
    return [
        Listing(
            listing_id="flux-cinematic-v1",
            title="FLUX Cinematic Look",
            model="FLUX.1-dev",
            kernel_spec="attn:1.14@0.42:0.86:0.35|mlp:1.06@0.55:0.92:0.28",
            price_cents=900,
            description=("Contrast-forward attention kernel tuned for "
                         "cinematic composition. 30 NFE, PickScore +4.2%."),
            scoring={"pickscore": 0.221, "imagereward": 0.912},
        ),
        Listing(
            listing_id="qwen-crisp-v1",
            title="Qwen-Image Crisp Text",
            model="Qwen-Image",
            kernel_spec="attn:1.22@0.31:0.90:0.22|mlp:1.10@0.62:0.95:0.4",
            price_cents=1200,
            description=("Sharpens glyph fidelity for poster/text-heavy "
                         "generations. 30 NFE, HPSv3 +5.1%."),
            scoring={"hpsv3": 0.841},
        ),
        Listing(
            listing_id="sd35-pastel-v1",
            title="SD3.5 Pastel Palette",
            model="SD3.5-Medium",
            kernel_spec="attn:0.94@0.68:0.88:0.3|mlp:1.02@0.5:1.0:1e6",
            price_cents=700,
            description=("Soft color palette emphasis with restrained "
                         "contrast. 15 NFE, aesthetic +6.8%."),
            scoring={"aesthetic": 0.603},
        ),
    ]


def cmd_seed(args: argparse.Namespace) -> int:
    store = JsonStore(args.store)
    if store.list_listings(active_only=False):
        print(f"store {args.store} already has listings; not seeding again")
        return 0
    for l in _demo_listings():
        store.add_listing(l)
        print(f"  + listing {l.listing_id}  ${l.price_cents / 100:.2f}  {l.title}")
    print(f"seeded {args.store}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    store = JsonStore(args.store)
    provider = make_provider()
    print(f"store: {args.store} | provider: {provider.name}")
    serve(store, provider, host=args.host, port=args.port)
    return 0


def cmd_order(args: argparse.Namespace) -> int:
    """Create a pending order and print a signed mock webhook to fulfill it."""
    store = JsonStore(args.store)
    listing = store.get_listing(args.listing)
    if listing is None:
        print(f"error: no listing {args.listing!r}", file=sys.stderr)
        return 1
    provider = make_provider()
    order = store.create_order(listing, args.email, provider.name)
    session = provider.create_checkout(
        listing, order, success_url="http://localhost/success",
        cancel_url="http://localhost/",
    )
    store.attach_session(order, session["session_id"])
    print(json.dumps({
        "order_id": order.order_id,
        "session_id": session["session_id"],
        "amount_cents": order.amount_cents,
    }, indent=2))
    if provider.name == "mock":
        payload, sig = build_mock_event(session["session_id"])
        print("\n# fulfill with:\n"
              f"curl -s -X POST http://localhost:{args.port}/webhook "
              f"-H 'X-Mock-Signature: {sig}' "
              f"--data-binary @- <<'EOF'\n{payload.decode()}\nEOF")
    return 0


def cmd_fulfill(args: argparse.Namespace) -> int:
    store = JsonStore(args.store)
    order = store.get_order(args.order)
    if order is None:
        print(f"error: no order {args.order!r}", file=sys.stderr)
        return 1
    order = store.fulfill_order(order, ttl_days=args.ttl_days)
    print(json.dumps({
        "order_id": order.order_id,
        "status": order.status,
        "license_id": order.license_id,
        "license_key": order.license_key,
    }, indent=2))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    result = verify_license(args.key, spec=args.spec,
                            require_entitlement=args.entitlement)
    print(json.dumps({
        "valid": result["valid"],
        "reason": result["reason"],
        "license_id": result["license"].license_id if result["license"] else None,
        "buyer": result["license"].buyer if result["license"] else None,
    }, indent=2))
    return 0 if result["valid"] else 1


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="calibrix-marketplace",
        description="Sell Calibrix kernels: listings, checkout, licenses",
    )
    p.add_argument("--store", default="marketplace_data.json",
                   help="path to the JSON store file")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --store accepted at top level OR after the subcommand (SUPPRESS keeps
    # the parent-level value when the subparser doesn't receive it).
    s = sub.add_parser("seed", help="create demo listings")
    s.add_argument("--store", default=argparse.SUPPRESS)
    s.set_defaults(fn=cmd_seed)

    sv = sub.add_parser("serve", help="run the storefront + webhook server")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8700)
    sv.add_argument("--store", default=argparse.SUPPRESS)
    sv.set_defaults(fn=cmd_serve)

    o = sub.add_parser("order", help="create an order (prints mock webhook curl)")
    o.add_argument("--listing", required=True)
    o.add_argument("--email", required=True)
    o.add_argument("--port", type=int, default=8700,
                   help="port used in the printed webhook curl")
    o.add_argument("--store", default=argparse.SUPPRESS)
    o.set_defaults(fn=cmd_order)

    f = sub.add_parser("fulfill", help="issue the license for an order")
    f.add_argument("--order", required=True)
    f.add_argument("--ttl-days", type=int, default=None)
    f.add_argument("--store", default=argparse.SUPPRESS)
    f.set_defaults(fn=cmd_fulfill)

    v = sub.add_parser("verify", help="verify a license key")
    v.add_argument("--key", required=True)
    v.add_argument("--spec", default=None,
                   help="kernel spec the license must cover")
    v.add_argument("--entitlement", default=None)
    v.add_argument("--store", default=argparse.SUPPRESS)
    v.set_defaults(fn=cmd_verify)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
