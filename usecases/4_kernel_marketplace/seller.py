# UC-4 marketplace seller console (self-contained).
#
# Wraps the calibrix.marketplace core into a seller workflow: stock listings
# from kernel exports, serve the storefront, complete mock/real payments,
# and reconcile revenue. All state lives in ./store.json inside this folder.

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "calibrix"))

from calibrix.marketplace.licenses import verify_license
from calibrix.marketplace.server import serve
from calibrix.marketplace.stripe_provider import (
    build_mock_event,
    make_provider,
)
from calibrix.marketplace.store import JsonStore, Listing, OrderStatus

HERE = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(HERE, "store.json")


def get_store() -> JsonStore:
    return JsonStore(STORE_PATH)


def cmd_stock(args) -> int:
    store = get_store()
    added = 0
    paths = []
    for d in args.from_dir:
        if os.path.isdir(d):
            paths.append(os.path.join(d, "listing.json"))
        else:
            paths.append(d)
    for path in paths:
        if not os.path.isfile(path):
            print(f"skip (missing): {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        listing = Listing(**{k: data[k] for k in Listing.__dataclass_fields__
                             if k in data})
        store.add_listing(listing)
        print(f"  + {listing.listing_id}  ${listing.price_cents / 100:.2f}  {listing.title}")
        added += 1
    print(f"stocked {added} listing(s) into {STORE_PATH}")
    return 0


def cmd_listings(args) -> int:
    store = get_store()
    for l in store.list_listings(active_only=not args.all):
        status = "" if l.active else " [inactive]"
        print(f"{l.listing_id:28s} ${l.price_cents / 100:>6.2f}  {l.title}{status}")
    return 0


def cmd_serve(args) -> int:
    store = get_store()
    provider = make_provider()
    print(f"provider: {provider.name} | store: {STORE_PATH}")
    serve(store, provider, host=args.host, port=args.port)
    return 0


def cmd_buy(args) -> int:
    store = get_store()
    listing = store.get_listing(args.listing)
    if listing is None:
        print(f"no listing {args.listing!r}", file=sys.stderr)
        return 1
    provider = make_provider()
    order = store.create_order(listing, args.email, provider.name)
    session = provider.create_checkout(
        listing, order,
        success_url=f"http://localhost:{args.port}/success",
        cancel_url=f"http://localhost:{args.port}/",
    )
    store.attach_session(order, session["session_id"])
    print(json.dumps({"order_id": order.order_id,
                      "session_id": session["session_id"],
                      "amount_cents": order.amount_cents}, indent=2))
    if provider.name == "mock":
        payload, sig = build_mock_event(session["session_id"])
        print("\n# complete payment (mock webhook):")
        print(f'curl -s -X POST http://localhost:{args.port}/webhook '
              f"-H 'X-Mock-Signature: {sig}' --data-binary '{payload.decode()}'")
        print("# or: python seller.py webhook "
              f"--session {session['session_id']}")
    return 0


def cmd_webhook(args) -> int:
    """Complete a mock payment without running curl."""
    store = get_store()
    provider = make_provider()
    order = store.get_order_by_session(args.session)
    if order is None:
        print(f"no order for session {args.session!r}", file=sys.stderr)
        return 1
    if provider.name != "mock":
        print("webhook helper is mock-only; real Stripe posts to /webhook",
              file=sys.stderr)
        return 1
    payload, sig = build_mock_event(args.session)
    from calibrix.marketplace.server import MarketplaceServer

    server = MarketplaceServer(store, provider)
    status, body = server.handle_webhook(payload, sig)
    print(f"[{status}] {body}")
    if status == 200:
        order = store.get_order(order.order_id)
        res = verify_license(order.license_key,
                             spec=store.get_listing(order.listing_id).kernel_spec)
        print(f"license valid: {res['valid']} | id: {order.license_id}")
        print(f"license key :\n{order.license_key}")
    return 0


def cmd_demo(args) -> int:
    """One-shot offline sale cycle: stock -> order -> mock webhook ->
    license -> revenue. Uses a throwaway store so CI stays idempotent."""
    store_path = os.path.join(HERE, "_demo_store.json")
    if os.path.exists(store_path):
        os.remove(store_path)
    store = JsonStore(store_path)

    listing = Listing(
        listing_id="uc4-demo-kernel",
        title="Demo Domain Kernel (offline)",
        model="FLUX.1-dev",
        kernel_spec="demo|seed=7|gamma=1.25|nfe=4",
        price_cents=1200,
        description="Emitted by seller.py demo",
    )
    store.add_listing(listing)
    print(f"stocked 1 listing: {listing.listing_id} "
          f"(${listing.price_cents / 100:.2f})")

    provider = make_provider()
    assert provider.name == "mock", "demo requires offline mock provider"
    order = store.create_order(listing, "buyer@example.com", provider.name)
    session = provider.create_checkout(
        listing, order,
        success_url=f"http://localhost:{args.port}/success",
        cancel_url=f"http://localhost:{args.port}/",
    )
    store.attach_session(order, session["session_id"])
    print(f"checkout session created for order {order.order_id} "
          f"(${order.amount_cents / 100:.2f})")

    payload, sig = build_mock_event(session["session_id"])
    from calibrix.marketplace.server import MarketplaceServer
    server = MarketplaceServer(store, provider)
    status, body = server.handle_webhook(payload, sig)
    if status != 200:
        print(f"webhook failed: [{status}] {body}", file=sys.stderr)
        return 1
    print(f"webhook accepted: {body}")

    order = store.get_order(order.order_id)
    res = verify_license(order.license_key,
                         spec=store.get_listing(order.listing_id).kernel_spec)
    lic = res.get("license")
    print(f"license verified: valid={res['valid']} "
          f"buyer={getattr(lic, 'buyer', 'n/a')}")
    if not res["valid"]:
        return 1

    orders = list(store._data["orders"].values())
    gross = sum(o["amount_cents"] for o in orders
                if o["status"] in (OrderStatus.PAID, OrderStatus.FULFILLED))
    print(f"revenue: {len(orders)} order(s), gross ${gross / 100:.2f}")
    os.remove(store_path)
    print("UC-4 demo sale cycle: OK")
    return 0


def cmd_revenue(args) -> int:
    store = get_store()
    orders = list(store._data["orders"].values())
    by_status = {}
    gross_cents = 0
    per_listing: dict = {}
    for o in orders:
        by_status[o["status"]] = by_status.get(o["status"], 0) + 1
        if o["status"] in (OrderStatus.PAID, OrderStatus.FULFILLED):
            gross_cents += o["amount_cents"]
            entry = per_listing.setdefault(
                o["listing_id"], {"units": 0, "gross_cents": 0})
            entry["units"] += 1
            entry["gross_cents"] += o["amount_cents"]
    print(json.dumps({
        "orders_by_status": by_status,
        "gross_usd": round(gross_cents / 100, 2),
        "per_listing": {k: {"units": v["units"],
                            "gross_usd": round(v["gross_cents"] / 100, 2)}
                        for k, v in sorted(per_listing.items())},
    }, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="UC-4 kernel marketplace seller")
    p.add_argument("--store", default=STORE_PATH)
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("stock", help="add listings from listing.json files/dirs")
    s.add_argument("--from-dir", nargs="+", required=True)
    s.set_defaults(fn=cmd_stock)

    l = sub.add_parser("listings", help="list inventory")
    l.add_argument("--all", action="store_true")
    l.set_defaults(fn=cmd_listings)

    sv = sub.add_parser("serve", help="run storefront + webhook server")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8700)
    sv.set_defaults(fn=cmd_serve)

    b = sub.add_parser("buy", help="simulate a buyer checkout")
    b.add_argument("--listing", required=True)
    b.add_argument("--email", required=True)
    b.add_argument("--port", type=int, default=8700)
    b.set_defaults(fn=cmd_buy)

    w = sub.add_parser("webhook", help="complete a mock payment by session id")
    w.add_argument("--session", required=True)
    w.set_defaults(fn=cmd_webhook)

    r = sub.add_parser("revenue", help="reconcile the order ledger")
    r.set_defaults(fn=cmd_revenue)

    dm = sub.add_parser("demo", help="run the full offline sale cycle once")
    dm.add_argument("--port", type=int, default=8700)
    dm.set_defaults(fn=cmd_demo)

    args = p.parse_args()
    if args.cmd is None:
        # Bare invocation (scripts/CI) defaults to the offline demo cycle.
        print("no subcommand given; running the offline demo (see --help)")
        args.cmd, args.fn, args.port = "demo", cmd_demo, 8700
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
