# SPDX-License-Identifier: MIT
# Calibrix product-photo kernel marketplace HTTP server (stdlib only).
#
# Routes:
#   GET  /                         -> focused catalog + filters
#   GET  /listing/<id>             -> product detail + buyer onboarding
#   GET  /api/listings             -> filtered JSON snapshot (CORS *)
#   POST /checkout                 -> create order + checkout session
#   GET  /success?order=<id>       -> fulfillment status + delivered license
#   POST /webhook                  -> payment webhook (Stripe or mock)

from __future__ import annotations

import html
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

from .store import JsonStore, Listing, OrderStatus
from .stripe_provider import BaseProvider, PaymentError


class MarketplaceServer:
    def __init__(self, store: JsonStore, provider: BaseProvider,
                 success_url: str = "http://localhost:8700/success") -> None:
        self.store = store
        self.provider = provider
        self.success_url = success_url

    # ------------------------------------------------------------------
    def handle_checkout(self, form: Dict[str, str]) -> Tuple[int, str, str]:
        """Returns (status, content_type, body). 302 Location is sent by handler."""
        listing = self.store.get_listing(form.get("listing_id", ""))
        if listing is None or not listing.active:
            return 404, "text/plain", "unknown listing"
        email = (form.get("email") or "").strip()
        if "@" not in email:
            return 400, "text/plain", "valid email required"
        order = self.store.create_order(listing, email, self.provider.name)
        session = self.provider.create_checkout(
            listing, order,
            success_url=self.success_url + f"?order={order.order_id}",
            cancel_url=self.success_url.replace("/success", "/"),
        )
        self.store.attach_session(order, session["session_id"])
        return 302, "text/plain", session["url"]

    def handle_webhook(self, payload: bytes, signature: str) -> Tuple[int, str]:
        """Verify payment and issue the offline-verifiable license exactly once."""
        try:
            event = self.provider.verify_webhook(payload, signature)
        except PaymentError as e:
            return 400, str(e)
        if event["type"] != "checkout.session.completed" or not event["session_id"]:
            return 200, "ignored"
        order = self.store.get_order_by_session(event["session_id"])
        if order is None:
            return 404, "order not found"
        if order.status == OrderStatus.PENDING:
            self.store.mark_paid(order)
        if order.status in (OrderStatus.PAID, OrderStatus.FULFILLED):
            order = self.store.fulfill_order(order)
        return 200, json.dumps({"status": "ok", "order_id": order.order_id,
                                "license_id": order.license_id})

    # ------------------------------------------------------------------
    @staticmethod
    def _category(listing: Listing) -> str:
        return (listing.category or "product-photo").strip().lower()

    @staticmethod
    def _matches(listing: Listing, query: str, category: str,
                 evidence: str) -> bool:
        haystack = " ".join([
            listing.title, listing.model, listing.description,
            MarketplaceServer._category(listing),
            " ".join(listing.tags),
        ]).lower()
        query_ok = not query or query.lower() in haystack
        category_ok = category in ("", "all") or MarketplaceServer._category(listing) == category.lower()
        evidence_ok = evidence in ("", "all") or listing.evidence_status == evidence.lower()
        return query_ok and category_ok and evidence_ok

    def filtered_listings(self, query: str = "", category: str = "product-photo",
                          evidence: str = "all") -> List[Listing]:
        return [
            listing for listing in self.store.list_listings(active_only=True)
            if self._matches(listing, query.strip(), category.strip(), evidence.strip())
        ]

    @staticmethod
    def _esc(value: Any) -> str:
        return html.escape(str(value), quote=True)

    @staticmethod
    def _fill_template(template: str, values: Dict[str, str]) -> str:
        """Resolve placeholders while restoring doubled CSS braces."""
        sentinels = {}
        rendered = template
        for key, value in values.items():
            marker = f"\x00CALIBRIX_{key.upper()}\x00"
            sentinels[marker] = value
            rendered = rendered.replace("{{" + key + "}}", marker)
        rendered = rendered.replace("{{", "{").replace("}}", "}")
        for marker, value in sentinels.items():
            rendered = rendered.replace(marker, value)
        return rendered

    def _evidence_html(self, listing: Listing) -> str:
        label = listing.evidence_status.replace("_", " ").upper()
        css = {
            "VERIFIED": "verified",
            "MEASURED": "measured",
            "SIMULATION": "simulation",
            "REFERENCE ONLY": "reference",
        }.get(label, "measured")
        note = listing.evidence_note or {
            "VERIFIED": "Real-model holdout passed",
            "MEASURED": "Real outputs recorded; promotion gate not passed",
            "SIMULATION": "Synthetic adapter only",
            "REFERENCE ONLY": "Reference result, not a local measurement",
        }.get(label, "Evidence status recorded")
        return (
            f"<div class='evidence {css}'><strong>{self._esc(label)}</strong> "
            f"{self._esc(note)}</div>"
        )

    def _score_html(self, listing: Listing) -> str:
        scores = []
        for name, value in listing.scoring.items():
            try:
                scores.append(
                    f"<span class='metric'><b>{self._esc(name)}</b> "
                    f"{float(value):.3f}</span>"
                )
            except (TypeError, ValueError):
                scores.append(f"<span class='metric'><b>{self._esc(name)}</b> {self._esc(value)}</span>")
        if listing.holdout_score is not None and listing.baseline_score is not None:
            delta = listing.holdout_score - listing.baseline_score
            scores.append(
                f"<span class='metric'><b>Holdout Δ</b> {delta:+.3f}</span>"
            )
        return "".join(scores) or "<span class='muted'>No score package attached</span>"

    def _card_html(self, listing: Listing) -> str:
        price = f"${listing.price_cents / 100:.2f}"
        models = listing.compatible_models or [listing.model]
        tags = "".join(f"<span class='tag'>{self._esc(tag)}</span>" for tag in listing.tags)
        compatibility = ", ".join(self._esc(model) for model in models)
        return f"""
<article class="product-card">
  <div class="card-top"><span class="category">PRODUCT PHOTO</span><span class="price">{price}</span></div>
  <h2>{self._esc(listing.title)}</h2>
  <p class="model">{self._esc(listing.model)} · portable ComfyUI kernel</p>
  <p>{self._esc(listing.description)}</p>
  {self._evidence_html(listing)}
  <div class="metrics">{self._score_html(listing)}</div>
  <div class="compat"><b>Works with:</b> {compatibility}</div>
  <div class="tags">{tags}</div>
  <div class="card-actions">
    <a class="secondary" href="/listing/{urllib.parse.quote(listing.listing_id)}">See proof &amp; install</a>
    <form method="post" action="/checkout">
      <input type="hidden" name="listing_id" value="{self._esc(listing.listing_id)}">
      <label class="sr-only" for="email-{self._esc(listing.listing_id)}">Email for license delivery</label>
      <input id="email-{self._esc(listing.listing_id)}" type="email" name="email" placeholder="you@example.com" required>
      <button type="submit">Buy {price}</button>
    </form>
  </div>
</article>"""

    def render_storefront(self, query: str = "", category: str = "product-photo",
                          evidence: str = "all") -> str:
        all_listings = self.store.list_listings(active_only=True)
        listings = self.filtered_listings(query, category, evidence)
        categories = sorted({self._category(listing) for listing in all_listings})
        category_options = "<option value='all'>All categories</option>" + "".join(
            f"<option value='{self._esc(item)}' {'selected' if category == item else ''}>{self._esc(item.replace('-', ' ').title())}</option>"
            for item in categories
        )
        evidence_options = "".join(
            f"<option value='{item}' {'selected' if evidence == item else ''}>{label}</option>"
            for item, label in (("all", "All evidence"), ("verified", "Verified only"),
                                ("measured", "Measured"), ("simulation", "Simulation"),
                                ("reference_only", "Reference only"))
        )
        cards = "".join(self._card_html(listing) for listing in listings)
        if not cards:
            cards = "<div class='empty'><h2>No matching product kernels</h2><p>Try clearing the search or selecting All categories.</p></div>"
        return self._fill_template(_PAGE, {
            "cards": cards,
            "count": str(len(listings)),
            "query": self._esc(query),
            "category_options": category_options,
            "evidence_options": evidence_options,
            "active_category": self._esc(category.replace('-', ' ').title()),
        })

    def render_listing_detail(self, listing_id: str) -> Optional[str]:
        listing = self.store.get_listing(listing_id)
        if listing is None or not listing.active:
            return None
        models = ", ".join(self._esc(model) for model in (listing.compatible_models or [listing.model]))
        tags = "".join(f"<span class='tag'>{self._esc(tag)}</span>" for tag in listing.tags)
        price = f"${listing.price_cents / 100:.2f}"
        return self._fill_template(_DETAIL_PAGE, {
            "title": self._esc(listing.title),
            "listing_id": self._esc(listing.listing_id),
            "model": self._esc(listing.model),
            "description": self._esc(listing.description),
            "price": price,
            "evidence": self._evidence_html(listing),
            "metrics": self._score_html(listing),
            "models": models,
            "tags": tags,
        })

    def render_success(self, order_id: str) -> Tuple[int, str]:
        order = self.store.get_order(order_id)
        if order is None:
            return 404, "<h1>Order not found</h1><p>Check the order link from checkout.</p>"
        listing = self.store.get_listing(order.listing_id)
        title = self._esc(listing.title if listing else order.listing_id)
        if order.status == OrderStatus.FULFILLED and order.license_key:
            body = f"""<h1>You're ready to install {title}</h1>
<p>Payment is fulfilled. Copy this license key into the CalibrixKernelScale node in ComfyUI.</p>
<pre class="license">{self._esc(order.license_key)}</pre>
<div class="onboarding"><b>Install checklist</b><ol><li>Copy <code>calibrix_node.py</code> into ComfyUI/custom_nodes.</li><li>Restart ComfyUI and load your compatible checkpoint.</li><li>Add Calibrix Kernel Scale after the checkpoint loader.</li><li>Paste the key above and use the purchased kernel's exact model.</li></ol></div>
<p class="muted">Order {self._esc(order.order_id)} · License {self._esc(order.license_id)}</p>"""
        else:
            body = f"""<h1>Payment received for {title}</h1>
<p>Your payment is recorded and fulfillment is still processing. Refresh this page after the payment webhook completes.</p>
<p class="muted">Order {self._esc(order.order_id)} · current status: {self._esc(order.status)}</p>"""
        return 200, self._fill_template(_SUCCESS_PAGE, {"body": body})

    def handle_api_listings(self, query: str = "", category: str = "product-photo",
                            evidence: str = "all") -> Tuple[int, str]:
        """Return filtered storefront data without kernel specs or licenses."""
        listings = self.filtered_listings(query, category, evidence)
        all_orders = list(self.store._data["orders"].values())
        paid = (OrderStatus.PAID, OrderStatus.FULFILLED)
        listing_ids = {listing.listing_id for listing in listings}
        orders = [order for order in all_orders if order["listing_id"] in listing_ids]
        gross_cents = sum(order["amount_cents"] for order in orders if order["status"] in paid)
        payload = {
            "filters": {"q": query, "category": category, "evidence": evidence},
            "listings": [
                {
                    "listing_id": listing.listing_id,
                    "title": listing.title,
                    "model": listing.model,
                    "category": self._category(listing),
                    "tags": listing.tags,
                    "compatible_models": listing.compatible_models or [listing.model],
                    "price_cents": listing.price_cents,
                    "price_usd": round(listing.price_cents / 100, 2),
                    "description": listing.description,
                    "vector": (listing.kernel_spec.split("|")[0].split(":")[0]
                               if listing.kernel_spec else ""),
                    "scoring": listing.scoring,
                    "evidence_status": listing.evidence_status,
                    "evidence_note": listing.evidence_note,
                    "report_path": listing.report_path,
                    "holdout_score": listing.holdout_score,
                    "baseline_score": listing.baseline_score,
                    "quality_gate": listing.quality_gate,
                    "active": listing.active,
                    "created_at": listing.created_at,
                    "orders": sum(1 for order in all_orders if order["listing_id"] == listing.listing_id),
                    "revenue_cents": sum(order["amount_cents"] for order in all_orders
                                          if order["listing_id"] == listing.listing_id
                                          and order["status"] in paid),
                }
                for listing in listings
            ],
            "stats": {
                "listing_count": len(listings),
                "active_count": len(listings),
                "order_count": len(orders),
                "gross_cents": gross_cents,
                "gross_usd": round(gross_cents / 100, 2),
            },
        }
        return 200, json.dumps(payload)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Calibrix Product Photo Kernels</title>
<style>
:root {{--bg:#0b1020;--panel:#121a2b;--panel2:#18233a;--ink:#edf3ff;--muted:#9aa9c4;--blue:#7db5ff;--green:#63d69a;--yellow:#f4c96b;--red:#ff8d8d;--line:#2b3a58}}
* {{box-sizing:border-box}} body {{margin:0;background:linear-gradient(130deg,#0b1020,#10192b);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}}
main {{max-width:1160px;margin:auto;padding:42px 22px 70px}} h1 {{font-size:34px;letter-spacing:-.03em;margin:0 0 6px}} h2 {{margin:8px 0;font-size:22px}} p {{color:#d5def0}} .muted {{color:var(--muted)}}
.hero {{display:grid;grid-template-columns:1.4fr 1fr;gap:22px;align-items:end;margin-bottom:28px}} .eyebrow,.category {{color:var(--blue);font-size:11px;font-weight:800;letter-spacing:.12em;text-transform:uppercase}}
.hero-copy {{max-width:720px}} .hero-copy p {{font-size:17px;color:var(--muted)}} .proof {{background:var(--panel);border:1px solid var(--line);padding:16px;border-radius:10px}}
.filters {{display:flex;gap:10px;flex-wrap:wrap;background:var(--panel);border:1px solid var(--line);padding:12px;border-radius:10px;margin-bottom:18px}} input,select {{background:#0c1424;color:var(--ink);border:1px solid var(--line);border-radius:6px;padding:9px 10px}} input[type=search] {{min-width:250px;flex:1}} button,.primary {{background:var(--blue);color:#07101e;border:0;border-radius:6px;padding:9px 14px;font-weight:800;cursor:pointer;text-decoration:none}}
.grid {{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}} .product-card,.empty,.onboarding {{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px}} .card-top {{display:flex;justify-content:space-between;align-items:center}} .price {{font-size:22px;font-weight:800}} .model {{color:var(--blue);font-size:13px}} .evidence {{margin:13px 0;padding:9px 10px;border-radius:6px;font-size:12px}} .evidence.verified {{background:#123d2b;color:var(--green)}} .evidence.measured,.evidence.simulation,.evidence.reference {{background:#3a2d12;color:var(--yellow)}}
.metrics {{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}} .metric,.tag {{display:inline-block;background:var(--panel2);border:1px solid var(--line);border-radius:20px;padding:3px 8px;font-size:11px;color:var(--muted)}} .metric b {{color:var(--ink)}} .tags {{display:flex;gap:5px;flex-wrap:wrap}} .compat {{font-size:12px;color:var(--muted);margin:10px 0}} .card-actions {{margin-top:16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}} .card-actions form {{display:flex;gap:6px;flex:1;min-width:250px}} .card-actions input {{min-width:0;flex:1}} a.secondary {{color:var(--blue);font-size:13px}} .empty {{text-align:center;padding:50px}} .onboarding {{margin-top:20px}} .sr-only {{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}}
footer {{margin-top:30px;color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:15px}} code,.license {{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}} @media(max-width:760px){{.hero{{grid-template-columns:1fr}}}}
</style></head><body><main>
<section class="hero"><div class="hero-copy"><div class="eyebrow">Calibrix · ComfyUI product-photo kernels</div><h1>Cleaner catalog images without retraining.</h1><p>Small, portable kernels for product photography workflows. Each card tells you whether its evidence is synthetic, measured, or verified—no inflated savings dashboard.</p></div><div class="proof"><b>What you buy</b><p class="muted">A signed kernel license for one compatible model, plus the exact install path. The base checkpoint is not included.</p><b>What we do not claim</b><p class="muted">A kernel score is not proof of sales lift or production ROI. Test it on your own catalog before scaling.</p></div></section>
<form class="filters" method="get" action="/"><input type="search" name="q" value="{{query}}" placeholder="Search product, model, or tag"><select name="category">{{category_options}}</select><select name="evidence">{{evidence_options}}</select><button type="submit">Filter catalog</button></form>
<p class="muted">Showing {{count}} kernels in <b>{{active_category}}</b></p><section class="grid">{{cards}}</section>
<section class="onboarding"><h2>Buyer onboarding</h2><p><b>1.</b> Confirm your checkpoint matches the card. <b>2.</b> Install the Calibrix custom node. <b>3.</b> Buy and complete checkout. <b>4.</b> Paste the delivered key into the node and compare against your baseline.</p><p class="muted">Every purchase is bound to the exact kernel spec. Changing the spec invalidates the license.</p></section>
<footer>Proof-first catalog · payments are processed by the configured provider · license fulfillment is recorded in the order ledger.</footer>
</main></body></html>"""

_DETAIL_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{{title}} · Calibrix</title><style>
body{{margin:0;background:#0b1020;color:#edf3ff;font:15px/1.5 system-ui;}}main{{max-width:850px;margin:auto;padding:42px 22px}}.card{{background:#121a2b;border:1px solid #2b3a58;border-radius:10px;padding:22px}}.muted{{color:#9aa9c4}}.model{{color:#7db5ff}}.evidence{{padding:10px;background:#3a2d12;color:#f4c96b;border-radius:6px;margin:14px 0}}.evidence.verified{{background:#123d2b;color:#63d69a}}.metrics,.tags{{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}}.metric,.tag{{border:1px solid #2b3a58;border-radius:20px;padding:4px 9px;color:#9aa9c4;font-size:12px}}.onboarding{{background:#18233a;border-radius:8px;padding:15px;margin:18px 0}}form{{display:flex;gap:8px;flex-wrap:wrap;margin-top:18px}}input{{flex:1;min-width:230px;background:#0c1424;color:#edf3ff;border:1px solid #2b3a58;border-radius:6px;padding:10px}}button,.primary{{background:#7db5ff;color:#07101e;border:0;border-radius:6px;padding:10px 15px;font-weight:800}}a{{color:#7db5ff}}</style></head><body><main><p><a href="/">← Back to catalog</a></p><article class="card"><p class="muted">PRODUCT PHOTO · {{model}}</p><h1>{{title}}</h1><p>{{description}}</p>{{evidence}}<div class="metrics">{{metrics}}</div><p><b>Compatible models:</b> {{models}}</p><div class="tags">{{tags}}</div><section class="onboarding"><h2>Before you buy</h2><ol><li>Install <code>calibrix_node.py</code> in your ComfyUI custom_nodes folder.</li><li>Have the compatible base checkpoint installed locally.</li><li>Use the exact model family listed above; the checkpoint is not included.</li><li>After fulfillment, paste the delivered license key into Calibrix Kernel Scale.</li></ol></section><h2>${{price}}</h2><form method="post" action="/checkout"><input type="hidden" name="listing_id" value="{{listing_id}}"><input type="email" name="email" placeholder="you@example.com" required><button type="submit">Continue to checkout</button></form></article></main></body></html>"""

_SUCCESS_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Calibrix fulfillment</title><style>body{{margin:0;background:#0b1020;color:#edf3ff;font:15px/1.5 system-ui}}main{{max-width:760px;margin:auto;padding:45px 22px}}.card{{background:#121a2b;border:1px solid #2b3a58;border-radius:10px;padding:22px}}.license{{white-space:pre-wrap;word-break:break-all;background:#07101e;border:1px solid #2b3a58;padding:14px;border-radius:6px}}.onboarding{{background:#18233a;border-radius:8px;padding:15px;margin-top:18px}}.muted{{color:#9aa9c4}}a{{color:#7db5ff}}</style></head><body><main><div class="card">{{body}}<p><a href="/">Return to product catalog</a></p></div></main></body></html>"""


def make_handler(store: JsonStore, provider: BaseProvider):
    server = MarketplaceServer(store, provider)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, status: int, body: str, ctype: str = "text/html",
                  location: Optional[str] = None,
                  headers: Optional[Dict[str, str]] = None) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            if location:
                self.send_header("Location", location)
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            params = urllib.parse.parse_qs(parsed.query)
            query = params.get("q", [""])[0]
            category = params.get("category", ["product-photo"])[0]
            evidence = params.get("evidence", ["all"])[0]
            if path == "/":
                self._send(200, server.render_storefront(query, category, evidence))
            elif path == "/api/listings":
                status, body = server.handle_api_listings(query, category, evidence)
                self._send(status, body, "application/json", headers={
                    "Access-Control-Allow-Origin": "*", "Cache-Control": "no-store",
                })
            elif path.startswith("/listing/"):
                listing_id = urllib.parse.unquote(path[len("/listing/"):])
                body = server.render_listing_detail(listing_id)
                self._send(200, body) if body is not None else self._send(404, "not found", "text/plain")
            elif path == "/success":
                status, body = server.render_success(params.get("order", [""])[0])
                self._send(status, body)
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            if self.path == "/checkout":
                form = {key: value[0] for key, value in urllib.parse.parse_qs(raw.decode("utf-8")).items()}
                status, ctype, body = server.handle_checkout(form)
                self._send(status, body, ctype, location=body if status == 302 else None)
            elif self.path == "/webhook":
                signature = self.headers.get("X-Mock-Signature", self.headers.get("Stripe-Signature", ""))
                status, body = server.handle_webhook(raw, signature or "")
                self._send(status, body, "application/json")
            else:
                self._send(404, "not found", "text/plain")

    return Handler


def serve(store: JsonStore, provider: BaseProvider,
          host: str = "127.0.0.1", port: int = 8700) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(store, provider))
    print(f"Calibrix product-photo marketplace on http://{host}:{port} (provider: {provider.name})")
    httpd.serve_forever()
