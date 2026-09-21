# Calibrix marketplace HTTP server (stdlib only).
#
# Routes:
#   GET  /                      -> storefront HTML (listings)
#   GET  /listing/<id>          -> listing detail + buy form
#   POST /checkout              -> create order + checkout session, 302 to payment
#   GET  /success               -> fulfillment confirmation page
#   POST /webhook               -> payment webhook (Stripe or mock)
#
# No frameworks, no deps: runs anywhere, mirrors how report.html needs zero
# infrastructure. For production put behind any TLS terminator.

from __future__ import annotations

import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

from .store import JsonStore, OrderStatus
from .stripe_provider import BaseProvider, PaymentError


class MarketplaceServer:
    def __init__(self, store: JsonStore, provider: BaseProvider,
                 success_url: str = "http://localhost:8700/success") -> None:
        self.store = store
        self.provider = provider
        self.success_url = success_url

    # ------------------------------------------------------------------
    def handle_checkout(self, form: Dict[str, str]) -> Tuple[int, str, str]:
        """Returns (status, content_type, body). 302 Location in headers via body."""
        listing = self.store.get_listing(form.get("listing_id", ""))
        if listing is None or not listing.active:
            return 404, "text/plain", "unknown listing"
        email = (form.get("email") or "").strip()
        if "@" not in email:
            return 400, "text/plain", "valid email required"
        order = self.store.create_order(listing, email, self.provider.name)
        urls = self.success_url
        session = self.provider.create_checkout(
            listing, order,
            success_url=urls + f"?order={order.order_id}",
            cancel_url=urls.replace("/success", "/"),
        )
        self.store.attach_session(order, session["session_id"])
        # In the mock provider the "payment URL" is the success page; real
        # Stripe returns a hosted checkout URL the buyer is redirected to.
        return 302, "text/plain", session["url"]

    def handle_webhook(self, payload: bytes, signature: str) -> Tuple[int, str]:
        """Returns (status, body)."""
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

    def render_storefront(self) -> str:
        listings = self.store.list_listings(active_only=True)
        rows = []
        for l in listings:
            price = f"${l.price_cents / 100:.2f}"
            scoring = " ".join(
                f"<span class='pill'>{k}: {v:.3f}</span>" for k, v in l.scoring.items()
            )
            rows.append(f"""
      <div class='card'>
        <h3>{l.title}</h3>
        <p class='model'>{l.model}</p>
        <p>{l.description}</p>
        <div>{scoring}</div>
        <form method='post' action='/checkout'>
          <input type='hidden' name='listing_id' value='{l.listing_id}'>
          <input type='email' name='email' placeholder='you@example.com' required>
          <button type='submit'>Buy for {price}</button>
        </form>
      </div>""")
        return _PAGE.replace("{{listings}}", "".join(rows)).replace(
            "{{count}}", str(len(listings)))


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Calibrix Kernel Marketplace</title>
<style>
  :root {{ --bg:#0d1117; --card:#161b22; --ink:#e6edf3; --mut:#8b949e;
          --acc:#58a6ff; --ok:#3fb950; }}
  * {{ box-sizing:border-box; margin:0; }}
  body {{ background:var(--bg); color:var(--ink);
         font:15px/1.5 -apple-system, 'Segoe UI', Roboto, sans-serif; }}
  main {{ max-width:1080px; margin:0 auto; padding:40px 20px; }}
  h1 {{ font-size:26px; margin-bottom:4px; }}
  p.sub {{ color:var(--mut); margin-bottom:28px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr));
          gap:16px; }}
  .card {{ background:var(--card); border:1px solid #30363d; border-radius:10px;
          padding:18px; }}
  .card h3 {{ font-size:17px; margin-bottom:2px; }}
  .model {{ color:var(--acc); font-size:13px; margin-bottom:8px; }}
  .pill {{ display:inline-block; background:#21262d; border:1px solid #30363d;
          border-radius:999px; padding:1px 10px; font-size:12px;
          color:var(--ok); margin:0 6px 6px 0; }}
  form {{ margin-top:12px; display:flex; gap:8px; }}
  input[type=email] {{ flex:1; background:#0d1117; color:var(--ink);
          border:1px solid #30363d; border-radius:6px; padding:7px 10px; }}
  button {{ background:var(--acc); color:#0d1117; font-weight:600;
          border:0; border-radius:6px; padding:7px 14px; cursor:pointer; }}
</style>
</head>
<body>
<main>
  <h1>Calibrix Kernel Marketplace</h1>
  <p class="sub">{{count}} calibrated kernels — portable ComfyUI specs with offline license keys</p>
  <div class="grid">{{listings}}</div>
</main>
</body>
</html>"""


def make_handler(store: JsonStore, provider: BaseProvider):
    server = MarketplaceServer(store, provider)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter logs
            pass

        def _send(self, status: int, body: str, ctype: str = "text/html",
                  location: Optional[str] = None) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            if location:
                self.send_header("Location", location)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            path = urllib.parse.urlparse(self.path).path
            if path == "/":
                self._send(200, server.render_storefront())
            elif path == "/success":
                self._send(200, "<h1>Payment received</h1><p>Your license key "
                                "arrives by email; paste it into the Calibrix "
                                "node's license_key field.</p>")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            if self.path == "/checkout":
                form = {k: v[0] for k, v in
                        urllib.parse.parse_qs(raw.decode("utf-8")).items()}
                status, ctype, body = server.handle_checkout(form)
                self._send(status, body, ctype,
                           location=body if status == 302 else None)
            elif self.path == "/webhook":
                sig = self.headers.get("X-Mock-Signature",
                                       self.headers.get("Stripe-Signature", ""))
                status, body = server.handle_webhook(raw, sig or "")
                self._send(status, body, "application/json")
            else:
                self._send(404, "not found", "text/plain")

    return Handler


def serve(store: JsonStore, provider: BaseProvider,
          host: str = "127.0.0.1", port: int = 8700) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(store, provider))
    print(f"Calibrix marketplace on http://{host}:{port} "
          f"(provider: {provider.name})")
    httpd.serve_forever()
