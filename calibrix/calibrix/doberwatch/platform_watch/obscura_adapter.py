# SPDX-License-Identifier: MIT
"""Obscura adapter — stealth evidence capture for OpenWatch.

Wraps https://github.com/h4ckf0r0day/obscura (Apache-2.0, Rust headless
browser, V8, CDP, no Chromium) as the *evidence* layer, not the judge.
The judge stays deterministic (grading/verification); Obscura only fetches
and renders so a grade can be checked by a human.

Why Obscura here: 30 MB vs 200+ MB, 85 ms vs 500 ms, built-in anti-detect
(fingerprint randomization, isTrusted=true, webdriver=undefined, tracker
blocking 3.5k domains), drops into Puppeteer/Playwright via CDP, and —
critically for your ask — exposes ``--stealth --proxy`` and ``serve --port``
so the runner can rotate identity per locale/platform without the platform
seeing "benchmark probe".

Anonymity stance: this module is for *auditing consumption* (fetch public
platform UIs, capture your own OpenWatch HTML as PNG/PDF) and for
*operator-anonymous* evidence collection. It is NOT for scraping private
user data. Use residential proxies you own, respect robots.txt when
``--obey-robots`` matters, and don't target login walls.

If the ``obscura`` binary is absent, every helper degrades to
``available=False`` with an install hint — tests and CI never hard-depend
on the binary.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

OBSCURA_BINARY = os.environ.get("OBSCURA_BIN", "obscura")
OBSCURA_CDP_URL = os.environ.get("OBSCURA_CDP_URL", "ws://127.0.0.1:9222/devtools/browser")

# Locales map to proxy/IP hints where available; keep as env-driven so
# deployments can inject NodeMaven/ProxyEmpire/Masklabs pools per region
# without hardcoding provider keys into the repo.
LOCALE_PROXY_ENV = {
    "en": "OBSCURA_PROXY_EN",
    "zh-Hans": "OBSCURA_PROXY_ZH",
    "es-419": "OBSCURA_PROXY_ES",
    "hi": "OBSCURA_PROXY_HI",
    "ar": "OBSCURA_PROXY_AR",
    "pt-BR": "OBSCURA_PROXY_PT",
    "fr": "OBSCURA_PROXY_FR",
    "ru": "OBSCURA_PROXY_RU",
    "ja": "OBSCURA_PROXY_JA",
    "bn": "OBSCURA_PROXY_BN",
}

INSTALL_HINT = (
    "Obscura binary not found. Install: "
    "curl -LO https://github.com/h4ckf0r0day/obscura/releases/latest/download/obscura-x86_64-linux.tar.gz && "
    "tar xzf obscura-x86_64-linux.tar.gz && ./obscura --help  "
    "| docker: docker run -p 127.0.0.1:9222:9222 -e OBSCURA_CDP_TOKEN=$(openssl rand -hex 32) h4ckf0r0day/obscura"
)

@dataclass(frozen=True)
class ObscuraStatus:
    available: bool
    binary: str
    version: str
    hint: str

def status(binary: str = OBSCURA_BINARY) -> ObscuraStatus:
    path = shutil.which(binary) or (binary if os.path.exists(binary) else None)
    if not path:
        return ObscuraStatus(False, binary, "", INSTALL_HINT)
    try:
        out = subprocess.run([path, "--help"], capture_output=True, text=True, timeout=5)
        ver = ""
        # --help prints usage; --version prints obscura X.Y.Z where available
        vproc = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
        if vproc.returncode == 0:
            ver = (vproc.stdout or vproc.stderr).strip().splitlines()[0][:120]
        return ObscuraStatus(True, path, ver, "")
    except Exception as e:
        return ObscuraStatus(False, binary, "", str(e))

def _proxy_for_locale(locale: str) -> Optional[str]:
    env = LOCALE_PROXY_ENV.get(locale, "")
    if env and os.environ.get(env):
        return os.environ[env].strip()
    # global fallback
    g = os.environ.get("OBSCURA_PROXY", "").strip()
    return g or None

def build_fetch_cmd(
    url: str,
    *,
    dump: str = "html",
    eval_js: Optional[str] = None,
    screenshot: Optional[str] = None,
    wait_until: str = "load",
    timeout: int = 30,
    locale: str = "en",
    stealth: bool = True,
    allow_private: bool = False,
    proxy: Optional[str] = None,
    binary: str = OBSCURA_BINARY,
) -> List[str]:
    """Build the CLI argv for ``obscura fetch <url>``.

    Returned argv is inspectable (tests) and runnable (subprocess).
    """
    cmd: List[str] = [binary]
    eff_proxy = proxy if proxy is not None else _proxy_for_locale(locale)
    if eff_proxy:
        cmd += ["--proxy", eff_proxy]
    if stealth:
        cmd += ["--stealth"]
    cmd += ["fetch", url, "--dump", dump, "--wait-until", wait_until, "--timeout", str(timeout)]
    if eval_js:
        cmd += ["--eval", eval_js]
    if screenshot:
        cmd += ["--screenshot", screenshot]
    if allow_private:
        cmd += ["--allow-private-network"]
    return cmd

def build_serve_cmd(
    *,
    port: int = 9222,
    stealth: bool = True,
    proxy: Optional[str] = None,
    workers: int = 1,
    allow_private: bool = False,
    binary: str = OBSCURA_BINARY,
) -> List[str]:
    cmd: List[str] = [binary]
    if proxy or os.environ.get("OBSCURA_PROXY"):
        cmd += ["--proxy", proxy or os.environ["OBSCURA_PROXY"]]
    if stealth:
        cmd += ["--stealth"]
    cmd += ["serve", "--port", str(port), "--workers", str(workers)]
    if allow_private:
        cmd += ["--allow-private-network"]
    return cmd

def build_scrape_cmd(
    urls: Sequence[str],
    *,
    concurrency: int = 10,
    eval_js: Optional[str] = None,
    fmt: str = "json",
    locale: str = "en",
    stealth: bool = True,
    proxy: Optional[str] = None,
    binary: str = OBSCURA_BINARY,
) -> List[str]:
    cmd: List[str] = [binary]
    eff_proxy = proxy if proxy is not None else _proxy_for_locale(locale)
    if eff_proxy:
        cmd += ["--proxy", eff_proxy]
    if stealth:
        cmd += ["--stealth"]
    cmd = cmd + ["scrape"] + list(urls) + ["--concurrency", str(concurrency), "--format", fmt]
    if eval_js:
        cmd += ["--eval", eval_js]
    return cmd

# CDP helpers — for Puppeteer/Playwright over Obscura's CDP bridge

def cdp_connect_snippet(kind: str = "puppeteer") -> str:
    if kind == "playwright":
        return (
            "import { chromium } from 'playwright-core';\n"
            f"const browser = await chromium.connectOverCDP({{ endpointURL: '{OBSCURA_CDP_URL}' }});\n"
        )
    return (
        "import puppeteer from 'puppeteer-core';\n"
        f"const browser = await puppeteer.connect({{ browserWSEndpoint: '{OBSCURA_CDP_URL}/devtools/browser' }});\n"
    )

def mcp_config_snippet() -> Dict[str, Any]:
    return {"mcpServers": {"obscura": {"command": OBSCURA_BINARY, "args": ["mcp"]}}}

def evidence_capture_plan(
    report_html_path: str,
    out_dir: str,
    *,
    locale: str = "en",
    stealth: bool = True,
) -> Dict[str, Any]:
    """Plan to capture the OpenWatch HTML as PNG + PDF evidence.

    The HTML is local (file://), so this does NOT need a proxy. Obscura's
    native rendering (no Chromium) does screenshot/PDF directly.
    """
    base = os.path.splitext(os.path.basename(report_html_path))[0]
    png = os.path.join(out_dir, f"{base}.{locale}.png")
    pdf = os.path.join(out_dir, f"{base}.{locale}.pdf")
    # file:// URL for local capture; allow private network so 127.0.0.1/file:// routes are not blocked
    url = "file://" + os.path.abspath(report_html_path).replace("\\", "/")
    return {
        "locale": locale,
        "source_html": report_html_path,
        "url": url,
        "png": png,
        "pdf": pdf,
        "fetch_png_cmd": build_fetch_cmd(url, dump="html", screenshot=png, locale=locale, stealth=stealth, allow_private=True),
        "fetch_pdf_hint": f"Use CDP Page.printToPDF over {OBSCURA_CDP_URL} after puppeteer.connect() — see cdp_connect_snippet()",
    }

# Operator anonymity checklist — returned for deployment docs/tests

ANONYMITY_CHECKLIST = [
    "Use OBSCURA_PROXY_* per locale (residential pool you control); never run locale checks from one datacenter IP.",
    "Enable --stealth (fingerprint randomization + tracker blocking 3,520 domains) so platform cannot tag 'benchmark'.",
    "Rotate OBSCURA_CDP_TOKEN per run; bind CDP to 127.0.0.1 only (docker -p 127.0.0.1:9222:9222).",
    "Stagger locale scrapes (--concurrency low, random jitter) — parallel bursts are a detection signal.",
    "Capture evidence locally (file://) — no proxy needed for your own HTML; keep network evidence minimal.",
    "Store reports on your disk/cloud (ResponseCache local_only); sharing needs explicit per-entry consent.",
]

__all__ = [
    "OBSCURA_BINARY", "OBSCURA_CDP_URL", "LOCALE_PROXY_ENV", "INSTALL_HINT",
    "ObscuraStatus", "status",
    "build_fetch_cmd", "build_serve_cmd", "build_scrape_cmd",
    "cdp_connect_snippet", "mcp_config_snippet", "evidence_capture_plan",
    "ANONYMITY_CHECKLIST",
]
