# SPDX-License-Identifier: MIT
import json
import os
import re
import tempfile
import unittest

from calibrix.doberwatch.platform_watch.blog import (
    build_openwatch_report,
    write_openwatch_report,
    write_openwatch_reports_all_locales,
)
from calibrix.doberwatch.platform_watch.locales import LOCALES, PACK_HASH_LOCALES_VERSION, STRINGS, get_locale, t
from calibrix.doberwatch.platform_watch.obscura_adapter import (
    ANONYMITY_CHECKLIST,
    INSTALL_HINT,
    LOCALE_PROXY_ENV,
    OBSCURA_BINARY,
    OBSCURA_CDP_URL,
    build_fetch_cmd,
    build_scrape_cmd,
    build_serve_cmd,
    cdp_connect_snippet,
    evidence_capture_plan,
    mcp_config_snippet,
    status,
)
from calibrix.doberwatch.platform_watch.prompts import PACK_HASH

# ---- locales ----

class TestLocales(unittest.TestCase):
    def test_ten_locales(self):
        self.assertEqual(len(LOCALES), 10)
        codes = [loc.code for loc in LOCALES]
        self.assertEqual(len(codes), len(set(codes)))

    def test_cultural_names_present(self):
        by = {loc.code: loc.name for loc in LOCALES}
        self.assertEqual(by["ar"], "Noor")
        self.assertEqual(by["zh-Hans"], "Mingjing")
        self.assertEqual(by["es-419"], "Vela")
        self.assertEqual(by["bn"], "Alo")
        self.assertEqual(by["ru"], "Dozor")

    def test_rtl_only_ar(self):
        rtl = [loc.code for loc in LOCALES if loc.dir == "rtl"]
        self.assertEqual(rtl, ["ar"])

    def test_get_locale_fallback(self):
        self.assertEqual(get_locale("ar").code, "ar")
        self.assertEqual(get_locale("pt-BR").code, "pt-BR")
        self.assertEqual(get_locale("pt").code, "pt-BR")  # lang fallback
        self.assertEqual(get_locale("xx").code, "en")

    def test_strings_have_ten(self):
        for loc in LOCALES:
            base = loc.lang
            self.assertIn(base, STRINGS)
            for key in ("title", "summary", "platforms", "best", "worst", "no_platforms"):
                self.assertIn(key, STRINGS[base])

    def test_coverage_speakers_sum(self):
        total = sum(loc.speakers_m for loc in LOCALES)
        self.assertGreater(total, 4000)  # covers >4B L1+L2

    def test_t_helper(self):
        self.assertEqual(t("ar", "title"), "نور اليومي")
        self.assertEqual(t("xx", "title"), "OpenWatch Daily")

    def test_locale_version_tag(self):
        self.assertTrue(PACK_HASH_LOCALES_VERSION.startswith("openwatch.locales."))

# ---- obscura adapter (no binary needed) ----

class TestObscuraAdapter(unittest.TestCase):
    def test_status_without_binary_is_graceful(self):
        st = status(binary="__no_such_binary__obscura")
        self.assertFalse(st.available)
        self.assertIn("Obscura", st.hint)

    def test_build_fetch_cmd_has_stealth_and_proxy(self):
        # per-locale proxy via env
        os.environ["OBSCURA_PROXY_AR"] = "socks5://127.0.0.1:1080"
        os.environ.pop("OBSCURA_PROXY", None)
        cmd = build_fetch_cmd("https://example.com", locale="ar", stealth=True, dump="text")
        self.assertIn("--stealth", cmd)
        self.assertIn("--proxy", cmd)
        self.assertIn("socks5://127.0.0.1:1080", cmd)
        self.assertIn("fetch", cmd)
        # global fallback when per-locale absent
        del os.environ["OBSCURA_PROXY_AR"]
        os.environ["OBSCURA_PROXY"] = "http://127.0.0.1:8080"
        cmd2 = build_fetch_cmd("https://example.com", locale="bn")
        self.assertIn("http://127.0.0.1:8080", cmd2)
        del os.environ["OBSCURA_PROXY"]
        # no proxy when none set
        cmd3 = build_fetch_cmd("https://example.com", locale="ja", proxy=None)
        # should not contain --proxy when env empty
        # proxy=None but no env — so no --proxy
        self.assertNotIn("--proxy", cmd3)

    def test_build_serve_cmd(self):
        cmd = build_serve_cmd(port=9223, stealth=True, workers=2, allow_private=True)
        self.assertIn("serve", cmd)
        self.assertIn("--port", cmd)
        self.assertIn("9223", cmd)
        self.assertIn("--stealth", cmd)
        self.assertIn("--allow-private-network", cmd)

    def test_build_scrape_cmd(self):
        cmd = build_scrape_cmd(["https://a.com", "https://b.com"], concurrency=5, eval_js="document.title", fmt="json", locale="fr", stealth=False)
        self.assertIn("scrape", cmd)
        self.assertIn("--concurrency", cmd)
        self.assertIn("5", cmd)
        self.assertIn("--eval", cmd)
        self.assertNotIn("--stealth", cmd)  # stealth False

    def test_explicit_proxy_overrides_env(self):
        os.environ["OBSCURA_PROXY_EN"] = "socks5://env"
        cmd = build_fetch_cmd("https://example.com", locale="en", proxy="http://explicit:8080")
        self.assertIn("http://explicit:8080", cmd)
        self.assertNotIn("socks5://env", cmd)
        del os.environ["OBSCURA_PROXY_EN"]

    def test_evidence_capture_plan_is_file_url(self):
        with tempfile.TemporaryDirectory() as d:
            html_path = os.path.join(d, "2026-09-24.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write("<html><body>hi</body></html>")
            plan = evidence_capture_plan(html_path, d, locale="ar", stealth=True)
            self.assertTrue(plan["url"].startswith("file://"))
            self.assertTrue(plan["png"].endswith(".ar.png"))
            self.assertTrue(plan["pdf"].endswith(".ar.pdf"))
            self.assertIn("--screenshot", plan["fetch_png_cmd"])
            self.assertIn("--allow-private-network", plan["fetch_png_cmd"])

    def test_cdp_and_mcp_snippets(self):
        s = cdp_connect_snippet("puppeteer")
        self.assertIn(OBSCURA_CDP_URL, s)
        self.assertIn("puppeteer", s.lower())
        self.assertIn("mcpServers", json.dumps(mcp_config_snippet()))
        self.assertEqual(mcp_config_snippet()["mcpServers"]["obscura"]["command"], OBSCURA_BINARY)

    def test_anonymity_checklist_pinned(self):
        self.assertGreaterEqual(len(ANONYMITY_CHECKLIST), 5)
        self.assertTrue(any("--stealth" in c for c in ANONYMITY_CHECKLIST))
        self.assertTrue(any("OBSCURA_CDP_TOKEN" in c or "CDP_TOKEN" in c for c in ANONYMITY_CHECKLIST))

    def test_locale_proxy_env_covers_all_locales(self):
        for loc in LOCALES:
            self.assertIn(loc.code, LOCALE_PROXY_ENV)

# ---- locale rendering ----

class TestLocaleRendering(unittest.TestCase):
    def _grade(self, verdict="PASS", score=0.85):
        return {
            "rubric_id": "platform_watch.v1", "domain": "platform_watch",
            "score": score, "verdict": verdict, "complaint": verdict != "PASS",
            "raw_score": score,
            "criteria": [{"criterion": "direction_following", "weight": 1.0, "judgement": 0.9, "credited": 0.9, "requires_evidence": False, "evidence_refs": 0, "requires_corroboration": False, "corroboration": None}],
            "evidence_gate_applied": False, "corroboration_gate_applied": False, "contradiction_veto": False, "corroboration_coverage": 0.8,
        }

    def test_per_locale_html_has_lang_and_dir(self):
        g = self._grade()
        report = build_openwatch_report("2026-09-24", PACK_HASH, [{"platform": "luna", "grade": g}], locale="ar")
        with tempfile.TemporaryDirectory() as d:
            paths = write_openwatch_report(report, d, locale="ar")
            with open(paths["html"], encoding="utf-8") as f:
                html_text = f.read()
            self.assertIn('lang="ar"', html_text)
            self.assertIn('dir="rtl"', html_text)
            self.assertIn("نور", html_text)  # cultural name
            self.assertIn("الملخص", html_text)

    def test_write_all_locales_creates_ten_bundles(self):
        g = self._grade()
        report = build_openwatch_report("2026-09-24", PACK_HASH, [{"platform": "luna", "grade": g}])
        with tempfile.TemporaryDirectory() as d:
            out = write_openwatch_reports_all_locales(report, d)
            self.assertIn("htmls", out)
            self.assertEqual(len(out["htmls"]), 10)  # one entry per locale (en == root)
            # each locale dir exists
            for loc in LOCALES:
                loc_html = os.path.join(d, loc.code, "2026-09-24.html")
                self.assertTrue(os.path.exists(loc_html), f"missing {loc_html}")
                with open(loc_html, encoding="utf-8") as f:
                    txt = f.read()
                self.assertIn(f'lang="{loc.code}"', txt)

    def test_switcher_contains_all_locales(self):
        g = self._grade()
        report = build_openwatch_report("2026-09-24", PACK_HASH, [{"platform": "luna", "grade": g}])
        with tempfile.TemporaryDirectory() as d:
            paths = write_openwatch_report(report, d, locale="en")
            with open(paths["html"], encoding="utf-8") as f:
                txt = f.read()
            # switcher has links for every locale's cultural name
            for loc in LOCALES:
                self.assertIn(loc.name, txt)

    def test_xss_locale_escaped(self):
        g = self._grade()
        g["criteria"][0]["criterion"] = "</script><script>alert(1)</script>"
        report = build_openwatch_report("2026-09-24", PACK_HASH, [{"platform": "</script>", "grade": g}], locale="ja")
        with tempfile.TemporaryDirectory() as d:
            paths = write_openwatch_report(report, d, locale="ja")
            with open(paths["html"], encoding="utf-8") as f:
                txt = f.read()
            self.assertNotIn("</script><script>alert(1)</script>", txt.split("<script>")[1].split("</script>")[0])
