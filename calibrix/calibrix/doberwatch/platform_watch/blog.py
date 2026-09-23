# SPDX-License-Identifier: MIT
"""OpenWatch Daily blog — JSON + self-contained HTML for one day's audit.

Reuses the Doberwatch report primitives (pill colours, json-for-script
escaping, verification tables) but renders *multiple platforms* side-by-side
against the same rubric and prompt pack. The JSON is the source of truth;
the HTML is a file://-viewable mirror with no build step or network.

Ownership: the written files live at the caller's ``out_dir`` (any disk or
cloud path the user controls). The ResponseCache integration is optional —
callers that want audit-chained prompt storage can pass a cache; the blog
itself is otherwise just files the user owns.

Locales: build is locale-agnostic; rendering is per-locale via
``locales.py`` (10 cultural names, LTR + RTL, language switcher, hashed pack).
"""

from __future__ import annotations

import html
import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

OPENWATCH_SCHEMA = "calibrix.openwatch.daily/1.0"

_STATUS_PILL = {
    "corroborated": "ok",
    "single_source": "warn",
    "contested": "warn",
    "contradicted": "bad",
    "unverified": "bad",
    "PASS": "ok",
    "GROWL": "warn",
    "BARK": "warn",
    "BITE": "bad",
}

def _as_number(value: Any) -> Optional[float]:
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None

def _json_for_script(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False)
    for char, esc in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                      ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        text = text.replace(char, esc)
    return text

def _pill(text: Any, value: Any = None) -> str:
    key = str(value if value is not None else text)
    css = _STATUS_PILL.get(key, "warn")
    return f'<span class="pill {css}">{html.escape(str(text))}</span>'

def _num(value: Any, digits: int = 2) -> str:
    return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else "—"

def _platform_card(platform: Mapping[str, Any]) -> str:
    grade = platform.get("grade") or {}
    verification = platform.get("verification") or {}
    name = html.escape(str(platform.get("platform", "unknown")))
    verdict = str(grade.get("verdict") or "NOT GRADED")
    score = _num(grade.get("score"))
    mgrade = _num(verification.get("material_grade") if verification else grade.get("corroboration_coverage"))
    if verification and verification.get("material_grade") is not None:
        mgrade = _num(verification.get("material_grade"))
    gates = []
    if grade.get("evidence_gate_applied"):
        gates.append("evidence gate")
    if grade.get("corroboration_gate_applied"):
        gates.append("corroboration gate")
    if grade.get("contradiction_veto"):
        gates.append("contradiction veto")
    meta = " · ".join(filter(None, [
        html.escape(str(grade.get("rubric_id") or "")),
        " · ".join(gates) or "no gate",
    ]))
    status_counts = verification.get("status_counts") if isinstance(verification, Mapping) else {}
    counts_line = ""
    if status_counts:
        parts = [f"{html.escape(str(k))} {int(v)}" for k, v in sorted(status_counts.items())]
        counts_line = f'<div class="mut" style="margin-top:4px">claims: {" · ".join(parts)}</div>'
    elif grade.get("criteria"):
        counts_line = f'<div class="mut" style="margin-top:4px">criteria: {len(grade.get("criteria") or [])} · complaint: {"yes" if grade.get("complaint") else "no"}</div>'
    criteria_rows = ""
    for row in grade.get("criteria") or []:
        if not isinstance(row, Mapping):
            continue
        needs = []
        if row.get("requires_evidence"):
            needs.append("evidence")
        if row.get("requires_corroboration"):
            needs.append("corroboration")
        supplied = f'{int(row.get("evidence_refs") or 0)} ref(s)'
        if row.get("corroboration") is not None:
            supplied += f' · corr {_num(row.get("corroboration"))}'
        criteria_rows += (
            "<tr>"
            f'<td>{html.escape(str(row.get("criterion","")))}</td>'
            f'<td>{_num(row.get("weight"))}</td>'
            f'<td>{_num(row.get("judgement"))}</td>'
            f'<td>{_num(row.get("credited"))}</td>'
            f'<td class="mut">{html.escape(", ".join(needs)) or "—"}</td>'
            f'<td class="mut">{supplied}</td>'
            "</tr>"
        )
    criteria_html = ""
    if criteria_rows:
        criteria_html = (
            '<table><tr><th>Criterion</th><th>Weight</th><th>Judged</th><th>Credited</th><th>Needs</th><th>Supplied</th></tr>'
            + criteria_rows + '</table>'
        )
    claims_rows = ""
    claims = verification.get("claims") if isinstance(verification, Mapping) else None
    if claims:
        for c in claims:
            if not isinstance(c, Mapping):
                continue
            status = str(c.get("status", "unverified"))
            claims_rows += (
                "<tr>"
                f'<td>{_pill(status)}</td>'
                f'<td>{_num(c.get("grade"))}</td>'
                f'<td>{html.escape(str(c.get("claim","")))}</td>'
                f'<td class="mut">{"material" if c.get("material") else "context"}</td>'
                f'<td class="mut">{html.escape(", ".join(c.get("support") or []) or "—")}</td>'
                f'<td class="mut">{html.escape(", ".join(c.get("contradict") or []) or "—")}</td>'
                "</tr>"
            )
    claims_html = ""
    if claims_rows:
        claims_html = (
            '<details style="margin-top:8px"><summary class="mut">Fact-check verdicts per claim</summary>'
            '<table><tr><th>Status</th><th>Grade</th><th>Claim</th><th>Kind</th><th>Supported by</th><th>Contradicted by</th></tr>'
            + claims_rows + '</table></details>'
        )
    diff_html = ""
    diff = platform.get("diff")
    if isinstance(diff, str) and diff.strip():
        diff_html = f'<details style="margin-top:8px"><summary class="mut">Diff</summary><pre style="white-space:pre-wrap; background:#0d1117; padding:8px; border-radius:6px; overflow:auto">{html.escape(diff)}</pre></details>'
    return f"""
<div class="card">
  <h3>{name} {_pill(verdict)} <span class="mut">score {score} · material {mgrade}</span></h3>
  <div class="mut">{meta}</div>
  {counts_line}
  <div style="margin-top:8px">{criteria_html}</div>
  {claims_html}
  {diff_html}
</div>
"""

# ---------- core report builder (locale-agnostic) ----------

def build_openwatch_report(
    date: str,
    pack_hash: str,
    platforms: Sequence[Mapping[str, Any]],
    ledger_digest: str = "",
    pack_version: str = "v1",
    notes: str = "",
    locale: str = "en",
) -> Dict[str, Any]:
    """Serializable daily report over multiple platforms."""
    normed: List[Dict[str, Any]] = []
    for p in platforms:
        d = dict(p)
        if hasattr(d.get("grade"), "to_dict"):
            d["grade"] = d["grade"].to_dict()  # type: ignore
        normed.append(d)
    normed.sort(key=lambda x: str(x.get("platform", "")))
    verdict_counts: Dict[str, int] = {}
    best = None
    worst = None
    for p in normed:
        v = str((p.get("grade") or {}).get("verdict", "NOT GRADED"))
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        score = (p.get("grade") or {}).get("score")
        if isinstance(score, (int, float)):
            if best is None or score > best[1]:
                best = (str(p.get("platform")), float(score))
            if worst is None or score < worst[1]:
                worst = (str(p.get("platform")), float(score))
    return {
        "schema": OPENWATCH_SCHEMA,
        "date": date,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "pack_version": pack_version,
        "pack_hash": pack_hash,
        "ledger_digest": ledger_digest,
        "locale": locale,
        "platforms": normed,
        "summary": {
            "platform_count": len(normed),
            "verdict_counts": verdict_counts,
            "best": {"platform": best[0], "score": round(best[1], 6)} if best else None,
            "worst": {"platform": worst[0], "score": round(worst[1], 6)} if worst else None,
        },
        "notes": notes,
        "claim_boundary": "Audit documentation. Scores are reproducible against the hashed prompt pack; do not market a platform's grade as certification.",
    }

# Backwards-compat alias and locale wrapper
def build_openwatch_report_localized(
    date: str,
    pack_hash: str,
    platforms: Sequence[Mapping[str, Any]],
    locale: str = "en",
    ledger_digest: str = "",
    pack_version: str = "v1",
    notes: str = "",
) -> Dict[str, Any]:
    return build_openwatch_report(date, pack_hash, platforms, ledger_digest, pack_version, notes, locale)

def write_openwatch_report(
    report: Mapping[str, Any],
    out_dir: str,
    locale: str = "en",
) -> Dict[str, str]:
    """Write report JSON + HTML into ``out_dir`` (single locale)."""
    os.makedirs(out_dir, exist_ok=True)
    date = str(report.get("date", "unknown"))
    stem = date if date != "unknown" else "openwatch"
    json_path = os.path.join(out_dir, f"{stem}.json")
    html_path = os.path.join(out_dir, f"{stem}.html")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_render_openwatch_html(dict(report), locale=locale))
    return {"json": json_path, "html": html_path}

def write_openwatch_reports_all_locales(
    report: Mapping[str, Any],
    out_dir: str,
) -> Dict[str, str]:
    """Write JSON once + per-locale HTML bundles under ``out_dir/<locale>/``.

    Creates ``out_dir/<date>.json`` and for each of the 10 locales
    ``out_dir/<locale>/<date>.html`` plus an ``out_dir/<locale>/index.html``
    mirror. The report's ``locale`` field is overridden per bundle. Returns
    paths including ``json`` and ``htmls``.
    """
    from .locales import LOCALES as _LOCALES
    os.makedirs(out_dir, exist_ok=True)
    date = str(report.get("date", "unknown"))
    stem = date if date != "unknown" else "openwatch"
    json_path = os.path.join(out_dir, f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    # also keep a root HTML (en) for backwards compat
    root_html = os.path.join(out_dir, f"{stem}.html")
    with open(root_html, "w", encoding="utf-8") as f:
        f.write(_render_openwatch_html(dict(report), locale="en"))
    htmls: Dict[str, str] = {"en": root_html}
    for loc in _LOCALES:
        loc_dir = os.path.join(out_dir, loc.code)
        os.makedirs(loc_dir, exist_ok=True)
        loc_report = dict(report)
        loc_report["locale"] = loc.code
        # also record the cultural short name for header use
        loc_report["locale_name"] = loc.name
        html_path = os.path.join(loc_dir, f"{stem}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(_render_openwatch_html(loc_report, locale=loc.code))
        htmls[loc.code] = html_path
        # index.html mirror so /ar/ serves the latest
        idx = os.path.join(loc_dir, "index.html")
        try:
            # cheap copy via write of same content (avoid symlink on Windows)
            with open(idx, "w", encoding="utf-8") as g:
                g.write(_render_openwatch_html(loc_report, locale=loc.code))
        except Exception:
            pass
    return {"json": json_path, "html": root_html, "htmls": htmls}  # type: ignore

def _language_switcher(current: str, date: str) -> str:
    from .locales import LOCALES as _LOCALES
    links = []
    for loc in _LOCALES:
        active = ' style="font-weight:700; text-decoration:underline"' if loc.code == current else ""
        # relative link from per-locale dir: ../<code>/<date>.html — but for root, use <code>/<date>.html
        # Keep it simple: absolute-ish per-locale path
        href = f"{html.escape(loc.code)}/{html.escape(date)}.html" if current == "en" else f"../{html.escape(loc.code)}/{html.escape(date)}.html"
        # For root page, en self-links to ./<date>.html
        if loc.code == "en" and current == "en":
            href = f"{html.escape(date)}.html"
        links.append(f'<a href="{href}"{active} lang="{html.escape(loc.code)}" title="{html.escape(loc.native)} — {html.escape(loc.name)}: {html.escape(loc.meaning)}">{html.escape(loc.native)} · {html.escape(loc.name)}</a>')
    return '<nav style="display:flex; flex-wrap:wrap; gap:8px; margin:10px 0" aria-label="Language">' + " · ".join(links) + '</nav>'

def _render_openwatch_html(report: Dict[str, Any], locale: str = "en") -> str:
    from .locales import get_locale, t
    loc = get_locale(locale)
    lang = html.escape(loc.code)
    dir_attr = html.escape(loc.dir)
    payload = _json_for_script(report)
    platforms = report.get("platforms") or []
    cards = "".join(_platform_card(p) for p in platforms) if platforms else f'<div class="card"><div class="mut">{html.escape(t(loc.lang, "no_platforms"))}</div></div>'
    summary = report.get("summary") or {}
    verdict_line = ", ".join(f"{html.escape(str(k))} {int(v)}" for k, v in sorted((summary.get("verdict_counts") or {}).items())) or ["—"]
    best = summary.get("best")
    worst = summary.get("worst")
    best_line = f'{html.escape(str(best["platform"]))} {best["score"]}' if isinstance(best, dict) else "—"
    worst_line = f'{html.escape(str(worst["platform"]))} {worst["score"]}' if isinstance(worst, dict) else "—"
    ledger = html.escape(str(report.get("ledger_digest") or "—"))
    pack_hash = html.escape(str(report.get("pack_hash") or "—"))
    pack_version = html.escape(str(report.get("pack_version") or "v1"))
    date = html.escape(str(report.get("date") or ""))
    title_local = html.escape(t(loc.lang, "title"))
    summary_local = html.escape(t(loc.lang, "summary"))
    platforms_local = html.escape(t(loc.lang, "platforms"))
    best_label = html.escape(t(loc.lang, "best"))
    worst_label = html.escape(t(loc.lang, "worst"))
    brand_suffix = html.escape(f" — {loc.name}") if loc.code != "en" else ""
    switcher = _language_switcher(loc.code, str(report.get("date") or "openwatch"))
    return f"""<!DOCTYPE html>
<html lang="{lang}" dir="{dir_attr}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_local}{brand_suffix} — {date}</title>
<style>
  :root {{ --bg:#0d1117; --card:#161b22; --ink:#e6edf3; --mut:#8b949e; --acc:#58a6ff; --ok:#3fb950; --warn:#d29922; --bad:#f85149; }}
  * {{ box-sizing:border-box; margin:0; }}
  body {{ background:var(--bg); color:var(--ink); font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; padding:24px; max-width:1100px; margin:0 auto; }}
  h1 {{ font-size:20px; }}
  h2 {{ font-size:14px; color:var(--mut); text-transform:uppercase; letter-spacing:.05em; margin:16px 0 8px; }}
  .mut {{ color:var(--mut); }}
  .grid {{ display:grid; gap:12px; }}
  .card {{ background:var(--card); border:1px solid #30363d; border-radius:8px; padding:14px; }}
  .card h3 {{ font-size:13px; margin-bottom:6px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; font-size:12px; }}
  th,td {{ text-align:left; padding:6px 10px; border-bottom:1px solid #21262d; }}
  th {{ color:var(--mut); font-weight:500; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:99px; font-size:11px; font-weight:600; }}
  .ok {{ background:#12351c; color:var(--ok); }}
  .bad {{ background:#3d1618; color:var(--bad); }}
  .warn {{ background:#3a2b0e; color:var(--warn); }}
  a {{ color:var(--acc); }}
  pre {{ font:12px/1.4 ui-monospace, SFMono-Regular, Menlo, monospace; }}
  summary {{ cursor:pointer; }}
</style>
</head>
<body>
<h1>{title_local}<span class="mut" style="font-weight:400"> — {date}{brand_suffix}</span></h1>
<div class="mut">pack {pack_version} · {pack_hash[:12]}… · ledger {ledger[:12]}… · {html.escape(str(report.get("generated_at","")))} · open this file anywhere — no server needed</div>
{switcher}
<div class="card" style="margin-top:12px">
  <h3>{summary_local}</h3>
  <div>Platforms: {summary.get("platform_count", 0)} · {verdict_line}</div>
  <div class="mut">{best_label}: {best_line} · {worst_label}: {worst_line}</div>
  <div class="mut" style="margin-top:6px">{html.escape(str(report.get("claim_boundary","")))}</div>
</div>
<h2>{platforms_local}</h2>
<div class="grid">
{cards}
</div>
<script>
const R = {payload};
</script>
</body>
</html>
"""
