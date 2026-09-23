# SPDX-License-Identifier: MIT
# Calibrix report generation: JSON + self-contained HTML dashboard.
#
# The HTML is a single file with inline JS that works from file:// with no
# server, no build step, and no network — free hosting (GitHub Pages,
# Netlify free, Cloudflare Pages) or just double-click the file.
#
# The report also carries a per-claim verification section: the Doberwatch
# grade (verdict, criteria, evidence/corroboration gates) plus the fact-check
# verdict for every material claim. It is rendered server-side so the same
# JSON the API returns is exactly what the dashboard shows.

from __future__ import annotations

import html
import json
import os
import time
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

VERIFICATION_REPORT_SCHEMA = "calibrix.report.verification/1.0"


def _evidence_metadata(adapter_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
    """Describe what a report proves without upgrading a demo into a claim.

    A clean test run proves the pipeline executed. It does not prove that a
    customer gained money or that an offline adapter represents a real model.
    The status is therefore deliberately conservative and can only become
    ``verified`` when a caller explicitly supplies that assertion after an
    external validation review.
    """
    name = adapter_name.lower()
    explicit = result.get("evidence") or {}
    if explicit.get("status"):
        status = str(explicit["status"])
    elif any(token in name for token in ("scripted", "offline", "simulated", "demo")):
        status = "simulation"
    else:
        status = "measured"

    best = result.get("best")
    overfit = result.get("overfit") or {}
    has_holdout = bool(best is not None and getattr(best, "holdout", None))
    quality_gate = "PASS" if best is not None and has_holdout and not overfit.get("flagged") else "REVIEW"
    if explicit.get("quality_gate"):
        quality_gate = str(explicit["quality_gate"])

    if status == "simulation":
        note = "Synthetic adapter signal only; not evidence of real-model quality, savings, or customer ROI."
    elif status == "reference_only":
        note = "Reference or paper-derived result; not measured by this run."
    elif status == "verified":
        note = "Explicitly reviewed external validation; retain the attached protocol and raw artifacts."
    else:
        note = "Measured by the named adapter on this prompt split; customer production ROI is not established."

    return {
        "status": status,
        "quality_gate": quality_gate,
        "holdout_evaluated": has_holdout,
        "customer_roi_proven": False,
        "note": explicit.get("note", note),
        "claim_boundary": "Do not market this artifact as customer ROI unless production baseline, holdout, cost, and acceptance evidence are attached.",
    }


# Every fact-check verdict maps onto the dashboard's pill colours; a verdict
# not listed stays amber rather than being silently promoted to green.
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


def _is_submission(payload: Any) -> bool:
    """True for a ``Doberwatch.submit()``/``grade_matters()`` payload, which
    nests the grade under ``grade`` beside the verification result."""
    return isinstance(payload, Mapping) and isinstance(payload.get("grade"), Mapping)


def build_verification_section(grade: Any = None, verification: Any = None) -> Optional[Dict[str, Any]]:
    """Normalize a Doberwatch grade + fact-check into a renderable section.

    ``grade`` takes a ``grading.Grade.to_dict()`` and ``verification`` a
    ``verification.verify_response()`` result; either may instead be the whole
    ``Doberwatch.submit()`` payload, which is unwrapped automatically. Returns
    ``None`` when there is nothing to report, so callers can pass through a
    plain SearchEngine result unchanged.
    """
    submission = grade if _is_submission(grade) else (
        verification if _is_submission(verification) else None)
    if submission is not None:
        if verification is None or verification is submission:
            verification = submission.get("verification")
        if grade is submission or grade is None:
            grade = submission.get("grade")

    grade = dict(grade) if isinstance(grade, Mapping) else {}
    verification = dict(verification) if isinstance(verification, Mapping) else {}
    if not grade and not verification:
        return None

    criteria: List[Dict[str, Any]] = []
    for row in grade.get("criteria") or []:
        if not isinstance(row, Mapping):
            continue
        criteria.append({
            "criterion": str(row.get("criterion", "")),
            "weight": _as_number(row.get("weight")) or 0.0,
            "judgement": _as_number(row.get("judgement")) or 0.0,
            "credited": _as_number(row.get("credited")) or 0.0,
            "requires_evidence": bool(row.get("requires_evidence")),
            "requires_corroboration": bool(row.get("requires_corroboration")),
            "evidence_refs": int(row.get("evidence_refs") or 0),
            "corroboration": _as_number(row.get("corroboration")),
        })

    claims: List[Dict[str, Any]] = []
    for claim in verification.get("claims") or []:
        if not isinstance(claim, Mapping):
            continue
        claims.append({
            "claim": str(claim.get("claim", "")),
            "status": str(claim.get("status", "unverified")),
            "grade": _as_number(claim.get("grade")) or 0.0,
            "material": bool(claim.get("material")),
            "independent_support": int(claim.get("independent_support") or 0),
            "support": [str(s) for s in (claim.get("support") or [])],
            "contradict": [str(s) for s in (claim.get("contradict") or [])],
            "reasons": [str(r) for r in (claim.get("reasons") or [])],
        })

    material = [c for c in claims if c["material"]]
    contradicted = [c for c in claims if c["status"] == "contradicted"]
    return {
        "schema": VERIFICATION_REPORT_SCHEMA,
        "domain": grade.get("domain"),
        "rubric_id": grade.get("rubric_id"),
        "verdict": grade.get("verdict"),
        "score": _as_number(grade.get("score")),
        "raw_score": _as_number(grade.get("raw_score")),
        "complaint": bool(grade.get("complaint", False)),
        "drift": _as_number(grade.get("drift")),
        "drift_penalty": _as_number(grade.get("drift_penalty")),
        "evidence_gate_applied": bool(grade.get("evidence_gate_applied", False)),
        "corroboration_gate_applied": bool(grade.get("corroboration_gate_applied", False)),
        "contradiction_veto": bool(grade.get("contradiction_veto", False)),
        "missing_evidence": list(grade.get("missing_evidence") or []),
        "missing_corroboration": list(grade.get("missing_corroboration") or []),
        "evidence_coverage": _as_number(grade.get("evidence_coverage")),
        "corroboration_coverage": _as_number(grade.get("corroboration_coverage")),
        "material_grade": _as_number(verification.get("material_grade")),
        "claim_count": len(claims),
        "material_count": int(verification.get("material_count", len(material)) or 0),
        "contradiction_count": len(contradicted),
        "refuted_count": len(verification.get("refuted") or []),
        "status_counts": dict(verification.get("status_counts") or {}),
        "all_corroborated": bool(verification.get("all_corroborated", False)),
        "criteria": criteria,
        "claims": claims,
    }


def build_report(result: Dict[str, Any], adapter_name: str,
                 scorer_names: List[str],
                 evidence: Optional[Dict[str, Any]] = None,
                 grade: Any = None,
                 verification: Any = None) -> Dict[str, Any]:
    """Serializable summary of a SearchEngine.run() result.

    Pass ``grade`` / ``verification`` (or one ``Doberwatch.submit()`` payload)
    to attach the per-claim grading and fact-check verdicts; the report is
    otherwise unchanged, so existing callers keep working.
    """
    if evidence:
        result = dict(result)
        result["evidence"] = evidence
    evidence_meta = _evidence_metadata(adapter_name, result)
    trials = result.get("trials", [])
    best: Any = result.get("best")
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "adapter": adapter_name,
        "scorers": scorer_names,
        "evidence": evidence_meta,
        "verification": build_verification_section(grade, verification),
        "baseline_train": result.get("baseline_train", {}),
        "baseline_holdout": result.get("baseline_holdout", {}),
        "n_params": result.get("n_params", 0),
        "n_evals": result.get("n_evals", 0),
        "elapsed_seconds": round(result.get("elapsed_seconds", 0.0), 2),
        "overfit": result.get("overfit", {}),
        "best": {
            "trial": best.trial,
            "fitness": round(best.fitness, 6),
            "objectives": {k: round(v, 6) for k, v in best.objectives.items()},
            "holdout": {k: round(v, 6) for k, v in best.holdout.items()},
            "steps_used": best.steps_used,
            "vector": [round(v, 6) for v in best.vector],
        } if best is not None else None,
        "trials": [
            {
                "trial": t.trial,
                "generation": t.generation,
                "fitness": round(t.fitness, 6),
                "objectives": {k: round(v, 6) for k, v in t.objectives.items()},
                "steps_used": t.steps_used,
            }
            for t in trials
        ],
    }


def write_report(result: Dict[str, Any], adapter_name: str,
                 scorer_names: List[str], out_dir: str,
                 evidence: Optional[Dict[str, Any]] = None,
                 grade: Any = None,
                 verification: Any = None) -> Dict[str, str]:
    """Write report.json + report.html into out_dir. Returns file paths."""
    report = build_report(result, adapter_name, scorer_names, evidence=evidence,
                          grade=grade, verification=verification)
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_render_html(report))

    return {"json": json_path, "html": html_path}


def _json_for_script(payload: Any) -> str:
    """Serialize JSON for an inline ``<script>`` block without letting any
    field close the tag.

    ``json.dumps`` escapes quotes and backslashes but leaves ``<`` alone, so a
    value like ``</script>`` (a cached response, a claim, a source id) would
    end the block and become live markup. Escaping ``<>&`` and the JS line
    separators keeps the payload inert while staying valid JSON.
    """
    text = json.dumps(payload, ensure_ascii=False)
    for char, escape in (("&", "\\u0026"), ("<", "\\u003c"), (">", "\\u003e"),
                         ("\u2028", "\\u2028"), ("\u2029", "\\u2029")):
        text = text.replace(char, escape)
    return text


def _pill(text: Any, value: Any = None) -> str:
    """A status chip coloured by verdict; unknown verdicts stay amber."""
    key = str(value if value is not None else text)
    css = _STATUS_PILL.get(key, "warn")
    return f'<span class="pill {css}">{html.escape(str(text))}</span>'


def _render_verification(section: Optional[Dict[str, Any]]) -> str:
    """Per-claim verification block: the grade plus every fact-check verdict.

    Rendered server-side so a report opened straight from ``file://`` shows
    exactly the JSON that was written beside it — no JS needed for the data.
    """
    if not section:
        return ""

    def num(value: Any, digits: int = 2) -> str:
        return (f"{float(value):.{digits}f}"
                if isinstance(value, (int, float)) else "—")

    gates = []
    if section.get("evidence_gate_applied"):
        gates.append("evidence gate")
    if section.get("corroboration_gate_applied"):
        gates.append("corroboration gate")
    if section.get("contradiction_veto"):
        gates.append("contradiction veto")
    meta = " · ".join(filter(None, [
        html.escape(str(section.get("domain") or "")),
        html.escape(str(section.get("rubric_id") or "")),
        " · ".join(gates) or "no gate applied",
    ]))

    verdict = section.get("verdict") or "NOT GRADEABLE"
    out = [f'''
<div class="card" style="margin-top:12px">
  <h3>Per-claim verification {_pill(verdict)} score {num(section.get("score"))}
    · material-claim grade {num(section.get("material_grade"))}</h3>
  <div class="mut">{meta}</div>
  <div class="mut" style="margin-top:6px">
    claims {section.get("claim_count", 0)}
    · material {section.get("material_count", 0)}
    · contradictions {section.get("contradiction_count", 0)}
    · refuted {section.get("refuted_count", 0)}
    · corroborated in full: {"yes" if section.get("all_corroborated") else "no"}
    {" · complaint filed" if section.get("complaint") else ""}
  </div>
</div>''']

    criteria = section.get("criteria") or []
    if criteria:
        rows = []
        for c in criteria:
            needs = []
            if c.get("requires_evidence"):
                needs.append("evidence")
            if c.get("requires_corroboration"):
                needs.append("corroboration")
            supplied = f'{c.get("evidence_refs", 0)} ref(s)'
            if c.get("corroboration") is not None:
                supplied += f' · corr {num(c.get("corroboration"))}'
            rows.append(
                "<tr>"
                f'<td>{html.escape(str(c.get("criterion", "")))}</td>'
                f'<td>{num(c.get("weight"))}</td>'
                f'<td>{num(c.get("judgement"))}</td>'
                f'<td>{num(c.get("credited"))}</td>'
                f'<td class="mut">{html.escape(", ".join(needs)) or "—"}</td>'
                f'<td class="mut">{supplied}</td>'
                "</tr>")
        out.append(f'''
<div class="card" style="margin-top:12px">
  <h3>Criteria — judged vs credited (caps applied by the gates)</h3>
  <table>
    <tr><th>Criterion</th><th>Weight</th><th>Judged</th><th>Credited</th>
        <th>Needs</th><th>Supplied</th></tr>
    {"".join(rows)}
  </table>
</div>''')

    claims = section.get("claims") or []
    if claims:
        rows = []
        for c in claims:
            status = str(c.get("status", "unverified"))
            rows.append(
                "<tr>"
                f'<td>{_pill(status)}</td>'
                f'<td>{num(c.get("grade"))}</td>'
                f'<td>{html.escape(str(c.get("claim", "")))}</td>'
                f'<td class="mut">{"material" if c.get("material") else "context"}</td>'
                f'<td class="mut">{html.escape(", ".join(c.get("support") or []) or "—")}</td>'
                f'<td class="mut">{html.escape(", ".join(c.get("contradict") or []) or "—")}</td>'
                f'<td class="mut">{html.escape("; ".join(c.get("reasons") or []))}</td>'
                "</tr>")
        out.append(f'''
<div class="card" style="margin-top:12px">
  <h3>Fact-check verdicts per claim (material claims first)</h3>
  <table>
    <tr><th>Status</th><th>Grade</th><th>Claim</th><th>Kind</th>
        <th>Supported by</th><th>Contradicted by</th><th>Why</th></tr>
    {"".join(rows)}
  </table>
</div>''')

    return "".join(out)


def _render_html(report: Dict[str, Any]) -> str:
    payload = _json_for_script(report)
    verification_html = _render_verification(report.get("verification"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Calibrix Report — {report['adapter']}</title>
<style>
  :root {{ --bg:#0d1117; --card:#161b22; --ink:#e6edf3; --mut:#8b949e;
          --acc:#58a6ff; --ok:#3fb950; --warn:#d29922; --bad:#f85149; }}
  * {{ box-sizing:border-box; margin:0; }}
  body {{ background:var(--bg); color:var(--ink);
         font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif; padding:24px; }}
  h1 {{ font-size:20px; margin-bottom:2px; }}
  .mut {{ color:var(--mut); }}
  .grid {{ display:grid; gap:12px; margin:16px 0;
          grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); }}
  .card {{ background:var(--card); border:1px solid #30363d;
          border-radius:8px; padding:14px; }}
  .card h3 {{ font-size:12px; text-transform:uppercase; color:var(--mut);
             margin-bottom:8px; letter-spacing:.05em; }}
  .big {{ font-size:22px; font-weight:600; }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
  th,td {{ text-align:left; padding:6px 10px; border-bottom:1px solid #21262d; }}
  th {{ color:var(--mut); font-weight:500; font-size:12px; }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:99px;
          font-size:11px; font-weight:600; }}
  .ok   {{ background:#12351c; color:var(--ok); }}
  .bad  {{ background:#3d1618; color:var(--bad); }}
  .warn {{ background:#3a2b0e; color:var(--warn); }}
</style>
</head>
<body>
<h1>Calibrix Report</h1>
<div class="mut">{report['adapter']} · {report['generated_at']} ·
open this file anywhere — no server needed</div>
<div class="card" style="margin-top:12px; border-color:var(--warn)">
  <h3>Evidence status: {report['evidence']['status']} · quality gate: {report['evidence']['quality_gate']}</h3>
  <div>{report['evidence']['note']}</div>
  <div class="mut" style="margin-top:6px">{report['evidence']['claim_boundary']}</div>
</div>
{verification_html}

<div class="grid" id="cards"></div>

<div class="card" style="margin-top:12px">
  <h3>Trials (fitness = scalarized co-objectives)</h3>
  <table id="trials"></table>
</div>

<div class="card" style="margin-top:12px">
  <h3>Selected kernel vector (lossless — reload via adapter.apply_to_vector)</h3>
  <code id="vec" class="mut" style="font-size:12px; word-break:break-all"></code>
</div>

<script>
const R = {payload};
const $ = (id) => document.getElementById(id);

// ---- summary cards --------------------------------------------------
const cards = [
  ["Free parameters", R.n_params, ""],
  ["Evaluations", R.n_evals, ""],
  ["Best fitness", R.best ? R.best.fitness : "—", ""],
  ["Elapsed", R.elapsed_seconds + "s", ""],
];
$("cards").innerHTML = cards.map(([k,v,u]) => `
  <div class="card"><h3>${{k}}</h3><div class="big">${{v}}<span class="mut">${{u}}</span></div></div>
`).join("");

// ---- baseline vs best -------------------------------------------------
if (R.best) {{
  const rows = Object.keys(R.baseline_train).map(name => {{
    const b = R.baseline_train[name], t = R.best.objectives[name] ?? "—",
          h = R.best.holdout[name] ?? "—";
    return `<tr><td>${{name}}</td><td>${{b}}</td><td>${{t}}</td><td>${{h}}</td></tr>`;
  }});
  $("trials").innerHTML = `<tr><th>Metric</th><th>Baseline</th>
     <th>Best (train)</th><th>Best (holdout)</th></tr>` + rows.join("");
}}

// ---- overfit alarm ---------------------------------------------------
if (R.overfit && R.overfit.flagged) {{
  const d = document.createElement("div");
  d.className = "card";
  d.style.borderColor = "var(--bad)";
  d.innerHTML = `<h3>⚠ Overfit alarm</h3>` +
    R.overfit.details.map(x =>
      `<div>${{x.metric}}: train ${{x.train}} vs holdout ${{x.holdout}} (gap ${{x.gap}})</div>`).join("");
  $("cards").appendChild(d);
}}

$("vec").textContent = R.best ? JSON.stringify(R.best.vector) : "";
</script>
</body>
</html>"""
