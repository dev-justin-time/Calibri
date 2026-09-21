# SPDX-License-Identifier: MIT
# Calibrix report generation: JSON + self-contained HTML dashboard.
#
# The HTML is a single file with inline JS that works from file:// with no
# server, no build step, and no network — free hosting (GitHub Pages,
# Netlify free, Cloudflare Pages) or just double-click the file.

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List


def build_report(result: Dict[str, Any], adapter_name: str,
                 scorer_names: List[str]) -> Dict[str, Any]:
    """Serializable summary of a SearchEngine.run() result."""
    trials = result.get("trials", [])
    best: Any = result.get("best")
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "adapter": adapter_name,
        "scorers": scorer_names,
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
                 scorer_names: List[str], out_dir: str) -> Dict[str, str]:
    """Write report.json + report.html into out_dir. Returns file paths."""
    report = build_report(result, adapter_name, scorer_names)
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(_render_html(report))

    return {"json": json_path, "html": html_path}


def _render_html(report: Dict[str, Any]) -> str:
    payload = json.dumps(report, ensure_ascii=False)
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
