# SPDX-License-Identifier: MIT
"""OpenWatch Daily — unbiased daily audit for every AI platform.

Reuses the Doberwatch primitives (similarity, verification, grading, cache,
ledger) but audits *platforms* instead of finance answers. Six criteria,
identical for every platform, three independent judges, PASS only if every
material claim is corroborated.

  criteria  — PLATFORM_RUBRIC (0.80 PASS, contested veto, evidence gates)
  prompts   — versioned 20-prompt pack (hashed)
  evaluate  — aggregate prompt scores -> per-criterion scores -> Grade
  blog      — build/write the daily HTML+JSON (your disk owns the data)

Everything is deterministic and offline unless you plug live model callers
into ``evaluate``; the grading gates are the same code as Doberwatch Finance.
"""

from .blog import (OPENWATCH_SCHEMA, build_openwatch_report,
                 build_openwatch_report_localized, write_openwatch_report,
                 write_openwatch_reports_all_locales)
from .criteria import PLATFORM_CRITERIA, PLATFORM_RUBRIC
from .evaluate import (aggregate_criterion_scores, grade_platform_run,
                       neutrality_score)
from .locales import LOCALES, get_locale, t
from .obscura_adapter import (ANONYMITY_CHECKLIST, INSTALL_HINT,
                              OBSCURA_BINARY, OBSCURA_CDP_URL, ObscuraStatus,
                              build_fetch_cmd, build_scrape_cmd,
                              build_serve_cmd, cdp_connect_snippet,
                              evidence_capture_plan, mcp_config_snippet, status)
from .prompts import PACK_HASH, PROMPT_PACK_V1, Prompt, prompts_by_criterion

__all__ = [
    "PLATFORM_CRITERIA", "PLATFORM_RUBRIC",
    "PROMPT_PACK_V1", "PACK_HASH", "Prompt", "prompts_by_criterion",
    "aggregate_criterion_scores", "grade_platform_run", "neutrality_score",
    "OPENWATCH_SCHEMA", "build_openwatch_report", "build_openwatch_report_localized",
    "write_openwatch_report", "write_openwatch_reports_all_locales",
    "LOCALES", "get_locale", "t",
    "OBSCURA_BINARY", "OBSCURA_CDP_URL", "ObscuraStatus", "ANONYMITY_CHECKLIST", "INSTALL_HINT",
    "status", "build_fetch_cmd", "build_serve_cmd", "build_scrape_cmd",
    "cdp_connect_snippet", "mcp_config_snippet", "evidence_capture_plan",
]
