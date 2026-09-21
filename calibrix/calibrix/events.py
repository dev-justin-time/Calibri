# Calibrix event log: reproducibility + audit trail (repurposed from Heretic's
# reproduce.py philosophy: everything needed to re-run and to audit a run).

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


class EventLog:
    """Append-only JSONL event log for a search run."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self.events: List[Dict[str, Any]] = []

    def log(self, kind: str, **data: Any) -> None:
        ev = {"t": time.time(), "kind": kind, **data}
        self.events.append(ev)
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")

    def all(self) -> List[Dict[str, Any]]:
        return list(self.events)
