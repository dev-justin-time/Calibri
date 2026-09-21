# SPDX-License-Identifier: MIT
"""Message bus for the agent layer.

Design constraints:

  * deterministic — synchronous wave dispatch (publish a batch, deliver,
    collect the new messages that produces, repeat until quiescent or a
    round cap). No threads, no locks, replayable order.
  * auditable — every message is hash-chainable; the Auditor agent feeds
    them into studio.governance.AuditChain.
  * wildcard — agents may subscribe to "*" to observe all traffic.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List

_SEQ = itertools.count(1)


@dataclass
class Message:
    """One routed event on the bus."""
    topic: str
    payload: Dict = field(default_factory=dict)
    sender: str = "system"
    seq: int = field(default_factory=lambda: next(_SEQ))
    ts: float = field(default_factory=time.time)


Handler = Callable[[Message, "Bus"], None]


class Bus:
    """Topic router: publish() delivers synchronously to subscribers.

    Handlers that publish create the next wave; `drain()` loops waves
    until the bus is quiet (or `max_rounds` — a runaway agent loop must
    surface as a truncated run, not a hang).
    """

    def __init__(self) -> None:
        self._subs: Dict[str, List[Handler]] = {}
        self._queue: List[Message] = []
        self.log: List[Message] = []

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subs.setdefault(topic, []).append(handler)

    def publish(self, topic: str, payload: Dict | None = None,
                sender: str = "system") -> Message:
        msg = Message(topic=topic, payload=dict(payload or {}), sender=sender)
        self.log.append(msg)
        self._queue.append(msg)
        return msg

    def _deliver(self) -> int:
        """Deliver one wave; returns messages queued by handlers."""
        wave, self._queue = self._queue, []
        queued = 0
        for msg in wave:
            for topic, handlers in list(self._subs.items()):
                if topic != "*" and topic != msg.topic:
                    continue
                for h in handlers.get("*", []) if topic == "*" else handlers:
                    h(msg, self)
                    queued = len(self._queue)
        return queued

    def drain(self, max_rounds: int = 64) -> int:
        """Run waves until quiet; returns rounds used (0 remaining messages
        means the pipeline reached a fixed point)."""
        rounds = 0
        while self._queue and rounds < max_rounds:
            self._deliver()
            rounds += 1
        return rounds

    @property
    def pending(self) -> int:
        return len(self._queue)
