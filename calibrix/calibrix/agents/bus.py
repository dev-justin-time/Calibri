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
        """Deliver one wave; returns messages queued by handlers.

        A message reaches the handlers registered for its own topic, then the
        wildcard observers — a fixed order, so the run is replayable rather
        than dependent on dict/subscription order. (Both keys used to be
        iterated together and the *subscriber list* was indexed with ``.get``,
        which raised AttributeError and took the whole pipeline down the
        moment anything subscribed to ``"*"``.)

        Handler lists are copied before dispatch, so a handler may subscribe
        during delivery without mutating the loop it is in.
        """
        wave, self._queue = self._queue, []
        for msg in wave:
            for h in (list(self._subs.get(msg.topic, ()))
                      + list(self._subs.get("*", ()))):
                h(msg, self)
        return len(self._queue)

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
