# SPDX-License-Identifier: MIT
"""Agent base class."""

from __future__ import annotations

from typing import Tuple

from .bus import Bus, Message
from .tools import ToolSurface


class Agent:
    """A role agent: subscribes to topics, handles messages, publishes outputs.

    Subclasses set `role` and `subscribes`, and implement `handle`.
    Agents are stateless across missions except for explicitly named
    state (e.g. the Auditor's chain) — the message payloads carry it.
    """

    role: str = "agent"
    subscribes: Tuple[str, ...] = ()

    def __init__(self, bus: Bus, tools: ToolSurface) -> None:
        self.bus = bus
        self.tools = tools
        self.handled: int = 0
        for topic in self.subscribes:
            bus.subscribe(topic, self._route)

    def _route(self, msg: Message, bus: Bus) -> None:
        self.handled += 1
        self.handle(msg, bus)

    def handle(self, msg: Message, bus: Bus) -> None:
        raise NotImplementedError

    def say(self, bus: Bus, topic: str, payload: Dict | None = None) -> None:
        bus.publish(topic, payload, sender=self.role)
