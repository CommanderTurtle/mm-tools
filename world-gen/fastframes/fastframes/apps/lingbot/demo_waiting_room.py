#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Per-server FIFO turns; ownership includes preparation and safe GPU teardown."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import OrderedDict
from typing import Any

from websockets.exceptions import ConnectionClosed

LOGGER = logging.getLogger(__name__)


class WaitingRoom:
    def __init__(self, application: Any, *, idle_seconds: float = 120):
        if not math.isfinite(idle_seconds) or idle_seconds <= 0:
            raise ValueError("waiting_room_idle_seconds must be positive and finite")
        self.application = application
        self.idle_seconds = float(idle_seconds)
        self.connections: dict[str, Any] = {}
        self.waiters: OrderedDict[str, None] = OrderedDict()
        self.owner: str | None = None
        self.last_activity = 0.0
        self.waiting_since: float | None = None
        self.start_task: asyncio.Task | None = None
        self.release_task: asyncio.Task | None = None
        self.watchdog: asyncio.Task | None = None
        self.shutting_down = False
        self._clock = time.monotonic
        self._notification_lock = asyncio.Lock()

    def connect(self, connection_id: str, websocket: Any) -> None:
        self.connections[connection_id] = websocket
        if self.watchdog is None and not self.shutting_down:
            self.watchdog = asyncio.create_task(self._watch_idle(), name="waiting-room-idle")

    def state(self, connection_id: str) -> dict[str, Any]:
        position = None
        if connection_id == self.owner:
            status = "owner"
        elif connection_id in self.waiters:
            status = "queued"
            position = list(self.waiters).index(connection_id) + 1
        else:
            status = "busy" if self.owner is not None or self.shutting_down else "available"
        remaining = None
        if (self.owner is not None and self.waiting_since is not None
                and self.start_task is None):
            baseline = max(self.last_activity, self.waiting_since)
            remaining = max(0, math.ceil(self.idle_seconds - (self._clock() - baseline)))
        return {
            "type": "queue_state", "enabled": True, "status": status,
            "position": position, "waiting_count": len(self.waiters),
            "idle_remaining_seconds": remaining,
            "idle_timeout_seconds": self.idle_seconds,
        }

    async def publish(self, *, released_owner: str | None = None, reason: str | None = None):
        # This lock serializes notifications only; it never protects GPU work.
        async with self._notification_lock:
            deliveries = []
            for connection_id, websocket in tuple(self.connections.items()):
                payload = self.state(connection_id)
                if connection_id == released_owner and reason:
                    payload["reason"] = reason
                deliveries.append(asyncio.wait_for(
                    self.application._send(websocket, payload), timeout=2,
                ))
            if deliveries:
                await asyncio.gather(*deliveries, return_exceptions=True)

    def _reserve(self, connection_id: str) -> None:
        self.owner = connection_id
        self.last_activity = self._clock()

    async def join(self, connection_id: str) -> None:
        if self.shutting_down:
            raise ValueError("The demo is shutting down")
        if self.owner is None:
            self._reserve(connection_id)
        elif self.owner != connection_id and connection_id not in self.waiters:
            if not self.waiters:
                self.waiting_since = self._clock()
            self.waiters[connection_id] = None
        await self.publish()

    async def activity(self, connection_id: str) -> None:
        if self.owner != connection_id:
            raise ValueError("Only the current player can update activity")
        if self.release_task is not None and not self.release_task.done():
            return
        self.last_activity = self._clock()
        if self.waiters:
            await self.publish()

    async def start(self, websocket: Any, connection_id: str, payload: dict[str, Any]) -> None:
        if (self.shutting_down or self.owner not in (None, connection_id)
                or self.start_task is not None
                or (self.release_task is not None and not self.release_task.done())):
            await self.application._error(
                websocket, "The GPU is in use or finishing another session", code="busy",
            )
            return
        if self.owner is None:
            self._reserve(connection_id)
        self.last_activity = self._clock()
        # Return to the socket receive loop immediately. Stop, prompt activity,
        # queue joins and disconnects remain responsive during engine.open().
        self.start_task = asyncio.create_task(
            self._open_world(websocket, connection_id, payload),
            name=f"world-start-{connection_id[:8]}",
        )
        await self.publish()

    async def _open_world(self, websocket, connection_id, payload):
        opened = False
        try:
            await self.application._start_session(websocket, connection_id, payload)
            active = self.application.active
            opened = (active is not None and active.owner_id == connection_id
                      and not active.stopping)
        except (ValueError, TypeError) as exc:
            try:
                await self.application._error(websocket, str(exc))
            except ConnectionClosed:
                pass
        except ConnectionClosed:
            pass
        except asyncio.CancelledError:
            # _start_session drains its shielded open and closes any abandoned
            # result before cancellation reaches here.
            raise
        except Exception:
            LOGGER.exception("Standalone world startup failed")
        finally:
            self.start_task = None
            if opened:
                # Preparation may take minutes on a cold worker. Start the
                # idle allowance when the player can actually use the world.
                self.last_activity = self._clock()
                await self.publish()
            if self.release_task is None or self.release_task.done():
                active = self.application.active
                if active is None or active.owner_id != connection_id:
                    await self.leave(connection_id, reason="start_failed")

    async def leave(self, connection_id: str, *, reason: str = "stopped", notify: bool = True):
        if connection_id in self.waiters:
            del self.waiters[connection_id]
            if not self.waiters:
                self.waiting_since = None
            await self.publish()
            return
        if self.owner != connection_id:
            return
        if self.release_task is None or self.release_task.done():
            self.release_task = asyncio.create_task(
                self._release_turn(connection_id, reason=reason, notify=notify),
                name=f"turn-release-{connection_id[:8]}",
            )
        # Disconnecting/cancelling the socket handler must not cancel cleanup.
        await asyncio.shield(self.release_task)

    async def _release_turn(self, connection_id: str, *, reason: str, notify: bool):
        try:
            opening = self.start_task
            if opening is not None and opening is not asyncio.current_task():
                opening.cancel()
                await asyncio.gather(opening, return_exceptions=True)
            active = self.application.active
            frames = active.frame_sequence if active is not None else 0
            await self.application._stop_session(connection_id, notify=None)
        except Exception:
            # Preserve ownership if teardown could not finish. A subsequent
            # Stop can retry; never admit a new model session over failed close.
            LOGGER.exception("Could not release standalone turn")
            return
        if self.owner != connection_id:
            return
        self.owner = None
        if not self.shutting_down:
            while self.waiters:
                next_owner, _ = self.waiters.popitem(last=False)
                if next_owner in self.connections:
                    self._reserve(next_owner)
                    break
        if not self.waiters:
            self.waiting_since = None
        # Teardown is complete. The promoted owner may start as soon as its
        # notification arrives, even while other notifications are flushing.
        self.release_task = None
        if notify and connection_id in self.connections:
            try:
                await asyncio.wait_for(self.application._send(
                    self.connections[connection_id],
                    {"type": "session", "status": "stopped", "frames": frames, "reason": reason},
                ), timeout=2)
            except Exception:
                pass
        await self.publish(released_owner=connection_id, reason=reason)

    async def session_released(self, connection_id: str):
        # Restart retains the turn through old-world close and new-world open.
        # Explicit turn cleanup owns promotion while it drains a model call.
        if self.start_task is not None or (self.release_task is not None and not self.release_task.done()):
            return
        await self.leave(connection_id, reason="complete", notify=False)

    async def disconnect(self, connection_id: str):
        self.connections.pop(connection_id, None)
        await self.leave(connection_id, reason="disconnected", notify=False)

    async def check_idle(self):
        if (self.owner is not None and self.waiting_since is not None
                and self.start_task is None):
            baseline = max(self.last_activity, self.waiting_since)
            if self._clock() - baseline >= self.idle_seconds:
                await self.leave(self.owner, reason="idle_timeout")

    async def _watch_idle(self):
        while not self.shutting_down:
            await asyncio.sleep(min(1.0, self.idle_seconds / 4))
            await self.check_idle()
            if self.waiters:
                await self.publish()

    async def shutdown(self):
        self.shutting_down = True
        self.waiters.clear()
        self.waiting_since = None
        if self.watchdog is not None:
            self.watchdog.cancel()
            await asyncio.gather(self.watchdog, return_exceptions=True)
            self.watchdog = None
        if self.owner is not None:
            await self.leave(self.owner, reason="shutdown", notify=False)
