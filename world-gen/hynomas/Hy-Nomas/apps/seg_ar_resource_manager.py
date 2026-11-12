"""Durable host leasing and FIFO waitlist for the multi-user demo."""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


class ResourceBusyError(RuntimeError):
    """Raised when every configured 8-GPU host is already assigned."""


class SessionBindingConflictError(RuntimeError):
    """Raised when another browser lease already owns a World session."""


class ResourceManager:
    """Assign one complete GPU host to one browser session at a time."""

    def __init__(
        self,
        path: str | Path,
        hosts: list[dict[str, str]],
        *,
        lease_seconds: float = 1800,
        host_stale_seconds: float = 30,
        allow_waitlist: bool = True,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lock_path = self.path.with_suffix(f"{self.path.suffix}.lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lease_seconds = float(lease_seconds)
        self.host_stale_seconds = float(host_stale_seconds)
        self.allow_waitlist = bool(allow_waitlist)
        normalized: dict[str, dict[str, Any]] = {}
        for item in hosts:
            host_id = str(item.get("id") or "").strip()
            url = str(item.get("url") or "").strip().rstrip("/")
            if not host_id or not url:
                raise ValueError("every host requires non-empty id and url")
            if host_id in normalized:
                raise ValueError(f"duplicate host id: {host_id}")
            normalized[host_id] = {"id": host_id, "url": url}
        if not normalized:
            raise ValueError("at least one inference host is required")
        self.configured_hosts = normalized
        with self._locked_state(write=True) as state:
            self._sync_hosts(state)

    def _empty_state(self) -> Dict[str, Any]:
        return {"hosts": {}, "users": {}, "waitlist": [], "sessions": {}}

    def _read_unlocked(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return self._empty_state()
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._empty_state()
        if not isinstance(value, dict):
            return self._empty_state()
        value.setdefault("hosts", {})
        value.setdefault("users", {})
        value.setdefault("waitlist", [])
        value.setdefault("sessions", {})
        if not isinstance(value["sessions"], dict):
            value["sessions"] = {}
        return value

    def _write_unlocked(self, state: Dict[str, Any]) -> None:
        fd, raw_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        tmp_path = Path(raw_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(state, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        finally:
            tmp_path.unlink(missing_ok=True)

    @contextmanager
    def _locked_state(self, *, write: bool) -> Iterator[Dict[str, Any]]:
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX if write else fcntl.LOCK_SH)
            state = self._read_unlocked()
            try:
                yield state
                if write:
                    self._write_unlocked(state)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _sync_hosts(self, state: Dict[str, Any]) -> None:
        hosts = state["hosts"]
        for host_id, configured in self.configured_hosts.items():
            current = hosts.setdefault(
                host_id,
                {
                    "id": host_id,
                    "url": configured["url"],
                    "online": False,
                    "backend_available": False,
                    "draining": False,
                    "heartbeat_at": None,
                    "owner_token": None,
                },
            )
            current["url"] = configured["url"]
            current["configured"] = True
            current.setdefault("backend_available", False)
            current.setdefault("draining", False)
        for host_id in list(hosts):
            if host_id not in self.configured_hosts:
                host = hosts[host_id]
                has_active_session = any(
                    isinstance(binding, dict)
                    and binding.get("host_id") == host_id
                    and binding.get("owner_token") in state["users"]
                    for binding in state.setdefault("sessions", {}).values()
                )
                # Configuration changes must not silently migrate an active
                # browser either. Preserve enough information to keep routing
                # its requests to the original machine until the lease ends.
                if host.get("owner_token") or has_active_session:
                    host["configured"] = False
                    host["online"] = False
                    host["backend_available"] = False
                    continue
                del hosts[host_id]

    def _host_online(self, host: Dict[str, Any], now: float) -> bool:
        heartbeat_at = host.get("heartbeat_at")
        return bool(
            host.get("online")
            and heartbeat_at is not None
            and now - float(heartbeat_at) <= self.host_stale_seconds
        )

    def _enqueue(
        self, state: Dict[str, Any], token: str, *, front: bool = False
    ) -> None:
        user = state["users"].get(token)
        if not isinstance(user, dict):
            return
        waitlist = state["waitlist"]
        if token in waitlist:
            waitlist.remove(token)
        if front:
            waitlist.insert(0, token)
        else:
            waitlist.append(token)
        user["status"] = "waiting"
        user["host_id"] = None

    def _release(self, state: Dict[str, Any], token: str) -> None:
        user = state["users"].get(token)
        if not isinstance(user, dict):
            return
        host_id = user.get("host_id")
        host = state["hosts"].get(host_id)
        if isinstance(host, dict) and host.get("owner_token") == token:
            host["owner_token"] = None
            # Browser release and CUDA cleanup are asynchronous. Keep the
            # physical node unavailable until a later health probe confirms
            # that its queue and resident interactive Session are empty.
            host["draining"] = True
        if token in state["waitlist"]:
            state["waitlist"].remove(token)
        sessions = state.setdefault("sessions", {})
        for session_id, binding in list(sessions.items()):
            if isinstance(binding, dict) and binding.get("owner_token") == token:
                del sessions[session_id]
        state["users"].pop(token, None)

    def _reconcile(self, state: Dict[str, Any], now: float) -> None:
        self._sync_hosts(state)
        for token, user in list(state["users"].items()):
            idle = now - float(user.get("last_seen_at") or 0) > self.lease_seconds
            if idle:
                self._release(state, token)

        # Host health never changes ownership. One browser lease owns exactly
        # one machine until explicit logout or lease expiry. A failed/restarting
        # Gateway therefore makes requests fail temporarily against the same
        # host instead of moving the World to another host.
        sessions = state.setdefault("sessions", {})
        for session_id, binding in list(sessions.items()):
            if (
                not isinstance(binding, dict)
                or binding.get("owner_token") not in state["users"]
                or binding.get("host_id") not in state["hosts"]
            ):
                del sessions[session_id]

        state["waitlist"] = [
            token
            for token in state["waitlist"]
            if token in state["users"]
            and state["users"][token].get("status") == "waiting"
        ]
        session_reserved_host_ids = {
            str(binding.get("host_id"))
            for binding in sessions.values()
            if isinstance(binding, dict)
            and binding.get("owner_token") in state["users"]
            and binding.get("host_id") in state["hosts"]
        }
        free_hosts = [
            host
            for host in state["hosts"].values()
            if self._host_online(host, now)
            and bool(host.get("backend_available"))
            and not bool(host.get("draining"))
            and not host.get("owner_token")
            and str(host["id"]) not in session_reserved_host_ids
        ]
        free_hosts.sort(key=lambda host: str(host["id"]))
        while free_hosts and state["waitlist"]:
            host = free_hosts.pop(0)
            token = state["waitlist"].pop(0)
            user = state["users"].get(token)
            if not isinstance(user, dict):
                continue
            host["owner_token"] = token
            user["status"] = "assigned"
            user["host_id"] = host["id"]
            user["assigned_at"] = now

    def _public_user(
        self, state: Dict[str, Any], token: str, now: float
    ) -> Optional[Dict[str, Any]]:
        user = state["users"].get(token)
        if not isinstance(user, dict):
            return None
        wait_position = None
        if user.get("status") == "waiting" and token in state["waitlist"]:
            wait_position = state["waitlist"].index(token) + 1
        return {
            "user_id": user["user_id"],
            "status": user["status"],
            "host_id": user.get("host_id"),
            "waitlist_position": wait_position,
            "waitlist_size": len(state["waitlist"]),
            "lease_seconds": self.lease_seconds,
            "lease_expires_at": float(user["last_seen_at"]) + self.lease_seconds,
            "server_time": now,
        }

    @staticmethod
    def _clean_page_id(page_id: str | None) -> str:
        clean_page_id = str(page_id or "").strip()
        if len(clean_page_id) > 128:
            raise ValueError("page_id must contain at most 128 characters")
        return clean_page_id

    @staticmethod
    def _observe_page(user: Dict[str, Any], page_id: str, now: float) -> None:
        if not page_id:
            return
        connected = user.setdefault("connected_pages", {})
        disconnected = user.setdefault("disconnected_pages", {})
        # A request that was already in flight when pagehide fired must not
        # revive the closed page. A reload gets a new page id.
        if page_id in disconnected:
            return
        connected[page_id] = now
        user["disconnect_deadline_at"] = None

    def login(
        self,
        user_id: str,
        *,
        page_id: str | None = None,
    ) -> tuple[str, Dict[str, Any]]:
        clean_id = user_id.strip()
        if not clean_id or len(clean_id) > 128:
            raise ValueError("user_id must contain 1 to 128 characters")
        clean_page_id = self._clean_page_id(page_id)
        now = time.time()
        with self._locked_state(write=True) as state:
            self._reconcile(state, now)
            for existing in state["users"].values():
                if existing.get("user_id") == clean_id:
                    raise ValueError("this user_id is already logged in")
            token = secrets.token_urlsafe(32)
            state["users"][token] = {
                "user_id": clean_id,
                "status": "waiting",
                "host_id": None,
                "created_at": now,
                "last_seen_at": now,
                "assigned_at": None,
                "disconnect_deadline_at": None,
                "connected_pages": ({clean_page_id: now} if clean_page_id else {}),
                "disconnected_pages": {},
            }
            state["waitlist"].append(token)
            self._reconcile(state, now)
            if (
                not self.allow_waitlist
                and state["users"][token].get("status") != "assigned"
            ):
                self._release(state, token)
                raise ResourceBusyError(
                    "All 8-GPU nodes are currently in use. Please try again later."
                )
            return token, self._public_user(state, token, now)  # type: ignore[arg-type]

    def touch(
        self,
        token: str,
        *,
        page_id: str | None = None,
    ) -> Optional[Dict[str, Any]]:
        clean_page_id = self._clean_page_id(page_id)
        now = time.time()
        with self._locked_state(write=True) as state:
            self._reconcile(state, now)
            user = state["users"].get(token)
            if not isinstance(user, dict):
                return None
            if clean_page_id and clean_page_id in user.setdefault(
                "disconnected_pages", {}
            ):
                # A request that was already in flight when pagehide fired
                # must not refresh the lease or cancel pending cleanup.
                return self._public_user(state, token, now)
            user["last_seen_at"] = now
            self._observe_page(user, clean_page_id, now)
            self._reconcile(state, now)
            return self._public_user(state, token, now)

    def status(
        self,
        token: str,
        *,
        page_id: str | None = None,
    ) -> Optional[Dict[str, Any]]:
        clean_page_id = self._clean_page_id(page_id)
        now = time.time()
        with self._locked_state(write=True) as state:
            self._reconcile(state, now)
            user = state["users"].get(token)
            if isinstance(user, dict):
                self._observe_page(user, clean_page_id, now)
            return self._public_user(state, token, now)

    def reconnect(
        self,
        token: str,
        *,
        page_id: str | None = None,
    ) -> Optional[Dict[str, Any]]:
        """Cancel page-close cleanup without extending the inactivity lease."""
        clean_page_id = self._clean_page_id(page_id)
        now = time.time()
        with self._locked_state(write=True) as state:
            self._reconcile(state, now)
            user = state["users"].get(token)
            if isinstance(user, dict):
                self._observe_page(user, clean_page_id, now)
            return self._public_user(state, token, now)

    def disconnect(
        self,
        token: str,
        *,
        page_id: str | None = None,
        grace_seconds: float = 3.0,
    ) -> Optional[Dict[str, Any]]:
        """Record a closed page without releasing its exclusive machine.

        Page-close signals are not reliable enough to transfer a resident
        World to another user. Only explicit logout or lease expiry releases
        the host; a later reload can therefore continue on the same machine.
        """
        clean_page_id = self._clean_page_id(page_id)
        now = time.time()
        del grace_seconds
        with self._locked_state(write=True) as state:
            self._reconcile(state, now)
            user = state["users"].get(token)
            if not isinstance(user, dict):
                return None
            connected = user.setdefault("connected_pages", {})
            disconnected = user.setdefault("disconnected_pages", {})
            if clean_page_id:
                connected.pop(clean_page_id, None)
                disconnected[clean_page_id] = now
            user["disconnect_deadline_at"] = None
            return self._public_user(state, token, now)

    def logout(self, token: str) -> None:
        now = time.time()
        with self._locked_state(write=True) as state:
            self._release(state, token)
            self._reconcile(state, now)

    def assigned_host(self, token: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._locked_state(write=True) as state:
            self._reconcile(state, now)
            user = state["users"].get(token)
            if not isinstance(user, dict) or user.get("status") != "assigned":
                return None
            host = state["hosts"].get(user.get("host_id"))
            return dict(host) if isinstance(host, dict) else None

    @staticmethod
    def _clean_session_id(session_id: str) -> str:
        clean_session_id = str(session_id or "").strip()
        if not clean_session_id or len(clean_session_id) > 256:
            raise ValueError("session_id must contain 1 to 256 characters")
        return clean_session_id

    def bind_session_host(
        self,
        token: str,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Atomically bind a World session to its first assigned host.

        The binding is stored in the same locked state file as leases, so it
        survives dispatcher restarts and concurrent POSTs cannot bind one
        session to different hosts. Existing bindings win over a user's
        current allocation to recover state created by older dispatchers that
        could migrate a lease after a failed health probe.
        """
        clean_session_id = self._clean_session_id(session_id)
        now = time.time()
        with self._locked_state(write=True) as state:
            self._reconcile(state, now)
            user = state["users"].get(token)
            if not isinstance(user, dict):
                return None
            sessions = state.setdefault("sessions", {})
            binding = sessions.get(clean_session_id)
            if isinstance(binding, dict):
                if binding.get("owner_token") != token:
                    raise SessionBindingConflictError(
                        "World session belongs to another browser lease"
                    )
                host = state["hosts"].get(binding.get("host_id"))
                if not isinstance(host, dict):
                    raise SessionBindingConflictError(
                        "World session host is no longer configured"
                    )
                reserved_by_another_browser = any(
                    other_session_id != clean_session_id
                    and isinstance(other_binding, dict)
                    and other_binding.get("host_id") == host["id"]
                    and other_binding.get("owner_token") != token
                    and other_binding.get("owner_token") in state["users"]
                    for other_session_id, other_binding in sessions.items()
                )
                mapped_owner = host.get("owner_token")
                if mapped_owner not in (None, token) or reserved_by_another_browser:
                    raise SessionBindingConflictError(
                        "World session host is assigned to another browser; "
                        "refusing to migrate"
                    )
                current_host_id = user.get("host_id")
                if current_host_id != host["id"]:
                    current_host = state["hosts"].get(current_host_id)
                    if (
                        isinstance(current_host, dict)
                        and current_host.get("owner_token") == token
                    ):
                        current_host["owner_token"] = None
                        current_host["draining"] = True
                    if token in state["waitlist"]:
                        state["waitlist"].remove(token)
                    host["owner_token"] = token
                    user["status"] = "assigned"
                    user["host_id"] = host["id"]
                    user["assigned_at"] = now
                binding["last_seen_at"] = now
                return dict(host)

            if user.get("status") != "assigned":
                return None
            host = state["hosts"].get(user.get("host_id"))
            if not isinstance(host, dict):
                return None
            sessions[clean_session_id] = {
                "host_id": str(host["id"]),
                "owner_token": token,
                "user_id": str(user["user_id"]),
                "bound_at": now,
                "last_seen_at": now,
            }
            return dict(host)

    def update_host(
        self,
        host_id: str,
        online: bool,
        *,
        available: Optional[bool] = None,
    ) -> None:
        now = time.time()
        with self._locked_state(write=True) as state:
            self._sync_hosts(state)
            host = state["hosts"].get(host_id)
            if host is None:
                raise KeyError(host_id)
            if online:
                host["online"] = True
                host["heartbeat_at"] = now
            elif host.get("heartbeat_at") is None:
                host["online"] = False
            # A single failed health probe must stop new assignments, but it
            # must not immediately evict the current browser to another host.
            # Keep the last successful heartbeat until host_stale_seconds has
            # elapsed; short Gateway hot restarts then preserve sticky routing.
            host["backend_available"] = (
                bool(online) if available is None else bool(online and available)
            )
            if host["backend_available"] and not host.get("owner_token"):
                host["draining"] = False
            host["last_probe_at"] = now
            self._reconcile(state, now)

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        with self._locked_state(write=True) as state:
            self._reconcile(state, now)
            session_reserved_host_ids = {
                str(binding.get("host_id"))
                for binding in state.setdefault("sessions", {}).values()
                if isinstance(binding, dict)
                and binding.get("owner_token") in state["users"]
                and binding.get("host_id") in state["hosts"]
            }
            hosts = []
            for host in sorted(state["hosts"].values(), key=lambda item: item["id"]):
                session_reserved = str(host["id"]) in session_reserved_host_ids
                hosts.append(
                    {
                        "id": host["id"],
                        "url": host["url"],
                        "online": self._host_online(host, now),
                        "available": bool(
                            self._host_online(host, now)
                            and host.get("backend_available")
                            and not host.get("draining")
                            and not host.get("owner_token")
                            and not session_reserved
                        ),
                        "busy": bool(
                            host.get("owner_token")
                            or host.get("draining")
                            or not host.get("backend_available")
                            or session_reserved
                        ),
                    }
                )
            return {
                "hosts": hosts,
                "waitlist_size": len(state["waitlist"]),
                "active_users": len(state["users"]),
            }

    def active_allocations(self) -> list[Dict[str, Any]]:
        """Return current assigned users without exposing session tokens."""
        now = time.time()
        with self._locked_state(write=True) as state:
            self._reconcile(state, now)
            allocations = []
            for user in state["users"].values():
                if user.get("status") != "assigned" or not user.get("host_id"):
                    continue
                allocations.append(
                    {
                        "user_id": str(user["user_id"]),
                        "host_id": str(user["host_id"]),
                        "assigned_at": user.get("assigned_at"),
                        "last_seen_at": user.get("last_seen_at"),
                    }
                )
            return sorted(
                allocations,
                key=lambda item: (item["host_id"], item["user_id"]),
            )
