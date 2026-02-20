from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class _ClientEntry:
    client_id: str
    first_seen: float
    last_seen: float
    remote_addr: str | None = None
    user_agent: str | None = None


class UiSessionQueue:
    def __init__(self, lease_seconds: int = 20) -> None:
        self.lease_seconds = int(lease_seconds)
        self._lock = threading.Lock()
        self._clients: dict[str, _ClientEntry] = {}
        self._order: list[str] = []

    def heartbeat(
        self,
        client_id: str,
        remote_addr: str | None = None,
        user_agent: str | None = None,
    ) -> dict:
        now = time.time()
        with self._lock:
            self._cleanup_locked(now)
            entry = self._clients.get(client_id)
            if entry is None:
                entry = _ClientEntry(
                    client_id=client_id,
                    first_seen=now,
                    last_seen=now,
                    remote_addr=remote_addr,
                    user_agent=user_agent,
                )
                self._clients[client_id] = entry
                self._order.append(client_id)
            else:
                entry.last_seen = now
                if remote_addr:
                    entry.remote_addr = remote_addr
                if user_agent:
                    entry.user_agent = user_agent
            return self._snapshot_locked(client_id, now)

    def leave(self, client_id: str) -> dict:
        with self._lock:
            removed = client_id in self._clients
            self._clients.pop(client_id, None)
            if client_id in self._order:
                self._order.remove(client_id)
            now = time.time()
            self._cleanup_locked(now)
            controller_client_id = self._order[0] if self._order else None
            return {
                "client_id": client_id,
                "left": removed,
                "queue_size": len(self._order),
                "controller_client_id": controller_client_id,
            }

    def is_controller(self, client_id: str) -> bool:
        now = time.time()
        with self._lock:
            self._cleanup_locked(now)
            if not self._order:
                return False
            return self._order[0] == client_id

    def _snapshot_locked(self, client_id: str, now: float) -> dict:
        queue_size = len(self._order)
        controller_client_id = self._order[0] if self._order else None
        try:
            queue_position = self._order.index(client_id) + 1
        except ValueError:
            queue_position = None
        role = "controller" if controller_client_id == client_id else "spectator"

        expires_in = self.lease_seconds
        entry = self._clients.get(client_id)
        if entry is not None:
            expires_in = max(
                0,
                int(self.lease_seconds - (now - entry.last_seen)),
            )

        return {
            "client_id": client_id,
            "role": role,
            "queue_position": queue_position,
            "queue_size": queue_size,
            "controller_client_id": controller_client_id,
            "lease_seconds": self.lease_seconds,
            "expires_in_seconds": expires_in,
        }

    def _cleanup_locked(self, now: float) -> None:
        stale_cutoff = now - float(self.lease_seconds)
        stale = []
        for client_id, entry in self._clients.items():
            if entry.last_seen < stale_cutoff:
                stale.append(client_id)

        for client_id in stale:
            self._clients.pop(client_id, None)
            if client_id in self._order:
                self._order.remove(client_id)
