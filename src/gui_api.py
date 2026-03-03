from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import httpx


@dataclass(frozen=True)
class ChatClient:
    base_url: str = "http://localhost:8000"
    timeout: float = 120.0

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def check_health(self) -> bool:
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(self._url("/agent/health"))
                resp.raise_for_status()
            return True
        except Exception:
            return False

    def send_message(self, message: str) -> Dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self._url("/agent/chat"), json={"message": message})
            resp.raise_for_status()
            data = resp.json()

        if not isinstance(data, dict) or "reply" not in data:
            raise RuntimeError(f"Unexpected response from server: {data!r}")
        return data

