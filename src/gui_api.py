from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import httpx


def _fastapi_error_detail(response: httpx.Response) -> str:
    """Extract ``detail`` from a FastAPI/Starlette JSON error body."""
    try:
        data = response.json()
    except Exception:
        return (response.text or "").strip()[:800]
    detail = data.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        parts: list[str] = []
        for item in detail:
            if isinstance(item, dict) and "msg" in item:
                parts.append(str(item["msg"]))
            else:
                parts.append(str(item))
        return "; ".join(parts) if parts else str(data)[:800]
    if detail is not None:
        return str(detail)
    return str(data)[:800]


@dataclass(frozen=True)
class ChatClient:
    # Use IPv4 loopback by default; some Docker Desktop/WSL setups reset IPv6 ::1 connections.
    base_url: str = "http://127.0.0.1:8000"
    timeout: float = 120.0
    # Must match backend `session_id` for reasoning cache and CLS-M scoping.
    session_id: str = "default"

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
            resp = client.post(
                self._url("/agent/chat"),
                json={"message": message, "session_id": self.session_id},
            )
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                msg = _fastapi_error_detail(exc.response)
                raise RuntimeError(
                    msg or f"Server error {exc.response.status_code} for {exc.request.url}"
                ) from exc
            data = resp.json()

        if not isinstance(data, dict) or "reply" not in data:
            raise RuntimeError(f"Unexpected response from server: {data!r}")
        return data

    def schedule_task(self, message: str) -> Dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                self._url("/agent/schedule"),
                json={"message": message, "session_id": self.session_id},
            )
            resp.raise_for_status()
            data = resp.json()
        if not isinstance(data, dict) or "note" not in data:
            raise RuntimeError(f"Unexpected schedule response from server: {data!r}")
        return data

    def get_prompt_debug(self, message: str) -> Dict[str, Any]:
        """Fetch the structured prompt for a message without generating a reply."""
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                self._url("/agent/prompt-debug"),
                json={"message": message, "session_id": self.session_id},
            )
            resp.raise_for_status()
            data = resp.json()
        if not isinstance(data, dict) or "prompt" not in data:
            raise RuntimeError(f"Unexpected prompt-debug response from server: {data!r}")
        return data

    def get_memory_debug(self, limit: int = 1) -> Dict[str, Any]:
        """Fetch recent CLS-M memory metrics for debugging.

        Returns the raw JSON payload from the backend debug endpoint.
        """
        params = {"limit": int(limit)}
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(self._url("/agent/memory/debug"), params=params)
            resp.raise_for_status()
            data = resp.json()
        if not isinstance(data, dict) or "samples" not in data:
            raise RuntimeError(f"Unexpected debug response from server: {data!r}")
        return data

    def get_reasoning_cache_debug(self, limit: int = 10) -> Dict[str, Any]:
        """Fetch topic-head cache snapshot for the configured session."""
        params = {"session_id": self.session_id, "limit": int(limit)}
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(self._url("/agent/reasoning-cache-debug"), params=params)
            resp.raise_for_status()
            data = resp.json()
        if not isinstance(data, dict) or "topic_heads" not in data:
            raise RuntimeError(f"Unexpected reasoning-cache response: {data!r}")
        return data

