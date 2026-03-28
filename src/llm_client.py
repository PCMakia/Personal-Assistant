"""
LLM client abstraction: talks to Ollama in Docker (http://ollama:11434)
and can be replaced later by a vLLM client.
"""
import os
import json
from typing import List, Dict, Any, Iterator, Optional, AsyncIterator

import httpx


class LLMClient:
    """
    Client for the local LLM backend (Ollama in Docker).
    Base URL is read from OLLAMA_BASE_URL (default http://localhost:11434 for local runs;
    Docker Compose sets http://ollama:11434 for container-to-container).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "qwen3:4b",
        timeout: float = 120.0,
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.model = model
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        stream: bool = False,
        **params: Any,
    ) -> Dict[str, Any]:
        """
        Send a chat request and return a single response.
        Returns dict with keys: completion (str), usage (dict with prompt_tokens, completion_tokens).
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if params:
            payload["options"] = params

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content = (data.get("message") or {}).get("content") or ""
        eval_count = data.get("eval_count", 0)
        prompt_eval_count = data.get("prompt_eval_count", 0)

        return {
            "completion": content,
            "usage": {
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": eval_count,
            },
        }

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **params: Any,
    ) -> Iterator[str]:
        """
        Send a chat request with stream=True and yield content chunks.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if params:
            payload["options"] = params

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    part = json.loads(line)
                    if part.get("error"):
                        raise RuntimeError(part["error"])
                    msg = (part.get("message") or {}).get("content") or ""
                    if msg:
                        yield msg
                    if part.get("done"):
                        break

    async def async_stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **params: Any,
    ) -> AsyncIterator[str]:
        """
        Async variant of stream_generate suitable for FastAPI websocket handlers.
        """
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if params:
            payload["options"] = params

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    part = json.loads(line)
                    if part.get("error"):
                        raise RuntimeError(part["error"])
                    msg = (part.get("message") or {}).get("content") or ""
                    if msg:
                        yield msg
                    if part.get("done"):
                        break