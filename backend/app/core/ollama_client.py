"""
Minimal HTTP client for a locally-running Ollama daemon.

No API key. No cloud provider. Talks only to OLLAMA_BASE_URL (default
http://localhost:11434), which is a process running on the same machine
(or same docker network) as this backend. All failure modes are caught and
turned into a structured, non-throwing result so a RAG failure never
propagates into an ML-prediction failure.
"""
from dataclasses import dataclass
from typing import Optional

import requests

from rag.config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    OLLAMA_TIMEOUT_SECONDS,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)


@dataclass
class OllamaResult:
    ok: bool
    text: str = ""
    error: Optional[str] = None  # internal-only; never sent verbatim to the frontend


class OllamaClient:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = LLM_MODEL,
        timeout: float = OLLAMA_TIMEOUT_SECONDS,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def health_check(self) -> bool:
        """True if the Ollama daemon is reachable and the configured model is pulled."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            # Ollama tags include a ":latest"/":3b" suffix; match on the base name too.
            return any(self.model == m or m.startswith(self.model.split(":")[0]) for m in models)
        except requests.RequestException:
            return False
        except Exception:
            return False

    def generate(self, prompt: str) -> OllamaResult:
        """
        Single-turn generation against /api/generate (non-streaming).
        Never raises -- all failure modes are surfaced via OllamaResult.ok=False.
        """
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": LLM_TEMPERATURE,
                        "num_predict": LLM_MAX_TOKENS,
                    },
                },
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError:
            return OllamaResult(ok=False, error="connection_error")
        except requests.exceptions.Timeout:
            return OllamaResult(ok=False, error="timeout")
        except requests.RequestException as e:
            return OllamaResult(ok=False, error=f"request_error: {e}")

        if resp.status_code == 404:
            return OllamaResult(ok=False, error="model_unavailable")
        if resp.status_code != 200:
            return OllamaResult(ok=False, error=f"http_{resp.status_code}")

        try:
            data = resp.json()
        except ValueError:
            return OllamaResult(ok=False, error="invalid_json_response")

        text = (data.get("response") or "").strip()
        if not text:
            return OllamaResult(ok=False, error="empty_response")

        return OllamaResult(ok=True, text=text)


_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    global _client
    if _client is None:
        _client = OllamaClient()
    return _client
