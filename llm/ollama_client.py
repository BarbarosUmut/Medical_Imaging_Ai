"""
Ollama REST API client with streaming support.
"""
from __future__ import annotations

import json
from typing import Iterator

import requests

OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaClient:
    """Thin wrapper around the Ollama REST API.

    Args:
        base_url: Base URL of the Ollama server.
        model:    Model name to use for generation (e.g. ``"llama3"``).
        timeout:  HTTP request timeout in seconds for non-streaming requests.
                  Streaming requests use the same value per chunk read.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = "llama3",
        timeout: int = 120,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, stream: bool = False, **kwargs) -> str | Iterator[str]:
        """Generate text from Ollama.

        Args:
            prompt: The input prompt string.
            stream: If ``False`` (default), blocks and returns the full
                    response as a string.  If ``True``, returns a generator
                    that yields text chunks as they arrive.
            **kwargs: Additional Ollama generation parameters, e.g.
                      ``temperature``, ``top_p``, ``num_predict``.

        Returns:
            Complete response string when ``stream=False``, or a
            ``Iterator[str]`` of incremental text chunks when ``stream=True``.

        Raises:
            RuntimeError: If the server returns a non-2xx HTTP status.
            requests.exceptions.ConnectionError: If the server is unreachable.
        """
        if stream:
            return self.generate_stream(prompt, **kwargs)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **kwargs,
        }
        response = self._post("/api/generate", payload)
        data = response.json()
        return data.get("response", "")

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Streaming generation — yields text chunks as they arrive.

        Args:
            prompt:  The input prompt string.
            **kwargs: Additional Ollama generation parameters.

        Yields:
            Incremental text chunks (``str``) until the stream is done.

        Raises:
            RuntimeError: If the server returns a non-2xx HTTP status.
            requests.exceptions.ConnectionError: If the server is unreachable.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            **kwargs,
        }

        with requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as resp:
            if not resp.ok:
                raise RuntimeError(
                    f"Ollama API error {resp.status_code}: {resp.text[:200]}"
                )
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                text = chunk.get("response", "")
                if text:
                    yield text
                if chunk.get("done", False):
                    break

    def is_available(self) -> bool:
        """Check if the Ollama server is running.

        Returns:
            ``True`` if the server responds with HTTP 200, ``False`` otherwise.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
        except requests.exceptions.Timeout:
            return False

    def list_models(self) -> list[str]:
        """Return a list of available model names on the Ollama server.

        Returns:
            List of model name strings (e.g. ``["llama3:latest", "mistral:latest"]``).

        Raises:
            RuntimeError: If the server returns a non-2xx HTTP status.
            requests.exceptions.ConnectionError: If the server is unreachable.
        """
        resp = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout)
        if not resp.ok:
            raise RuntimeError(
                f"Ollama API error {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        models = data.get("models", [])
        return [m["name"] for m in models if "name" in m]

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from the Ollama registry if not already available.

        This call blocks until the pull completes (can take several minutes
        for large models).

        Args:
            model_name: Name of the model to pull (e.g. ``"llama3"``).

        Returns:
            ``True`` on success, ``False`` on any error.
        """
        try:
            payload = {"name": model_name, "stream": False}
            resp = self._post("/api/pull", payload)
            return resp.ok
        except (requests.exceptions.ConnectionError, RuntimeError):
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, endpoint: str, payload: dict) -> requests.Response:
        """Send a POST request to *endpoint* with *payload* as JSON.

        Args:
            endpoint: URL path, e.g. ``"/api/generate"``.
            payload:  Dict to serialise as JSON body.

        Returns:
            The :class:`requests.Response` object.

        Raises:
            RuntimeError: If the server returns a non-2xx HTTP status.
            requests.exceptions.ConnectionError: Propagated as-is.
        """
        url = f"{self.base_url}{endpoint}"
        resp = requests.post(url, json=payload, timeout=self.timeout)
        if not resp.ok:
            raise RuntimeError(
                f"Ollama API error {resp.status_code} at {endpoint}: {resp.text[:200]}"
            )
        return resp
