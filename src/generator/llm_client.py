"""LLM wrapper supporting Qwen, Gemini, OpenAI, Ollama, and mock backends."""

from __future__ import annotations

import json
from collections.abc import Generator, Iterator
from typing import Any

from src.config import get_config


class LLMClient:
    """Unified LLM client with streaming and JSON-mode support."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        cfg = get_config().llm
        self.provider = provider or cfg.provider
        self.model = model or cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self._client: Any = None

    def generate(self, prompt: str, system: str | None = None) -> str:
        chunks = list(self.stream(prompt, system=system))
        return "".join(chunks)

    def stream(self, prompt: str, system: str | None = None) -> Iterator[str]:
        if self.provider == "mock":
            yield from self._mock_stream(prompt)
            return
        if self.provider == "qwen":
            yield from self._qwen_stream(prompt, system)
            return
        if self.provider == "openai":
            yield from self._openai_stream(prompt, system)
            return
        if self.provider == "gemini":
            yield from self._gemini_stream(prompt, system)
            return
        if self.provider == "ollama":
            yield from self._ollama_stream(prompt, system)
            return
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate_json(self, prompt: str, system: str | None = None) -> dict:
        json_prompt = (
            f"{prompt}\n\nRespond with valid JSON only, no markdown fences."
        )
        raw = self.generate(json_prompt, system=system)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            raw = raw.rsplit("```", 1)[0].strip()
        return json.loads(raw)

    def _mock_stream(self, prompt: str) -> Generator[str, None, None]:
        answer = (
            "This is a mock RAG response based on the retrieved context. "
            f"Query snippet: {prompt[:120]}..."
        )
        for word in answer.split():
            yield word + " "

    def _build_messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _qwen_stream(self, prompt: str, system: str | None) -> Iterator[str]:
        """Stream from Qwen via DashScope OpenAI-compatible API."""
        from openai import OpenAI

        cfg = get_config().llm
        if not cfg.qwen_api_key:
            raise ValueError(
                "Qwen API key missing. Set DASHSCOPE_API_KEY or QWEN_API_KEY in .env"
            )

        client = OpenAI(api_key=cfg.qwen_api_key, base_url=cfg.qwen_base_url)
        stream = client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, system),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _openai_stream(self, prompt: str, system: str | None) -> Iterator[str]:
        from openai import OpenAI

        cfg = get_config().llm
        client = OpenAI(api_key=cfg.openai_api_key)

        stream = client.chat.completions.create(
            model=self.model,
            messages=self._build_messages(prompt, system),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _gemini_stream(self, prompt: str, system: str | None) -> Iterator[str]:
        cfg = get_config().llm
        try:
            from google import genai
        except ImportError:
            import google.generativeai as genai_legacy

            genai_legacy.configure(api_key=cfg.gemini_api_key)
            model = genai_legacy.GenerativeModel(self.model)
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            response = model.generate_content(full_prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            return

        client = genai.Client(api_key=cfg.gemini_api_key)
        contents = prompt if not system else f"{system}\n\n{prompt}"
        for chunk in client.models.generate_content_stream(
            model=self.model,
            contents=contents,
        ):
            if chunk.text:
                yield chunk.text

    def _ollama_stream(self, prompt: str, system: str | None) -> Iterator[str]:
        import httpx

        cfg = get_config().llm
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": self.temperature},
        }
        if system:
            payload["system"] = system

        with httpx.stream(
            "POST",
            f"{cfg.ollama_base_url}/api/generate",
            json=payload,
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if token := data.get("response"):
                    yield token
