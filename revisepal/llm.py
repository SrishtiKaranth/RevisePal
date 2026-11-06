"""Thin wrapper over an OpenAI-compatible API for chat and embeddings.

Works with Ollama (default, fully local), LM Studio, vLLM, OpenAI, Groq, etc.
"""
from __future__ import annotations

import json
import re

from .config import Settings


class LLM:
    def __init__(self, settings: Settings, client=None):
        self.s = settings
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=settings.base_url, api_key=settings.api_key)
        self.client = client
        # nomic-embed-text is trained with task prefixes; they noticeably improve retrieval
        self._nomic = "nomic" in settings.embed_model.lower()

    def embed(self, texts: list[str], kind: str = "document") -> list[list[float]]:
        if self._nomic:
            prefix = "search_query: " if kind == "query" else "search_document: "
            texts = [prefix + t for t in texts]
        out = []
        for i in range(0, len(texts), 32):  # batch to keep requests small
            resp = self.client.embeddings.create(model=self.s.embed_model, input=texts[i:i + 32])
            out.extend(d.embedding for d in resp.data)
        return out

    def chat(self, system: str, user: str, json_mode: bool = False, temperature: float = 0.2) -> str:
        kwargs = dict(
            model=self.s.chat_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def chat_json(self, system: str, user: str, retries: int = 2) -> dict:
        last = None
        for _ in range(retries + 1):
            raw = self.chat(system, user, json_mode=True)
            try:
                return parse_json(raw)
            except ValueError as e:
                last = e
        raise ValueError(f"Model did not return valid JSON: {last}")


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)  # small models sometimes add chatter around the JSON
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Not JSON: {raw[:200]}")
