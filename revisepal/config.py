"""Settings, read from environment variables (see .env.example)."""
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    # Any OpenAI-compatible server works; default is a local Ollama.
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "ollama"))
    chat_model: str = field(default_factory=lambda: os.getenv("CHAT_MODEL", "llama3.2:3b"))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "nomic-embed-text"))

    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    chunk_chars: int = int(os.getenv("CHUNK_CHARS", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def deck_path(self) -> Path:
        return self.data_dir / "deck.json"
