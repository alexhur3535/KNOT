# src/models/__init__.py
import os
from .GPT import GPT
from .GPT5 import GPT5
from .CLAUDE import Claude
from .GEMINI import Gemini
from .HF_CAUSAL import HFCausal
from .VICUNA import Vicuna
from src.utils import load_json

__all__ = ["create_model", "GPT", "GPT5", "Claude", "Gemini", "HFCausal"]

_REGISTRY = {
    "gpt": GPT,
    "gpt-4": GPT,
    "gpt-5": GPT5,
    "claude": Claude,   # https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names
    "gemini": Gemini,
    "llama": HFCausal,
    "mistral": HFCausal,
    "vicuna": Vicuna
}

def create_model(config_or_path):
    """Create an LLM instance from a config dict or path."""
    config = (
        load_json(config_or_path)
        if isinstance(config_or_path, (str, os.PathLike))
        else config_or_path
    )
    provider = (
        config.get("model_info", {}).get("provider", "").strip().lower()
    )
    if provider not in _REGISTRY:
        raise ValueError(f"Unknown provider '{provider}'. "
                         f"Choose one of: {sorted(_REGISTRY)}")
    return _REGISTRY[provider](config)
