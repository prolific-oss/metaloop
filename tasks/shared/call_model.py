"""
Shared model calling — routes to the right API based on model ID.

- anthropic/* or bare claude-* → Anthropic direct API (with extended thinking)
- openai/* or bare gpt-* / o1-* / o3-* → OpenAI direct API
- Everything else (google/*, qwen/*, deepseek/*, etc.) → OpenRouter
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load API keys from the repo-root .env, overriding any pre-existing shell vars.
_DOTENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_DOTENV_PATH, override=True)

# Map short names to canonical IDs for convenience
MODEL_ALIASES = {
    # Anthropic
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4.6",
    "claude-opus-4-6": "anthropic/claude-opus-4.6",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    # OpenAI
    "gpt-5.4": "openai/gpt-5.4",
    "gpt-4.1": "openai/gpt-4.1",
    "gpt-4o": "openai/gpt-4o-2024-11-20",
}

# All models we're testing
ALL_MODELS = [
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-sonnet-4",
    "openai/gpt-5.4",
    "openai/gpt-4.1",
    "qwen/qwen3-235b-a22b",
    "deepseek/deepseek-v3.2",
    "mistralai/mistral-large-2512",
    "x-ai/grok-4",
    "google/gemini-3.1-pro-preview",
    "google/gemini-2.5-pro",
    "google/gemma-4-31b-it",
]


def resolve_model(model: str) -> str:
    """Resolve aliases to canonical IDs."""
    return MODEL_ALIASES.get(model, model)


def call_model(model: str, messages: list, max_tokens: int = 4096) -> tuple[str, dict]:
    """Call a model with a list of messages. Returns (text, usage_dict).

    All models are routed through OpenRouter for parity with other tasks.
    """
    model = resolve_model(model)
    return _call_openrouter(model, messages, max_tokens)


def _call_anthropic(model: str, messages: list, max_tokens: int) -> tuple[str, dict]:
    from anthropic import Anthropic
    client = Anthropic()

    # Strip provider prefix for API
    api_model = model.replace("anthropic/", "").replace(".", "-", 1).replace(".", "-")
    # anthropic/claude-sonnet-4.6 → claude-sonnet-4-6
    # anthropic/claude-opus-4.6 → claude-opus-4-6
    # anthropic/claude-sonnet-4 → claude-sonnet-4

    response = client.messages.create(
        model=api_model, max_tokens=max(max_tokens, 16384),
        thinking={"type": "enabled", "budget_tokens": 8000},
        messages=messages,
    )
    text = ""
    for block in response.content:
        if block.type == "text":
            text = block.text
    return text, {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def _call_openai(model: str, messages: list, max_tokens: int) -> tuple[str, dict]:
    from openai import OpenAI
    client = OpenAI()

    # Strip provider prefix
    api_model = model.replace("openai/", "")

    response = client.chat.completions.create(
        model=api_model, max_completion_tokens=max_tokens, temperature=0, messages=messages,
    )
    return response.choices[0].message.content, {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
    }


def _call_openrouter(model: str, messages: list, max_tokens: int) -> tuple[str, dict]:
    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    response = client.chat.completions.create(
        model=model, max_completion_tokens=max_tokens, temperature=0, messages=messages,
    )
    return response.choices[0].message.content or "", {
        "input_tokens": response.usage.prompt_tokens if response.usage else 0,
        "output_tokens": response.usage.completion_tokens if response.usage else 0,
    }
