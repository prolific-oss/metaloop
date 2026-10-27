"""
Shared model calls for the MetaLoop v2 rerun (October 2026).

Every task calls `call(model_key, messages, ...)`. Settings come only from
config/models.yaml, so each model is served the same way in every task.
Each attempt is appended to a JSONL log with the resolved request (minus API
keys), the serving provider and model version, finish reason, token usage,
latency and errors, plus the config hash.

Messages use the OpenAI shape: [{"role": "system"|"user"|"assistant",
"content": str}]. Prefilled assistant turns (Strategy-Switching) and long
conversations (The-Loop) are passed through unchanged.

Retries cover transport failures, HTTP 429 and 5xx only. Anything else is
returned as an error result, never silently retried until it succeeds.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "models.yaml"
_ENV = {**dotenv_values(ROOT / ".env")}
_LOG_LOCK = threading.Lock()
_SEMAPHORES: dict[str, threading.BoundedSemaphore] = {}

CONCURRENCY = {"mistral-large": 2, "mistral-large-4": 2, "mistral-large-4-32k": 2,  # Mistral's endpoint is rate-limited
               "gemma-4-31b": 1,  # Gemini API per-minute quota: 429 on most calls with 6 parallel Loop sessions (2026-10-16)
               "gemma-4-31b-crusoe": 2}  # Crusoe upstream 429s killed all 6 Loop sessions at 6 parallel calls (2026-10-19)
DEFAULT_CONCURRENCY = 8


def _load_config():
    raw = CONFIG_PATH.read_bytes()
    return yaml.safe_load(raw), hashlib.sha256(raw).hexdigest()[:12]


CONFIG, CONFIG_HASH = _load_config()


def spec(model_key: str) -> dict:
    """Resolved settings for a model: defaults overridden by the model's entry."""
    if model_key not in CONFIG["models"]:
        raise KeyError(f"{model_key!r} not in config/models.yaml (models: {sorted(CONFIG['models'])})")
    s = {**CONFIG["defaults"], **CONFIG["models"][model_key]}
    s["key"] = model_key
    return s


def models(cohort: str | None = "core") -> list[str]:
    """Model keys in a cohort ("core" = the original panel; "extension"; None = all)."""
    return [k for k, v in CONFIG["models"].items() if cohort is None or v.get("cohort", "core") == cohort]


@dataclass
class Result:
    text: str | None
    status: str                      # ok | truncated | empty | blocked | error
    finish_reason: str | None
    provider: str | None
    model_version: str | None
    usage: dict = field(default_factory=dict)
    error: str | None = None
    attempts: int = 1
    latency_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# ---------------------------------------------------------------------------
# Request builders: (url, headers, body) per route
# ---------------------------------------------------------------------------

def _temperature(s):
    t = s.get("temperature")
    return None if t in (None, "omit") else float(t)


def _reasoning(s):
    r = s.get("reasoning")
    return None if r in (None, "none") else r


def _build_anthropic(s, messages, max_tokens):
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    body = {"model": s["id"], "max_tokens": max_tokens,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]}
    if system:
        body["system"] = system
    if (t := _temperature(s)) is not None:
        body["temperature"] = t
    r = _reasoning(s)
    if r:
        body["thinking"] = {"type": r.get("thinking", "adaptive")}
        if "effort" in r:
            body["output_config"] = {"effort": r["effort"]}
    headers = {"x-api-key": _ENV["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"}
    return "https://api.anthropic.com/v1/messages", headers, body


def _build_openai(s, messages, max_tokens):
    body = {"model": s["id"], "max_completion_tokens": max_tokens, "messages": messages}
    if (t := _temperature(s)) is not None:
        body["temperature"] = t
    r = _reasoning(s)
    if r and "reasoning_effort" in r:
        body["reasoning_effort"] = r["reasoning_effort"]
    return "https://api.openai.com/v1/chat/completions", {"Authorization": f"Bearer {_ENV['OPENAI_API_KEY']}"}, body


def _build_google(s, messages, max_tokens):
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages if m["role"] != "system"]
    gen = {"maxOutputTokens": max_tokens}
    if (t := _temperature(s)) is not None:
        gen["temperature"] = t
    r = _reasoning(s)
    tc = {k: v for k, v in (r or {}).items() if k in ("thinkingBudget", "thinkingLevel")}
    if tc:  # models that think by default with no configurable level (Gemma 4) get no thinkingConfig
        gen["thinkingConfig"] = tc
    body = {"contents": contents, "generationConfig": gen}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{s['id']}:generateContent"
    return url, {"x-goog-api-key": _ENV["GEMINI_API_KEY"]}, body


def _build_openrouter(s, messages, max_tokens):
    body = {"model": s["id"], "max_tokens": max_tokens, "messages": messages, "usage": {"include": True},
            "provider": {"order": [s["provider"]], "allow_fallbacks": bool(s.get("allow_fallbacks", False))}}
    if (t := _temperature(s)) is not None:
        body["temperature"] = t
    r = _reasoning(s)
    if r and "effort" in r:
        body["reasoning"] = {"effort": r["effort"]}
    if r and "enabled" in r:  # explicit on/off (reasoning-off ablation)
        body["reasoning"] = {"enabled": bool(r["enabled"])}
    return "https://openrouter.ai/api/v1/chat/completions", {"Authorization": f"Bearer {_ENV['OPENROUTER_API_KEY']}"}, body


BUILDERS = {"anthropic": _build_anthropic, "openai": _build_openai, "google": _build_google, "openrouter": _build_openrouter}


_UNSET = object()


def build_request(model_key: str, messages: list[dict], max_tokens: int | None = None, temperature=_UNSET):
    """`temperature` (a number or "omit") replaces the model's configured temperature for this call."""
    s = spec(model_key)
    if temperature is not _UNSET:
        s["temperature"] = temperature
    return s, BUILDERS[s["route"]](s, messages, int(max_tokens or s["max_tokens"]))


def task_temperature(model_key: str, meta: dict | None):
    """A model entry may set `task_temperature: {<task>: t, "<task>:<condition>": t}` (t a number or
    "omit"), matched on meta["task"] and meta["condition"]; the condition-level key wins. Returns
    _UNSET when the entry sets neither, so the model's own temperature applies."""
    table = spec(model_key).get("task_temperature") or {}
    task, cond = (meta or {}).get("task"), (meta or {}).get("condition")
    for k in (f"{task}:{cond}", task):
        if k in table:
            return table[k]
    return _UNSET


# ---------------------------------------------------------------------------
# Response parsers -> Result
# ---------------------------------------------------------------------------

TRUNCATED = {"length", "max_tokens", "MAX_TOKENS"}


def _status(text, finish):
    if finish in TRUNCATED:
        return "truncated"
    return "ok" if (text or "").strip() else "empty"


def request_limit(body: dict) -> int | None:
    """The output-token limit a request body asked for, whatever the route calls it."""
    gc = body.get("generationConfig") or body.get("generation_config") or {}
    return (body.get("max_tokens") or body.get("max_completion_tokens") or body.get("max_output_tokens")
            or gc.get("maxOutputTokens"))


def provider_fault(status, usage, finish) -> bool:
    """A reply the provider broke rather than the model: provider_empty, or a response that came back
    with finish_reason "error" (the provider's stream failed mid-answer; seen on Vertex, October 2026,
    where a partial answer was otherwise stored as ok). Retried like a transport error."""
    return provider_empty(status, usage) or finish == "error"


def provider_empty(status, usage) -> bool:
    """An "empty" reply for which the provider reports generated tokens. Seen on DeepSeek V3.2 via
    Google Vertex (October 2026): the complete answer arrives in the reasoning field and the
    content field is null, mostly on step-by-step prompts. The model did answer, so this is a
    provider fault, retried like an API error rather than kept as an empty response."""
    return status == "empty" and bool((usage or {}).get("output"))


def _parse_anthropic(s, r):
    text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
    u = r.get("usage", {})
    usage = {"input": u.get("input_tokens"), "output": u.get("output_tokens"),
             "thinking_blocks": sum(b.get("type") == "thinking" for b in r.get("content", []))}
    fin = r.get("stop_reason")
    return Result(text, "blocked" if fin == "refusal" else _status(text, fin), fin, "anthropic", r.get("model"), usage)


def _parse_openai_like(s, r, provider):
    if not r.get("choices"):
        return Result(None, "error", None, provider, r.get("model"), {}, error=json.dumps(r)[:300])
    c = r["choices"][0]
    text = c["message"].get("content")
    u = r.get("usage") or {}
    usage = {"input": u.get("prompt_tokens"), "output": u.get("completion_tokens"),
             "reasoning": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"), "cost": u.get("cost")}
    fin = c.get("finish_reason")
    return Result(text, "blocked" if fin == "content_filter" else _status(text, fin), fin, provider, r.get("model"), usage)


def _parse_google(s, r):
    u = r.get("usageMetadata", {})
    usage = {"input": u.get("promptTokenCount"), "output": u.get("candidatesTokenCount"), "reasoning": u.get("thoughtsTokenCount")}
    if not r.get("candidates"):
        return Result(None, "blocked", (r.get("promptFeedback") or {}).get("blockReason"), "google", r.get("modelVersion"), usage)
    c = r["candidates"][0]
    text = "".join(p.get("text", "") for p in (c.get("content") or {}).get("parts", []) if not p.get("thought"))
    fin = c.get("finishReason")
    status = "blocked" if fin in ("SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST") else _status(text, fin)
    return Result(text, status, fin, "google", r.get("modelVersion"), usage)


def _parse(s, r):
    route = s["route"]
    if route == "anthropic":
        return _parse_anthropic(s, r)
    if route == "openai":
        return _parse_openai_like(s, r, "openai")
    if route == "google":
        return _parse_google(s, r)
    return _parse_openai_like(s, r, r.get("provider"))


# ---------------------------------------------------------------------------
# Call
# ---------------------------------------------------------------------------

REFUSAL_MARKERS = ("permission-denied", "can't help with that", "Output blocked by content filtering")


def _is_refusal(code: int, detail: str) -> bool:
    return code in (400, 403) and any(m.lower() in detail.lower() for m in REFUSAL_MARKERS)


def _post(url, headers, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={**headers, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _semaphore(model_key):
    if model_key not in _SEMAPHORES:
        _SEMAPHORES[model_key] = threading.BoundedSemaphore(CONCURRENCY.get(model_key, DEFAULT_CONCURRENCY))
    return _SEMAPHORES[model_key]


def _log(log_path, record):
    if log_path is None:
        return
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK, open(log_path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def call(model_key: str, messages: list[dict], *, max_tokens: int | None = None, timeout_s: int | None = None,
         log_path: str | Path | None = None, meta: dict | None = None) -> Result:
    """One model call under the shared config. `meta` (task, item id, condition, repeat...) is logged.

    A model entry may set `task_max_tokens: {<task>: n}` (matched on meta["task"]) for a model whose
    reasoning routinely overruns the shared limit on that task; such calls get a 1800 s timeout.
    It may also set `task_temperature` (see task_temperature) to reproduce per-task settings."""
    override = (spec(model_key).get("task_max_tokens") or {}).get((meta or {}).get("task"))
    if max_tokens is None and override:
        max_tokens, timeout_s = override, timeout_s or 1800
    s, (url, headers, body) = build_request(model_key, messages, max_tokens, task_temperature(model_key, meta))
    retries, timeout = int(s.get("retries", 4)), int(timeout_s or s.get("timeout_s", 300))
    t0, attempt = time.time(), 0
    with _semaphore(model_key):
        while True:
            attempt += 1
            ta = time.time()
            base = {"ts": datetime.now(timezone.utc).isoformat(), "model_key": model_key, "route": s["route"],
                    "config_hash": CONFIG_HASH, "attempt": attempt, "meta": meta or {}, "request": body}
            try:
                raw = _post(url, headers, body, timeout)
                res = _parse(s, raw)
                res.attempts, res.latency_s = attempt, round(time.time() - t0, 2)
                lim = request_limit(body)
                if res.status == "empty" and lim and (res.usage or {}).get("output", 0) >= lim:
                    # the whole limit went on reasoning but the provider said "stop" (DeepSeek V4 Pro,
                    # October 2026): a cut-off reply, final like any truncation, not a provider fault
                    res.status = "truncated"
                misfiled = provider_fault(res.status, res.usage, res.finish_reason)
                _log(log_path, {**base, "latency_s": round(time.time() - ta, 2),
                                "status": "error" if misfiled else res.status, "finish_reason": res.finish_reason,
                                "provider": res.provider, "model_version": res.model_version, "usage": res.usage,
                                "text": res.text, "raw": raw, **({"retryable": True} if misfiled else {})})
                if not misfiled:
                    return res
                # tokens were generated but no answer text came back: a provider fault, retried
                # like a transport error (see provider_empty)
                retryable, err = True, ("provider_error: stream ended with finish_reason error" if res.finish_reason == "error"
                                        else f"provider_empty: no answer text, {res.usage.get('output')} output tokens")
                if attempt > retries:
                    return Result(None, "error", res.finish_reason, res.provider, res.model_version, res.usage,
                                  error=err, attempts=attempt, latency_s=round(time.time() - t0, 2))
                time.sleep(min(60, 5 * 2 ** (attempt - 1)))
                continue
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                err = f"HTTP {e.code}: {detail}"
                if _is_refusal(e.code, detail):
                    # A refusal delivered as an HTTP error (xAI 403 permission-denied, Anthropic
                    # 400 output blocked): model behaviour, final, never retried.
                    _log(log_path, {**base, "latency_s": round(time.time() - ta, 2), "status": "blocked",
                                    "finish_reason": "refusal_http", "error": err})
                    return Result(None, "blocked", "refusal_http", s["route"], None, {}, error=err, attempts=attempt,
                                  latency_s=round(time.time() - t0, 2))
                retryable = e.code == 429 or e.code >= 500
            except Exception as e:  # transport: timeouts, resets, DNS
                retryable, err = True, f"{type(e).__name__}: {str(e)[:300]}"
            _log(log_path, {**base, "latency_s": round(time.time() - ta, 2), "status": "error", "error": err,
                            "retryable": retryable})
            if not retryable or attempt > retries:
                return Result(None, "error", None, None, None, {}, error=err, attempts=attempt,
                              latency_s=round(time.time() - t0, 2))
            time.sleep(min(60, 5 * 2 ** (attempt - 1)))
