"""
Shared extraction and scoring for the MetaLoop v2 rerun (October 2026).

One rule set for every task, applied after the runs:

- JSON tasks (Wagering, Deferral): strict JSON parse (whole response, then fenced
  blocks, then the first decodable object). No field-by-field regex rescue and no
  default values: a missing or out-of-range field is missing.
- Multiple-choice answers map to a letter only when unambiguous: the letter itself
  ("C", "C)", "C) text") or the exact text of one option. Anything else goes to
  the judge panel.
- Free-text answers (SimpleQA) count as correct on a normalised exact match;
  everything else, plus a seeded 10% audit of exact matches, is graded by the
  panel with a SimpleQA-style grader prompt.
- Judge panel: three models, majority vote; a three-way split leaves the strict
  result in place. Judge errors are never cached, so a rerun retries them.

Ground truth in bank v2 is option text; `gt_letter` maps it through `choices`.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import string
import unicodedata
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
JUDGES = ["anthropic/claude-sonnet-4.6", "openai/gpt-5.4", "google/gemini-3.1-pro-preview"]
AUDIT_RATE = 0.10

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_DEC = json.JSONDecoder()


def parse_json(text: str | None) -> dict | None:
    t = (text or "").strip()
    if not t:
        return None
    cands = [t]
    for b in t.split("```")[1::2]:
        b = b.strip()
        cands.append(b[4:].strip() if b.lower().startswith("json") else b)
    for c in cands:
        try:
            d = json.loads(c)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass
    for m in re.finditer(r"\{", t):
        try:
            d, _ = _DEC.raw_decode(t[m.start():])
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            continue
    return None


def gt_letter(q: dict) -> str | None:
    ch = q.get("choices")
    if not isinstance(ch, dict):
        return None
    a = str(q.get("answer", "")).strip()
    if a in ch:
        return a
    hits = [k for k, v in ch.items() if str(v).strip() == a]
    return hits[0] if len(hits) == 1 else None


def mc_letter(ans, choices: dict) -> str | None:
    """Letter for an answer field, or None if it is not unambiguous."""
    if ans is None:
        return None
    s = str(ans).strip()
    by_text = [k for k, v in choices.items() if str(v).strip().lower() == s.lower()]
    if len(by_text) == 1:
        return by_text[0]
    m = re.fullmatch(r"\(?([A-J]|\d{1,2})\)?[.):]?", s, re.I)
    if m and m.group(1).upper() in choices:
        return m.group(1).upper()
    m = re.fullmatch(r"\(?([A-J]|\d{1,2})[.):]\s*(.+)", s, re.I | re.S)   # "C) text": text must be option C's
    if m and m.group(1).upper() in choices and str(choices[m.group(1).upper()]).strip().lower() == m.group(2).strip().lower():
        return m.group(1).upper()
    return None


def confidence(v) -> int | None:
    """0-100 integer, else None (out-of-range values such as DeepSeek's -1 are invalid)."""
    if isinstance(v, bool):
        return None
    try:
        f = float(str(v).strip().rstrip("%"))
    except (TypeError, ValueError):
        return None
    return int(round(f)) if 0 <= f <= 100 else None


def enum(v, allowed) -> str | None:
    s = str(v or "").strip().upper()
    return s if s in allowed else None


STRICT_ANSWER = re.compile(r"ANSWER\s*\**\s*:\s*\**\s*[\[\(]?\s*([A-J])(?![A-Za-z])", re.I)
STRICT_CONF = re.compile(r"CONFIDENCE\s*\**\s*:\s*\**\s*(-?\d+(?:\.\d+)?)", re.I)


def strict_answer(text, letters) -> str | None:
    hits = [h.upper() for h in STRICT_ANSWER.findall(text or "") if h.upper() in letters]
    return hits[-1] if hits else None


def strict_conf(text) -> int | None:
    hits = STRICT_CONF.findall(text or "")
    return confidence(hits[-1]) if hits else None


def norm_text(s) -> str:
    s = "".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in str(s or "").lower())
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


# ---------------------------------------------------------------------------
# Judge panel
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """Below is a model's answer to a multiple-choice question. The options are:
{options}

Your job is ONLY to extract which option the response commits to as its final answer. Do not solve the question or judge correctness.

Reply with exactly one token:
- a single option label from {letters}, if the response commits to one option
- NONE, if it commits to no option or to more than one

RESPONSE:
<<<
{response}
>>>"""

GRADE_PROMPT = """Grade a predicted answer against the gold target for a short factual question.

Question: {question}
Gold target: {gold}
Predicted answer: {predicted}

CORRECT: the prediction contains the gold target's meaning without contradicting it (extra detail is fine; spelling, formatting and equivalent units are fine).
INCORRECT: the prediction contradicts the gold target or gives a different answer, including a hedge between several answers.
NOT_ATTEMPTED: no answer given (e.g. "I don't know", empty).

Reply with exactly one word: CORRECT, INCORRECT or NOT_ATTEMPTED."""


def parse_extract(raw, letters) -> str:
    tok = [t for t in re.findall(r"\b(NONE|[A-J]|\d{1,2})\b", (raw or "").upper()) if t == "NONE" or t in letters]
    return tok[-1] if tok else "UNPARSED"


def parse_grade(raw) -> str:
    tok = re.findall(r"\b(NOT_ATTEMPTED|INCORRECT|CORRECT)\b", (raw or "").upper())
    return tok[-1] if tok else "UNPARSED"


def majority(votes):
    top = max(set(votes), key=votes.count)
    return top if votes.count(top) >= 2 else None


class Panel:
    """Three-judge panel over OpenRouter with a persistent cache (errors not cached)."""

    def __init__(self, cache_path: Path, workers: int = 8, judges: list[str] | None = None, max_tokens: int = 2000):
        self.judges, self.max_tokens = judges or JUDGES, max_tokens
        load_dotenv(ROOT / ".env", override=True)
        self.key = os.environ["OPENROUTER_API_KEY"]
        self.path = Path(cache_path)
        self.lock = threading.Lock()
        self.workers = workers
        self.cache: dict[tuple, str] = {}
        if self.path.exists():
            for line in self.path.open():
                c = json.loads(line)
                self.cache[(c["judge"], c["h"])] = c["raw"]

    @staticmethod
    def _h(prompt):
        return hashlib.sha256(prompt.encode()).hexdigest()[:20]

    def _call(self, judge, prompt):
        body = {"model": judge, "messages": [{"role": "user", "content": prompt}], "max_tokens": self.max_tokens,
                "temperature": 0}
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        for attempt in range(5):
            try:
                resp = json.load(urllib.request.urlopen(req, timeout=300))
                return (resp["choices"][0]["message"]["content"] or "").strip()
            except Exception:  # transport (incl. dropped connections), bad JSON, missing fields: retry
                time.sleep(3 * 2 ** attempt)
        return None

    def run(self, prompts: list[str]) -> dict[str, list[str | None]]:
        """prompt -> [raw per judge] (None where a judge failed)."""
        todo = sorted({(j, p) for p in prompts for j in self.judges if (j, self._h(p)) not in self.cache})

        def work(jp):
            j, p = jp
            raw = self._call(j, p)
            if raw is not None:
                with self.lock:
                    self.cache[(j, self._h(p))] = raw
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a") as fh:
                        fh.write(json.dumps({"judge": j, "h": self._h(p), "raw": raw}) + "\n")
        if todo:
            print(f"  judging: {len(todo)} calls", flush=True)
            with ThreadPoolExecutor(self.workers) as ex:
                list(ex.map(work, todo))
        return {p: [self.cache.get((j, self._h(p))) for j in self.judges] for p in prompts}


def audit_pick(*key) -> bool:
    return random.Random("metaloop-v2-audit:" + ":".join(map(str, key))).random() < AUDIT_RATE


def options_block(choices: dict) -> str:
    return "\n".join(f"{k}) {v}".strip() for k, v in sorted(choices.items()))


def wilson(k, n):
    """Wilson 95% interval for k/n, in percent (statsmodels, via stats.wilson)."""
    from stats import wilson as _wilson
    return _wilson(k, n)
