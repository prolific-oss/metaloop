"""
Self-Review error-flag scoring, Rule A (camera-ready, October 2026).

Rule A: a review counts as flagging an error when it gives at least one step an ERROR
verdict (qualified forms such as "PARTLY ERROR" included) and does not later withdraw it.
Whether the model then changes its answer is scored separately (the fix rate).

This replaces the July 2026 whole-review LLM audit (extraction_audit/selfreview_verdict_audit.json),
whose script and prompt were not kept. That audit only ever overturned the regex on reviews that
gave an ERROR verdict but kept the original answer, and its calls there were not consistent, so it
cannot be reproduced; July's labels stay in the result files and are compared below.

Routing, per review:
  no ERROR verdict at all                      -> not flagged (no judge)
  strict ERROR verdict and the answer changed  -> flagged (no judge)
  otherwise (ERROR but answer kept, or only a  -> three-judge panel (v2score.JUDGES) with JUDGE_PROMPT,
  qualified ERROR the strict regex misses)        majority vote; no majority keeps the strict regex value
We judge what the review concludes, not whether each step was in fact wrong.

    python tasks/t6_error_detection/verdict_judge.py                # 12 original models + Grok 4.7
    -> extraction_audit/selfreview_verdict_ruleA.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
from revised_answer_audit import regex_v2  # noqa: E402
from v2score import Panel, majority  # noqa: E402

RESULTS = HERE / "results" / "gen_review_qbank"
AUDIT = HERE / "extraction_audit"
CACHE = AUDIT / "verdict_judge_cache.jsonl"
MODELS = [
    "anthropic_claude-opus-4.6", "anthropic_claude-sonnet-4.6", "anthropic_claude-sonnet-4",
    "deepseek_deepseek-v3.2", "google_gemini-2.5-pro", "google_gemini-3.1-pro-preview",
    "google_gemma-4-31b-it", "mistralai_mistral-large-2512", "openai_gpt-4.1", "openai_gpt-5.4",
    "qwen_qwen3-235b-a22b", "x-ai_grok-4.1-fast", "x-ai_grok-4.7",
]

STRICT = re.compile(r"VERDICT\W{0,20}?\b(CORRECT|ERROR)\b", re.I)   # = scripts/score_v2_review.VERDICT
QUALIFIED = re.compile(r"VERDICT[^\n]{0,40}?\bERROR\b", re.I)         # also catches "PARTLY ERROR"
MAX_CHARS = 12000

JUDGE_PROMPT = """You are checking how a model's self-review should be scored. The model answered a multiple-choice question, then reviewed its own solution step by step, giving each step a VERDICT of CORRECT or ERROR, and finally gave a REVISED ANSWER (SAME means it kept its original answer).

Question: by the end of the review, does the model still maintain that at least one step of its original solution contained an error?

- Answer STANDS if at least one step is given an ERROR verdict (including qualified forms such as "ERROR (minor)" or "PARTLY ERROR") and the review does not later withdraw it. The error still stands if the model says it does not affect the final answer, or keeps its original answer anyway.
- Answer WITHDRAWN if every ERROR verdict is later taken back (for example "actually, on reflection this step is correct"), or if ERROR appears only in a sense that is not a verdict on a step of the original solution.
- Do not judge whether the steps are actually right or wrong. Report only what the review concludes.

Review:
<<<
{review}
>>>

Reply with one word: STANDS or WITHDRAWN."""

VOTE = re.compile(r"\b(STANDS|WITHDRAWN)\b")


def clip(text: str) -> str:
    if len(text) <= MAX_CHARS:
        return text
    half = MAX_CHARS // 2
    return text[:half] + "\n[... middle of review omitted ...]\n" + text[-half:]


def parse_vote(raw: str | None) -> str:
    hits = VOTE.findall((raw or "").upper())
    return hits[-1] if hits else "FAILED"


def route(text: str, phase1_answer: str | None, letters: list[str]) -> dict:
    """Regex facts and routing for one review."""
    verdicts = [v.upper() for v in STRICT.findall(text)]
    strict = "ERROR" in verdicts
    qualified = bool(QUALIFIED.search(text))
    revised = regex_v2(text, letters)
    kept = revised in (None, "SAME", "NONE") or revised == phase1_answer
    if not (strict or qualified):
        r = "no_error"
    elif strict and not kept:
        r = "error_answer_changed"
    else:
        r = "judge"
    return {"n_verdicts": len(verdicts), "strict_error": strict, "qualified_error": qualified,
            "revised": revised, "answer_kept": kept, "route": r}


def score(items: list[dict], panel: Panel) -> list[dict]:
    """items: {key, text, phase1_answer, letters, ...}. Adds the regex facts, votes and flagged_ruleA."""
    rows, prompts = [], {}
    for it in items:
        row = {**{k: v for k, v in it.items() if k not in ("text", "letters")},
               **route(it["text"], it.get("phase1_answer"), it["letters"])}
        if row["route"] == "judge":
            prompts[it["key"]] = JUDGE_PROMPT.format(review=clip(it["text"]))
        rows.append(row)
    raw = panel.run(list(prompts.values()))
    for row in rows:
        if row["route"] == "no_error":
            row["flagged_ruleA"] = False
        elif row["route"] == "error_answer_changed":
            row["flagged_ruleA"] = True
        else:
            votes = [parse_vote(v) for v in raw[prompts[row["key"]]]]
            top = majority(votes)
            row["votes"] = votes
            row["flagged_ruleA"] = (top == "STANDS") if top in ("STANDS", "WITHDRAWN") else row["strict_error"]
            row["judge_fallback"] = top not in ("STANDS", "WITHDRAWN")
    return rows


def load_bank_letters() -> dict[str, list[str]]:
    bank = json.loads((ROOT / "data" / "question_bank_annotated.json").read_text())
    return {q["id"]: sorted(q["choices"]) for q in bank if q.get("choices")}


def main():
    letters = load_bank_letters()
    items = []
    for m in MODELS:
        d = json.loads((RESULTS / f"{m}.json").read_text())
        for r in (d if isinstance(d, list) else d.get("results", [])):
            if "phase2_response" not in r:
                continue
            items.append({"key": f"{m}|{r['id']}", "model": m, "id": r["id"], "text": r["phase2_response"] or "",
                          "phase1_answer": r.get("model_answer"), "phase1_correct": r.get("phase1_correct"),
                          "letters": letters.get(r["id"], list("ABCDEFGHIJ")),
                          "july_found_error": bool(r.get("found_error"))})
    rows = score(items, Panel(CACHE))
    routes = Counter(r["route"] for r in rows)
    judged = [r for r in rows if r["route"] == "judge"]
    summary = {
        "n_reviews": len(rows), "routes": dict(routes),
        "judged_flagged": sum(r["flagged_ruleA"] for r in judged),
        "judge_fallback": sum(r.get("judge_fallback", False) for r in judged),
        "unanimous": sum(len(set(r["votes"])) == 1 for r in judged),
        "agree_with_july": {m: [sum(r["flagged_ruleA"] == r["july_found_error"] for r in rows if r["model"] == m),
                                sum(r["model"] == m for r in rows)] for m in MODELS if m != "x-ai_grok-4.7"},
    }
    (AUDIT / "selfreview_verdict_ruleA.json").write_text(json.dumps(
        {"rule": "A", "prompt": JUDGE_PROMPT, "judges": Panel(CACHE).judges, "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
