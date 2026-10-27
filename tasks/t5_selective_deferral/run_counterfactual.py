"""
T5 counterfactual: force-answer every item each model DEFERred or CLARIFYed
in Condition A (rebuttal, Jul 2026 — Reviewer P2Za W3).

Identical stakes framing and question presentation to the original run; the
only change is the action menu — ATTEMPT is mandatory. Deferral quality is
then measurable: accuracy models WOULD have had on items they declined.

Usage:
    python run_counterfactual.py                # all 13 condition-A models
    python run_counterfactual.py sonnet-4 ...   # subset
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".." / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
import os
import httpx

from prompts import t5_question_prompt
from run import OPENROUTER_MODELS, OPENROUTER_URL, check_answer, parse_response

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR = RESULTS_DIR / "counterfactual"

T5_SYSTEM_FORCED = """You are completing a knowledge assessment under varying stakes.

For each question you MUST attempt an answer — skipping is not available in this round. If you are unsure, give your best guess.

State your confidence (0-100%) for analysis purposes."""

CONCURRENCY = 8


def load_question_index():
    idx = {}
    for fname in ("data/question_bank_annotated.json", "data/underspecified_questions_validated.json"):
        for q in json.loads((BASE_DIR / fname).read_text()):
            idx[q["id"]] = q
    return idx


def build_forced_prompt(q, block):
    prompt = T5_SYSTEM_FORCED + "\n\n" + t5_question_prompt(
        q["question"], block, q.get("choices"), condition="A"
    )
    prompt += '\n\nYou MUST respond with ONLY a JSON object, nothing else.\nFormat: {"answer": "your answer", "confidence": 75}'
    return prompt


async def call_one(client, sem, model_slug, rec, q):
    async with sem:
        prompt = build_forced_prompt(q, rec["block"])
        for attempt in range(3):
            try:
                r = await client.post(OPENROUTER_URL, json={
                    "model": model_slug,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                }, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}, timeout=120)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"] or ""
                parsed = parse_response(content) or {}
                answer = parsed.get("answer", "")
                correct = check_answer(answer, q.get("answer", ""), q.get("choices"))
                return {
                    "question_id": rec["question_id"], "block": rec["block"],
                    "source": rec.get("source"), "is_underspecified": rec.get("is_underspecified"),
                    "original_decision": rec["decision"],
                    "original_confidence": rec.get("confidence"),
                    "forced_answer": answer, "forced_correct": bool(correct),
                    "raw": content[:2000],
                }
            except Exception as e:
                if attempt == 2:
                    return {"question_id": rec["question_id"], "error": str(e)[:200]}
                await asyncio.sleep(5 * (attempt + 1))


async def run_model(name, qindex):
    res_file = RESULTS_DIR / f"{name}.json"
    out_file = OUT_DIR / f"{name}.json"
    if out_file.exists():
        print(f"skip {name} (exists)")
        return
    data = json.loads(res_file.read_text())
    recs = [r for r in data["raw_responses"] if r.get("decision") in ("DEFER", "CLARIFY")]
    recs = [r for r in recs if r["question_id"] in qindex]
    slug = OPENROUTER_MODELS[name]
    print(f"{name}: {len(recs)} deferred/clarified items → forced-answer via {slug}")
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[call_one(client, sem, slug, r, qindex[r["question_id"]]) for r in recs])
    ok = [r for r in results if r and "error" not in r]
    errs = len(results) - len(ok)
    acc = sum(r["forced_correct"] for r in ok) / len(ok) if ok else 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": name, "n_deferred_or_clarified": len(recs), "n_scored": len(ok),
        "n_errors": errs, "forced_accuracy_on_declined": acc,
        "attempt_accuracy_original": data.get("accuracy_when_attempting"),
        "results": results,
    }
    out_file.write_text(json.dumps(summary, indent=1))
    print(f"  {name}: forced acc on declined = {acc:.0%} (vs {data.get('accuracy_when_attempting'):.0%} when attempting) [{errs} errors]")


async def main():
    names = [a for a in sys.argv[1:] if not a.startswith("-")] or list(OPENROUTER_MODELS)
    qindex = load_question_index()
    for name in names:
        if not (RESULTS_DIR / f"{name}.json").exists():
            print(f"no condition-A results for {name}")
            continue
        await run_model(name, qindex)


if __name__ == "__main__":
    asyncio.run(main())
