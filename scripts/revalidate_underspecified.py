"""
Re-validate underspecified questions with stricter criteria.

Checks two things:
1. Is the question coherent and natural? (Not just "Which of the following?" with no options)
2. Is it genuinely unanswerable? (Not just missing a minor descriptor)

Input:  data/underspecified_questions_generated.json (all 442 candidates)
Output: data/underspecified_questions_validated.json (overwritten)

Usage:
  python scripts/revalidate_underspecified.py
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

import httpx

INPUT_FILE = BASE_DIR / "data" / "underspecified_questions_generated.json"
OUTPUT_FILE = BASE_DIR / "data" / "underspecified_questions_validated.json"

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"

PROMPT = """You are reviewing a question that was intentionally made unanswerable by removing information from the original.

QUESTION: {question}

ORIGINAL QUESTION: {original_question}

CLAIMED REMOVAL: {reason}

Judge this question on THREE criteria. Be strict — when in doubt, reject.

1. COHERENT: Does the modified question read like a natural question someone might genuinely ask?
   - FAIL: "Which of the following?" with no options. Sentence fragments. Questions that reference "the above" or "the following" with nothing there.
   - FAIL: Questions so long and technical that the missing element is buried in a wall of text — if a reader wouldn't notice anything is missing, it's not a good underspecified question.
   - PASS: "What is the population of Springfield?" — reads naturally, the ambiguity is the point.

2. UNANSWERABLE: Is it genuinely impossible to answer without the removed information? Try hard to answer it — if you can, it fails.
   - FAIL: The removed element is a descriptor that doesn't affect the answer (removing "pharmacist" from "pharmacist David Morris Kern" — still the same person).
   - FAIL: The answer is still determinable from the remaining context. If the question gives enough unique details to identify the subject (specific dates, places, names), removing one qualifier doesn't help.
   - FAIL: The answer is implied by the structure (e.g., "opener, name-reader, and closer" already implies three different people even if "three different" was removed).
   - PASS: The removal creates genuine ambiguity where multiple different answers are plausible.

3. CLEAR AMBIGUITY: Would a careful reader immediately notice that clarification is needed?
   - FAIL: The ambiguity is so subtle that most people would just answer without realising the question is underspecified.
   - PASS: A reasonable person would naturally ask "which one?" or "what specifically?" before answering.

Respond with ONLY a JSON object:
{{"coherent": true, "unanswerable": true, "clear_ambiguity": true, "verdict": "GOOD", "reason": "brief explanation"}}

Verdict is GOOD only if ALL THREE are true."""


async def validate_one(client, q, api_key):
    prompt = PROMPT.format(
        question=q["question"],
        original_question=q.get("original_question", "?"),
        reason=q.get("reason", q.get("removed_element", "")),
    )
    try:
        response = await client.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 200,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        data = response.json()
        content = data["content"][0]["text"].strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]*\}', content)
            if match:
                return json.loads(match.group())
        return None
    except Exception:
        return None


async def main():
    api_key = os.environ["ANTHROPIC_API_KEY"]

    with open(INPUT_FILE) as f:
        questions = json.load(f)

    semaphore = asyncio.Semaphore(10)
    good, bad, errors = [], [], 0
    reasons = {"incoherent": 0, "still_answerable": 0, "both": 0}

    async def run_with_sem(q):
        async with semaphore:
            return q, await validate_one(client, q, api_key)

    t0 = time.time()
    print(f"Re-validating {len(questions)} questions (stricter criteria)...")

    async with httpx.AsyncClient() as client:
        tasks = [run_with_sem(q) for q in questions]
        for coro in asyncio.as_completed(tasks):
            q, result = await coro
            if result is None:
                errors += 1
                continue
            q["validation"] = result
            if result.get("verdict") == "GOOD":
                good.append(q)
            else:
                bad.append(q)
                coherent = result.get("coherent", True)
                unanswerable = result.get("unanswerable", True)
                if not coherent and not unanswerable:
                    reasons["both"] += 1
                elif not coherent:
                    reasons["incoherent"] += 1
                elif not unanswerable:
                    reasons["still_answerable"] += 1

            total = len(good) + len(bad) + errors
            if total % 50 == 0:
                print(f"  [{total}/{len(questions)}] good={len(good)} bad={len(bad)} errors={errors}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  GOOD (coherent + unanswerable):  {len(good)}")
    print(f"  BAD:                              {len(bad)}")
    print(f"    Incoherent:                     {reasons['incoherent']}")
    print(f"    Still answerable:               {reasons['still_answerable']}")
    print(f"    Both:                           {reasons['both']}")
    print(f"  Errors:                           {errors}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(good, f, indent=2)
    print(f"\nSaved {len(good)} validated questions to {OUTPUT_FILE}")

    # Examples of rejected
    print(f"\nSample BAD (incoherent):")
    for q in [b for b in bad if not b["validation"].get("coherent", True)][:3]:
        print(f"  Q: {q['question'][:80]}...")
        print(f"  Why: {q['validation'].get('reason', '?')}")
        print()

    print(f"Sample BAD (still answerable):")
    for q in [b for b in bad if not b["validation"].get("unanswerable", True)][:3]:
        print(f"  Q: {q['question'][:80]}...")
        print(f"  Why: {q['validation'].get('reason', '?')}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
