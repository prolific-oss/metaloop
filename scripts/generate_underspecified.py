"""
Generate underspecified versions of existing questions by stripping key premises.

Uses Claude to rewrite questions with a critical piece of information removed,
making them unanswerable without clarification.

Input:  data/question_bank_annotated.json
Output: data/underspecified_questions_generated.json

Usage:
  python scripts/generate_underspecified.py
  python scripts/generate_underspecified.py --n 60
  python scripts/generate_underspecified.py --concurrency 10
"""

import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

import httpx

INPUT_FILE = BASE_DIR / "data" / "question_bank_annotated.json"
OUTPUT_FILE = BASE_DIR / "data" / "underspecified_questions_generated.json"

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"

PROMPT = """Take this question and rewrite it so that it is UNANSWERABLE — remove or obscure one critical piece of information so that the question cannot be answered without asking for clarification.

Rules:
- The rewritten question should LOOK like a real question (not obviously broken)
- Remove exactly ONE key element (a specific referent, a necessary condition, a measurement, etc.)
- The question should be clearly unanswerable to someone paying attention, but tempting to guess at
- Do NOT make it trivially obvious (don't just delete half the sentence)

Original question: {question}
Original answer: {answer}

Respond with ONLY a JSON object:
{{"underspecified_question": "the rewritten question", "removed_element": "what was removed", "clarification_needed": "what someone would need to ask to answer it"}}"""


async def generate_one(
    client: httpx.AsyncClient,
    q: dict,
    api_key: str,
) -> dict | None:
    prompt = PROMPT.format(question=q["question"], answer=q["answer"])
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
                "max_tokens": 256,
                "temperature": 0.3,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        data = response.json()
        content = data["content"][0]["text"].strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        parsed = json.loads(content)
        return {
            "id": "",  # assigned after collection
            "question": parsed["underspecified_question"],
            "reason": parsed["removed_element"],
            "clarification_needed": parsed["clarification_needed"],
            "original_id": q["id"],
            "original_question": q["question"],
            "original_answer": q["answer"],
            "source": q.get("source", "unknown"),
        }
    except Exception as e:
        return None


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    api_key = os.environ["ANTHROPIC_API_KEY"]

    with open(INPUT_FILE) as f:
        bank = json.load(f)

    random.seed(42)
    random.shuffle(bank)
    candidates = bank[:args.n * 2]  # oversample for failures

    semaphore = asyncio.Semaphore(args.concurrency)
    results = []
    errors = 0

    async def run_with_sem(q):
        async with semaphore:
            return await generate_one(client, q, api_key)

    t0 = time.time()
    print(f"Generating {args.n} underspecified questions (concurrency={args.concurrency})...")

    async with httpx.AsyncClient() as client:
        tasks = [run_with_sem(q) for q in candidates]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is not None and len(results) < args.n:
                results.append(result)
                if len(results) % 10 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{len(results)}/{args.n}] {elapsed:.1f}s", flush=True)
            elif result is None:
                errors += 1

    # Assign IDs
    for i, r in enumerate(results):
        r["id"] = f"underspec_gen_{i+1:03d}"

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s — {len(results)} generated, {errors} errors")

    # Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")

    # Show a few examples
    print(f"\nExamples:")
    for r in results[:5]:
        print(f"  Original: {r['original_question'][:80]}...")
        print(f"  Underspec: {r['question'][:80]}...")
        print(f"  Removed: {r['reason']}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
