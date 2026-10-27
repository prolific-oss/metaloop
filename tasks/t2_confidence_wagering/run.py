"""
Local test of T2 wagering pipeline across multiple models via OpenRouter.
Uses async for concurrent API calls (~10x faster than sequential).

Usage:
  python run.py                          # 10 Qs, default model
  python run.py --n 30                   # 30 Qs
  python run.py --all                    # full bank
  python run.py --model gpt-4o           # specific model
  python run.py --compare --n 15         # run all models
  python run.py --concurrency 20         # more parallel requests
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

TASK_DIR = Path(__file__).resolve().parent
BASE_DIR = TASK_DIR.parent.parent
load_dotenv(BASE_DIR / ".env", override=True)
sys.path.insert(0, str(BASE_DIR / "src"))

import httpx
from prompts import T2_SYSTEM, T2_SYSTEM_CONDITION_B, t2_question_prompt
from scoring import t2_score_all

OPENROUTER_MODELS = {
    "sonnet-4.6": "anthropic/claude-sonnet-4.6",
    "opus-4.6": "anthropic/claude-opus-4.6",
    "sonnet-4": "anthropic/claude-sonnet-4",
    "gpt-5.4": "openai/gpt-5.4",
    "gpt-4.1": "openai/gpt-4.1",
    "qwen3-235b": "qwen/qwen3-235b-a22b",
    "deepseek-v3.2": "deepseek/deepseek-v3.2",
    "mistral-large": "mistralai/mistral-large-2512",
    "grok-4": "x-ai/grok-4",
    "grok-4.1-fast": "x-ai/grok-4.1-fast",
    "gemini-3.1-pro": "google/gemini-3.1-pro-preview",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemma-4-31b": "google/gemma-4-31b-it",
}

DEFAULT_MODEL = "sonnet-4.6"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def build_prompt(question: dict, condition: str = "A") -> str:
    system = T2_SYSTEM_CONDITION_B if condition == "B" else T2_SYSTEM
    prompt = system + "\n\n" + t2_question_prompt(question["question"], question.get("choices"))
    prompt += '\n\nYou MUST respond with ONLY a JSON object, nothing else. No reasoning, no explanation.\nFormat: {"answer": "your answer", "confidence": 75, "wager": "MEDIUM"}'
    return prompt


def parse_response(content: str) -> dict | None:
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    if "```" in content:
        block = content.split("```")[1]
        if block.startswith("json"):
            block = block[4:]
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            pass

    match = re.search(r'\{[^{}]*\}', content)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def check_answer(resp_answer, truth: str, choices: dict | None) -> bool:
    resp = str(resp_answer).strip()
    resp_upper = resp.upper()
    truth_upper = truth.strip().upper()

    if resp_upper == truth_upper:
        return True

    if choices:
        label_to_text = {l.upper(): t.upper() for l, t in choices.items()}

        correct_label = None
        for label, text in choices.items():
            if text.upper() == truth_upper:
                correct_label = label.upper()
                break

        if correct_label:
            if resp_upper == correct_label:
                return True
            cleaned = re.sub(r'[.)(]', '', resp_upper).strip()
            if cleaned == correct_label:
                return True

        if resp_upper in label_to_text.values() and resp_upper == truth_upper:
            return True

    if len(truth_upper) >= 4 and truth_upper in resp_upper:
        return True

    return False


async def run_question_async(
    client: httpx.AsyncClient,
    model_id: str,
    question: dict,
    api_key: str,
    max_retries: int = 3,
    condition: str = "A",
) -> tuple[dict, dict | None]:
    """Run a single question async with retry + backoff."""
    prompt = build_prompt(question, condition)

    for attempt in range(max_retries):
        try:
            response = await client.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model_id,
                    "max_tokens": 2048,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": "You are a JSON-only responder. Output valid JSON and nothing else."},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=30,
            )
            if response.status_code == 429 or response.status_code >= 500:
                await asyncio.sleep(2 ** attempt)
                continue
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return question, parse_response(content)
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            continue
    return question, None


async def run_model_async(
    model_name: str,
    model_id: str,
    questions: list[dict],
    concurrency: int = 10,
    verbose: bool = True,
    condition: str = "A",
) -> dict:
    """Run T2 on a single model with async concurrency."""
    api_key = os.environ["OPENROUTER_API_KEY"]
    semaphore = asyncio.Semaphore(concurrency)

    answered = 0
    errors = 0
    results_list = []

    async def run_with_sem(q):
        async with semaphore:
            return await run_question_async(client, model_id, q, api_key, condition=condition)

    t0 = time.time()
    print(f"\n  {model_name} ({model_id}) — {len(questions)} questions, concurrency={concurrency}")

    async with httpx.AsyncClient() as client:
        tasks = [run_with_sem(q) for q in questions]
        for coro in asyncio.as_completed(tasks):
            q, result = await coro
            answered += 1
            if result is None:
                errors += 1
            results_list.append((q, result))

            if answered % 25 == 0 or answered == len(questions):
                elapsed = time.time() - t0
                qps = answered / elapsed if elapsed > 0 else 0
                print(f"  [{answered}/{len(questions)}] {qps:.1f} q/s, {errors} errors", flush=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(questions)/elapsed:.1f} q/s, {errors} errors)")

    # Process results (maintain original order)
    q_id_to_result = {q["id"]: (q, r) for q, r in results_list}

    confidences, correct_list, wagers, raw_responses = [], [], [], []
    for q in questions:
        _, result = q_id_to_result.get(q["id"], (q, None))
        if result is None:
            continue

        is_correct = check_answer(result.get("answer", ""), q["answer"], q.get("choices"))
        conf = result.get("confidence", 50)
        wager = result.get("wager", "SAFE")

        confidences.append(conf)
        correct_list.append(is_correct)
        wagers.append(wager)
        raw_responses.append({
            "question_id": q["id"],
            "question": q["question"],
            "source": q.get("source", "unknown"),
            "domain": q.get("domain", "unknown"),
            "ground_truth": q["answer"],
            "model_answer": result.get("answer", ""),
            "correct": is_correct,
            "confidence": conf,
            "wager": wager,
        })

    if not confidences:
        print(f"  No valid responses for {model_name}")
        return {}

    scores = t2_score_all(confidences, correct_list, wagers)
    scores["model"] = model_name
    scores["model_id"] = model_id
    scores["condition"] = condition
    scores["accuracy"] = sum(correct_list) / len(correct_list)
    scores["n_answered"] = len(confidences)
    scores["n_errors"] = errors
    scores["raw_responses"] = raw_responses
    scores["n_questions"] = len(questions)
    scores["elapsed_seconds"] = round(elapsed, 1)

    # Save (separate dirs for A and B)
    suffix = f"_condB" if condition == "B" else ""
    results_dir = TASK_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{model_name}{suffix}.json", "w") as f:
        json.dump(scores, f, indent=2, default=str)

    return scores


def print_scores(scores: dict):
    if not scores:
        return
    cd = scores.get("confidence_distribution", {})
    print(f"\n{'─'*55}")
    print(f"  {scores.get('model', '?')} — {scores['n_answered']}/{scores['n_questions']} answered ({scores.get('elapsed_seconds', '?')}s)")
    print(f"{'─'*55}")
    print(f"  Accuracy:              {scores['accuracy']:.0%}")
    print(f"  ECE:                   {scores['ece']:.4f}")
    print(f"  Wagering Rationality:  {scores['wagering_rationality']:.1%}")
    print(f"  Total Payout:          {scores['total_payout']}")
    print(f"  Optimal Payout:        {scores['optimal_payout']}")
    print(f"  Payout Gap:            {scores['payout_gap']}")
    if cd:
        print(f"  Confidence:            mean={cd['mean']:.0f}% median={cd['median']}% std={cd['std']:.0f}")
    print(f"  Monitoring→Control:    {scores['monitoring_control_dissociation']['interpretation']}")
    print(f"{'─'*55}")


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model shortname: {', '.join(OPENROUTER_MODELS.keys())}")
    parser.add_argument("--compare", action="store_true",
                        help="Run all models and compare")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent API requests (default: 10)")
    parser.add_argument("--condition", type=str, default="A", choices=["A", "B"],
                        help="A=payoffs only (default), B=explicit thresholds")
    args = parser.parse_args()

    if args.condition == "B":
        print(f"  CONDITION B: explicit thresholds included in prompt")

    with open(BASE_DIR / "data" / "question_bank_annotated.json") as f:
        questions = json.load(f)

    if not args.all:
        questions = questions[:args.n]

    if args.compare:
        all_scores = []
        for name, model_id in OPENROUTER_MODELS.items():
            scores = await run_model_async(name, model_id, questions, args.concurrency, condition=args.condition)
            print_scores(scores)
            all_scores.append(scores)

        print(f"\n{'='*80}")
        print(f"  COMPARISON — {len(questions)} questions")
        print(f"{'='*80}")
        print(f"  {'Model':<22} {'Acc':>5} {'ECE':>6} {'WR':>6} {'Payout':>7} {'Gap':>5} {'Conf μ':>6} {'Time':>6}")
        print(f"  {'─'*68}")
        for s in all_scores:
            if not s:
                continue
            cd = s.get("confidence_distribution", {})
            print(f"  {s['model']:<22} {s['accuracy']:>4.0%} {s['ece']:>6.3f} {s['wagering_rationality']:>5.0%} "
                  f"{s['total_payout']:>7} {s['payout_gap']:>5} {cd.get('mean', 0):>5.0f}% {s.get('elapsed_seconds', ''):>5}s")
        print(f"{'='*80}")

        out_path = TASK_DIR / "results" / "comparison.json"
        with open(out_path, "w") as f:
            json.dump(all_scores, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")
    else:
        model_id = OPENROUTER_MODELS.get(args.model, args.model)
        scores = await run_model_async(args.model, model_id, questions, args.concurrency, condition=args.condition)
        print_scores(scores)


if __name__ == "__main__":
    asyncio.run(main_async())
