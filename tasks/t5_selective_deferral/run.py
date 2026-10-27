"""
T5 Selective Deferral pilot runner — async, multi-model via OpenRouter.

Usage:
  python run.py                          # 10 Qs per block, default model
  python run.py --n 30                   # 30 Qs per block
  python run.py --all                    # full bank
  python run.py --model gpt-4o           # specific model
  python run.py --compare --all          # all models, full bank
  python run.py --concurrency 15         # more parallel requests
"""

import argparse
import asyncio
import json
import os
import random
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
from prompts import T5_SYSTEM, t5_question_prompt, T5_BLOCK_TEMPLATES_CONDITION_B
from scoring import t5_score_all

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
BLOCKS = ["low", "medium", "high"]


def load_questions(n_per_block: int | None = None) -> list[dict]:
    """Load and assign questions to blocks.

    Well-specified questions are randomly distributed across blocks.
    Underspecified questions are split 10 per block.
    """
    with open(BASE_DIR / "data" / "question_bank_annotated.json") as f:
        well_specified = json.load(f)

    # Load underspecified questions (validated generated only)
    with open(BASE_DIR / "data" / "underspecified_questions_validated.json") as f:
        underspecified = json.load(f)

    random.seed(42)

    # Assign well-specified to blocks randomly
    shuffled = list(well_specified)
    random.shuffle(shuffled)
    for i, q in enumerate(shuffled):
        q["block"] = BLOCKS[i % 3]
        q["is_underspecified"] = False

    # Assign underspecified: 10 per block
    for i, q in enumerate(underspecified):
        q["block"] = BLOCKS[i % 3]
        q["is_underspecified"] = True
        q["choices"] = None
        q["format"] = "open_ended"
        # Ground truth for underspecified: CLARIFY is correct, no "answer"
        if "answer" not in q:
            q["answer"] = ""

    all_questions = shuffled + underspecified

    # Subsample per block if requested (n_per_block = well-specified only)
    if n_per_block is not None:
        sampled = []
        for block in BLOCKS:
            block_qs = [q for q in all_questions if q["block"] == block]
            well = [q for q in block_qs if not q["is_underspecified"]]
            under = [q for q in block_qs if q["is_underspecified"]]
            sampled.extend(well[:n_per_block])
            sampled.extend(under)  # always include all underspecified
        all_questions = sampled

    return all_questions


def build_prompt(question: dict, condition: str = "A") -> str:
    prompt = T5_SYSTEM + "\n\n" + t5_question_prompt(
        question["question"], question["block"], question.get("choices"), condition=condition
    )
    prompt += '\n\nYou MUST respond with ONLY a JSON object, nothing else.\nFormat: {"decision": "ATTEMPT", "answer": "your answer", "confidence": 75}'
    prompt += '\nIf you DEFER or CLARIFY, set answer to "".'
    return prompt


def parse_response(content: str) -> dict | None:
    content = content.strip()

    # 1. Direct JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. Markdown code block
    if "```" in content:
        for block in content.split("```")[1::2]:
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue

    # 3. Find any JSON object in the text (greedy — grab the largest match)
    matches = list(re.finditer(r'\{', content))
    for m in matches:
        start = m.start()
        # Try progressively larger substrings from this {
        for end in range(len(content), start, -1):
            if content[end - 1] == '}':
                try:
                    return json.loads(content[start:end])
                except json.JSONDecodeError:
                    continue

    # 4. Try to reconstruct from known field patterns
    decision_match = re.search(r'"?decision"?\s*[:=]\s*"?(ATTEMPT|DEFER|CLARIFY)"?', content, re.IGNORECASE)
    conf_match = re.search(r'"?confidence"?\s*[:=]\s*(\d+)', content)
    answer_match = re.search(r'"?answer"?\s*[:=]\s*"([^"]*)"', content)

    if decision_match:
        return {
            "decision": decision_match.group(1).upper(),
            "confidence": int(conf_match.group(1)) if conf_match else 50,
            "answer": answer_match.group(1) if answer_match else "",
        }

    return None


def check_answer(resp_answer, truth: str, choices: dict | None) -> bool:
    if not truth:  # underspecified questions have no correct answer
        return False
    resp = str(resp_answer).strip().upper()
    truth_upper = truth.strip().upper()
    if resp == truth_upper:
        return True
    if choices:
        correct_label = None
        for label, text in choices.items():
            if text.upper() == truth_upper:
                correct_label = label.upper()
                break
        if correct_label:
            if resp == correct_label:
                return True
            cleaned = re.sub(r'[.)(]', '', resp).strip()
            if cleaned == correct_label:
                return True
    if len(truth_upper) >= 4 and truth_upper in resp:
        return True
    return False


async def run_question_async(
    client: httpx.AsyncClient,
    model_id: str,
    question: dict,
    api_key: str,
    max_retries: int = 3,
    condition: str = "A",
    provider: dict | None = None,
) -> tuple[dict, dict | None]:
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
                    # Optional OpenRouter provider routing (None = OpenRouter default).
                    **({"provider": provider} if provider else {}),
                },
                timeout=30,
            )
            if response.status_code == 429 or response.status_code >= 500:
                print(f"    HTTP {response.status_code} (attempt {attempt+1}): {response.text[:200]}", flush=True)
                await asyncio.sleep(2 ** attempt)
                continue
            data = response.json()
            if "error" in data:
                print(f"    API error (attempt {attempt+1}): {data['error']}", flush=True)
                await asyncio.sleep(2 ** attempt)
                continue
            content = data["choices"][0]["message"]["content"]
            parsed = parse_response(content)
            if parsed is None:
                print(f"    PARSE FAIL: {content[:200]}", flush=True)
            return question, parsed
        except Exception as e:
            print(f"    Exception (attempt {attempt+1}): {type(e).__name__}: {e}", flush=True)
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            continue
    return question, None


async def run_model_async(
    model_name: str,
    model_id: str,
    questions: list[dict],
    concurrency: int = 10,
    condition: str = "A",
) -> dict:
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
            if answered % 50 == 0 or answered == len(questions):
                elapsed = time.time() - t0
                qps = answered / elapsed if elapsed > 0 else 0
                print(f"  [{answered}/{len(questions)}] {qps:.1f} q/s, {errors} errors", flush=True)

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(questions)/elapsed:.1f} q/s, {errors} errors)")

    # Process results in original order
    q_id_to_result = {}
    for q, r in results_list:
        key = q.get("id", id(q))
        q_id_to_result[key] = (q, r)

    decisions, correct_list, confidences, blocks, is_underspecified = [], [], [], [], []
    raw_responses = []

    for q in questions:
        key = q.get("id", id(q))
        _, result = q_id_to_result.get(key, (q, None))
        if result is None:
            continue

        decision = result.get("decision", "ATTEMPT").upper()
        answer = result.get("answer", "")
        conf = result.get("confidence", 50)

        if decision == "ATTEMPT":
            is_correct = check_answer(answer, q.get("answer", ""), q.get("choices"))
        else:
            is_correct = False  # didn't attempt

        decisions.append(decision)
        correct_list.append(is_correct)
        confidences.append(conf)
        blocks.append(q["block"])
        is_underspecified.append(q["is_underspecified"])

        raw_responses.append({
            "question_id": q.get("id", ""),
            "question": q["question"],
            "source": q.get("source", "underspecified"),
            "domain": q.get("domain", ""),
            "block": q["block"],
            "is_underspecified": q["is_underspecified"],
            "ground_truth": q.get("answer", ""),
            "decision": decision,
            "model_answer": answer,
            "correct": is_correct,
            "confidence": conf,
        })

    if not decisions:
        print(f"  No valid responses for {model_name}")
        return {}

    scores = t5_score_all(decisions, correct_list, confidences, blocks, is_underspecified)
    scores["model"] = model_name
    scores["model_id"] = model_id
    scores["condition"] = condition
    scores["n_answered"] = len(decisions)
    scores["n_errors"] = errors
    scores["n_questions"] = len(questions)
    scores["elapsed_seconds"] = round(elapsed, 1)
    scores["raw_responses"] = raw_responses

    # Save
    suffix = "_condB" if condition == "B" else ""
    results_dir = TASK_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{model_name}{suffix}.json", "w") as f:
        json.dump(scores, f, indent=2, default=str)

    return scores


def print_scores(scores: dict):
    if not scores:
        return
    dr = scores.get("deferral_rates", {})
    clar = scores.get("clarification", {})
    conf = scores.get("deferral_clarify_confusion", {})
    ev = scores.get("expected_value", {})

    print(f"\n{'─'*60}")
    print(f"  {scores.get('model', '?')} — {scores['n_answered']}/{scores['n_questions']} answered ({scores.get('elapsed_seconds', '?')}s)")
    print(f"{'─'*60}")
    print(f"  DAI:                   {scores.get('dai', 0):.3f}")
    print(f"  Deferral rates:        low={dr.get('low',0):.0%}  med={dr.get('medium',0):.0%}  high={dr.get('high',0):.0%}")
    print(f"  Accuracy (attempted):  {scores.get('accuracy_when_attempting', 0):.0%}")
    print(f"  Deferral quality:      {scores.get('deferral_quality', 0):.0%}")
    print(f"  EV:                    low={ev.get('low',0)}  med={ev.get('medium',0)}  high={ev.get('high',0)}  total={ev.get('total',0)}")
    print(f"  Clarification P/R/F1:  P={clar.get('precision',0):.0%}  R={clar.get('recall',0):.0%}  F1={clar.get('f1',0):.0%}")
    print(f"  Defer-when-CLARIFY:    {conf.get('defer_when_should_clarify',0):.0%}")
    print(f"  CLARIFY-when-defer:    {conf.get('clarify_when_should_defer',0):.0%}")
    print(f"{'─'*60}")


async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Questions per block (well-specified)")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--condition", type=str, default="A", choices=["A", "B"],
                        help="A=payoffs only (default), B=explicit thresholds")
    args = parser.parse_args()

    if args.condition == "B":
        print(f"  CONDITION B: explicit thresholds included in prompt")

    n_per_block = None if args.all else args.n
    questions = load_questions(n_per_block)

    block_counts = {}
    under_counts = {}
    for q in questions:
        block_counts[q["block"]] = block_counts.get(q["block"], 0) + 1
        if q["is_underspecified"]:
            under_counts[q["block"]] = under_counts.get(q["block"], 0) + 1

    print(f"Questions loaded: {len(questions)}")
    for b in BLOCKS:
        print(f"  {b}: {block_counts.get(b, 0)} ({under_counts.get(b, 0)} underspecified)")

    if args.compare:
        all_scores = []
        for name, model_id in OPENROUTER_MODELS.items():
            scores = await run_model_async(name, model_id, questions, args.concurrency, condition=args.condition)
            print_scores(scores)
            all_scores.append(scores)

        # Comparison table
        print(f"\n{'='*85}")
        print(f"  T5 COMPARISON — {len(questions)} questions")
        print(f"{'='*85}")
        print(f"  {'Model':<22} {'DAI':>6} {'Def_L':>6} {'Def_M':>6} {'Def_H':>6} {'Acc':>5} {'Clar_P':>7} {'Clar_R':>7} {'EV':>6} {'Time':>5}")
        print(f"  {'─'*75}")
        for s in all_scores:
            if not s:
                continue
            dr = s.get("deferral_rates", {})
            clar = s.get("clarification", {})
            ev = s.get("expected_value", {})
            print(f"  {s['model']:<22} {s.get('dai',0):>6.3f} {dr.get('low',0):>5.0%} {dr.get('medium',0):>5.0%} "
                  f"{dr.get('high',0):>5.0%} {s.get('accuracy_when_attempting',0):>4.0%} "
                  f"{clar.get('precision',0):>6.0%} {clar.get('recall',0):>6.0%} "
                  f"{ev.get('total',0):>6} {s.get('elapsed_seconds',''):>4}s")
        print(f"{'='*85}")

        out_path = TASK_DIR / "results" / "comparison.json"
        with open(out_path, "w") as f:
            json.dump(all_scores, f, indent=2, default=str)
        print(f"\nSaved to {out_path}")
    else:
        model_id = OPENROUTER_MODELS.get(args.model, args.model)
        scores = await run_model_async(args.model, model_id, questions, args.concurrency, condition=args.condition)
        print_scores(scores)


if __name__ == "__main__":
    asyncio.run(main_async())
