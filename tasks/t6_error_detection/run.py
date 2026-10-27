"""
T6 Error Detection — Generate-then-Review on Q_BANK

Model answers MC questions with step-by-step reasoning, then reviews
its own work. Auto-scored against ground truth.

Usage:
    python run.py claude-sonnet-4-6
    python run.py gpt-5.4
    python run.py google/gemini-3.1-pro-preview
    python run.py gpt-5.4 --sources gpqa_diamond mmlu_pro --limit 10
"""

import json
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_path, override=True)

RESULTS_DIR = Path(__file__).parent / "results" / "gen_review_qbank"
QBANK_PATH = Path(__file__).parent / "qbank.json"


# ---------------------------------------------------------------------------
# Load and sample questions
# ---------------------------------------------------------------------------

def load_questions(sources=None, limit=None, seed=42):
    data = json.loads(QBANK_PATH.read_text())

    # Filter to MC only
    data = [q for q in data if q.get("choices")]

    if sources:
        data = [q for q in data if q["source"] in sources]

    random.seed(seed)
    random.shuffle(data)

    if limit:
        data = data[:limit]

    return data


# ---------------------------------------------------------------------------
# Model calling (multi-turn)
# ---------------------------------------------------------------------------

def call_model(model, messages, max_tokens=8192):
    provider = model.split("/")[0] if "/" in model else None
    is_openai = provider == "openai" or (not provider and (model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3")))
    is_anthropic = provider == "anthropic" or (not provider and model.startswith("claude"))

    if is_openai:
        from openai import OpenAI
        client = OpenAI()
        api_model = model.replace("openai/", "")
        response = client.chat.completions.create(
            model=api_model, max_completion_tokens=max_tokens, messages=messages,
        )
        return response.choices[0].message.content, {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
    elif is_anthropic:
        from anthropic import Anthropic
        client = Anthropic()
        api_model = model.replace("anthropic/", "").replace(".", "-", 1).replace(".", "-")
        # Resolve short names to full model IDs with date suffix
        ANTHROPIC_ALIASES = {
            "claude-sonnet-4": "claude-sonnet-4-20250514",
            "claude-sonnet-4-6": "claude-sonnet-4-6",
            "claude-opus-4-6": "claude-opus-4-6",
        }
        api_model = ANTHROPIC_ALIASES.get(api_model, api_model)
        response = client.messages.create(
            model=api_model, max_tokens=16384,
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
    else:
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
        response = client.chat.completions.create(
            model=model, max_completion_tokens=max_tokens, messages=messages,
        )
        return response.choices[0].message.content or "", {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }


# ---------------------------------------------------------------------------
# Format question
# ---------------------------------------------------------------------------

def format_question(q):
    choices_text = "\n".join(f"  {k}) {v}" for k, v in q["choices"].items())
    return f"{q['question']}\n\n{choices_text}"


# ---------------------------------------------------------------------------
# Phase 1: Generate answer with reasoning
# ---------------------------------------------------------------------------

PHASE1_PROMPT = """Answer the following multiple-choice question. Think through it step by step, then give your final answer.

{question}

Show your reasoning step by step (number each step), then on the final line write exactly:
ANSWER: [letter]"""


# ---------------------------------------------------------------------------
# Phase 2: Self-review
# ---------------------------------------------------------------------------

PHASE2_PROMPT = """Now carefully review your solution above. Check each step of your reasoning for correctness.

For each step, provide:
- VERDICT: CORRECT or ERROR
- CONFIDENCE: 0-100 (how confident you are in this verdict)
- REASONING: brief explanation

If you find any errors, also state what your corrected answer would be.

Respond in this format:
Step 1:
  VERDICT: [CORRECT/ERROR]
  CONFIDENCE: [0-100]
  REASONING: [analysis]

Step 2:
  VERDICT: [CORRECT/ERROR]
  CONFIDENCE: [0-100]
  REASONING: [analysis]

...

REVISED ANSWER: [letter, or SAME if no change]"""


# ---------------------------------------------------------------------------
# Extract answer letter from response
# ---------------------------------------------------------------------------

def extract_answer(text):
    """Extract the answer letter from model response. Multi-strategy with LLM fallback."""
    if not text:
        return None

    # Strategy 1: ANSWER: X (strict format)
    m = re.search(r'ANSWER:\s*\**\s*([A-Za-z])\b', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Strategy 2: "answer is X" / "the answer is (X)" / "I choose X"
    m = re.search(r'(?:answer|choice|select|choose|pick)\s*(?:is|:|-|=)\s*\**\s*\(?([A-Za-z])\)?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Strategy 3: **X** or **X)** (markdown bold letter)
    m = re.search(r'\*\*([A-D])\)?\.?\*\*', text)
    if m:
        return m.group(1).upper()

    # Strategy 4: "X)" or "(X)" near end
    last_500 = text[-500:]
    m = re.search(r'(?:^|\s)\(?([A-D])\)\s*', last_500, re.MULTILINE)
    if m:
        return m.group(1).upper()

    # Strategy 5: standalone letter on last non-empty lines
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    for line in reversed(lines[-5:]):
        clean = line.replace('*', '').replace('#', '').strip()
        m = re.match(r'^([A-D])\.?\s*$', clean)
        if m:
            return m.group(1).upper()

    # Strategy 6: LLM extraction as last resort
    return extract_answer_llm(text)


def extract_answer_llm(text):
    """Use a fast cheap model to extract the answer when regex fails."""
    try:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            max_completion_tokens=10,
            messages=[
                {"role": "user", "content": (
                    f"What multiple choice answer (single letter A-D) was selected in this text? "
                    f"Reply with ONLY the letter.\n\n{text[-1000:]}"
                )}
            ],
        )
        letter = response.choices[0].message.content.strip().upper()
        if len(letter) == 1 and letter in "ABCDEFGHIJ":
            return letter
    except Exception:
        pass
    return None


def extract_revised_answer(text):
    """Extract revised answer from self-review."""
    m = re.search(r'REVISED\s*ANSWER:\s*(?:\**)?\s*([A-Za-z]|SAME)', text, re.IGNORECASE)
    if m:
        val = m.group(1).upper()
        return None if val == "SAME" else val
    return None


def extract_found_errors(text):
    """Check if the review found any errors."""
    verdicts = re.findall(r'VERDICT:\s*(CORRECT|ERROR)', text, re.IGNORECASE)
    n_errors = sum(1 for v in verdicts if v.upper() == "ERROR")
    n_correct = sum(1 for v in verdicts if v.upper() == "CORRECT")
    confidences = [int(m) for m in re.findall(r'CONFIDENCE:\s*(\d+)', text)]
    return {
        "n_errors": n_errors,
        "n_correct": n_correct,
        "n_steps_reviewed": len(verdicts),
        "found_error": n_errors > 0,
        "confidences": confidences,
        "mean_confidence": round(sum(confidences) / len(confidences), 1) if confidences else None,
    }


# ---------------------------------------------------------------------------
# Process one question (both phases)
# ---------------------------------------------------------------------------

def process_question(model, q):
    qid = q["id"]
    question_text = format_question(q)

    # Normalise ground truth to letter
    gt_answer = q["answer"].strip().upper()
    if len(gt_answer) > 1:
        # Answer might be the text, find which choice matches
        for k, v in q["choices"].items():
            if v.strip() == q["answer"].strip():
                gt_answer = k.upper()
                break

    # Phase 1: Generate
    p1_messages = [{"role": "user", "content": PHASE1_PROMPT.format(question=question_text)}]
    try:
        p1_response, p1_usage = call_model(model, p1_messages)
    except Exception as e:
        return {"id": qid, "error": f"Phase 1: {e}"}

    model_answer = extract_answer(p1_response or "")
    phase1_correct = model_answer == gt_answer if model_answer else None

    # Phase 2: Self-review
    p2_messages = [
        {"role": "user", "content": PHASE1_PROMPT.format(question=question_text)},
        {"role": "assistant", "content": p1_response},
        {"role": "user", "content": PHASE2_PROMPT},
    ]
    try:
        p2_response, p2_usage = call_model(model, p2_messages)
    except Exception as e:
        return {"id": qid, "error": f"Phase 2: {e}"}

    review = extract_found_errors(p2_response or "")
    revised = extract_revised_answer(p2_response or "")

    # Determine final answer after self-review
    final_answer = revised if revised else model_answer
    phase2_correct = final_answer == gt_answer if final_answer else None

    # Classify outcome
    if phase1_correct and not review["found_error"]:
        outcome = "TRUE_APPROVE"       # correct + approved = good
    elif phase1_correct and review["found_error"]:
        outcome = "FALSE_CRITICISM"    # correct but flagged error = bad
    elif not phase1_correct and review["found_error"]:
        outcome = "SELF_CORRECTED"     # wrong + found error = good metacognition
    elif not phase1_correct and not review["found_error"]:
        outcome = "BLIND_SPOT"         # wrong + approved = bad metacognition
    else:
        outcome = "UNCLEAR"

    return {
        "id": qid,
        "source": q["source"],
        "gt_answer": gt_answer,
        "model_answer": model_answer,
        "phase1_response": p1_response,
        "phase2_response": p2_response,
        "phase1_correct": phase1_correct,
        "found_error": review["found_error"],
        "n_errors_flagged": review["n_errors"],
        "n_steps_reviewed": review["n_steps_reviewed"],
        "mean_confidence": review["mean_confidence"],
        "revised_answer": revised,
        "final_answer": final_answer,
        "phase2_correct": phase2_correct,
        "outcome": outcome,
        "tokens": {
            "gen": p1_usage,
            "review": p2_usage,
        },
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(model, questions):
    print(f"\n{'='*60}")
    print(f"  T6 Generate-then-Review (Q_BANK)")
    print(f"  Model: {model}")
    print(f"  Questions: {len(questions)}")
    by_source = Counter(q["source"] for q in questions)
    for s, n in sorted(by_source.items()):
        print(f"    {s}: {n}")
    print(f"{'='*60}")

    results = []

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_question, model, q): q["id"] for q in questions}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            if "error" in r:
                print(f"  [{r['id']}] ERROR: {r['error']}")
            else:
                mark = {"TRUE_APPROVE": ".", "SELF_CORRECTED": "S", "BLIND_SPOT": "X", "FALSE_CRITICISM": "?", "UNCLEAR": "-"}
                sym = mark.get(r["outcome"], "-")
                correct_str = "correct" if r["phase1_correct"] else "WRONG"
                print(f"  [{r['id']:20s}] {sym} {correct_str:7s} → {r['outcome']}")

    # Sort by ID
    results.sort(key=lambda r: r.get("id", ""))

    # --- Summary ---
    valid = [r for r in results if "error" not in r]
    outcomes = Counter(r["outcome"] for r in valid)
    phase1_acc = sum(1 for r in valid if r.get("phase1_correct")) / len(valid) if valid else 0
    phase2_acc = sum(1 for r in valid if r.get("phase2_correct")) / len(valid) if valid else 0

    wrong = [r for r in valid if not r.get("phase1_correct")]
    blind_spots = [r for r in valid if r["outcome"] == "BLIND_SPOT"]
    self_corrected = [r for r in valid if r["outcome"] == "SELF_CORRECTED"]
    false_crit = [r for r in valid if r["outcome"] == "FALSE_CRITICISM"]

    blind_spot_rate = len(blind_spots) / len(wrong) if wrong else 0
    self_correction_rate = len(self_corrected) / len(wrong) if wrong else 0

    print(f"\n{'='*60}")
    print(f"  RESULTS — {model}")
    print(f"{'='*60}")
    n_p1_correct = sum(1 for r in valid if r.get("phase1_correct"))
    print(f"  Phase 1 accuracy:     {phase1_acc:.0%} ({n_p1_correct}/{len(valid)})")
    print(f"  Phase 2 accuracy:     {phase2_acc:.0%} (after self-correction)")
    print(f"  Errors made:          {len(wrong)}")
    print(f"  Self-corrected:       {len(self_corrected)} ({self_correction_rate:.0%} of errors)")
    print(f"  Blind spots:          {len(blind_spots)} ({blind_spot_rate:.0%} of errors)")
    print(f"  False self-criticism: {len(false_crit)}")
    print(f"\n  Outcomes: {dict(outcomes)}")

    # By source
    print(f"\n  By source:")
    for source in sorted(set(r.get("source", "?") for r in valid)):
        src_items = [r for r in valid if r.get("source") == source]
        src_correct = sum(1 for r in src_items if r.get("phase1_correct"))
        src_wrong = [r for r in src_items if not r.get("phase1_correct")]
        src_blind = sum(r["outcome"] == "BLIND_SPOT" for r in src_items)
        src_self = sum(r["outcome"] == "SELF_CORRECTED" for r in src_items)
        print(f"    {source:20s} acc={src_correct}/{len(src_items)}  wrong={len(src_wrong)}  blind={src_blind}  self-corrected={src_self}")

    # Confidence on blind spots vs self-corrected
    blind_confs = [r["mean_confidence"] for r in blind_spots if r["mean_confidence"] is not None]
    correct_confs = [r["mean_confidence"] for r in valid if r["outcome"] == "TRUE_APPROVE" and r["mean_confidence"] is not None]
    if blind_confs:
        print(f"\n  Confidence (blind spots):   {sum(blind_confs)/len(blind_confs):.1f}")
    if correct_confs:
        print(f"  Confidence (true approves): {sum(correct_confs)/len(correct_confs):.1f}")

    return results


def main():
    args = sys.argv[1:]
    model = "claude-sonnet-4-6"
    sources = ["gpqa_diamond", "mmlu_pro", "mmlu"]
    limit = 153  # default: match other model runs

    i = 0
    while i < len(args):
        if args[i] == "--sources":
            sources = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                sources.append(args[i])
                i += 1
        elif args[i] == "--limit":
            limit = int(args[i + 1])
            i += 2
        elif not args[i].startswith("--"):
            model = args[i]
            i += 1
        else:
            i += 1

    questions = load_questions(sources=sources, limit=limit)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    outfile = RESULTS_DIR / f"{model.replace('/', '_')}.json"

    # --retry: load existing results, only rerun items with errors/nulls
    retry = "--retry" in sys.argv
    if retry and outfile.exists():
        existing = json.loads(outfile.read_text())
        failed_ids = {r["id"] for r in existing if "error" in r or r.get("phase1_correct") is None}
        ok_items = [r for r in existing if r["id"] not in failed_ids]
        retry_qs = [q for q in questions if q["id"] in failed_ids]
        print(f"\n  Retry mode: {len(ok_items)} ok, {len(failed_ids)} to retry")
        if not retry_qs:
            print("  Nothing to retry.")
            return
        retried = run(model, retry_qs)
        merged = ok_items + retried
        merged.sort(key=lambda r: r.get("id", ""))
        outfile.write_text(json.dumps(merged, indent=2))
        print(f"\n  Saved retried results to {outfile}")
    else:
        results = run(model, questions)
        outfile.write_text(json.dumps(results, indent=2))
        print(f"\n  Saved to {outfile}")


if __name__ == "__main__":
    main()
