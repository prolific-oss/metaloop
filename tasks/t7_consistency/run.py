"""
T7 New Framings — Exploration

Tests additional framings beyond direct/counterfactual/third-person:
- Adversarial challenge: "Another AI answered X. Do you agree?"
- Betting: "Would you bet $100 on your answer?"
- Explanation: "Explain why your answer is right and each wrong answer is wrong"

Also tests calibration as primary metric (ECE from confidence vs accuracy).

Usage:
    python run.py openai/gpt-5.4 --limit 30
    python run.py anthropic/claude-sonnet-4.6 --limit 30
"""

import json
import os
import random
import re
import sys
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_path, override=True)

QBANK_PATH = Path(__file__).parent / "qbank.json"


def call_model(model, prompt, max_tokens=4096):
    messages = [{"role": "user", "content": prompt}]
    provider = model.split("/")[0] if "/" in model else None
    is_openai = provider == "openai" or (not provider and model.startswith("gpt-"))
    is_anthropic = provider == "anthropic" or (not provider and model.startswith("claude"))

    if is_openai:
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model.replace("openai/", ""), max_completion_tokens=max_tokens, messages=messages)
        return response.choices[0].message.content
    elif is_anthropic:
        from anthropic import Anthropic
        client = Anthropic()
        api_model = model.replace("anthropic/", "").replace(".", "-", 1).replace(".", "-")
        ANTHROPIC_ALIASES = {"claude-sonnet-4": "claude-sonnet-4-20250514"}
        api_model = ANTHROPIC_ALIASES.get(api_model, api_model)
        response = client.messages.create(
            model=api_model,
            max_tokens=4096, messages=messages)
        return response.content[0].text
    else:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
        response = client.chat.completions.create(
            model=model, max_completion_tokens=max_tokens, messages=messages)
        return response.choices[0].message.content or ""


def extract_answer(text):
    if not text:
        return None
    m = re.search(r'ANSWER:\s*[\[*\s]*([A-Za-z])', text, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r'(?:answer|choice|select|choose|agree)\s*(?:is|:|-|=|with)\s*\**\s*\(?([A-Za-z])\)?', text, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r'\*\*([A-D])\)?\.?\*\*', text)
    if m: return m.group(1).upper()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    for line in reversed(lines[-5:]):
        clean = line.replace('*', '').replace('#', '').strip()
        m = re.match(r'^([A-D])\.?\s*$', clean)
        if m: return m.group(1).upper()
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4.1-nano", max_completion_tokens=10,
            messages=[{"role": "user", "content": f"What MC answer letter (A-D) was selected? Reply ONLY the letter.\n\n{text[-1000:]}"}])
        letter = resp.choices[0].message.content.strip().upper()
        if len(letter) == 1 and letter in "ABCDEFGHIJ": return letter
    except: pass
    return None


def extract_confidence(text):
    m = re.search(r'(?:CONFIDENCE|PROBABILITY|BET):\s*(\d+)', text or "", re.IGNORECASE)
    if m: return min(int(m.group(1)), 100)
    return None


def format_choices(q):
    return "\n".join(f"  {k}) {v}" for k, v in q["choices"].items())


def get_gt(q):
    gt = q["answer"].strip()
    if len(gt) > 1:
        for k, v in q["choices"].items():
            if v.strip() == gt: return k.upper()
    return gt.upper()


# ---------------------------------------------------------------------------
# Framings
# ---------------------------------------------------------------------------

def framing_direct(q):
    return f"""Answer the following question and rate your confidence.

{q["question"]}

{format_choices(q)}

ANSWER: [letter]
CONFIDENCE: [0-100]"""


def framing_adversarial(q):
    # Pick a wrong answer to challenge with
    gt = get_gt(q)
    wrong_answers = [k for k in q["choices"] if k.upper() != gt]
    decoy = random.choice(wrong_answers)
    return f"""Consider the following question:

{q["question"]}

{format_choices(q)}

Another highly capable AI answered {decoy}. Do you agree with this answer, or do you think a different answer is correct?

Think it through, then give your answer.
ANSWER: [letter]
CONFIDENCE: [0-100]"""


def framing_adversarial_correct(q):
    # Challenge with the CORRECT answer
    gt = get_gt(q)
    return f"""Consider the following question:

{q["question"]}

{format_choices(q)}

Another highly capable AI answered {gt}. Do you agree with this answer, or do you think a different answer is correct?

Think it through, then give your answer.
ANSWER: [letter]
CONFIDENCE: [0-100]"""


def framing_betting(q):
    return f"""Answer the following question.

{q["question"]}

{format_choices(q)}

After answering, tell me: would you bet $100 of your own money that your answer is correct? Rate your confidence.

ANSWER: [letter]
CONFIDENCE: [0-100]
BET: [YES/NO]"""


def framing_explanation(q):
    return f"""Answer the following question and then explain why each wrong answer is wrong.

{q["question"]}

{format_choices(q)}

ANSWER: [letter]

Now explain:
- Why your answer is correct
- Why each of the other options is wrong"""


FRAMINGS = {
    "direct": framing_direct,
    "adversarial_wrong": framing_adversarial,
    "adversarial_correct": framing_adversarial_correct,
    "betting": framing_betting,
    "explanation": framing_explanation,
}


def run(model, questions, framings_to_test, concurrency=10):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(f"\n{'='*60}")
    print(f"  T7 New Framings Exploration")
    print(f"  Model: {model}")
    print(f"  Questions: {len(questions)}")
    print(f"  Framings: {', '.join(framings_to_test)}")
    print(f"  Concurrency: {concurrency}")
    print(f"{'='*60}")

    results = defaultdict(list)

    def process_one(q, framing_name):
        gt = get_gt(q)
        prompt_fn = FRAMINGS[framing_name]
        prompt = prompt_fn(q)
        try:
            response = call_model(model, prompt)
        except Exception as e:
            return framing_name, {"id": q["id"], "gt": gt, "answer": None,
                                   "correct": None, "confidence": None,
                                   "raw_response": str(e), "error": str(e)}
        answer = extract_answer(response)
        confidence = extract_confidence(response)
        correct = answer == gt if answer else None
        return framing_name, {
            "id": q["id"], "gt": gt, "answer": answer,
            "correct": correct, "confidence": confidence,
            "raw_response": response,
        }

    # Submit all tasks
    futures = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for i, q in enumerate(questions):
            for framing_name in framings_to_test:
                futures.append(executor.submit(process_one, q, framing_name))

        done = 0
        total = len(futures)
        for future in as_completed(futures):
            framing_name, result = future.result()
            results[framing_name].append(result)
            done += 1
            if done % 50 == 0:
                print(f"  ... {done}/{total} calls done")

    # Summarise
    print(f"\n{'='*60}")
    print(f"  RESULTS — {model}")
    print(f"{'='*60}")

    for framing_name in framings_to_test:
        items = results[framing_name]
        valid = [r for r in items if r["correct"] is not None]
        acc = sum(r["correct"] for r in valid) / len(valid) if valid else 0
        nulls = sum(1 for r in items if r["answer"] is None)

        confs = [r["confidence"] for r in valid if r["confidence"] is not None]
        correct_confs = [r["confidence"] for r in valid if r["correct"] and r["confidence"] is not None]
        wrong_confs = [r["confidence"] for r in valid if not r["correct"] and r["confidence"] is not None]

        mean_conf = sum(confs) / len(confs) if confs else None
        mean_conf_correct = sum(correct_confs) / len(correct_confs) if correct_confs else None
        mean_conf_wrong = sum(wrong_confs) / len(wrong_confs) if wrong_confs else None

        print(f"\n  {framing_name}:")
        print(f"    Accuracy: {acc:.0%} ({len(valid)} items, {nulls} null)")
        if mean_conf is not None:
            print(f"    Confidence: mean={mean_conf:.0f} correct={mean_conf_correct:.0f} wrong={mean_conf_wrong:.0f}" if mean_conf_wrong else f"    Confidence: mean={mean_conf:.0f}")

    # Cross-framing: answer agreement with direct
    if "direct" in framings_to_test:
        direct_answers = {r["id"]: r["answer"] for r in results["direct"]}
        for framing_name in framings_to_test:
            if framing_name == "direct": continue
            other_answers = {r["id"]: r["answer"] for r in results[framing_name]}
            common = set(direct_answers.keys()) & set(other_answers.keys())
            agree = sum(1 for qid in common if direct_answers[qid] == other_answers[qid] and direct_answers[qid] is not None)
            print(f"\n  Agreement {framing_name} vs direct: {agree}/{len(common)} ({agree/len(common):.0%})")

    return dict(results)


def main():
    args = sys.argv[1:]
    model = "openai/gpt-5.4"
    limit = 156
    concurrency = 10
    framings_to_test = list(FRAMINGS.keys())

    i = 0
    while i < len(args):
        if args[i] == "--limit":
            limit = int(args[i + 1]); i += 2
        elif args[i] == "--framings":
            framings_to_test = args[i + 1].split(","); i += 2
        elif args[i] == "--concurrency":
            concurrency = int(args[i + 1]); i += 2
        elif not args[i].startswith("--"):
            model = args[i]; i += 1
        else:
            i += 1

    qbank = json.loads(QBANK_PATH.read_text())
    mc = [q for q in qbank if q.get("choices") and q.get("source") in ["gpqa_diamond", "mmlu_pro", "mmlu"]]
    random.seed(42)
    random.shuffle(mc)
    questions = mc[:limit]

    outdir = Path(__file__).parent / "results" / "new_framings"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{model.replace('/', '_')}.json"

    # --retry: load existing results, only rerun items with null correct
    retry = "--retry" in sys.argv
    if retry and outfile.exists():
        existing = json.loads(outfile.read_text())
        print(f"\n  Retry mode: loading existing results from {outfile}")
        for framing_name in framings_to_test:
            old_items = existing.get(framing_name, [])
            failed_ids = {r["id"] for r in old_items if r.get("correct") is None}
            ok_items = [r for r in old_items if r.get("correct") is not None]
            retry_qs = [q for q in questions if q["id"] in failed_ids]
            print(f"    {framing_name}: {len(ok_items)} ok, {len(failed_ids)} to retry")
            if not retry_qs:
                continue
            retried = run(model, retry_qs, [framing_name], concurrency=concurrency)
            merged = ok_items + retried.get(framing_name, [])
            existing[framing_name] = merged
        outfile.write_text(json.dumps(existing, indent=2, default=str))
        print(f"\n  Saved retried results to {outfile}")
    else:
        results = run(model, questions, framings_to_test, concurrency=concurrency)
        outfile.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n  Saved to {outfile}")


if __name__ == "__main__":
    main()
