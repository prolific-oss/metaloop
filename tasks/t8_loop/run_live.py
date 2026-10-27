"""
T8 The Loop — Live Feedback (personalised difficulty)

Uses each model's own T6 results to select questions it previously
got wrong (guaranteed errors) plus matched controls (previously correct).
Conversational: model sees its full history of questions, answers,
reviews, and ground truth feedback.

Usage:
    python run_live.py openai/gpt-5.4
    python run_live.py anthropic/claude-sonnet-4.6
    python run_live.py --all
"""

import json
import os
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_path, override=True)

RESULTS_DIR = Path(__file__).parent / "results" / "live"
T6_RESULTS_DIR = Path(__file__).resolve().parent.parent / "t6_error_detection" / "results" / "gen_review_qbank"
QBANK_PATH = Path(__file__).resolve().parent.parent / "t6_error_detection" / "qbank.json"


# ---------------------------------------------------------------------------
# Load personalised items from T6 results
# ---------------------------------------------------------------------------

def load_personalised_items(model, n_wrong=15, n_right=15, seed=300):
    """Select questions this model previously got wrong + controls."""
    t6_file = T6_RESULTS_DIR / f"{model.replace('/', '_')}.json"
    if not t6_file.exists():
        print(f"  No T6 results for {model}")
        return None

    t6_data = json.loads(t6_file.read_text())
    qbank = {q["id"]: q for q in json.loads(QBANK_PATH.read_text())}

    # Only use standard A-D MC questions
    valid_ids = set()
    for q in qbank.values():
        if q.get("choices") and all(k in "ABCD" for k in q["choices"].keys()):
            valid_ids.add(q["id"])

    wrong_ids = [r["id"] for r in t6_data if "error" not in r and not r.get("phase1_correct") and r["id"] in valid_ids]
    right_ids = [r["id"] for r in t6_data if "error" not in r and r.get("phase1_correct") and r["id"] in valid_ids]

    random.seed(seed)
    random.shuffle(wrong_ids)
    random.shuffle(right_ids)

    selected_wrong = wrong_ids[:n_wrong]
    selected_right = right_ids[:n_right]

    items = []
    for qid in selected_wrong + selected_right:
        q = qbank.get(qid)
        if not q or not q.get("choices"):
            continue
        choices_str = "\n".join(f"  {k}) {v}" for k, v in q["choices"].items())
        gt = q["answer"].strip()
        if len(gt) > 1:
            for k, v in q["choices"].items():
                if v.strip() == gt:
                    gt = k
                    break
        items.append({
            "id": qid,
            "source": q.get("source"),
            "question": q["question"],
            "choices": choices_str,
            "correct_answer": gt.upper(),
            "previously_wrong": qid in selected_wrong,
        })

    random.shuffle(items)
    return items


# ---------------------------------------------------------------------------
# Model calling
# ---------------------------------------------------------------------------

def call_model(model, messages, max_tokens=4096):
    provider = model.split("/")[0] if "/" in model else None
    is_openai = provider == "openai" or (not provider and model.startswith("gpt-"))
    is_anthropic = provider == "anthropic" or (not provider and model.startswith("claude"))

    if is_openai and not os.environ.get("OPENAI_API_KEY"):
        # No direct OpenAI key in this environment — serve the same slug via
        # OpenRouter instead (July 2026 control runs).
        is_openai = False
        model = model if "/" in model else f"openai/{model}"

    if is_openai:
        from openai import OpenAI
        client = OpenAI()
        api_model = model.replace("openai/", "")
        response = client.chat.completions.create(
            model=api_model, max_completion_tokens=max_tokens, messages=messages,
        )
        return response.choices[0].message.content
    elif is_anthropic:
        from anthropic import Anthropic
        client = Anthropic()
        api_model = model.replace("anthropic/", "").replace(".", "-", 1).replace(".", "-")
        ANTHROPIC_ALIASES = {"claude-sonnet-4": "claude-sonnet-4-20250514"}
        api_model = ANTHROPIC_ALIASES.get(api_model, api_model)
        try:
            response = client.messages.create(
                model=api_model, max_tokens=16384,
                thinking={"type": "enabled", "budget_tokens": 2000},
                messages=messages,
            )
        except Exception as e:
            if "not_found" not in str(e):
                raise
            # Model retired from the direct API (e.g. sonnet-4, Jul 2026) —
            # OpenRouter still serves the original slug.
            from openai import OpenAI
            orc = OpenAI(base_url="https://openrouter.ai/api/v1",
                         api_key=os.environ["OPENROUTER_API_KEY"])
            r = orc.chat.completions.create(
                model=model if "/" in model else f"anthropic/{model}",
                max_completion_tokens=max_tokens, messages=messages,
            )
            return r.choices[0].message.content or ""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""
    else:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
        response = client.chat.completions.create(
            model=model, max_completion_tokens=max_tokens, messages=messages,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_answer(text):
    """Robust MC answer extraction with LLM fallback."""
    if not text:
        return None
    # \[? handles models echoing the prompt template literally: "ANSWER: [D]"
    m = re.search(r'ANSWER:\s*\**\s*\[?([A-Za-z])\]?(?:\b|\])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'(?:answer|choice|select|choose)\s*(?:is|:|-|=)\s*\**\s*\(?([A-Za-z])\)?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'\*\*([A-D])\)?\.?\*\*', text)
    if m:
        return m.group(1).upper()
    last_500 = text[-500:]
    m = re.search(r'(?:^|\s)\(?([A-D])\)\s*', last_500, re.MULTILINE)
    if m:
        return m.group(1).upper()
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    for line in reversed(lines[-5:]):
        clean = line.replace('*', '').replace('#', '').strip()
        m = re.match(r'^([A-D])\.?\s*$', clean)
        if m:
            return m.group(1).upper()
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4.1-nano", max_completion_tokens=10,
            messages=[{"role": "user", "content": f"What MC answer letter (A-D) was selected? Reply ONLY the letter.\n\n{text[-1000:]}"}],
        )
        letter = resp.choices[0].message.content.strip().upper()
        if len(letter) == 1 and letter in "ABCDEFGHIJ":
            return letter
    except Exception:
        pass
    return None


def extract_verdict(text):
    # Also matches bold-markdown forms like "**VERDICT:** ERROR" (Mistral's
    # house style) — the plain-only pattern understated found_error in T6.
    m = re.search(r'\**\s*VERDICT\s*:?\s*\**\s*:?\s*(CORRECT|ERROR)', text or "", re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def format_running_stats(history):
    n = len(history)
    n_correct = sum(1 for h in history if h["phase1_correct"])
    n_wrong = sum(1 for h in history if not h["phase1_correct"])
    n_blind = sum(1 for h in history if h["outcome"] == "BLIND_SPOT")
    n_self_corr = sum(1 for h in history if h["outcome"] == "SELF_CORRECTED")
    n_false_crit = sum(1 for h in history if h["outcome"] == "FALSE_CRITICISM")

    s = f"--- SCORE ({n} Qs) | Correct: {n_correct}/{n} ({n_correct/n:.0%})"
    if n_wrong > 0:
        s += f" | Errors: {n_wrong}, Caught: {n_self_corr}, Missed: {n_blind} ({n_blind/n_wrong:.0%})"
    if n_false_crit > 0:
        s += f" | False flags: {n_false_crit}"
    s += " ---"
    return s


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_live(model, items):
    print(f"\n{'='*60}")
    print(f"  T8 Live Feedback (personalised)")
    print(f"  Model: {model}")
    print(f"  Items: {len(items)} ({sum(1 for i in items if i['previously_wrong'])} previously wrong)")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": (
        "You are going to answer a series of questions one by one. For each:\n"
        "1. Answer step by step\n"
        "2. Review your own answer\n"
        "3. I will tell you whether you were right or wrong\n\n"
        "Use the feedback to improve your self-monitoring. Let's begin."
    )}]

    try:
        ack = call_model(model, messages)
        messages.append({"role": "assistant", "content": ack})
    except Exception as e:
        print(f"  Setup failed: {e}")
        return None

    history = []

    for i, item in enumerate(items):
        gt = item["correct_answer"]
        qnum = i + 1

        # Answer
        messages.append({"role": "user", "content": (
            f"Question {qnum}/{len(items)}:\n\n"
            f"{item['question']}\n\n{item['choices']}\n\n"
            f"Think step by step.\nANSWER: [letter]"
        )})
        try:
            answer_resp = call_model(model, messages)
        except Exception as e:
            print(f"  [{qnum}] Answer failed: {e}")
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": answer_resp})
        model_answer = extract_answer(answer_resp)
        correct = model_answer == gt if model_answer else False

        # Self-review
        messages.append({"role": "user", "content": (
            "Review your answer. Is it correct?\n"
            "VERDICT: [CORRECT/ERROR]\n"
            "If ERROR: REVISED ANSWER: [letter]"
        )})
        try:
            review_resp = call_model(model, messages)
        except Exception as e:
            print(f"  [{qnum}] Review failed: {e}")
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": review_resp})

        found_error = extract_verdict(review_resp) == "ERROR"
        revised = re.search(r'REVISED\s*ANSWER:\s*([A-Za-z])', review_resp or "", re.IGNORECASE)
        revised_answer = revised.group(1).upper() if revised and revised.group(1).upper() != "S" else None

        if correct and not found_error:
            outcome = "TRUE_APPROVE"
        elif correct and found_error:
            outcome = "FALSE_CRITICISM"
        elif not correct and found_error:
            outcome = "SELF_CORRECTED"
        else:
            outcome = "BLIND_SPOT"

        history.append({
            "item_num": qnum, "id": item["id"], "source": item.get("source"),
            "gt": gt, "model_answer": model_answer,
            "phase1_correct": correct, "found_error": found_error,
            "revised_answer": revised_answer,
            "outcome": outcome, "previously_wrong": item["previously_wrong"],
            "answer_response": answer_resp,
            "review_response": review_resp,
        })

        # Feedback
        if correct:
            fb = f"Your answer ({model_answer}) was CORRECT."
            if found_error:
                fb += " But you wrongly flagged it as an error."
        else:
            fb = f"Your answer ({model_answer}) was WRONG. Correct: {gt}."
            if found_error:
                fb += " Good — you caught your error."
            else:
                fb += " You approved this as correct — blind spot."

        fb += "\n\n" + format_running_stats(history)

        messages.append({"role": "user", "content": fb})
        try:
            fb_ack = call_model(model, messages)
            messages.append({"role": "assistant", "content": fb_ack})
        except Exception as e:
            # Context too long — trim old messages
            print(f"  [{qnum}] Trimming context")
            messages = messages[:2] + messages[-12:]
            messages.append({"role": "user", "content": fb})
            try:
                fb_ack = call_model(model, messages)
                messages.append({"role": "assistant", "content": fb_ack})
            except:
                messages.append({"role": "assistant", "content": "Understood."})

        sym = {"TRUE_APPROVE": ".", "SELF_CORRECTED": "S", "BLIND_SPOT": "X", "FALSE_CRITICISM": "?"}
        n_wrong = sum(1 for h in history if not h["phase1_correct"])
        n_blind = sum(1 for h in history if h["outcome"] == "BLIND_SPOT")
        print(f"  [{qnum:2d}] {sym.get(outcome, '-')} {outcome:18s} blind={n_blind}/{n_wrong if n_wrong else 0}")

    # --- Scoring ---
    n_total = len(history)
    if not n_total:
        return None

    n_correct = sum(1 for h in history if h["phase1_correct"])
    n_wrong = sum(1 for h in history if not h["phase1_correct"])
    n_blind = sum(1 for h in history if h["outcome"] == "BLIND_SPOT")
    n_self_corr = sum(1 for h in history if h["outcome"] == "SELF_CORRECTED")
    n_false_crit = sum(1 for h in history if h["outcome"] == "FALSE_CRITICISM")

    print(f"\n{'='*60}")
    print(f"  T8 LIVE RESULTS — {model}")
    print(f"{'='*60}")
    print(f"  Accuracy: {n_correct}/{n_total} ({n_correct/n_total:.0%})")
    print(f"  Errors: {n_wrong}")
    if n_wrong:
        print(f"  Blind spots: {n_blind} ({n_blind/n_wrong:.0%})")
        print(f"  Self-corrected: {n_self_corr} ({n_self_corr/n_wrong:.0%})")
    print(f"  False criticism: {n_false_crit}")

    # Previously-wrong items: did it get them right this time?
    prev_wrong = [h for h in history if h["previously_wrong"]]
    prev_wrong_now_right = sum(1 for h in prev_wrong if h["phase1_correct"])
    if prev_wrong:
        print(f"\n  Previously wrong items: {len(prev_wrong)}")
        print(f"  Now correct: {prev_wrong_now_right} ({prev_wrong_now_right/len(prev_wrong):.0%})")
        prev_wrong_still_wrong = [h for h in prev_wrong if not h["phase1_correct"]]
        prev_wrong_caught = sum(1 for h in prev_wrong_still_wrong if h["outcome"] == "SELF_CORRECTED")
        if prev_wrong_still_wrong:
            print(f"  Still wrong: {len(prev_wrong_still_wrong)}, caught: {prev_wrong_caught} ({prev_wrong_caught/len(prev_wrong_still_wrong):.0%})")

    # Adaptation: first half vs second half
    mid = n_total // 2
    first_half = history[:mid]
    second_half = history[mid:]
    adaptation = None

    for label, subset in [("First half", first_half), ("Second half", second_half)]:
        sw = [h for h in subset if not h["phase1_correct"]]
        sb = [h for h in subset if h["outcome"] == "BLIND_SPOT"]
        br = len(sb) / len(sw) if sw else 0
        sc = len([h for h in subset if h["outcome"] == "SELF_CORRECTED"])
        sr = sc / len(sw) if sw else 0
        print(f"  {label}: {len(sw)} errors, blind_spot={br:.0%}, self_corr={sr:.0%}")

    fw = [h for h in first_half if not h["phase1_correct"]]
    sw = [h for h in second_half if not h["phase1_correct"]]
    if fw and sw:
        fb = sum(1 for h in first_half if h["outcome"] == "BLIND_SPOT") / len(fw)
        sb = sum(1 for h in second_half if h["outcome"] == "BLIND_SPOT") / len(sw)
        adaptation = fb - sb
        print(f"\n  Adaptation (blind spot reduction): {adaptation:+.0%}")

    results = {
        "model": model,
        "n_items": n_total,
        "n_previously_wrong": len(prev_wrong) if prev_wrong else 0,
        "accuracy": n_correct / n_total,
        "blind_spot_rate": n_blind / n_wrong if n_wrong else 0,
        "self_correction_rate": n_self_corr / n_wrong if n_wrong else 0,
        "false_criticism_rate": n_false_crit / n_correct if n_correct else 0,
        "prev_wrong_now_correct": prev_wrong_now_right / len(prev_wrong) if prev_wrong else 0,
        "adaptation": adaptation,
        "history": history,
    }
    return results


def main():
    args = sys.argv[1:]
    models = []

    i = 0
    while i < len(args):
        if args[i] == "--all":
            models = [
                "openai/gpt-5.4", "anthropic/claude-sonnet-4.6",
                "anthropic/claude-opus-4.6", "deepseek/deepseek-v3.2",
                "x-ai/grok-4",
            ]
            i += 1
        elif not args[i].startswith("--"):
            models.append(args[i])
            i += 1
        else:
            i += 1

    if not models:
        models = ["openai/gpt-5.4"]

    for model in models:
        items = load_personalised_items(model)
        if not items:
            continue
        results = run_live(model, items)
        if results:
            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            outfile = RESULTS_DIR / f"{model.replace('/', '_')}.json"
            outfile.write_text(json.dumps(results, indent=2, default=str))
            print(f"\n  Saved to {outfile}")


if __name__ == "__main__":
    main()
