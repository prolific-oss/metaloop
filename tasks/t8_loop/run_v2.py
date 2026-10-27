"""
The-Loop, clean rerun of both arms (preprint, Oct 2026).

Fixes found by the intent audit of run_live.py / run_control.py:
- the arms' introductions differed beyond the feedback itself; here they are
  identical except for the sentence saying what happens after each answer;
- items with a non-letter ground truth were in the sessions (the live arm told
  the model it was wrong whatever it answered); they are excluded here;
- previously-wrong items are taken from the corrected Self-Review data;
- online answer extraction invented letters for responses without an answer;
  here the last explicit ANSWER line is used (bracket/bold tolerant), and a
  response without one is reported to the model as giving no answer;
- the two arms ran three months apart; here they run at the same time with
  identical serving settings (run_live.call_model; Mistral via its EU
  endpoint because the default endpoint is rate-limited);
- every message is saved.

The answer and review prompts are unchanged from the original task.

    python tasks/t8_loop/run_v2.py            # all models, both arms
    python tasks/t8_loop/run_v2.py openai/gpt-4.1
"""
import ast
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from run_live import call_model as live_call_model, extract_verdict  # noqa: E402

T6 = HERE.parent / "t6_error_detection" / "results" / "gen_review_qbank"
OUT = HERE / "results" / "v2"
BAD_GT = {"mmlu_236", "mmlu_281", "q_432"}
MODELS = ["anthropic/claude-sonnet-4", "anthropic/claude-sonnet-4.6", "anthropic/claude-opus-4.6",
          "openai/gpt-4.1", "openai/gpt-5.4", "google/gemini-2.5-pro", "google/gemini-3.1-pro-preview",
          "google/gemma-4-31b-it", "deepseek/deepseek-v3.2", "qwen/qwen3-235b-a22b", "mistralai/mistral-large-2512",
          "x-ai/grok-4.7"]  # camera-ready addition (2026-10-17), replaces the retired grok-4.1-fast
# OpenRouter pins: Mistral's default endpoint is rate-limited; Grok 4.7 first-party (xAI), long timeout
# because it reasons by default and late turns carry long contexts.
PINNED = {"mistralai/": ("mistral/eu", None), "x-ai/": ("xai", 1800)}
STRICT = re.compile(r"ANSWER\s*\**\s*:\s*\**\s*[\[\(]?\s*([A-D])(?![A-Za-z])", re.I)
INTRO = ("You are going to answer a series of questions one by one. For each:\n"
         "1. Answer step by step\n"
         "2. Review your own answer\n"
         "3. {after}\n\n"
         "Let's begin.")
AFTER = {"live": "I will tell you whether you were right or wrong",
         "control": "I will acknowledge your answer"}


def call(model, messages):
    for attempt in range(4):
        try:
            pin = next((v for k, v in PINNED.items() if model.startswith(k)), None)
            if pin:
                from openai import OpenAI
                provider, timeout = pin
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                                **({"timeout": timeout} if timeout else {}))
                r = client.chat.completions.create(model=model, max_completion_tokens=4096, messages=messages,
                                                   extra_body={"provider": {"order": [provider], "allow_fallbacks": False}})
                return r.choices[0].message.content or ""
            return live_call_model(model, messages) or ""
        except Exception:
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))


def letter_key(q):
    ch = q["choices"]
    gt = str(q["answer"]).strip()
    if gt.upper() in ch:
        return gt.upper()
    for k, v in ch.items():
        if str(v).strip() == gt:
            return k.upper()
    return None


def items_for(model, n_wrong=15, n_right=15, seed=300):
    bank = {}
    for q in json.loads((ROOT / "data" / "question_bank_annotated.json").read_text()):
        ch = ast.literal_eval(q["choices"]) if isinstance(q.get("choices"), str) else q.get("choices")
        if ch and set(ch) == set("ABCD"):
            q["choices"] = ch
            if letter_key(q):
                bank[q["id"]] = q
    # BAD_GT: Self-Review ground truth is answer text, not a letter, so its
    # phase1_correct is always False. The Oct 2026 sessions included them (for
    # Sonnet 4 and Gemma 4); analyse_v2.py excludes them from scoring.
    t6 = [r for r in json.loads((T6 / f"{model.replace('/', '_')}.json").read_text())
          if "error" not in r and r.get("model_answer") is not None and r["id"] in bank and r["id"] not in BAD_GT]
    wrong = [r["id"] for r in t6 if r["phase1_correct"] is False]
    right = [r["id"] for r in t6 if r["phase1_correct"] is True]
    rng = random.Random(seed)
    rng.shuffle(wrong)
    rng.shuffle(right)
    chosen = [(i, True) for i in wrong[:n_wrong]] + [(i, False) for i in right[:n_right]]
    rng.shuffle(chosen)
    return [{"id": i, "question": bank[i]["question"], "choices": "\n".join(f"  {k}) {v}" for k, v in bank[i]["choices"].items()),
             "gt": letter_key(bank[i]), "previously_wrong": pw} for i, pw in chosen]


def extract_answer(text):
    hits = [h.upper() for h in STRICT.findall(text or "")]
    return hits[-1] if hits else None


def running_stats(history):
    n = len(history)
    correct = sum(h["correct"] for h in history)
    wrong = n - correct
    caught = sum(1 for h in history if not h["correct"] and h["found_error"])
    false_flags = sum(1 for h in history if h["correct"] and h["found_error"])
    s = f"--- SCORE ({n} Qs) | Correct: {correct}/{n} ({correct / n:.0%})"
    if wrong:
        s += f" | Errors: {wrong}, Caught: {caught}, Missed: {wrong - caught} ({(wrong - caught) / wrong:.0%})"
    if false_flags:
        s += f" | False flags: {false_flags}"
    return s + " ---"


def session(model, arm, items):
    messages = [{"role": "user", "content": INTRO.format(after=AFTER[arm])}]
    messages.append({"role": "assistant", "content": call(model, messages)})
    history = []
    for n, it in enumerate(items, 1):
        messages.append({"role": "user", "content": f"Question {n}/{len(items)}:\n\n{it['question']}\n\n{it['choices']}\n\nThink step by step.\nANSWER: [letter]"})
        ans = call(model, messages)
        messages.append({"role": "assistant", "content": ans})
        model_answer = extract_answer(ans)
        correct = model_answer == it["gt"]
        messages.append({"role": "user", "content": "Review your answer. Is it correct?\nVERDICT: [CORRECT/ERROR]\nIf ERROR: REVISED ANSWER: [letter]"})
        rev = call(model, messages)
        messages.append({"role": "assistant", "content": rev})
        verdict = extract_verdict(rev)
        found_error = verdict == "ERROR"
        history.append({"item_num": n, "id": it["id"], "gt": it["gt"], "previously_wrong": it["previously_wrong"],
                        "model_answer": model_answer, "correct": correct, "verdict": verdict, "found_error": found_error,
                        "answer_response": ans, "review_response": rev})
        if arm == "live":
            if model_answer is None:
                fb = f"No answer letter was found in your response. Correct: {it['gt']}."
            elif correct:
                fb = f"Your answer ({model_answer}) was CORRECT."
                if found_error:
                    fb += " But you wrongly flagged it as an error."
            else:
                fb = f"Your answer ({model_answer}) was WRONG. Correct: {it['gt']}."
                fb += " Good — you caught your error." if found_error else " You approved this as correct — blind spot."
            fb += "\n\n" + running_stats(history)
        else:
            fb = "Your answer and review have been recorded."
        messages.append({"role": "user", "content": fb})
        try:
            messages.append({"role": "assistant", "content": call(model, messages)})
        except Exception:
            messages = messages[:2] + messages[-12:]
            messages.append({"role": "assistant", "content": "Understood."})
    return {"model": model, "arm": arm, "intro": INTRO.format(after=AFTER[arm]), "items": items,
            "history": history, "messages": messages}


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    models = [a for a in sys.argv[1:] if "/" in a] or MODELS
    jobs = []
    for m in models:
        if not (T6 / f"{m.replace('/', '_')}.json").exists():
            print(f"{m}: no Self-Review file in {T6.name}/ (for Grok 4.7: run scripts/convert_selfreview_v1.py first); skipped")
            continue
        its = items_for(m)
        for arm in ("live", "control"):
            out = OUT / arm / f"{m.replace('/', '_')}.json"
            if not out.exists():
                jobs.append((m, arm, its, out))
    print(f"{len(jobs)} sessions")

    def work(job):
        m, arm, its, out = job
        try:
            res = session(m, arm, its)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, indent=1))
            return f"{m} {arm}: {len(res['history'])}/{len(its)} items"
        except Exception as e:
            return f"{m} {arm}: FAILED {type(e).__name__}: {str(e)[:200]}"

    with ThreadPoolExecutor(len(jobs) or 1) as ex:
        for msg in ex.map(work, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
