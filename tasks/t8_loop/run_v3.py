"""
The-Loop, v3 rerun (October 2026): question bank v2, shared model config.

Same session design and prompts as run_v2.py (identical introductions apart
from the sentence on what follows each answer; live feedback vs neutral
acknowledgement; single-verdict review), with these changes:
- previously-wrong items come from the v2 Self-Review run (scored Phase-1
  answers in tasks/t6_error_detection/results_v2/phase1_scored/<model>.json),
  four-option items only, the three non-letter items excluded;
- a failed call ends the session with status "error" and the whole session is
  rerun (run_v2 truncated the history and inserted a fixed reply);
- each arm is run three times (main run + 2 repeats) on the same item set.

    python tasks/t8_loop/run_v3.py [model ...] [--pilot --limit N]
"""
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import llm  # noqa: E402
import v2run  # noqa: E402
from run_v2 import AFTER, INTRO, running_stats  # noqa: E402

TASK = "t8_loop"

# Online parsers decide the live feedback, so they tolerate markdown: bold
# markers and line breaks between a label and its value, and a "Final Answer"
# heading instead of an ANSWER line. Feedback accuracy is still audited
# afterwards by the three-judge re-extraction.
# The option letter must be upper case: with a case-insensitive letter, prose such
# as "to answer a question" was read as ANSWER: A (one control-arm item in
# Gemini 2.5 Pro's sessions; fixed 2026-10-12 before the remaining models ran).
_ANS = re.compile(r"ANSWER\W{0,12}?\b((?-i:[A-D]))\b(?![A-Za-z])", re.I)
_FINAL = re.compile(r"final\s+answer\W{0,20}?\(?\b((?-i:[A-D]))\b(?:\)|\*|\.|\s|$)", re.I)
_VERDICT = re.compile(r"VERDICT\W{0,20}?\b(CORRECT|ERROR)\b", re.I)


def extract_answer(text):
    hits = _ANS.findall(text or "") or _FINAL.findall(text or "")
    return hits[-1].upper() if hits else None


def extract_verdict(text):
    hits = _VERDICT.findall(text or "")
    return hits[-1].upper() if hits else None
SCORED = HERE.parent / "t6_error_detection" / "results_v2" / "phase1_scored"


def items_for(model, n_wrong=15, n_right=15, seed=300):
    bank = v2run.bank(include_removed=True)
    scored = json.loads((SCORED / f"{llm.spec(model).get('items_from', model)}.json").read_text())   # {id: {"answer": letter|None, "correct": bool}}
    ok = [i for i in sorted(scored) if i in bank and i not in v2run.BAD_GT and set(bank[i]["choices"]) == set("ABCD")
          and scored[i]["answer"] is not None]
    wrong = [i for i in ok if not scored[i]["correct"]]
    right = [i for i in ok if scored[i]["correct"]]
    rng = random.Random(seed)
    rng.shuffle(wrong)
    rng.shuffle(right)
    chosen = [(i, True) for i in wrong[:n_wrong]] + [(i, False) for i in right[:n_right]]
    rng.shuffle(chosen)
    out = []
    for i, pw in chosen:
        q = bank[i]
        gt = [k for k, v in q["choices"].items() if str(v).strip() == str(q["answer"]).strip()]
        out.append({"id": i, "question": q["question"], "choices": "\n".join(f"  {k}) {v}" for k, v in q["choices"].items()),
                    "gt": gt[0] if gt else str(q["answer"]).upper(), "previously_wrong": pw})
    return out


class SessionError(Exception):
    pass


def session(model, arm, items, store, rep):
    def call(messages, step):
        r = llm.call(model, messages, log_path=store.log, meta={"task": TASK, "arm": arm, "repeat": rep, "step": step})
        if r.status == "truncated" and not r.text:
            # Cut off at the output limit before any visible text (DeepSeek V3.2 on one GPQA item):
            # model behaviour, so the session goes on. The turn is kept empty rather than filled
            # with text the model did not write; it scores as no answer (wrong), and the live arm
            # gets the usual "No answer letter was found" feedback.
            calls.append({"step": step, **v2run.result_fields(r)})
            return ""
        if r.status == "error" or r.text is None:
            raise SessionError(f"{step}: {r.status} {r.error or r.finish_reason}")
        calls.append({"step": step, **v2run.result_fields(r)})
        return r.text

    calls = []
    messages = [{"role": "user", "content": INTRO.format(after=AFTER[arm])}]
    messages.append({"role": "assistant", "content": call(messages, "intro")})
    history = []
    for n, it in enumerate(items, 1):
        messages.append({"role": "user", "content": f"Question {n}/{len(items)}:\n\n{it['question']}\n\n{it['choices']}\n\nThink step by step.\nANSWER: [letter]"})
        ans = call(messages, f"q{n}")
        messages.append({"role": "assistant", "content": ans})
        model_answer = extract_answer(ans)
        correct = model_answer == it["gt"]
        messages.append({"role": "user", "content": "Review your answer. Is it correct?\nVERDICT: [CORRECT/ERROR]\nIf ERROR: REVISED ANSWER: [letter]"})
        rev = call(messages, f"r{n}")
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
        messages.append({"role": "assistant", "content": call(messages, f"ack{n}")})
    status = "truncated" if any(c["status"] == "truncated" for c in calls) else "ok"
    return {"items": items, "history": history, "messages": messages, "calls": calls, "status": status}


def main():
    a = v2run.cli(__doc__)
    for model in a.models:
        items = items_for(model)
        jobs = [(arm, "session", rep) for rep in (0, *v2run.REPEATS) for arm in ("live", "control")]

        def work(arm, cond, rep, store, model=model, items=items):
            try:
                return session(model, arm, items, store, rep)
            except SessionError as e:
                return {"status": "error", "error": str(e), "items": items}
        v2run.run(TASK, model, jobs, work, workers=6, retry_errors=not a.no_retry_errors, limit=a.limit, pilot=a.pilot, units=a.units)


if __name__ == "__main__":
    main()
