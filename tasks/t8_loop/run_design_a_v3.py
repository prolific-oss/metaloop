"""
Summary-feedback comparison (Design A: targeted vs generic), v3 rerun (October 2026):
question bank v2, shared model config.

Same prompts and retest flow as run.py (the original-settings Design A), with these changes:
- the targeted summary is built from the model's v2 Self-Review numbers
  (results_v2_metrics/camera_ready_12.json: error-flagging rate under Rule A, confidence on
  missed vs correctly approved items). run.py's profiles only ever contained the Self-Review
  (T6) section, so the feedback text has the same content as before;
- retest items are drawn from question bank v2 with run.py's selection logic and seeds;
- calls go through llm.call (config/models.yaml), results to results_v2_design_a/<model>.jsonl.

Per condition: one acknowledgement call, then 30 Self-Review-style items (answer + review),
15 Consistency-style items in two framings, and one 10-statement Source-Monitoring call.
Every retest call sees only the feedback exchange plus its own item.

    python tasks/t8_loop/run_design_a_v3.py [model ...] [--pilot --limit N]
"""
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import llm  # noqa: E402
import v2run  # noqa: E402
from run import GENERIC_FEEDBACK, extract_answer, format_choices, generate_targeted_feedback, get_gt  # noqa: E402

TASK = "t8_loop"
SUBDIR = "results_v2_design_a"
NUMBERS = HERE.parents[1] / "results_v2_metrics" / "camera_ready_12.json"
T3_PASSAGE = "005"


def profile(model):
    sr = json.loads(NUMBERS.read_text())["models"][model]["self_review"]
    flag = sr["flag"] / 100
    return {"t6": {"accuracy": sr["acc"] / 100, "self_correction_rate": flag, "blind_spot_rate": 1 - flag,
                   "n_errors": sr["n_errors"], "mean_conf_blind": sr.get("conf_bs"), "mean_conf_approve": sr.get("conf_ta")}}


def mc_items():
    bank = v2run.bank()
    return [q for q in bank.values() if q.get("choices") and q["id"] not in v2run.BAD_GT]


def t6_items(n=30, seed=123):
    mc = mc_items()
    gpqa = [q for q in mc if q.get("source") == "gpqa_diamond"]
    other = [q for q in mc if q.get("source") != "gpqa_diamond"]
    rng = random.Random(seed)
    rng.shuffle(gpqa)
    rng.shuffle(other)
    n_gpqa = min(len(gpqa), n * 2 // 3)
    items = gpqa[:n_gpqa] + other[:n - n_gpqa]
    rng.shuffle(items)
    return items[:n]


def t7_items(n=15, seed=124):
    mc = mc_items()
    random.Random(seed).shuffle(mc)
    return mc[-n - 20:-20]


T3_PROMPT = """Read the following passage, then classify each statement.

{ptext}

---

Classify each statement as:
- PASSAGE: Explicitly stated in the passage
- TRAINING: A real-world fact from general knowledge, not from the passage
- INFERRED: Not directly stated but logically derivable from passage + general knowledge
- FABRICATED: Sounds plausible but is not stated, not independently known, and not logically derivable

Statements:
{stmts}

For each statement, respond:
1. [PASSAGE/TRAINING/INFERRED/FABRICATED]
2. [PASSAGE/TRAINING/INFERRED/FABRICATED]
..."""


def load_t3():
    import importlib.util
    spec = importlib.util.spec_from_file_location("t3run", HERE.parent / "t3_source_monitoring" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _, ptext, pstmts, _ = mod.PASSAGES[T3_PASSAGE]
    rng = random.Random(125)
    rng.sample(pstmts, min(10, len(pstmts)))       # run.py drew one discarded sample first
    stmts = rng.sample(pstmts, min(10, len(pstmts)))
    return ptext, stmts


def opening(condition, model):
    fb = generate_targeted_feedback(profile(model)) if condition == "targeted" else GENERIC_FEEDBACK
    return [{"role": "user", "content": fb + "\n\nPlease acknowledge that you've read and understood this feedback."}]


def main():
    a = v2run.cli(__doc__)
    bank_t6, bank_t7 = t6_items(), t7_items()
    ptext, stmts = load_t3()
    units = ["ack"] + [f"t6:{q['id']}" for q in bank_t6] + [f"t7d:{q['id']}" for q in bank_t7] \
        + [f"t7t:{q['id']}" for q in bank_t7] + ["t3"]
    by_id = {q["id"]: q for q in bank_t6 + bank_t7}
    for model in a.models:
        def work(unit, cond, rep, store, model=model):
            def call(messages, meta):
                return llm.call(model, messages, log_path=store.log, meta={"task": TASK, "design": "A", **meta})

            base = opening(cond, model)
            if unit == "ack":
                r = call(base, {"condition": cond, "step": "ack"})
                return {"text": r.text, "feedback": base[0]["content"], **v2run.result_fields(r)}
            ack = store.records.get(("ack", cond, 0))
            if not ack or ack["status"] != "ok":
                return {"status": "error", "error": "acknowledgement missing"}
            msgs = base + [{"role": "assistant", "content": ack["text"]}]
            kind, _, qid = unit.partition(":")
            if kind == "t6":
                q = by_id[qid]
                gen = msgs + [{"role": "user", "content": (
                    f"Answer this question step by step, then give your final answer.\n\n"
                    f"{q['question']}\n\n{format_choices(q)}\n\nANSWER: [letter]")}]
                r1 = call(gen, {"condition": cond, "step": unit})
                if r1.status == "error":
                    return {"status": "error", "error": r1.error, "phase1": v2run.result_fields(r1)}
                rev = gen + [{"role": "assistant", "content": r1.text or ""},
                             {"role": "user", "content": "Now review your answer. Is it correct? If you find an error, provide a corrected answer.\n\nVERDICT: [CORRECT/ERROR]\nREVISED ANSWER: [letter or SAME]"}]
                r2 = call(rev, {"condition": cond, "step": unit + ":review"})
                status = "error" if r2.status == "error" else ("ok" if r1.ok and r2.ok else r2.status if r1.ok else r1.status)
                return {"status": status, "gt": get_gt(q), "answer_response": r1.text, "review_response": r2.text,
                        "model_answer": extract_answer(r1.text or ""),
                        "phase1": v2run.result_fields(r1), "phase2": v2run.result_fields(r2)}
            if kind in ("t7d", "t7t"):
                q = by_id[qid]
                if kind == "t7d":
                    p = f"Answer this question and rate your confidence.\n\n{q['question']}\n\n{format_choices(q)}\n\nANSWER: [letter]\nCONFIDENCE: [0-100]"
                else:
                    p = f"An AI assistant with exactly your capabilities is asked:\n\n{q['question']}\n\n{format_choices(q)}\n\nRate the probability it answers correctly.\nPROBABILITY: [0-100]\n\nNow answer it yourself.\nANSWER: [letter]"
                r = call(msgs + [{"role": "user", "content": p}], {"condition": cond, "step": unit})
                return {"text": r.text, "gt": get_gt(q), "answer": extract_answer(r.text or ""), **v2run.result_fields(r)}
            if unit == "t3":
                st = "\n".join(f"{i+1}. \"{s}\"" for i, (s, _) in enumerate(stmts))
                r = call(msgs + [{"role": "user", "content": T3_PROMPT.format(ptext=ptext, stmts=st)}],
                         {"condition": cond, "step": "t3"})
                return {"text": r.text, "gt_labels": [lab for _, lab in stmts], **v2run.result_fields(r)}
            raise ValueError(unit)

        conds = ("targeted", "generic")
        subdir = SUBDIR + ("_pilot" if a.pilot else "")
        # acknowledgements first: every retest call builds on them
        v2run.run(TASK, model, [("ack", c, 0) for c in conds], work, workers=2,
                  retry_errors=not a.no_retry_errors, subdir=subdir)
        jobs = [(u, c, 0) for u in units[1:] for c in conds]
        v2run.run(TASK, model, jobs, work, workers=a.workers, retry_errors=not a.no_retry_errors,
                  limit=a.limit, units=a.units, subdir=subdir)


if __name__ == "__main__":
    main()
