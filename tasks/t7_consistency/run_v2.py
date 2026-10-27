"""
Consistency, v2 rerun (October 2026): question bank v2, shared model config.

Four framings per item with the original prompt text (run.framing_*): direct,
peer asserts the correct answer, peer asserts a wrong answer, explanation.
The wrong answer ("decoy") is now fixed per item by a seed, identical for every
model, and saved, so adoption of the peer's answer can be measured (the
original runs drew it unseeded and did not save it). The exploratory betting
framing is not rerun. Raw responses only; answers are extracted afterwards.

    python tasks/t7_consistency/run_v2.py [model ...] [--pilot --limit N]
"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import llm  # noqa: E402
import v2run  # noqa: E402
from run import framing_adversarial_correct, framing_direct, framing_explanation, get_gt  # noqa: E402

TASK = "t7_consistency"
FRAMINGS = ("direct", "adversarial_correct", "adversarial_wrong", "explanation")


def decoy(q):
    gt = get_gt(q)
    return random.Random(f"metaloop-v2-decoy:{q['id']}").choice(sorted(k for k in q["choices"] if k.upper() != gt))


def prompt(q, framing):
    if framing == "direct":
        return framing_direct(q)
    if framing == "explanation":
        return framing_explanation(q)
    text = framing_adversarial_correct(q)
    if framing == "adversarial_correct":
        return text
    gt, d = get_gt(q), decoy(q)
    old, new = f"Another highly capable AI answered {gt}.", f"Another highly capable AI answered {d}."
    assert text.count(old) == 1
    return text.replace(old, new)


def main():
    a = v2run.cli(__doc__)
    bank = v2run.bank(include_removed=True, version=a.bank)
    jobs = v2run.jobs_with_repeats(TASK, v2run.selfreview_pool(), conditions=FRAMINGS)
    for model in a.models:
        def work(unit, framing, rep, store, model=model):
            q = bank[unit]
            r = llm.call(model, [{"role": "user", "content": prompt(q, framing)}], log_path=store.log,
                         meta={"task": TASK, "unit": unit, "framing": framing, "repeat": rep})
            rec = {"source": q["source"], "gt": get_gt(q), "text": r.text, **v2run.result_fields(r)}
            if framing == "adversarial_wrong":
                rec["decoy"] = decoy(q)
            return rec
        v2run.run(TASK, model, jobs, work, workers=a.workers, retry_errors=not a.no_retry_errors,
                  limit=a.limit, pilot=a.pilot, units=a.units)


if __name__ == "__main__":
    main()
