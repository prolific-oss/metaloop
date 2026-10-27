"""
Wagering, v2 rerun (October 2026): question bank v2, shared model config.

Prompts are run.build_prompt (Conditions A and B) with two agreed changes:
"No reasoning, no explanation." becomes "Include no explanation." (reasoning is
now on wherever supported, so only the visible output is constrained), and the
format line's stale example ({"confidence": 75, "wager": "MEDIUM"}) becomes a
neutral placeholder. Same system message as the original. Items: the 451 with a
letter ground truth; 2 repeats of a fixed 30-item subset. Raw responses only.

    python tasks/t2_confidence_wagering/run_v2.py [model ...] [--pilot --limit N]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import llm  # noqa: E402
import v2run  # noqa: E402
import run  # noqa: E402

TASK = "t2_confidence_wagering"
SYSTEM = "You are a JSON-only responder. Output valid JSON and nothing else."
OLD = '\n\nYou MUST respond with ONLY a JSON object, nothing else. No reasoning, no explanation.\nFormat: {"answer": "your answer", "confidence": 75, "wager": "MEDIUM"}'
NEW = '\n\nYou MUST respond with ONLY a JSON object, nothing else. Include no explanation.\nFormat: {"answer": "<your answer>", "confidence": <integer 0-100>, "wager": "<SAFE|MEDIUM|BOLD>"}'


# camera-ready v1 arms (`v1_prompt` + `v1_format: placeholder`): the original prompt with only the format
# example replaced, as in the corrected neutral-format rerun (run_placeholder_sensitivity.py)
PH_OLD = '\nFormat: {"answer": "your answer", "confidence": 75, "wager": "MEDIUM"}'
PH_NEW = '\nFormat: {"answer": "<your answer>", "confidence": <integer 0-100>, "wager": "<SAFE|MEDIUM|BOLD>"}'


def v1_prompt(q, condition, spec):
    p = run.build_prompt(q, condition)
    if spec.get("v1_format") == "placeholder":
        assert p.count(PH_OLD) == 1, "original format example not found"
        p = p.replace(PH_OLD, PH_NEW)
    return p


def prompt(q, condition):
    p = run.build_prompt(q, condition)
    assert p.count(OLD) == 1, "original format line not found"
    return p.replace(OLD, NEW)


def main():
    a = v2run.cli(__doc__)
    bank = v2run.bank(version=a.bank)
    units = sorted(i for i in bank if i not in v2run.BAD_GT)
    jobs = v2run.jobs_with_repeats(TASK, units, conditions=tuple(a.conditions or ("A", "B")))
    for model in a.models:
        def work(unit, cond, rep, store, model=model):
            q = bank[unit]
            r = llm.call(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": (v1_prompt(q, cond, llm.spec(model)) if llm.spec(model).get("v1_prompt") else prompt(q, cond))}],
                         log_path=store.log, meta={"task": TASK, "unit": unit, "condition": cond, "repeat": rep})
            return {"source": q["source"], "text": r.text, **v2run.result_fields(r)}
        v2run.run(TASK, model, jobs, work, workers=a.workers, retry_errors=not a.no_retry_errors,
                  limit=a.limit, pilot=a.pilot, units=a.units)


if __name__ == "__main__":
    main()
