"""
Self-Review, v2 rerun (October 2026): question bank v2, shared model config.

Phase 1 (answer step by step) and Phase 2 (review own steps) with the original
prompts (run.PHASE1_PROMPT, run.PHASE2_PROMPT, run.format_question). Raw
responses only; answers, verdicts and revised answers are extracted afterwards
by the corrected pipeline. Pool: the 153 letter-ground-truth items (incl.
q_451), plus 2 repeats of a fixed 30-item subset.

    python tasks/t6_error_detection/run_v2.py [model ...] [--pilot --limit N]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import llm  # noqa: E402
import v2run  # noqa: E402
from run import PHASE1_PROMPT, PHASE2_PROMPT, format_question  # noqa: E402

TASK = "t6_error_detection"


def main():
    a = v2run.cli(__doc__)
    bank = v2run.bank(include_removed=True, version=a.bank)
    pool = v2run.selfreview_pool()
    jobs = v2run.jobs_with_repeats(TASK, pool)

    for model in a.models:
        def work(unit, cond, rep, store, model=model):
            q = bank[unit]
            meta = {"task": TASK, "unit": unit, "repeat": rep}
            p1 = [{"role": "user", "content": PHASE1_PROMPT.format(question=format_question(q))}]
            r1 = llm.call(model, p1, log_path=store.log, meta={**meta, "phase": 1})
            rec = {"source": q["source"], "phase1": {"text": r1.text, **v2run.result_fields(r1)}}
            if not r1.ok:
                return {**rec, "status": r1.status}
            p2 = p1 + [{"role": "assistant", "content": r1.text}, {"role": "user", "content": PHASE2_PROMPT}]
            r2 = llm.call(model, p2, log_path=store.log, meta={**meta, "phase": 2})
            return {**rec, "phase2": {"text": r2.text, **v2run.result_fields(r2)}, "status": r2.status}
        v2run.run(TASK, model, jobs, work, workers=a.workers, retry_errors=not a.no_retry_errors,
                  limit=a.limit, pilot=a.pilot, units=a.units)


if __name__ == "__main__":
    main()
