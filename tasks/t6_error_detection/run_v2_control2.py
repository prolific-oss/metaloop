"""
Self-Review Control 2, v2 rerun (October 2026): a fresh attempt at each item
the model got wrong in the v2 Self-Review Phase 1 (scored answers in
results_v2/phase1_scored/<model>.json), with the Phase-1 prompt and the shared
model config. Raw responses only.

    python tasks/t6_error_detection/run_v2_control2.py [model ...] [--pilot --limit N]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import llm  # noqa: E402
import v2run  # noqa: E402
from run import PHASE1_PROMPT, format_question  # noqa: E402

TASK = "t6_control2"
SCORED = HERE / "results_v2" / "phase1_scored"


def main():
    a = v2run.cli(__doc__)
    bank = v2run.bank(include_removed=True)
    for model in a.models:
        scored = json.loads((SCORED / f"{model}.json").read_text())
        wrong = sorted(i for i, s in scored.items() if s["answer"] is not None and not s["correct"])
        jobs = [(i, "fresh", 0) for i in wrong]

        def work(unit, cond, rep, store, model=model):
            q = bank[unit]
            r = llm.call(model, [{"role": "user", "content": PHASE1_PROMPT.format(question=format_question(q))}],
                         log_path=store.log, meta={"task": TASK, "unit": unit})
            return {"source": q["source"], "text": r.text, **v2run.result_fields(r)}
        v2run.run(TASK, model, jobs, work, workers=a.workers, retry_errors=not a.no_retry_errors, limit=a.limit, pilot=a.pilot, units=a.units)


if __name__ == "__main__":
    main()
