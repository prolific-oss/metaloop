"""
Strategy-Switching, v2 rerun (October 2026): shared model config.

The eight task modules (tasks/*.py) are run unchanged, with their model calls
routed through src/llm.py; the max_tokens a module requests is ignored in
favour of the shared config. Task data: tasks/*_tasks.json, except Letter
Guess, whose file was regenerated after the original runs; it uses the data
the original runs used (judge_validation/original_tasks/wordle.json).
Transcripts only; switching and correctness are judged afterwards (Opus judge,
Knights panel and programmatic colouring check, as in rescore_correctness.py).

    python tasks/t4_strategy_switching/run_v2.py [model ...] [--pilot --limit N]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import llm  # noqa: E402
import v2run  # noqa: E402
from run import TASK_MODULES, TASKS_DIR  # noqa: E402

TASK = "t4_strategy_switching"


def problems():
    out = {}
    for tt, mod in TASK_MODULES.items():
        path = HERE / "judge_validation" / "original_tasks" / "wordle.json" if tt == "letter_guess" else TASKS_DIR / mod.TASKS_JSON_PATH
        for i, p in enumerate(json.loads(path.read_text())["problems"]):
            out[f"{tt}_{i}"] = (tt, p)
    return out


# Cipher: with reasoning on, most models ran out of the shared 16,384-token limit on
# every Cipher problem (2026-10-14: 10/10 for Gemini 2.5 Pro, Sonnet 4.6, GPT-5.4 and
# DeepSeek V3.2, 8/10 for Opus 4.6). The shared limit is primary (option A, 2026-10-15);
# --limit-32k runs the Cipher problems only at 32,768, the largest limit every model in
# the panel accepts (GPT-4.1's maximum), into results_v2_32k/ as the sensitivity check.
CIPHER_MAX_TOKENS = 32768
CIPHER_TIMEOUT_S = 1800


def main():
    a = v2run.cli(__doc__)
    probs = problems()
    jobs = v2run.jobs_with_repeats(TASK, sorted(probs))
    if a.limit_32k:
        jobs = [j for j in jobs if probs[j[0]][0] == "cipher"]
    for model in a.models:
        def work(unit, cond, rep, store, model=model):
            tt, task = probs[unit]
            calls = []

            def call_model(_model, messages, max_tokens=None):
                big = tt == "cipher" and a.limit_32k
                r = llm.call(model, messages, log_path=store.log,
                             max_tokens=CIPHER_MAX_TOKENS if big else None, timeout_s=CIPHER_TIMEOUT_S if big else None,
                             meta={"task": TASK, "unit": unit, "repeat": rep, "turn": len(calls)})
                calls.append(v2run.result_fields(r))
                if r.status == "error":
                    raise RuntimeError(r.error)
                return r.text or "", r.usage

            result = TASK_MODULES[tt].run(model, task, call_model=call_model)
            status = "ok" if all(c["status"] == "ok" for c in calls) else next(c["status"] for c in calls if c["status"] != "ok")
            n_prefill = result.get("n_prefill", len(task.get("prefill_messages", [])))
            keep = {k: v for k, v in result.items() if k not in ("messages",)}
            return {"task_type": tt, "messages": result["messages"], "n_prefill": n_prefill,
                    "module_result": json.loads(json.dumps(keep, default=str)), "calls": calls, "status": status}
        v2run.run(TASK, model, jobs, work, workers=a.workers, retry_errors=not a.no_retry_errors,
                  limit=a.limit, pilot=a.pilot, units=a.units, subdir="results_v2_32k" if a.limit_32k else None)


if __name__ == "__main__":
    main()
