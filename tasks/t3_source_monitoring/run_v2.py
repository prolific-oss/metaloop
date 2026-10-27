"""
Source-Monitoring, v2 rerun (October 2026): shared model config.

Open-book condition on the five passages reported in the paper (001-005), with
the original prompt (run.build_reasoned_prompt). The statement order is now
shuffled with a per-passage seed and saved (the original shuffle was unseeded
and not saved), so responses can be mapped back to statements and labels.
Raw responses only; labels are parsed afterwards. All five passages are repeated
twice (the repeat subset covers every unit when there are fewer than 30).

Output limit: the shared 16384 is primary. At 16384 Gemini 3.8 Flash spent up to
15.7k tokens on hidden reasoning and was cut off on 5 of 15 calls (2026-10-14).
--limit-32k reruns at 32768 tokens with a 1800 s timeout into results_v2_32k/, as
the output-limit sensitivity check.

    python tasks/t3_source_monitoring/run_v2.py [model ...] [--pilot --limit N]
"""
import random
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
import llm  # noqa: E402
import v2run  # noqa: E402
import run  # noqa: E402

TASK = "t3_source_monitoring"
PASSAGES = ["001", "002", "003", "004", "005"]
BIG_MAX_TOKENS = 32768
BIG_TIMEOUT_S = 1800
_LOCK = threading.Lock()  # build_reasoned_prompt uses the global random module


def prompt(pid):
    name, text, statements, tiers = run.PASSAGES[pid]
    with _LOCK:
        random.seed(f"metaloop-v2-sm:{pid}")
        p, indexed = run.build_reasoned_prompt(text, statements)
    return p, [orig for orig, _ in indexed], statements


def main():
    a = v2run.cli(__doc__)
    jobs = v2run.jobs_with_repeats(TASK, PASSAGES)
    for model in a.models:
        def work(unit, cond, rep, store, model=model):
            p, order, statements = prompt(unit)
            r = llm.call(model, [{"role": "user", "content": p}], log_path=store.log,
                         max_tokens=BIG_MAX_TOKENS if a.limit_32k else None,
                         timeout_s=BIG_TIMEOUT_S if a.limit_32k else None, meta={"task": TASK, "unit": unit, "repeat": rep})
            return {"shuffle_order": order, "labels": [l for _, l in statements], "text": r.text,
                    **v2run.result_fields(r)}
        v2run.run(TASK, model, jobs, work, workers=a.workers, retry_errors=not a.no_retry_errors,
                  limit=a.limit, pilot=a.pilot, units=a.units, subdir="results_v2_32k" if a.limit_32k else None)


if __name__ == "__main__":
    main()
