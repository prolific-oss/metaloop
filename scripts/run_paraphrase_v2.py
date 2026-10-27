"""
Camera-ready paraphrase arm on the v2 settings (2026-10-19).

Wordings: config/paraphrases.yaml (PARAPHRASES.md, approved by John), each an assert-exactly-once
replacement of one line (two for the Self-Review V*). Prompts and serving are the v2 runs':
- Wagering: tasks/t2_confidence_wagering/run_v2.prompt, Condition A, system message as there;
  7 wordings (original + C1-C3 + W1-W3) x the 451 letter-ground-truth items of bank v2.
- Self-Review: Phase 2 only, PHASE2_PROMPT with the variant lines, replaying the model's stored
  v2 Phase-1 answer (tasks/t6_error_detection/results_v2/<model>.jsonl, repeat 0); 4 wordings
  (original + V1-V3) x the 153-item pool. Items whose stored Phase 1 is not ok are skipped.
All calls through llm.call (config/models.yaml). The original wording is rerun in the same
window. Results: tasks/<task>/results_v2_paraphrase/<model>.jsonl, condition = wording.

    python scripts/run_paraphrase_v2.py {wagering,selfreview} MODEL [MODEL ...] [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import llm  # noqa: E402
import v2run  # noqa: E402
import yaml  # noqa: E402

CONFIG = yaml.safe_load((ROOT / "config" / "paraphrases.yaml").read_text())
SUBDIR = "results_v2_paraphrase"


def replace_once(text, old, new):
    assert text.count(old) == 1, f"expected exactly one occurrence of {old!r}"
    return text.replace(old, new)


def apply(text, section, variant):
    for slot, new in (CONFIG[section]["variants"][variant] or {}).items():
        text = replace_once(text, CONFIG[section]["original_lines"][slot], new)
    return text


def wagering(models, limit, workers):
    sys.path.insert(0, str(ROOT / "tasks" / "t2_confidence_wagering"))
    import run_v2
    task = "t2_confidence_wagering"
    bank = v2run.bank()
    units = sorted(i for i in bank if i not in v2run.BAD_GT)
    jobs = [(u, v, 0) for v in CONFIG["wagering"]["variants"] for u in units]
    for model in models:
        def work(unit, variant, rep, store, model=model):
            q = bank[unit]
            msgs = [{"role": "system", "content": run_v2.SYSTEM},
                    {"role": "user", "content": apply(run_v2.prompt(q, "A"), "wagering", variant)}]
            r = llm.call(model, msgs, log_path=store.log, meta={"task": task, "paraphrase": variant, "unit": unit})
            return {"source": q["source"], "text": r.text, **v2run.result_fields(r)}
        v2run.run(task, model, jobs, work, workers=workers, limit=limit, subdir=SUBDIR)


def selfreview(models, limit, workers):
    sys.path.insert(0, str(ROOT / "tasks" / "t6_error_detection"))
    from run import PHASE1_PROMPT, PHASE2_PROMPT, format_question
    task = "t6_error_detection"
    bank = v2run.bank(include_removed=True)
    pool = v2run.selfreview_pool()
    for model in models:
        stored = {}
        for line in (ROOT / "tasks" / task / "results_v2" / f"{model}.jsonl").open():
            r = json.loads(line)
            if r["repeat"] == 0:
                stored[r["unit"]] = r
        units = [u for u in pool if u in stored and stored[u].get("phase1", {}).get("status") == "ok"]
        print(f"[paraphrase] {model}: {len(units)}/{len(pool)} items with an ok stored Phase 1", flush=True)
        jobs = [(u, v, 0) for v in CONFIG["self_review"]["variants"] for u in units]

        def work(unit, variant, rep, store, model=model):
            q = bank[unit]
            msgs = [{"role": "user", "content": PHASE1_PROMPT.format(question=format_question(q))},
                    {"role": "assistant", "content": stored[unit]["phase1"]["text"]},
                    {"role": "user", "content": apply(PHASE2_PROMPT, "self_review", variant)}]
            r = llm.call(model, msgs, log_path=store.log, meta={"task": task, "paraphrase": variant, "unit": unit, "phase": 2})
            return {"source": q["source"], "phase2": {"text": r.text, **v2run.result_fields(r)}, "status": r.status}
        v2run.run(task, model, jobs, work, workers=workers, limit=limit, subdir=SUBDIR)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["wagering", "selfreview"])
    p.add_argument("models", nargs="+")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    (wagering if a.task == "wagering" else selfreview)(a.models, a.limit, a.workers)


if __name__ == "__main__":
    main()
