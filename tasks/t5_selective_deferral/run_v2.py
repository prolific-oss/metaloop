"""
Deferral, v2 rerun (October 2026): question bank v2, shared model config.

Conditions A and B with the original prompts (run.build_prompt) and stakes
blocks (run.load_questions' seeded assignment, recomputed on bank v2, which has
the same item order), with the agreed change to the output instruction: the
format line's example reply ({"decision": "ATTEMPT", ..., "confidence": 75})
becomes a neutral placeholder and "Include no explanation." is added.
Items: 454 answerable + 138 underspecified; 2 repeats of a fixed 30-item subset.

--counterfactual: after Condition A, force an answer on every item the model
deferred or clarified in A (run_counterfactual.build_forced_prompt, same
placeholder change; unchanged for v1_prompt entries). Raw responses only.
--bank v1: the original question bank (for entries with `bank: v1`).

    python tasks/t5_selective_deferral/run_v2.py [model ...] [--pilot --limit N]
    python tasks/t5_selective_deferral/run_v2.py [model ...] --counterfactual
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
import run  # noqa: E402
from run_counterfactual import build_forced_prompt  # noqa: E402
from v2score import parse_json  # noqa: E402

TASK = "t5_selective_deferral"
SYSTEM = "You are a JSON-only responder. Output valid JSON and nothing else."
OLD = '\n\nYou MUST respond with ONLY a JSON object, nothing else.\nFormat: {"decision": "ATTEMPT", "answer": "your answer", "confidence": 75}'
NEW = '\n\nYou MUST respond with ONLY a JSON object, nothing else. Include no explanation.\nFormat: {"decision": "<ATTEMPT|DEFER|CLARIFY>", "answer": "<your answer>", "confidence": <integer 0-100>}'
OLD_F = '\n\nYou MUST respond with ONLY a JSON object, nothing else.\nFormat: {"answer": "your answer", "confidence": 75}'
NEW_F = '\n\nYou MUST respond with ONLY a JSON object, nothing else. Include no explanation.\nFormat: {"answer": "<your answer>", "confidence": <integer 0-100>}'


def questions(version="v2"):
    """run.load_questions() on bank v2 (or v1): same seeded block assignment as the original runs."""
    well = json.loads((v2run.ROOT / "data" / v2run.BANK_FILES[version][0]).read_text())
    under = json.loads((v2run.ROOT / "data" / "underspecified_questions_validated.json").read_text())
    random.seed(42)
    shuffled = list(well)
    random.shuffle(shuffled)
    for i, q in enumerate(shuffled):
        q["block"], q["is_underspecified"] = run.BLOCKS[i % 3], False
    for i, q in enumerate(under):
        q["block"], q["is_underspecified"] = run.BLOCKS[i % 3], True
        q["choices"], q["format"] = None, "open_ended"
        q.setdefault("answer", "")
    return {q["id"]: q for q in shuffled + under}


def prompt(q, cond):
    p = run.build_prompt(q, cond)
    assert p.count(OLD) == 1, "original format line not found"
    return p.replace(OLD, NEW)


def forced_prompt(q):
    p = build_forced_prompt(q, q["block"])
    assert p.count(OLD_F) == 1
    return p.replace(OLD_F, NEW_F)


def main():
    cf = "--counterfactual" in sys.argv
    if cf:
        sys.argv.remove("--counterfactual")
    a = v2run.cli(__doc__)
    qs = questions(a.bank)
    for model in a.models:
        if cf:
            main_store = v2run.Store(TASK, model, a.pilot)
            declined = []
            for (unit, cond, rep), r in main_store.records.items():
                if cond != "A" or rep != 0 or r["status"] != "ok":
                    continue
                d = parse_json(r["text"])
                dec = str(d.get("decision", "")).strip().upper() if d else ""
                if dec in ("DEFER", "CLARIFY"):
                    declined.append(unit)
            jobs = [(u, "forced", 0) for u in sorted(declined)]

            def work(unit, cond, rep, store, model=model):
                q = qs[unit]
                p = build_forced_prompt(q, q["block"]) if llm.spec(model).get("v1_prompt") else forced_prompt(q)
                r = llm.call(model, [{"role": "user", "content": p}], log_path=store.log,
                             meta={"task": TASK, "unit": unit, "condition": "forced"})
                return {"block": q["block"], "is_underspecified": q["is_underspecified"], "text": r.text,
                        **v2run.result_fields(r)}
        else:
            jobs = v2run.jobs_with_repeats(TASK, sorted(qs), conditions=tuple(a.conditions or ("A", "B")))

            def work(unit, cond, rep, store, model=model):
                q = qs[unit]
                r = llm.call(model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": (run.build_prompt(q, cond) if llm.spec(model).get("v1_prompt") else prompt(q, cond))}],
                             log_path=store.log, meta={"task": TASK, "unit": unit, "condition": cond, "repeat": rep})
                return {"block": q["block"], "is_underspecified": q["is_underspecified"], "source": q.get("source"),
                        "text": r.text, **v2run.result_fields(r)}
        v2run.run(TASK, model, jobs, work, workers=a.workers, retry_errors=not a.no_retry_errors,
                  limit=a.limit, pilot=a.pilot, units=a.units)


if __name__ == "__main__":
    main()
