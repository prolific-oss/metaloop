"""
Harness for the MetaLoop v2 rerun (October 2026): item pools on question bank
v2, the repeat subset, and a resumable runner.

Runners collect raw responses only. Scoring happens afterwards with the
corrected extraction pipeline, so no inline parser or fallback shapes the data.

Results: tasks/<task>/results_v2/<model>.jsonl, one record per
(item, condition, repeat); call logs: tasks/<task>/results_v2/logs/<model>.jsonl.
Only records with status "error" (API or transport failure, or an empty reply for
which the provider reports generated tokens) are retried on a later run;
refusals, truncations and genuinely empty responses are final outcomes.
Every attempt is in the log.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import llm  # noqa: E402

BAD_GT = {"mmlu_236", "mmlu_281", "q_432"}   # stored answer is option text, not a letter
REPEATS = (1, 2)                            # repeat 0 is the main run
REPEAT_N = 30
REPEAT_SEED = "metaloop-v2-repeats"


BANK_FILES = {  # version -> (bank, removed items)
    "v2": ("question_bank_v2.json", "question_bank_v2_removed.json"),
    # the original bank of the accepted paper (GPQA items list the correct answer first);
    # non-GPQA items are identical to v2. Used to add a model to the original panel.
    "v1": ("question_bank_annotated.json", "question_bank_removed_2026-04-10.json"),
}


def bank(include_removed: bool = False, version: str = "v2") -> dict[str, dict]:
    main, removed = BANK_FILES[version]
    items = json.loads((ROOT / "data" / main).read_text())
    if include_removed:
        items += json.loads((ROOT / "data" / removed).read_text())["items"]
    return {q["id"]: q for q in items}


def bank_version(model: str) -> str:
    """The question bank a model's runs use: the config entry's `bank`, else v2. Scorers use this
    so that ground-truth letters (which differ between banks on GPQA items) match the run."""
    return llm.spec(model).get("bank", "v2")


def selfreview_pool() -> list[str]:
    """The 153 letter-ground-truth items of the Self-Review/Consistency pool (incl. q_451)."""
    ids = [r["id"] for r in json.loads((ROOT / "tasks" / "t6_error_detection" / "results" / "gen_review_qbank"
                                         / "anthropic_claude-opus-4.6.json").read_text())]
    return sorted(i for i in ids if i not in BAD_GT)


def repeat_subset(task: str, units: list[str]) -> list[str]:
    """Fixed 30-unit subset per task for the two extra repeats."""
    rng = random.Random(f"{REPEAT_SEED}:{task}")
    return sorted(rng.sample(sorted(units), min(REPEAT_N, len(units))))


def jobs_with_repeats(task: str, units: list[str], conditions=("main",)) -> list[tuple[str, str, int]]:
    rep = set(repeat_subset(task, units))
    out = [(u, c, 0) for u in units for c in conditions]
    out += [(u, c, r) for r in REPEATS for u in sorted(rep) for c in conditions]
    return out


class Store:
    """Append-only JSONL of results; the latest record per key wins."""

    def __init__(self, task: str, model: str, pilot: bool = False, subdir: str | None = None):
        self.dir = ROOT / "tasks" / task / (subdir or ("results_v2_pilot" if pilot else "results_v2"))
        self.path = self.dir / f"{model}.jsonl"
        self.log = self.dir / "logs" / f"{model}.jsonl"
        self.lock = threading.Lock()
        self.records: dict[tuple, dict] = {}
        if self.path.exists():
            for line in self.path.open():
                r = json.loads(line)
                self.records[(r["unit"], r["condition"], r["repeat"])] = r

    def done(self, key, retry_errors: bool) -> bool:
        """Only API/transport failures ("error") are retried. Refusals ("blocked"), truncated and
        empty responses are model behaviour and final: retrying them until a usable answer
        appears would select on the outcome. The exception is an empty reply for which the
        provider reports generated tokens (llm.provider_empty), which is a provider fault;
        records saved before llm.call retried these are rerun here."""
        r = self.records.get(key)
        return r is not None and ((r["status"] != "error" and not misfiled(r)) or not retry_errors)

    def add(self, rec: dict):
        with self.lock:
            self.records[(rec["unit"], rec["condition"], rec["repeat"])] = rec
            self.dir.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")


def misfiled(rec: dict) -> bool:
    """True if any call in the record was an llm.provider_fault reply: flat records, Self-Review
    phases, and multi-call sessions (Strategy-Switching, Loop)."""
    parts = [rec] + [rec[p] for p in ("phase1", "phase2") if isinstance(rec.get(p), dict)] + list(rec.get("calls") or [])
    return any(llm.provider_fault(x.get("status"), x.get("usage"), x.get("finish_reason")) for x in parts)


def result_fields(res: llm.Result) -> dict:
    return {"status": res.status, "finish_reason": res.finish_reason, "provider": res.provider,
            "model_version": res.model_version, "usage": res.usage, "error": res.error, "attempts": res.attempts}


def run(task: str, model: str, jobs: list[tuple], work, *, workers: int = 8, retry_errors: bool = True,
        limit: int | None = None, pilot: bool = False, units: list[str] | None = None,
        subdir: str | None = None) -> Store:
    """`work(unit, condition, repeat, store) -> dict` returns the record (must include 'status')."""
    store = Store(task, model, pilot, subdir)
    todo = [j for j in jobs if not store.done(j, retry_errors) and (not units or j[0] in units)]
    if limit:
        todo = todo[:limit]
    print(f"[{task}] {model}: {len(jobs)} jobs, {len(todo)} to run", flush=True)

    def one(job):
        unit, cond, rep = job
        try:
            rec = work(unit, cond, rep, store)
        except Exception as e:  # a bug in task code, not an API failure: record it, keep going
            rec = {"status": "error", "error": f"{type(e).__name__}: {str(e)[:300]}"}
        rec.update({"unit": unit, "condition": cond, "repeat": rep, "model": model, "config_hash": llm.CONFIG_HASH})
        store.add(rec)
        return rec["status"]

    n = 0
    with ThreadPoolExecutor(workers) as ex:
        for status in ex.map(one, todo):
            n += 1
            if n % 50 == 0 or n == len(todo):
                bad = sum(r["status"] != "ok" for r in store.records.values())
                print(f"[{task}] {model}: {n}/{len(todo)} done, {bad} records not ok", flush=True)
    return store


def cli(description: str):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("models", nargs="*", help="model keys from config/models.yaml (default: the core panel)")
    p.add_argument("--cohort", choices=["core", "extension", "all"], default="core")
    p.add_argument("--limit", type=int, help="run at most this many jobs per model (pilot)")
    p.add_argument("--pilot", action="store_true", help="pilot mode: results under results_v2_pilot/")
    p.add_argument("--no-retry-errors", action="store_true")
    p.add_argument("--limit-32k", action="store_true",
                   help="output-limit sensitivity run (32768 tokens, 1800 s), results under results_v2_32k/; "
                        "the shared 16384 limit is primary (option A, 2026-10-15)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--units", type=lambda s: s.split(","), help="comma-separated unit ids to run (pilot)")
    p.add_argument("--conditions", type=lambda s: s.split(","), help="comma-separated conditions to run (pilot; QA tasks)")
    p.add_argument("--bank", choices=sorted(BANK_FILES), default="v2",
                   help="question bank (default v2); must match each model's `bank` in config/models.yaml")
    a = p.parse_args()
    a.models = a.models or llm.models(None if a.cohort == "all" else a.cohort)
    unknown = [m for m in a.models if m not in llm.models(None)]
    if unknown:
        p.error(f"unknown model keys {unknown}; known: {llm.models(None)}")
    mismatch = [m for m in a.models if bank_version(m) != a.bank]
    if mismatch:
        p.error(f"--bank {a.bank} does not match the configured bank of {mismatch} "
                f"({[bank_version(m) for m in mismatch]}); scoring uses the configured bank")
    return a
