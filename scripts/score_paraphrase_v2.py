"""
Score the camera-ready paraphrase arm on the v2 settings (run_paraphrase_v2.py) with the rules
behind the main v2 numbers, writing rows in the schema analyse_paraphrase.py reads.

Wagering (tasks/t2_confidence_wagering/results_v2_paraphrase/<model>.jsonl), per (item, wording):
  scored by score_v2_qa.first_pass / finish exactly as the main v2 Wagering run (strict JSON parse,
  multiple-choice extraction panel where the answer is not a letter, SimpleQA grading panel on
  non-exact answers plus a seeded 10% audit). A row is usable ("parsed") under metrics_v2_qa's
  rule: status ok, a scorable answer, a 0-100 confidence and a valid wager.
Self-Review (tasks/t6_error_detection/results_v2_paraphrase/<model>.jsonl), per (item, wording):
  Phase 1 is the stored v2 answer the run replayed (results_v2/phase1_scored); rows as
  score_v2_review.self_review builds them (BAD_GT and unanswered Phase 1 excluded).
  flagged   Rule A (verdict_judge.score), as ruleA_v2.py; no verdict at all -> None
  revised   regex_v2, then the revised-answer panel on every Phase-1 error, every flagged
            Phase-1-correct item and a seeded 10% of the rest; SAME/NONE keep the Phase-1 answer

    python scripts/score_paraphrase_v2.py [--dry-run] [model ...]
    -> tasks/<task>/results_v2_paraphrase/scored/<model>.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "scripts", ROOT / "tasks" / "t6_error_detection"):
    sys.path.insert(0, str(p))
import v2run  # noqa: E402
import verdict_judge  # noqa: E402
from score_v2_qa import finish, first_pass  # noqa: E402
from score_v2_review import REVISED_PROMPT, VERDICT, latest, parse_revised_votes, regex_v2  # noqa: E402
from v2score import Panel, audit_pick, gt_letter, majority  # noqa: E402

T2 = ROOT / "tasks" / "t2_confidence_wagering"
T6 = ROOT / "tasks" / "t6_error_detection"
SUBDIR = "results_v2_paraphrase"
MODELS = ["sonnet-4.6", "gpt-4.1", "gpt-5.4", "mistral-large"]


class DryPanel:
    """Counts the prompts a panel would read; returns no votes."""
    def __init__(self, name):
        self.name, self.n = name, 0

    def run(self, prompts):
        self.n += len(prompts)
        return {p: [None, None, None] for p in prompts}


def wagering(model, panel):
    bank = v2run.bank(include_removed=True, version=v2run.bank_version(model))
    rows = []
    pairs = [first_pass("t2_confidence_wagering", r, bank[r["unit"]]) + (bank[r["unit"]], r)
             for r in latest(T2 / SUBDIR / f"{model}.jsonl")]
    raw = panel.run([n[1] for _, n, _, _ in pairs if n and n[1]])
    for row, need, q, r in pairs:
        if need and need[1]:
            row = finish(row, need, raw[need[1]], q)
        if isinstance(q.get("choices"), dict) and row["answer"] is not None:
            gt = gt_letter(q)
            row["correct"] = (row["answer"] == gt) if gt else None
        if row["unit"] in v2run.BAD_GT:
            row["correct"] = None
        usable = row["status"] == "ok" and row["correct"] is not None and row["confidence"] is not None \
            and row["wager"] is not None
        rows.append({"id": row["unit"], "variant": row["condition"], "source": q.get("source"),
                     "outcome": "parsed" if usable else "unusable", "invalid_confidence": False,
                     "strict_ok": usable, "defaulted": False, "confidence": row["confidence"],
                     "correct": row["correct"], "wager": row["wager"], "method": row["method"],
                     "judge_votes": row["judge_votes"], "status": row["status"]})
    return rows


def self_review(model, revised_panel, verdict_panel):
    bank = v2run.bank(include_removed=True, version=v2run.bank_version(model))
    scored = json.loads((T6 / "results_v2" / "phase1_scored" / f"{model}.json").read_text())
    rows, prompts, items = [], {}, []
    for r in latest(T6 / SUBDIR / f"{model}.jsonl"):
        s = scored.get(r["unit"])
        if r["unit"] in v2run.BAD_GT or s is None or s["answer"] is None:
            continue
        q = bank[r["unit"]]
        letters = sorted(q["choices"])
        text = (r.get("phase2") or {}).get("text") or ""
        verdicts = [v.upper() for v in VERDICT.findall(text)]
        key = f"{model}|{r['unit']}|{r['condition']}"
        row = {"key": key, "id": r["unit"], "variant": r["condition"], "source": q.get("source"),
               "status": r["status"], "p1": s["answer"], "gt": s["gt"], "p1_correct": s["correct"],
               "flagged_regex": ("ERROR" in verdicts) if verdicts else None,
               "revised_regex": regex_v2(text, letters) if text else "NONE", "revised_votes": None}
        if row["flagged_regex"] is not None:
            items.append({"key": key, "text": text, "phase1_answer": s["answer"], "letters": letters})
        if text and (not s["correct"] or row["flagged_regex"] or audit_pick("sr2p", model, r["unit"], r["condition"])):
            prompts[key] = REVISED_PROMPT.format(letters=", ".join(letters), review=text[-4000:])
        rows.append(row)
    raw = revised_panel.run(list(prompts.values()))
    judged = {j["key"]: j for j in verdict_judge.score(items, verdict_panel)}
    for row in rows:
        rev = row["revised_regex"]
        if row["key"] in prompts:
            votes = parse_revised_votes(raw[prompts[row["key"]]], sorted(bank[row["id"]]["choices"]))
            row["revised_votes"] = votes
            top = majority(votes)
            if top not in (None, "FAILED", "UNPARSED"):
                rev = top
        row["revised"] = rev
        row["final_correct"] = (row["p1"] if rev in ("SAME", "NONE") else rev) == row["gt"]
        j = judged.get(row["key"])
        row["flagged"] = j["flagged_ruleA"] if j else None
        row["verdict_route"] = j["route"] if j else None
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="*")
    ap.add_argument("--dry-run", action="store_true", help="count panel prompts, call nothing, write nothing")
    a = ap.parse_args()
    models = a.models or MODELS
    if a.dry_run:
        pq, pr, pv = DryPanel("qa"), DryPanel("revised"), DryPanel("verdict")
    else:
        pq = Panel(T2 / SUBDIR / "judge_cache.jsonl")
        pr = Panel(ROOT / "results_v2_metrics" / "judge_cache_paraphrase_v2.jsonl")
        pv = Panel(verdict_judge.CACHE)
    for m in models:
        w = wagering(m, pq)
        if a.dry_run:
            # the verdict router needs no votes to decide which reviews go to the panel
            sr = self_review(m, pr, type("P", (), {"run": lambda self, ps: (setattr(pv, "n", pv.n + len(ps)) or
                                                                              {p: [None] * 3 for p in ps})})())
        else:
            sr = self_review(m, pr, pv)
        uw = sum(r["outcome"] == "parsed" for r in w)
        print(f"{m}: Wagering {len(w)} rows, {uw} usable; Self-Review {len(sr)} rows, "
              f"{sum(r['flagged'] is not None for r in sr)} with a verdict", flush=True)
        if not a.dry_run:
            for d, rows in ((T2, w), (T6, sr)):
                out = d / SUBDIR / "scored" / f"{m}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps({"model": m, "rows": rows}, indent=1))
    if a.dry_run:
        print(f"panel prompts (x3 judges): QA {pq.n}, revised answer {pr.n}, Rule A verdict {pv.n}")


if __name__ == "__main__":
    main()
