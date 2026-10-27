"""
Score the v2 question-answering tasks (Wagering, Deferral, Consistency) with the
shared rules in src/v2score.py. Reads tasks/<task>/results_v2/<model>.jsonl and
writes tasks/<task>/results_v2/scored/<model>.jsonl, one row per
(unit, condition, repeat):

  status        run outcome (ok | truncated | empty | blocked | error)
  parsed        response gave the required structure
  answer        letter (multiple choice) or text (SimpleQA); None if none given
  correct       True/False; None when there is no answer, no ground truth or a
                panel split. A SimpleQA non-answer ("unknown") is incorrect, flagged
                not_attempted.
  confidence    0-100 or None (missing or out of range)
  wager / decision   Wagering / Deferral fields, None if invalid
  method        how the answer was obtained: strict | judge | none
  judge_votes   per-judge parse, where the panel read the response
  audit         judged only as part of the 10% audit

Deferral's 3 non-letter-ground-truth items and the underspecified items have
correct=None. The Deferral counterfactual is scored under condition "forced".

    python scripts/score_v2_qa.py [--tasks t2,t5,t7] [model ...]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tasks" / "t5_selective_deferral"))
import v2run  # noqa: E402
from v2score import (GRADE_PROMPT, EXTRACT_PROMPT, Panel, audit_pick, confidence, enum, gt_letter,  # noqa: E402
                     majority, mc_letter, norm_text, options_block, parse_extract, parse_grade, parse_json,
                     strict_answer, strict_conf)

TASKS = {"t2": "t2_confidence_wagering", "t5": "t5_selective_deferral", "t7": "t7_consistency"}
WAGERS = {"SAFE", "MEDIUM", "BOLD"}
DECISIONS = {"ATTEMPT", "DEFER", "CLARIFY"}


def load(task, model):
    p = ROOT / "tasks" / task / "results_v2" / f"{model}.jsonl"
    recs = {}
    for line in p.open():
        r = json.loads(line)
        recs[(r["unit"], r["condition"], r["repeat"])] = r
    return list(recs.values())


def first_pass(task, r, q):
    """Strict extraction. Returns the row plus what (if anything) the panel must read."""
    row = {k: r[k] for k in ("unit", "condition", "repeat", "status")}
    row.update(source=q.get("source") if q else None, answer=None, correct=None, confidence=None, method="none",
               judge_votes=None, audit=False, parsed=False, strict=None)
    text = r.get("text")
    if task == "t5_selective_deferral":
        row.update(block=q["block"], is_underspecified=q["is_underspecified"], decision=None)
    if task == "t2_confidence_wagering":
        row["wager"] = None
    if r["status"] in ("error", "blocked", "empty") or not text:
        return row, None
    mc = isinstance(q.get("choices"), dict)
    gt = gt_letter(q) if mc else None
    key = (task, r["model"], r["unit"], r["condition"], r["repeat"])

    if task == "t7_consistency":
        letters = sorted(q["choices"])
        row["confidence"] = strict_conf(text)
        a = strict_answer(text, letters)
        row["parsed"] = a is not None
        row.update(answer=a, strict=a, method="strict" if a else "none")
        need = (a is None) or audit_pick(*key)
        row["audit"] = a is not None and need
        prompt = EXTRACT_PROMPT.format(options=options_block(q["choices"]), letters=", ".join(letters),
                                       response=text[-4000:]) if need else None
        return row, ("extract", prompt, gt)

    d = parse_json(text)
    row["parsed"] = d is not None
    if d is None:
        return row, None          # no structure: nothing to score; counted as a parse failure
    row["confidence"] = confidence(d.get("confidence"))
    if task == "t2_confidence_wagering":
        row["wager"] = enum(d.get("wager"), WAGERS)
    if task == "t5_selective_deferral" and r["condition"] != "forced":
        row["decision"] = enum(d.get("decision"), DECISIONS)
        if row["decision"] != "ATTEMPT":
            return row, None
    raw_ans = d.get("answer")
    if raw_ans is None or not str(raw_ans).strip():
        return row, None
    if task == "t5_selective_deferral" and q["is_underspecified"]:
        row.update(answer=str(raw_ans), method="strict")
        return row, None
    if mc:
        a = mc_letter(raw_ans, q["choices"])
        if a:
            row.update(answer=a, strict=a, method="strict")
            return row, None
        prompt = EXTRACT_PROMPT.format(options=options_block(q["choices"]), letters=", ".join(sorted(q["choices"])),
                                       response=str(raw_ans)[-4000:])
        return row, ("extract", prompt, gt)
    # free text (SimpleQA)
    row["answer"] = str(raw_ans)
    exact = norm_text(raw_ans) == norm_text(q["answer"]) and norm_text(q["answer"]) != ""
    if exact:
        row.update(correct=True, method="strict")
        if not audit_pick(*key):
            return row, None
        row["audit"] = True
    prompt = GRADE_PROMPT.format(question=q["question"], gold=q["answer"], predicted=str(raw_ans)[:2000])
    return row, ("grade", prompt, None)


def finish(row, need, raw_votes, q):
    """Where the panel read a response, its majority wins (as in Self-Review); a split keeps the strict result."""
    kind, prompt, _ = need
    if kind == "extract":
        votes = [parse_extract(v, sorted(q["choices"])) if v is not None else "FAILED" for v in raw_votes]
        row["judge_votes"] = votes
        top = majority(votes)
        if top is not None and top not in ("FAILED", "UNPARSED"):
            ans = None if top == "NONE" else top
            if ans != row["answer"]:
                row.update(answer=ans, method="judge" if ans else "none")
    else:
        votes = [parse_grade(v) if v is not None else "FAILED" for v in raw_votes]
        row["judge_votes"] = votes
        top = majority(votes)
        if top in ("CORRECT", "INCORRECT", "NOT_ATTEMPTED"):
            # A non-answer ("unknown") given where an answer was required earns nothing: it
            # is scored incorrect (as in the original scoring) and flagged.
            row["correct"] = top == "CORRECT"
            row["not_attempted"] = top == "NOT_ATTEMPTED"
            row["method"] = "judge"
        elif not row["audit"]:
            row["correct"] = None   # panel split or failed: unscorable, reported as such
    return row


def score(task, model, panel_workers):
    if task == "t5_selective_deferral":
        import run_v2 as t5   # noqa: E402
        qs = t5.questions(v2run.bank_version(model))
    else:
        qs = v2run.bank(include_removed=True, version=v2run.bank_version(model))
    recs = load(task, model)
    panel = Panel(ROOT / "tasks" / task / "results_v2" / "judge_cache.jsonl", workers=panel_workers)
    pairs = [first_pass(task, r, qs[r["unit"]]) + (qs[r["unit"]],) for r in recs]
    raw = panel.run([n[1] for _, n, _ in pairs if n and n[1]])
    out = []
    for row, need, q in pairs:
        if need and need[1]:
            row = finish(row, need, raw[need[1]], q)
        if isinstance(q.get("choices"), dict) and row["answer"] is not None and row["unit"] not in v2run.BAD_GT:
            gt = gt_letter(q)
            row["correct"] = (row["answer"] == gt) if gt else None
        if row["unit"] in v2run.BAD_GT or q.get("is_underspecified"):
            row["correct"] = None
        out.append(row)
    dest = ROOT / "tasks" / task / "results_v2" / "scored" / f"{model}.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(json.dumps(r) + "\n" for r in sorted(out, key=lambda r: (r["unit"], r["condition"], r["repeat"]))))
    return out


def report(task, model, rows):
    n = len(rows)
    ok = [r for r in rows if r["status"] == "ok"]
    judged = [r for r in rows if r["judge_votes"]]
    aud = [r for r in rows if r["audit"]]
    scored = [r for r in rows if r["correct"] is not None]
    print(f"[{task}] {model}: {n} rows, ok {len(ok)}, parsed {sum(r['parsed'] for r in rows)}, "
          f"judged {len(judged)} (audit {len(aud)}), scorable {len(scored)}, "
          f"accuracy {sum(r['correct'] for r in scored) / max(len(scored), 1):.1%}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("models", nargs="*")
    p.add_argument("--tasks", default="t2,t5,t7")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    for t in a.tasks.split(","):
        task = TASKS[t]
        models = a.models or sorted(f.stem for f in (ROOT / "tasks" / task / "results_v2").glob("*.jsonl"))
        for m in models:
            report(task, m, score(task, m, a.workers))


if __name__ == "__main__":
    main()
