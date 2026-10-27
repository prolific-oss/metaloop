"""
Score the v2 review tasks (October 2026): Self-Review Phase 2, Control 2 and
The-Loop v3. Phase-1 answers come from score_v2_phase1.py.

Self-Review Phase 2 (main run):
  flagged     any step VERDICT: ERROR (markdown-tolerant); a review with no
              verdict at all is unparsed and excluded from flag rates
  revised     REVISED ANSWER via revised_answer_audit.regex_v2; the three-judge
              panel (its JUDGE_PROMPT) re-reads every Phase-1 error, every
              flagged Phase-1-correct item and a seeded 10% of the rest; the
              panel majority wins where it read. SAME/NONE keep the Phase-1 answer.
  confidence  mean step CONFIDENCE (0-100 values only)
  Metrics: Phase-1 accuracy, flag rate on Phase-1 errors (blind-spot rate is its
  complement), false-criticism rate, fix rate (revised answer correct on Phase-1
  errors), confidence on blind spots vs true approves.

Control 2: a fresh Phase-1 attempt on each Phase-1 error, scored like Phase 1
  (strict ANSWER line, panel where missing plus 10% audit). Compared with the
  review's fix on the same items: discordant counts and exact McNemar p.

The-Loop v3: answers by the strict ANSWER-line rule, re-read by the panel where
  there is none (plus 10% audit). The live parser's reading, which set the
  feedback, is compared with the scored answer to count feedback errors per arm. Recovery = correct on previously-wrong items; each
  item has 3 sessions per arm, so per item we take the mean over sessions and
  compare arms with a paired sign-flip permutation test over items.

    python scripts/score_v2_review.py [model ...]   -> results_v2_metrics/review/<model>.json
"""
import json

PREC = 9   # stored precision; papers round once at display (round-half-up)
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tasks" / "t6_error_detection"))
import stats  # noqa: E402
import v2run  # noqa: E402
from revised_answer_audit import JUDGE_PROMPT as REVISED_PROMPT, regex_v2  # noqa: E402
from v2score import (EXTRACT_PROMPT, Panel, audit_pick, confidence, majority, options_block,  # noqa: E402
                     parse_extract, strict_answer, wilson)

T6 = ROOT / "tasks" / "t6_error_detection" / "results_v2"
C2 = ROOT / "tasks" / "t6_control2" / "results_v2"
LOOP = ROOT / "tasks" / "t8_loop" / "results_v2"
OUT = ROOT / "results_v2_metrics" / "review"
VERDICT = re.compile(r"VERDICT\W{0,20}?\b(CORRECT|ERROR)\b", re.I)
CONF = re.compile(r"CONFIDENCE\W{0,12}?(-?\d+(?:\.\d+)?)", re.I)


def latest(path):
    recs = {}
    for line in path.open():
        r = json.loads(line)
        recs[(r["unit"], r["condition"], r["repeat"])] = r
    return list(recs.values())


def parse_revised_votes(raws, letters):
    out = []
    for v in raws:
        if v is None:
            out.append("FAILED")
            continue
        tok = [t for t in re.findall(r"\b(SAME|NONE|[A-J])\b", v.upper()) if t in ("SAME", "NONE") or t in letters]
        out.append(tok[-1] if tok else "UNPARSED")
    return out


def mcnemar(b, c):
    return stats.mcnemar_exact(b, c)


def signflip(diffs):
    """Two-sided paired permutation p for mean(diffs) != 0 (scipy)."""
    return stats.signflip_p(diffs)


def self_review(model, bank, panel):
    scored = json.loads((T6 / "phase1_scored" / f"{model}.json").read_text())
    recs = [r for r in latest(T6 / f"{model}.jsonl") if r["repeat"] == 0]
    rows, prompts = [], {}
    for r in recs:
        s = scored.get(r["unit"])
        if r["unit"] in v2run.BAD_GT or s is None or s["answer"] is None:
            continue
        q = bank[r["unit"]]
        letters = sorted(q["choices"])
        p2 = (r.get("phase2") or {})
        text = p2.get("text") or ""
        verdicts = [v.upper() for v in VERDICT.findall(text)]
        confs = [c for c in (confidence(x) for x in CONF.findall(text)) if c is not None]
        row = {"id": r["unit"], "gt": s["gt"], "p1": s["answer"], "p1_correct": s["correct"], "p2_status": p2.get("status"),
               "n_verdicts": len(verdicts), "flagged": ("ERROR" in verdicts) if verdicts else None,
               "mean_conf": round(sum(confs) / len(confs), PREC) if confs else None,
               "revised_regex": regex_v2(text, letters) if text else "NONE", "revised_votes": None}
        if text and (not s["correct"] or row["flagged"] or audit_pick("sr2", model, r["unit"])):
            prompts[r["unit"]] = REVISED_PROMPT.format(letters=", ".join(letters), review=text[-4000:])
        rows.append(row)
    raw = panel.run(list(prompts.values()))
    for row in rows:
        rev = row["revised_regex"]
        if row["id"] in prompts:
            votes = parse_revised_votes(raw[prompts[row["id"]]], sorted(bank[row["id"]]["choices"]))
            row["revised_votes"] = votes
            top = majority(votes)
            if top not in (None, "FAILED", "UNPARSED"):
                rev = top
        row["revised"] = rev
        row["final"] = row["p1"] if rev in ("SAME", "NONE") else rev
        row["final_correct"] = row["final"] == row["gt"]
    errs = [x for x in rows if not x["p1_correct"]]
    errs_p = [x for x in errs if x["flagged"] is not None]
    corr_p = [x for x in rows if x["p1_correct"] and x["flagged"] is not None]
    flag = sum(x["flagged"] for x in errs_p)
    blind = [x["mean_conf"] for x in errs_p if not x["flagged"] and x["mean_conf"] is not None]
    tapp = [x["mean_conf"] for x in corr_p if not x["flagged"] and x["mean_conf"] is not None]
    m = {"n_answered": len(rows), "accuracy": round(100 * sum(x["p1_correct"] for x in rows) / len(rows), PREC),
         "n_errors": len(errs), "review_unparsed": sum(x["flagged"] is None for x in rows),
         "flag_rate": round(100 * flag / len(errs_p), PREC) if errs_p else None,
         "flag_ci": [round(v) for v in wilson(flag, len(errs_p))] if errs_p else None,
         "false_criticism": round(100 * sum(x["flagged"] for x in corr_p) / len(corr_p), PREC) if corr_p else None,
         "fix_rate": round(100 * sum(x["final_correct"] for x in errs) / len(errs), PREC) if errs else None,
         "n_fixed": sum(x["final_correct"] for x in errs),
         "conf_blind_spot": round(sum(blind) / len(blind), PREC) if blind else None,
         "conf_true_approve": round(sum(tapp) / len(tapp), PREC) if tapp else None,
         "revised_judge_disagree": sum(1 for x in rows if x["revised_votes"] and x["revised"] != x["revised_regex"])}
    return m, rows


def control2(model, bank, panel, sr_rows):
    path = C2 / f"{model}.jsonl"
    if not path.exists():
        return None
    recs = latest(path)
    prompts, rows = {}, []
    for r in recs:
        q = bank[r["unit"]]
        letters = sorted(q["choices"])
        a = strict_answer(r.get("text"), letters)
        rows.append({"id": r["unit"], "status": r["status"], "answer": a})
        if r.get("text") and (a is None or audit_pick("c2", model, r["unit"])):
            prompts[r["unit"]] = EXTRACT_PROMPT.format(options=options_block(q["choices"]), letters=", ".join(letters),
                                                       response=r["text"][-4000:])
    raw = panel.run(list(prompts.values()))
    fix = {x["id"]: x for x in sr_rows}
    b = c = both = neither = 0
    for row in rows:
        if row["id"] in prompts:
            top = majority([parse_extract(v, sorted(bank[row["id"]]["choices"])) if v else "FAILED" for v in raw[prompts[row["id"]]]])
            if top not in (None, "FAILED", "UNPARSED"):
                row["answer"] = None if top == "NONE" else top
        gt = fix[row["id"]]["gt"] if row["id"] in fix else None
        row["correct"] = row["answer"] is not None and row["answer"] == gt
        fx = fix.get(row["id"], {}).get("final_correct", False)
        both += fx and row["correct"]
        neither += not fx and not row["correct"]
        b += fx and not row["correct"]
        c += row["correct"] and not fx
    n = len(rows)
    return {"n": n, "fresh_correct": sum(r["correct"] for r in rows), "fix_correct": both + b,
            "fix_only": b, "fresh_only": c, "both": both, "neither": neither, "mcnemar_p": round(mcnemar(b, c), 3),
            "fresh_unanswered": sum(r["answer"] is None for r in rows)}


def loop(model, bank, panel):
    path = LOOP / f"{model}.jsonl"
    if not path.exists():
        return None, None
    recs = [r for r in latest(path) if r["status"] in ("ok", "truncated")]
    prompts, rows = {}, []
    for r in recs:
        arm, rep = r["unit"], r["repeat"]
        for h in r["history"]:
            q = bank[h["id"]]
            letters = sorted(q["choices"])
            strict = strict_answer(h["answer_response"], letters)
            row = {"arm": arm, "repeat": rep, "id": h["id"], "gt": h["gt"], "pw": h["previously_wrong"],
                   "online": h["model_answer"], "answer": strict, "found_error": h["found_error"],
                   "verdict": h["verdict"]}
            key = (arm, rep, h["id"])
            if strict is None or audit_pick("loop", model, *key):
                prompts[key] = EXTRACT_PROMPT.format(options=options_block(q["choices"]), letters=", ".join(letters),
                                                     response=(h["answer_response"] or "")[-4000:])
            rows.append(row)
    raw = panel.run(list(prompts.values()))
    audited = audit_disagree = 0
    for row in rows:
        key = (row["arm"], row["repeat"], row["id"])
        if key in prompts:
            top = majority([parse_extract(v, sorted(bank[row["id"]]["choices"])) if v else "FAILED" for v in raw[prompts[key]]])
            if top not in (None, "FAILED", "UNPARSED"):
                judged = None if top == "NONE" else top
                if row["answer"] is not None:
                    audited += 1
                    audit_disagree += judged != row["answer"]
                row["answer"] = judged
        row["correct"] = row["answer"] == row["gt"]
    per = defaultdict(lambda: {"live": [], "control": []})
    for row in rows:
        if row["pw"]:
            per[row["id"]][row["arm"]].append(row["correct"])
    items = {i: d for i, d in per.items() if d["live"] and d["control"]}
    diffs = [sum(d["live"]) / len(d["live"]) - sum(d["control"]) / len(d["control"]) for d in items.values()]
    pw_rows = lambda arm: [r for r in rows if r["pw"] and r["arm"] == arm]
    pc_rows = lambda arm: [r for r in rows if not r["pw"] and r["arm"] == arm]
    m = {"n_items_pw": len(items), "sessions": {a: len({r["repeat"] for r in rows if r["arm"] == a}) for a in ("live", "control")},
         "recovery_live": round(100 * sum(r["correct"] for r in pw_rows("live")) / max(len(pw_rows("live")), 1), PREC),
         "recovery_control": round(100 * sum(r["correct"] for r in pw_rows("control")) / max(len(pw_rows("control")), 1), PREC),
         "prev_correct_acc_live": round(100 * sum(r["correct"] for r in pc_rows("live")) / max(len(pc_rows("live")), 1), PREC),
         "prev_correct_acc_control": round(100 * sum(r["correct"] for r in pc_rows("control")) / max(len(pc_rows("control")), 1), PREC),
         "diff_pp": round(100 * sum(diffs) / len(diffs), PREC) if diffs else None,
         "signflip_p": round(signflip(diffs), 3) if diffs else None,
         "no_answer_online": sum(r["online"] is None for r in rows), "no_answer_after_judge": sum(r["answer"] is None for r in rows),
         # feedback errors: the live parser's answer (which set the feedback) differs from the scored answer
         "feedback_misread": {a: sum(r["online"] != r["answer"] for r in rows if r["arm"] == a) for a in ("live", "control")},
         "strict_audit": {"audited": audited, "judge_disagreed": audit_disagree},
         # in-session review on previously-wrong items still answered wrongly: [ERROR verdicts, items]
         "flag_still_wrong": {a: [sum(r["verdict"] == "ERROR" for r in pw_rows(a) if not r["correct"]),
                                  sum(not r["correct"] for r in pw_rows(a))] for a in ("live", "control")},
         # pooled counts behind recovery_*: [correct, item-sessions]
         "recovery_counts": {a: [sum(r["correct"] for r in pw_rows(a)), len(pw_rows(a))] for a in ("live", "control")}}
    return m, {"diffs": diffs, "items": {i: {a: [sum(d[a]), len(d[a])] for a in ("live", "control")} for i, d in items.items()}}


def main():
    models = sys.argv[1:] or sorted(p.stem for p in (T6 / "phase1_scored").glob("*.json"))
    panel = Panel(ROOT / "results_v2_metrics" / "judge_cache_review.jsonl")
    OUT.mkdir(parents=True, exist_ok=True)
    for model in models:
        bank = v2run.bank(include_removed=True, version=v2run.bank_version(model))
        sr, sr_rows = self_review(model, bank, panel)
        c2 = control2(model, bank, panel, sr_rows)
        lp, lp_raw = loop(model, bank, panel)
        (OUT / f"{model}.json").write_text(json.dumps({"model": model, "self_review": sr, "control2": c2, "loop": lp,
                                                       "loop_item_diffs": (lp_raw or {}).get("diffs"),
                                                       "loop_items": (lp_raw or {}).get("items"),
                                                       "self_review_rows": [{k: x[k] for k in ("id", "gt", "p1_correct", "flagged", "mean_conf", "final_correct", "p2_status")}
                                                                            for x in sr_rows]}, indent=1))
        print(f"{model}\n  Self-Review: {sr}\n  Control 2: {c2}\n  Loop: {lp}", flush=True)


if __name__ == "__main__":
    main()
