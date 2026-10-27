"""
Paper metrics for the v2 question-answering tasks, from the scored rows written
by score_v2_qa.py. Main run only (repeat 0); every exclusion is counted.

Wagering (per condition): accuracy, ECE, wagering rationality, payout, payout gap
  (src/scoring.t2_score_all). Rows need a scorable answer, a 0-100 confidence and
  a valid wager.
Deferral (per condition): deferral rate by block and its LOW->HIGH change,
  clarification precision/recall, total points, and, from the forced-answer
  counterfactual, accuracy on the items the model declined in A.
Consistency: accuracy per framing on the items scorable in all four, the paired
  explanation effect, the direct-condition confidence gap (correct minus wrong),
  and on adversarial_wrong the share answering the planted decoy.

    python scripts/metrics_v2_qa.py [model ...]     -> results_v2_metrics/qa/<model>.json
"""
import json

PREC = 9   # stored precision; papers round once at display (round-half-up)
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scoring import (T5_PAYOFFS, clarification_precision_recall, deferral_rates_by_block,  # noqa: E402
                     t2_score_all)

OUT = ROOT / "results_v2_metrics" / "qa"


def rows(task, model):
    p = ROOT / "tasks" / task / "results_v2" / "scored" / f"{model}.jsonl"
    return [json.loads(l) for l in p.open()] if p.exists() else []


def pct(a, b):
    return round(100 * a / b, PREC) if b else None


def wagering(model):
    out = {}
    rs = rows("t2_confidence_wagering", model)
    for cond in ("A", "B"):
        rr = [r for r in rs if r["condition"] == cond and r["repeat"] == 0]
        why = Counter()
        use = []
        for r in rr:
            if r["status"] != "ok":
                why[f"status_{r['status']}"] += 1
            elif r["correct"] is None:
                why["no_scorable_answer"] += 1
            elif r["confidence"] is None:
                why["invalid_confidence"] += 1
            elif r["wager"] is None:
                why["invalid_wager"] += 1
            else:
                use.append(r)
        m = t2_score_all([r["confidence"] for r in use], [r["correct"] for r in use], [r["wager"] for r in use])
        out[cond] = {"n_items": len(rr), "n_scored": len(use), "excluded": dict(why),
                     "accuracy": pct(sum(r["correct"] for r in use), len(use)),
                     "ece": m["ece"], "wr": round(100 * m["wagering_rationality"], PREC),
                     "brier": round(sum((r["confidence"] / 100 - r["correct"]) ** 2 for r in use) / len(use), 4) if use else None,
                     "payout": m["total_payout"], "payout_gap": m["payout_gap"],
                     "mean_conf_correct": _mean([r["confidence"] for r in use if r["correct"]]),
                     "mean_conf_wrong": _mean([r["confidence"] for r in use if not r["correct"]]),
                     "wagers": dict(Counter(r["wager"] for r in use))}
    return out


def _mean(xs):
    return round(sum(xs) / len(xs), PREC) if xs else None


def deferral(model):
    out = {}
    rs = rows("t5_selective_deferral", model)
    forced = {r["unit"]: r for r in rs if r["condition"] == "forced"}
    for cond in ("A", "B"):
        rr = [r for r in rs if r["condition"] == cond and r["repeat"] == 0]
        valid = [r for r in rr if r["decision"] is not None]
        why = Counter(f"status_{r['status']}" if r["status"] != "ok" else "invalid_decision"
                      for r in rr if r["decision"] is None)
        well = [r for r in valid if not r["is_underspecified"]]
        rates = deferral_rates_by_block([r["decision"] for r in well], [r["block"] for r in well])
        clar = clarification_precision_recall([r["decision"] for r in valid], [r["is_underspecified"] for r in valid])
        pts, unscorable = Counter(), 0
        for r in well:
            if r["decision"] != "ATTEMPT":
                continue
            if r["correct"] is None:
                unscorable += 1
                continue
            pay = T5_PAYOFFS[r["block"]]
            pts[r["block"]] += pay["correct"] if r["correct"] else pay["wrong"]
        att = [r for r in well if r["decision"] == "ATTEMPT" and r["correct"] is not None]
        o = {"n_items": len(rr), "n_valid": len(valid), "excluded": dict(why),
             "decisions_wellspec": dict(Counter(r["decision"] for r in well)),
             "defer_rate": {b: round(100 * v, PREC) for b, v in rates.items()},
             "defer_change_low_to_high_pp": round(100 * (rates["high"] - rates["low"]), PREC),
             "clar_recall": round(100 * clar["recall"], PREC), "clar_precision": round(100 * clar["precision"], PREC),
             "points": {**pts, "total": sum(pts.values())}, "attempted_unscorable": unscorable,
             "accuracy_attempted": pct(sum(r["correct"] for r in att), len(att))}
        if cond == "A":
            dec = [r for r in well if r["decision"] in ("DEFER", "CLARIFY")]
            f = [forced[r["unit"]] for r in dec if r["unit"] in forced and forced[r["unit"]]["correct"] is not None]
            o["counterfactual"] = {"n_declined": len(dec), "n_forced_scored": len(f),
                                   "accuracy_declined": pct(sum(x["correct"] for x in f), len(f))}
        out[cond] = o
    return out


def consistency(model):
    rs = [r for r in rows("t7_consistency", model) if r["repeat"] == 0]
    by = defaultdict(dict)
    for r in rs:
        by[r["unit"]][r["condition"]] = r
    frames = ("direct", "explanation", "adversarial_correct", "adversarial_wrong")
    full = {u: d for u, d in by.items() if all(f in d and d[f]["correct"] is not None for f in frames)}
    acc = {f: pct(sum(d[f]["correct"] for d in full.values()), len(full)) for f in frames}
    direct = [r for r in rs if r["condition"] == "direct" and r["correct"] is not None and r["confidence"] is not None]
    cg, cw = _mean([r["confidence"] for r in direct if r["correct"]]), _mean([r["confidence"] for r in direct if not r["correct"]])
    decoys = _decoys(model)
    aw = [d["adversarial_wrong"] for d in full.values()]
    return {"n_items": len(by), "n_paired": len(full), "accuracy": acc,
            "explanation_effect_pp": round(acc["explanation"] - acc["direct"], PREC) if full else None,
            "conf_correct": cg, "conf_wrong": cw, "conf_gap": round(cg - cw, PREC) if cg is not None and cw is not None else None,
            "n_wrong_direct": sum(not r["correct"] for r in direct),
            "decoy_rate_adversarial_wrong": pct(sum(r["answer"] == decoys.get(r["unit"]) for r in aw), len(aw))}


def _decoys(model):
    sys.path.insert(0, str(ROOT / "tasks" / "t7_consistency"))
    import v2run
    from run_v2 import decoy
    bank = v2run.bank(include_removed=True, version=v2run.bank_version(model))
    return {u: decoy(bank[u]) for u in v2run.selfreview_pool()}


def main():
    models = sys.argv[1:] or sorted(p.stem for p in (ROOT / "tasks" / "t2_confidence_wagering" / "results_v2" / "scored").glob("*.jsonl"))
    OUT.mkdir(parents=True, exist_ok=True)
    for m in models:
        res = {"model": m, "wagering": wagering(m), "deferral": deferral(m), "consistency": consistency(m)}
        (OUT / f"{m}.json").write_text(json.dumps(res, indent=1))
        w, d, c = res["wagering"], res["deferral"], res["consistency"]
        print(f"{m}\n  Wagering A: acc {w['A']['accuracy']} ECE {w['A']['ece']} WR {w['A']['wr']} gap {w['A']['payout_gap']} "
              f"(n {w['A']['n_scored']}/{w['A']['n_items']}, excl {w['A']['excluded']})\n"
              f"  Wagering B: WR {w['B']['wr']} ECE {w['B']['ece']} (n {w['B']['n_scored']})\n"
              f"  Deferral A: defer {d['A']['defer_rate']} change {d['A']['defer_change_low_to_high_pp']}pp clarR {d['A']['clar_recall']} "
              f"clarP {d['A']['clar_precision']} pts {d['A']['points']['total']} cf {d['A']['counterfactual']}\n"
              f"  Deferral B: change {d['B']['defer_change_low_to_high_pp']}pp clarR {d['B']['clar_recall']} pts {d['B']['points']['total']}\n"
              f"  Consistency: acc {c['accuracy']} expl {c['explanation_effect_pp']}pp conf gap {c['conf_gap']} "
              f"(wrong n {c['n_wrong_direct']}) decoy {c['decoy_rate_adversarial_wrong']}% (paired {c['n_paired']}/{c['n_items']})")


if __name__ == "__main__":
    main()
