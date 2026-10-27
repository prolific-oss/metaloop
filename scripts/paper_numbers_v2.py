"""
Every model number in the corrected paper, from the v2 results (October 2026).

Reads the scored v2 records and results_v2_metrics/*, computes the statistics
the paper reports that no other script does (paired explanation and peer-cue
effects with item-bootstrap CIs, pooled Loop and Control 2 tests with Holm
adjustment, Self-Review answer changes, Strategy-Switching per task type,
Source-Monitoring on answered statements), and writes
  results_v2_metrics/paper_numbers.json     all numbers, by section
  <paper>/tables_v2/*.tex                    the per-model tables, \\input by the paper
Main run (repeat 0), core and extension panels; the ablation and 32k
sensitivity runs are reported from their own scripts' outputs.

    python scripts/paper_numbers_v2.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "results_v2_metrics"
T = ROOT / "tasks"
PAPER = ROOT.parents[1] / "papers" / "neurips-2026-metaloop"
TAB = PAPER / "tables_v2"
sys.path.insert(0, str(ROOT / "src"))
import stats  # noqa: E402
from scoring import T5_PAYOFFS, WAGER_PAYOFFS, optimal_wager  # noqa: E402
from stats import wilson  # noqa: E402

B = 2000
ORIGINAL = ["sonnet-4", "sonnet-4.6", "opus-4.6", "gpt-4.1", "gpt-5.4", "gemini-2.5-pro", "gemini-3.1-pro",
            "gemma-4-31b", "deepseek-v3.2", "mistral-large"]
NEW = ["opus-5.5", "sonnet-5.5", "gpt-6.1-sol", "gpt-6-luna", "gemini-3.8-flash", "deepseek-v4-pro",
       "mistral-large-4", "qwen-3.8-max", "grok-4.7"]
MODELS = ORIGINAL + NEW
NAME = {"sonnet-4": "Sonnet 4", "sonnet-4.6": "Sonnet 4.6", "opus-4.6": "Opus 4.6", "gpt-4.1": "GPT-4.1",
        "gpt-5.4": "GPT-5.4", "gemini-2.5-pro": "Gemini 2.5 Pro", "gemini-3.1-pro": "Gemini 3.1 Pro",
        "gemma-4-31b": "Gemma 4 31B", "deepseek-v3.2": "DeepSeek V3.2", "mistral-large": "Mistral Large",
        "opus-5.5": "Opus 5.5", "sonnet-5.5": "Sonnet 5.5", "gpt-6.1-sol": "GPT-6.1 Sol", "gpt-6-luna": "GPT-6 Luna",
        "gemini-3.8-flash": "Gemini 3.8 Flash", "deepseek-v4-pro": "DeepSeek V4 Pro",
        "mistral-large-4": "Mistral Large 4", "qwen-3.8-max": "Qwen 3.8 Max", "grok-4.7": "Grok 4.7"}


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def scored(task, model):
    p = T / task / "results_v2" / "scored" / f"{model}.jsonl"
    return [r for r in map(json.loads, p.open()) if r.get("repeat", 0) == 0] if p.exists() else []


def r1(x):
    return None if x is None else round(float(x), 1)


holm, mcnemar, signflip = stats.holm, stats.mcnemar_exact, stats.signflip_p


def boot_paired(a, b):
    """Mean of a - b (0/1 arrays paired by item) in pp, with a 95% item-bootstrap CI (scipy)."""
    m, (lo, hi) = stats.paired_mean_ci(a, b, n_resamples=B)
    return r1(100 * m), [r1(100 * lo), r1(100 * hi)]


# ---------------------------------------------------------------------------

def wagering(m):
    w = load(M / "qa" / f"{m}.json")["wagering"]
    a, b = w["A"], w["B"]
    acc = a["accuracy"] / 100
    mean_conf = acc * a["mean_conf_correct"] + (1 - acc) * a["mean_conf_wrong"]
    return {"n": a["n_scored"], "n_items": a["n_items"], "acc": a["accuracy"], "ece": a["ece"], "brier": a["brier"],
            "wr_a": a["wr"], "wr_b": b["wr"], "n_b": b["n_scored"], "payout": a["payout"], "gap": a["payout_gap"],
            "gap_per_item": r1(a["payout_gap"] / a["n_scored"]), "mean_conf": r1(mean_conf),
            "overconf": r1(mean_conf - a["accuracy"]), "wagers": a["wagers"], "conf_gap": r1(a["mean_conf_correct"] - a["mean_conf_wrong"]),
            **wager_regret(m)}


RISK = {"SAFE": 0, "MEDIUM": 1, "BOLD": 2}


def expected_points(wager, conf):
    """Expected points of a wager if the stated confidence were the probability of being correct."""
    p = conf / 100
    return p * WAGER_PAYOFFS[wager]["correct"] + (1 - p) * WAGER_PAYOFFS[wager]["wrong"]


def wager_regret(m):
    """Condition A, the rows metrics_v2_qa scores: expected regret per item under the model's own stated
    confidence (best expected points minus the chosen wager's), and how many suboptimal wagers are more
    cautious than the optimal one."""
    use = [r for r in scored("t2_confidence_wagering", m) if r["condition"] == "A" and r["status"] == "ok"
           and r["correct"] is not None and r["confidence"] is not None and r["wager"] is not None]
    reg = [expected_points(optimal_wager(r["confidence"]), r["confidence"]) - expected_points(r["wager"], r["confidence"])
           for r in use]
    sub = [r for r in use if r["wager"] != optimal_wager(r["confidence"])]
    cautious = sum(RISK[r["wager"]] < RISK[optimal_wager(r["confidence"])] for r in sub)
    # less risky than optimal, by stated-confidence band where BOLD is optimal: [cautious, items]
    bands = {"66-79": (66, 79), "80-89": (80, 89), "90-100": (90, 100)}
    by_band = {k: [sum(lo <= r["confidence"] <= hi and RISK[r["wager"]] < 2 for r in use),
                   sum(lo <= r["confidence"] <= hi for r in use)] for k, (lo, hi) in bands.items()}
    return {"n_valid_conf": len(use), "regret_per_item": round(float(np.mean(reg)), 2), "n_suboptimal": len(sub),
            "n_cautious": cautious, "cautious_pct": r1(100 * cautious / len(sub)) if sub else None,
            "cautious_by_band": by_band, "points_per_item": round(sum(WAGER_PAYOFFS[r["wager"]]["correct" if r["correct"] else "wrong"]
                                                                    for r in use) / len(use), 2)}


def deferral(m):
    d = load(M / "qa" / f"{m}.json")["deferral"]
    a, b = d["A"], d["B"]
    k = round(a["clar_recall"] * 138 / 100)
    cf = a.get("counterfactual") or {}
    out = {"defer": a["defer_rate"], "delta": a["defer_change_low_to_high_pp"], "att_acc": a["accuracy_attempted"],
           "clar_p": a["clar_precision"], "clar_r": a["clar_recall"], "clar_r_ci": [round(x) for x in wilson(k, 138)],
           "ev": a["points"]["total"], "ev_b": b["points"]["total"], "clar_r_b": b["clar_recall"],
           "false_clar": r1(100 * a["decisions_wellspec"].get("CLARIFY", 0) / sum(a["decisions_wellspec"].values())),
           "n_declined": cf.get("n_declined"), "forced_acc": cf.get("accuracy_declined")}
    # policy adherence on well-specified items: attempting is optimal when the stated confidence exceeds
    # penalty / (reward + penalty) for the item's block (50%, 75%, 90.9%), as in Wagering's derivation
    for cond in ("A", "B"):
        rs = [r for r in scored("t5_selective_deferral", m) if r["condition"] == cond and not r["is_underspecified"]
              and r["decision"] in ("ATTEMPT", "DEFER", "CLARIFY") and r["confidence"] is not None]
        thr = {b: -v["wrong"] / (v["correct"] - v["wrong"]) for b, v in T5_PAYOFFS.items()}
        # at exactly the threshold (50 or 75) attempting and declining have equal expected points: both optimal
        ok_ = [r["confidence"] / 100 == thr[r["block"]] or (r["decision"] == "ATTEMPT") == (r["confidence"] / 100 > thr[r["block"]])
               for r in rs]
        under = sum(r["decision"] != "ATTEMPT" and r["confidence"] / 100 > thr[r["block"]] for r in rs)
        out[f"attempted_below_threshold_{cond}"] = sum(r["decision"] == "ATTEMPT" and r["confidence"] / 100 < thr[r["block"]] for r in rs)
        out[f"adherence_{cond}"] = r1(100 * sum(ok_) / len(rs)) if rs else None
        out[f"declined_above_threshold_{cond}"] = under
        out[f"n_adherence_{cond}"] = len(rs)
    if cf.get("n_declined"):
        kk = round(cf["accuracy_declined"] * cf["n_forced_scored"] / 100)
        out["forced_correct"] = kk
        out["n_forced_scored"] = cf["n_forced_scored"]
        out["forced_ci"] = [round(x) for x in wilson(kk, cf["n_forced_scored"])]
        out["cf_gap"] = r1(a["accuracy_attempted"] - cf["accuracy_declined"])
    return out


def consistency(m):
    by = defaultdict(dict)
    for r in scored("t7_consistency", m):
        by[r["unit"]][r["condition"]] = r
    ans = lambda r: r is not None and r.get("answer") is not None  # noqa: E731
    out = {"n_items": len(by)}
    for cond in ("explanation", "adversarial_correct", "adversarial_wrong"):
        pairs = [(d[cond]["correct"], d["direct"]["correct"]) for d in by.values()
                 if ans(d.get("direct")) and ans(d.get(cond))]
        eff, ci = boot_paired([p[0] for p in pairs], [p[1] for p in pairs])
        # discordant pairs: right only with the manipulation, right only without it
        disc = [sum(a and not b for a, b in pairs), sum(b and not a for a, b in pairs)]
        # sensitivity: every item with both records, an unanswered response counted as wrong
        both = [(bool(ans(d[cond]) and d[cond]["correct"]), bool(ans(d["direct"]) and d["direct"]["correct"]))
                for d in by.values() if d.get("direct") is not None and d.get(cond) is not None]
        eff_all, ci_all = boot_paired([p[0] for p in both], [p[1] for p in both])
        # with no discordant pair the bootstrap is degenerate; |effect| <= discordance rate, so the Wilson
        # upper bound of 0/n discordant pairs bounds the effect
        if not sum(disc):
            ci = [r1(-wilson(0, len(pairs))[1]), r1(wilson(0, len(pairs))[1])]
        out[cond] = {"n": len(pairs), "pp": eff, "ci": ci, "discordant": disc, "ci_method": "wilson_bound" if not sum(disc) else "bootstrap",
                     "all_items": {"n": len(both), "pp": eff_all, "ci": ci_all}}
    dr = [d["direct"] for d in by.values() if "direct" in d]
    er = [d["explanation"] for d in by.values() if "explanation" in d]
    out["direct_acc"] = r1(100 * sum(r["correct"] for r in dr if ans(r)) / max(sum(ans(r) for r in dr), 1))
    out["answered"] = [r1(100 * sum(ans(r) for r in dr) / len(dr)), r1(100 * sum(ans(r) for r in er) / len(er))]
    q = load(M / "qa" / f"{m}.json")["consistency"]
    out["conf_gap_direct"] = q["conf_gap"]
    out["decoy_rate"] = q["decoy_rate_adversarial_wrong"]
    return out


def self_review(m):
    rv = load(M / "review" / f"{m}.json")
    s, c2, rows = rv["self_review"], rv["control2"], rv["self_review_rows"]
    cw = sum(1 for x in rows if x["p1_correct"] and not x["final_correct"])
    wc = sum(1 for x in rows if not x["p1_correct"] and x["final_correct"])
    n_flagged = sum(1 for x in rows if not x["p1_correct"] and x["flagged"])
    n_flag_den = sum(1 for x in rows if not x["p1_correct"] and x["flagged"] is not None)  # errors with a parsed review
    flagged_fixed = sum(1 for x in rows if not x["p1_correct"] and x["flagged"] and x["final_correct"])
    acc_all = r1(100 * sum(x["p1_correct"] for x in rows) / 153)  # unanswered of the 153 counted as wrong
    return {"items": s["n_answered"], "n_errors": s["n_errors"], "acc": s["accuracy"], "acc_all153": acc_all, "flag": s["flag_rate"],
            "flag_ci": s["flag_ci"], "fix": s["fix_rate"], "n_fixed": s["n_fixed"], "fc": s["false_criticism"],
            "c_to_w": cw, "w_to_c": wc, "net_pp": r1(100 * (wc - cw) / s["n_answered"]),
            "n_flagged": n_flagged, "n_flag_den": n_flag_den, "flagged_fixed": flagged_fixed, "review_unparsed": s["review_unparsed"],
            "conf_bs": s["conf_blind_spot"], "conf_ta": s["conf_true_approve"],
            "c2": None if not c2 else {k: c2[k] for k in ("n", "fix_correct", "fresh_correct", "fix_only", "fresh_only", "mcnemar_p")}}


def loop(m):
    rv = load(M / "review" / f"{m}.json")
    lp = rv.get("loop")
    if not lp or not lp.get("n_items_pw"):
        return None
    return {"items": lp["n_items_pw"], "rec_live": lp["recovery_live"], "rec_control": lp["recovery_control"],
            "diff_pp": lp["diff_pp"], "p": lp["signflip_p"], "counts": lp["recovery_counts"],
            "flag": lp["flag_still_wrong"], "prev_correct": [lp["prev_correct_acc_live"], lp["prev_correct_acc_control"]],
            "item_rows": rv.get("loop_items") or {}, "feedback_misread": lp["feedback_misread"]["live"]}


def switching(m):
    rows = scored("t4_strategy_switching", m)
    tt = Counter(r["task_type"] for r in rows)

    def block(rs):
        n = len(rs)
        sw = sum(r["switched"] for r in rs)
        sbw = sum(r["switched"] and not r["correct"] for r in rs)
        cor = sum(r["correct"] for r in rs)
        return {"n": n, "switch": r1(100 * sw / n), "sbw": r1(100 * sbw / n), "sbw_ci": [round(x) for x in wilson(sbw, n)],
                "conv": [sum(r["switched"] and r["correct"] for r in rs), sw], "acc": r1(100 * cor / n),
                "no_switch_correct": sum(r["correct"] and not r["switched"] for r in rs),
                "usable": sum(bool(r.get("judged")) for r in rs)}
    full = block(rows)
    excl = block([r for r in rows if r["task_type"] != "cipher"])
    cipher = Counter(r["status"] for r in rows if r["task_type"] == "cipher")
    return {"full": full, "excl_cipher": excl, "cipher_status": dict(cipher),
            "cipher_correct": sum(r["correct"] for r in rows if r["task_type"] == "cipher"), "task_types": dict(tt),
            "rows": [(r["task_type"], r["switched"], r["correct"], bool(r.get("judged"))) for r in rows]}


def source_monitoring(m):
    rows = scored("t3_source_monitoring", m)
    got = [r for r in rows if r["pred"]]
    fab = [r for r in got if r["gold"] == "FABRICATED"]
    inf = [r for r in got if r["gold"] == "INFERRED"]
    called = [r for r in got if r["pred"] == "FABRICATED"]
    det = sum(r["pred"] == "FABRICATED" for r in fab)
    return {"n": len(rows), "answered": len(got), "lost_pct": r1(100 * (len(rows) - len(got)) / len(rows)),
            "acc": r1(100 * sum(r["pred"] == r["gold"] for r in got) / len(got)),
            "confab": r1(100 * det / len(fab)), "confab_ci": [round(x) for x in wilson(det, len(fab))], "n_fab": len(fab),
            "inferred": r1(100 * sum(r["pred"] == "INFERRED" for r in inf) / len(inf)),
            "fab_prec": r1(100 * sum(r["gold"] == "FABRICATED" for r in called) / max(len(called), 1)),
            # statements share a passage, so the per-passage spread replaces a statement-level interval
            "confab_by_passage": {u: [sum(r["pred"] == "FABRICATED" for r in fab if r["unit"] == u),
                                      sum(r["unit"] == u for r in fab)] for u in sorted({r["unit"] for r in fab})}}


# ---------------------------------------------------------------------------

def spearman_model_ci(a, b):
    """95% interval of Spearman's rho, resampling models (scipy via stats.cluster_bootstrap)."""
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    _, lo, hi = stats.cluster_bootstrap(len(a), lambda i: spearmanr(a[i], b[i]).statistic, n_resamples=5000)
    return [round(float(lo), 2), round(float(hi), 2)]


def pooled(per):
    out = {}
    # Control 2
    c2 = {m: v["self_review"]["c2"] for m, v in per.items() if v["self_review"]["c2"]}
    ps = [c2[m]["mcnemar_p"] for m in c2]
    for m, h in zip(c2, holm(ps)):
        per[m]["self_review"]["c2"]["p_holm"] = round(h, 3)
    fo, fr = sum(v["fix_only"] for v in c2.values()), sum(v["fresh_only"] for v in c2.values())
    out["control2"] = {"n_models": len(c2), "errors": sum(v["n"] for v in c2.values()),
                       "fix": sum(v["fix_correct"] for v in c2.values()), "fresh": sum(v["fresh_correct"] for v in c2.values()),
                       "fix_only": fo, "fresh_only": fr,
                       # inference with models as the units: the per-model differences (fix - fresh) / n
                       "model_level_mean_pp": r1(100 * np.mean([(v["fix_correct"] - v["fresh_correct"]) / v["n"] for v in c2.values()])),
                       "model_level_signflip_p": round(signflip([(v["fix_correct"] - v["fresh_correct"]) / v["n"] for v in c2.values()]), 3),
                       "models_fresh_better": sum(v["fresh_correct"] > v["fix_correct"] for v in c2.values()),
                       "models_fix_better": sum(v["fix_correct"] > v["fresh_correct"] for v in c2.values()),
                       "diff_range_pp": [r1(min(100 * (v["fix_correct"] - v["fresh_correct"]) / v["n"] for v in c2.values())),
                                         r1(max(100 * (v["fix_correct"] - v["fresh_correct"]) / v["n"] for v in c2.values()))]}
    # Loop
    lp = {m: v["loop"] for m, v in per.items() if v["loop"]}
    ps = [lp[m]["p"] for m in lp]
    for m, h in zip(lp, holm(ps)):
        per[m]["loop"]["p_holm"] = round(h, 3)
    diffs, nl, cl, nc, cc = [], 0, 0, 0, 0
    for v in lp.values():
        for it in v["item_rows"].values():
            diffs.append(it["live"][0] / it["live"][1] - it["control"][0] / it["control"][1])
        nl += v["counts"]["live"][1]; cl += v["counts"]["live"][0]
        nc += v["counts"]["control"][1]; cc += v["counts"]["control"][0]
    fl = [sum(v["flag"]["live"][0] for v in lp.values()), sum(v["flag"]["live"][1] for v in lp.values())]
    fc = [sum(v["flag"]["control"][0] for v in lp.values()), sum(v["flag"]["control"][1] for v in lp.values())]
    out["loop"] = {"n_models": len(lp), "items": len(diffs), "rec_live": [cl, nl, r1(100 * cl / nl)],
                   "rec_control": [cc, nc, r1(100 * cc / nc)], "mean_item_diff_pp": r1(100 * np.mean(diffs)),
                   # inference with models as the units (sessions are conversations, so items are not independent)
                   "model_level_mean_pp": r1(np.mean([v["diff_pp"] for v in lp.values()])),
                   "model_level_signflip_p": round(signflip([v["diff_pp"] for v in lp.values()]), 3),
                   "models_live_lower": sum(v["diff_pp"] < 0 for v in lp.values()),
                   "models_live_higher": sum(v["diff_pp"] > 0 for v in lp.values()),
                   "flag_live": fl, "flag_control": fc,
                   "per_model_range_pp": [min(v["diff_pp"] for v in lp.values()), max(v["diff_pp"] for v in lp.values())],
                   "min_p_holm": min(per[m]["loop"]["p_holm"] for m in lp)}
    # Self-Review pooled
    sr = {m: v["self_review"] for m, v in per.items()}
    out["self_review"] = {"flagged": sum(v["n_flagged"] for v in sr.values()),
                          "flagged_fixed": sum(v["flagged_fixed"] for v in sr.values()),
                          "errors": sum(v["n_errors"] for v in sr.values()),
                          "net_range_pp": [min(v["net_pp"] for v in sr.values()), max(v["net_pp"] for v in sr.values())],
                          "flag_zero": [m for m, v in sr.items() if v["flag"] == 0]}
    # monitoring-control
    w = {m: v["wagering"] for m, v in per.items()}
    out["monitoring_control"] = {"both": [m for m, x in w.items() if x["ece"] < 0.10 and x["wr_a"] >= 90],
                                 "ece_only": [m for m, x in w.items() if x["ece"] < 0.10 and x["wr_a"] < 90],
                                 "wr_only": [m for m, x in w.items() if x["ece"] >= 0.10 and x["wr_a"] >= 90],
                                 "neither": [m for m, x in w.items() if x["ece"] >= 0.10 and x["wr_a"] < 90],
                                 "wr_b_min": min(x["wr_b"] for x in w.values()),
                                 "ece_wr_spearman": [round(v, 3) for v in stats.spearman([x["ece"] for x in w.values()], [x["wr_a"] for x in w.values()])],
                                 "ece_wr_spearman_ci95_models": spearman_model_ci([x["ece"] for x in w.values()], [x["wr_a"] for x in w.values()]),
                                 "n_valid_conf": sum(x["n_valid_conf"] for x in w.values()),
                                 "n_suboptimal": sum(x["n_suboptimal"] for x in w.values()),
                                 "n_cautious": sum(x["n_cautious"] for x in w.values()),
                                 "regret_range": [min(x["regret_per_item"] for x in w.values()), max(x["regret_per_item"] for x in w.values())],
                                 "cautious_by_band": {b: [sum(x["cautious_by_band"][b][0] for x in w.values()),
                                                          sum(x["cautious_by_band"][b][1] for x in w.values())] for b in ("66-79", "80-89", "90-100")},
                                 "overconf_range": [min(x["overconf"] for x in w.values()), max(x["overconf"] for x in w.values())]}
    # Deferral
    d = {m: v["deferral"] for m, v in per.items()}
    gaps = [x["cf_gap"] for x in d.values() if x.get("cf_gap") is not None]
    out["deferral"] = {"ev_b_up": [m for m, x in d.items() if x["ev_b"] > x["ev"]],
                       "ev_b_down": [m for m, x in d.items() if x["ev_b"] < x["ev"]],
                       "clar_r_change_max": max(abs(x["clar_r_b"] - x["clar_r"]) for x in d.values()),
                       "forced_range": [min(x["forced_acc"] for x in d.values() if x.get("forced_acc") is not None),
                                        max(x["forced_acc"] for x in d.values() if x.get("forced_acc") is not None)],
                       "att_range": [min(x["att_acc"] for x in d.values()), max(x["att_acc"] for x in d.values())],
                       "cf_gap_median": r1(np.median(gaps)), "n_cf": len(gaps),
                       "ev_positive": [m for m, x in d.items() if x["ev"] > 0],
                       "ev_per_item_range": [r1(min(x["ev"] for x in d.values()) / 592), r1(max(x["ev"] for x in d.values()) / 592)],
                       "clar_r_range": [min(x["clar_r"] for x in d.values()), max(x["clar_r"] for x in d.values())],
                       "adherence_A_range": [min(x["adherence_A"] for x in d.values()), max(x["adherence_A"] for x in d.values())],
                       "adherence_B_range": [min(x["adherence_B"] for x in d.values()), max(x["adherence_B"] for x in d.values())],
                       "adherence_B_minus_A": {m: r1(x["adherence_B"] - x["adherence_A"]) for m, x in d.items()},
                       "departures_A": {"attempted_below": sum(x["attempted_below_threshold_A"] for x in d.values()),
                                        "declined_above": sum(x["declined_above_threshold_A"] for x in d.values())}}
    # Consistency
    c = {m: v["consistency"] for m, v in per.items()}
    excl = lambda e: e["ci"][0] > 0 or e["ci"][1] < 0  # noqa: E731
    out["consistency"] = {"expl_ci_excludes_zero": [m for m, x in c.items() if excl(x["explanation"])],
                          "expl_all_items_ci_excludes_zero": [m for m, x in c.items() if excl(x["explanation"]["all_items"])],
                          "expl_all_items_range": [min(x["explanation"]["all_items"]["pp"] for x in c.values()),
                                                   max(x["explanation"]["all_items"]["pp"] for x in c.values())],
                          "expl_discordant_max": max(sum(x["explanation"]["discordant"]) for x in c.values()),
                          "expl_no_discordant": [m for m, x in c.items() if not sum(x["explanation"]["discordant"])],
                          "expl_range": [min(x["explanation"]["pp"] for x in c.values()), max(x["explanation"]["pp"] for x in c.values())],
                          "adv_wrong_range": [min(x["adversarial_wrong"]["pp"] for x in c.values()), max(x["adversarial_wrong"]["pp"] for x in c.values())],
                          "adv_correct_range": [min(x["adversarial_correct"]["pp"] for x in c.values()), max(x["adversarial_correct"]["pp"] for x in c.values())],
                          "decoy_range": [min(x["decoy_rate"] for x in c.values()), max(x["decoy_rate"] for x in c.values())]}
    # Strategy-Switching by task type, pooled over models (full suite)
    by = defaultdict(lambda: [0, 0, 0, 0, 0])  # n, switched, correct, sbw, correct without switch
    for v in per.values():
        for t_, sw, cor, _ in v["switching"]["rows"]:
            b_ = by[t_]
            b_[0] += 1; b_[1] += sw; b_[2] += cor; b_[3] += sw and not cor; b_[4] += cor and not sw
    out["switching_by_type"] = {t_: {"n": b_[0], "switch": r1(100 * b_[1] / b_[0]), "acc": r1(100 * b_[2] / b_[0]),
                                     "sbw": r1(100 * b_[3] / b_[0]), "correct_no_switch": [b_[4], b_[2]]} for t_, b_ in sorted(by.items())}
    return out


# ---------------------------------------------------------------------------
# LaTeX tables

def pct(x, d=0):
    return "---" if x is None else f"{x:.{d}f}\\%"


def sgn(x, d=0):
    if x is None:
        return "---"
    v = round(x, d)
    return "$0$" if v == 0 else f"${v:+.{d}f}$"


def ci(c, signed=False):
    if signed:
        return f"$[{c[0]:+.0f}, {c[1]:+.0f}]$"
    return f"[{c[0]}, {c[1]}]"


def rows_tex(fn):
    out = []
    for i, m in enumerate(MODELS):
        if i == len(ORIGINAL):
            out.append("\\midrule")
        out.append(NAME[m] + " & " + " & ".join(fn(m)) + " \\\\")
    return "\n".join(out)


def write_tables(per, pool):
    TAB.mkdir(exist_ok=True)
    w = lambda m: per[m]["wagering"]  # noqa: E731
    (TAB / "wagering.tex").write_text(rows_tex(lambda m: [
        str(w(m)["n"]), pct(w(m)["acc"]), f"{w(m)['ece']:.3f}"[1:], f"{w(m)['brier']:.3f}"[1:], pct(w(m)["wr_a"]),
        pct(w(m)["wr_b"]), f"{w(m)['regret_per_item']:.2f}", f"{w(m)['n_cautious']}/{w(m)['n_suboptimal']}",
        sgn(w(m)["overconf"])]) + "\n")

    def sr(m):
        s = per[m]["self_review"]
        c2 = s["c2"]
        return [str(s["items"]), str(s["n_errors"]), pct(s["acc"]), f"{s['n_flagged']}/{s['n_flag_den']}", ci(s["flag_ci"]), pct(s["fix"]),
                f"{c2['fix_correct']} / {c2['fresh_correct']} ({c2['n']})" if c2 else "---", pct(s["fc"]), str(s["c_to_w"])]
    (TAB / "self_review.tex").write_text(rows_tex(sr) + "\n")

    c1 = load(M / "control1.json")["per_model"]
    (TAB / "control1.tex").write_text(rows_tex(lambda m: [
        str(c1[m]["n_blind_spot"]), str(c1[m]["n_true_approve"]), f"${c1[m]['gap']:+.1f}$",
        "$[" + ", ".join(f"{x:+.2f}" if round(x, 1) == 0 else f"{x:+.1f}" for x in c1[m]["ci95"]) + "]$"]) + "\n")

    def c2(m):
        x = per[m]["self_review"]["c2"]
        return [str(x["n"]), str(x["fix_correct"]), str(x["fresh_correct"]), str(x["fix_only"]), str(x["fresh_only"]),
                f"{x['mcnemar_p']:.2f}", f"{x['p_holm']:.2f}"]
    (TAB / "control2.tex").write_text(rows_tex(c2) + "\n")

    def df(m):
        d = per[m]["deferral"]
        return [pct(d["defer"]["low"]), pct(d["defer"]["medium"]), pct(d["defer"]["high"]), f"${d['delta']:+.0f}$",
                pct(d["att_acc"]), pct(d["clar_p"]), pct(d["clar_r"]), ci(d["clar_r_ci"]), pct(d["false_clar"], 1),
                f"${d['ev']}$".replace("$-", "$-")]
    (TAB / "deferral.tex").write_text(rows_tex(df) + "\n")

    def cf(m):
        d = per[m]["deferral"]
        if not d.get("n_declined"):
            return [str(d.get("n_declined") or 0), "---", "---", pct(d["att_acc"]), "---"]
        return [str(d["n_declined"]), f"{d['forced_correct']}/{d['n_forced_scored']} ({d['forced_acc']:.0f}\\%)", ci(d["forced_ci"]),
                pct(d["att_acc"]), f"${d['cf_gap']:+.0f}$"]
    (TAB / "deferral_cf.tex").write_text(rows_tex(cf) + "\n")

    def dab(m):
        d = per[m]["deferral"]
        return [f"${d['ev']}$", f"${d['ev_b']}$", f"${d['ev_b'] - d['ev']:+d}$", pct(d["adherence_A"]), pct(d["adherence_B"]),
                pct(d["clar_r"]), pct(d["clar_r_b"])]
    (TAB / "deferral_ab.tex").write_text(rows_tex(dab) + "\n")

    hi = load(M / "deferral_human_items.json")

    def hum(m):
        if m == "sonnet-4":
            return ["\\multicolumn{3}{c}{not served (see caption)}"]
        x = hi[m]
        u, a = x["underspecified"], x["answerable"]
        return [f"{u['clarify']}/{u['n']}", f"{u['clarify']} / {u['defer']} / {u['attempt']}", f"{a['clarify']}/{a['n']}"]
    (TAB / "deferral_human.tex").write_text(rows_tex(hum) + "\n")

    def smt(m):
        s = per[m]["source_monitoring"]
        rates = [100 * k / n for k, n in s["confab_by_passage"].values() if n]
        return [pct(s["acc"]), pct(s["lost_pct"]), pct(s["confab"]), f"{min(rates):.0f}--{max(rates):.0f}\\%",
                pct(s["inferred"]), pct(s["fab_prec"])]
    (TAB / "source_monitoring.tex").write_text(rows_tex(smt) + "\n")

    def con(m):
        c = per[m]["consistency"]
        e = c["explanation"]
        return [pct(c["direct_acc"]), sgn(c["adversarial_correct"]["pp"]), sgn(c["adversarial_wrong"]["pp"]),
                pct(c["decoy_rate"]), sgn(e["pp"], 1),
                f"$[{e['ci'][0]:+.1f}, {e['ci'][1]:+.1f}]$" + ("$^\\ast$" if e["ci_method"] == "wilson_bound" else ""),
                f"{e['discordant'][0]} / {e['discordant'][1]}", sgn(e["all_items"]["pp"], 1),
                f"{c['answered'][0]:.0f}/{c['answered'][1]:.0f}"]
    (TAB / "consistency.tex").write_text(rows_tex(con) + "\n")

    def sw(m):
        f, x = per[m]["switching"]["full"], per[m]["switching"]
        return [pct(f["switch"]), pct(f["sbw"]), ci(f["sbw_ci"]), f"{f['conv'][0]} / {f['conv'][1]}", pct(f["acc"]),
                str(f["no_switch_correct"]), str(f["usable"]), pct(x["excl_cipher"]["acc"])]
    (TAB / "switching.tex").write_text(rows_tex(sw) + "\n")

    def lp(m):
        x = per[m]["loop"]
        if not x:
            return ["\\multicolumn{6}{c}{no completed sessions (see caption)}"]
        # per-model differences are descriptive: three sessions per arm leave no valid per-model test
        return [str(x["items"]), f"{x['rec_live']:.0f}\\%", f"{x['rec_control']:.0f}\\%", sgn(x["diff_pp"]),
                f"{x['flag']['live'][0]} / {x['flag']['live'][1]}", f"{x['flag']['control'][0]} / {x['flag']['control'][1]}"]
    (TAB / "loop.tex").write_text(rows_tex(lp) + "\n")

    fails = load(M / "failures.json")
    keys = [("t2_confidence_wagering", "Wager"), ("t5_selective_deferral", "Defer"), ("t7_consistency", "Consist"),
            ("t3_source_monitoring", "SM"), ("t4_strategy_switching", "Switch"), ("t4_cipher", "Cipher"),
            ("t6_error_detection", "SR"), ("t6_control2", "C2"), ("t8_loop", "Loop")]

    def fl(m):
        f = fails.get(m, {})
        return [("---" if k not in f else f"{f[k]['lost_pct']:.0f}") for k, _ in keys]
    (TAB / "failures.tex").write_text(rows_tex(fl) + "\n")

    def summ(m):
        s, wg, d, lpp, c, sw_ = (per[m][k] for k in ("self_review", "wagering", "deferral", "loop", "consistency", "switching"))
        return [f"{s['acc']:.0f}", f"{wg['ece']:.3f}"[1:], f"{wg['wr_a']:.0f}", f"{d['clar_r']:.0f}", f"{s['flag']:.0f}",
                "---" if not lpp else sgn(lpp["diff_pp"]), sgn(c["explanation"]["pp"]), f"{sw_['full']['sbw']:.0f}"]
    (TAB / "summary.tex").write_text(rows_tex(summ) + "\n")


def write_extra_tables(per):
    # settings ablation
    ab = load(M / "ablation.json") or {}
    lines = []
    for m in ("gpt-5.4", "sonnet-4.6", "deepseek-v3.2"):
        a = ab.get(m)
        if not a:
            continue
        for k, arm_name in (("v1", "accepted version"), ("v1set", "original settings"), ("noreason", "no reasoning"), ("v2", "main run")):
            x = a.get(k) or {}
            f = lambda key, fmt: "---" if x.get(key) is None else fmt.format(x[key])  # noqa: E731
            lines.append(" & ".join([NAME[m] if k == "v1" else "", arm_name, f("ECE", "{:.3f}").replace("0.", "."),
                                     f("Brier", "{:.3f}").replace("0.", "."), f("WR", "{:.0f}"), f("Regret", "{:.2f}"),
                                     f("Clar", "{:.1f}"), f("FalseClar", "{:.1f}"), f("DefPts", "${:.0f}$"),
                                     "---" if x.get("Expl") is None else f"${x['Expl']:+.1f}$"]) + " \\\\")
        if m != "deepseek-v3.2":
            lines.append("\\midrule")
    (TAB / "ablation.tex").write_text("\n".join(lines) + "\n")

    # lost responses by cause, per task, summed over the 19 models, in each task's own unit
    fails = load(M / "failures.json")
    units = [("t2_confidence_wagering", "Wagering", "calls"), ("t5_selective_deferral", "Deferral", "calls"),
             ("t7_consistency", "Consistency", "calls"), ("t3_source_monitoring", "Source-Monitoring", "passages"),
             ("t4_strategy_switching", "Strategy-Switching (excl.\\ Cipher)", "problems"), ("t4_cipher", "Cipher", "problems"),
             ("t6_error_detection", "Self-Review", "items"), ("t6_control2", "Control~2", "fresh attempts"),
             ("t8_loop", "The-Loop", "sessions")]
    rows = []
    for k, label, unit in units:
        fs = [fails[m][k] for m in MODELS if (fails.get(m) or {}).get(k)]
        tot = lambda key: sum(f.get(key) or 0 for f in fs)  # noqa: E731
        unusable = f"{tot('unusable')}" + (f" of {tot('n_statements')} statements" if k == "t3_source_monitoring" else
                                           f" of {tot('n_answers')} answers" if k == "t8_loop" else "")
        if any("unusable_truncated" in f for f in fs) and tot("unusable"):
            unusable += f" ({tot('unusable_truncated')} cut off)"
        if tot("blocked_judged"):
            unusable += f"; {tot('blocked_judged')} blocked, judged"
        rows.append(f"{label} & {unit} & {tot('n')} & {tot('error')} & {tot('empty')} & {tot('blocked')} & {tot('truncated')} & {unusable} \\\\")
    (TAB / "failures_by_cause.tex").write_text("\n".join(rows) + "\n")

    # 32k sensitivity: Strategy-Switching full suite, 16k vs 32k
    s16, s32 = load(M / "strategy_switching.json"), load(M / "strategy_switching_32k.json") or {}
    rows = []
    for m in MODELS:
        if m not in s32:
            continue
        a, b = s16[m]["full_suite"], s32[m]["full_suite"]
        rows.append(f"{NAME[m]} & {a['correct']} / {b['correct']} & {a['usable']} / {b['usable']} & "
                    f"{a['truncated']} / {b['truncated']} & {a['refused']} / {b['refused']} \\\\")
    (TAB / "sens32_switching.tex").write_text("\n".join(rows) + "\n")

    # repeat stability: per task, median and max range of the headline rate over three runs, median agreement
    rv = load(M / "repeat_variance.json")
    tasks = [("t2_confidence_wagering", "correct", "Wagering (accuracy)"), ("t5_selective_deferral", "decision", "Deferral (attempt rate)"),
             ("t7_consistency", "correct", "Consistency (accuracy)"), ("t3_source_monitoring", "correct", "Source-Monitoring (accuracy)"),
             ("t4_strategy_switching", "correct", "Strategy-Switching (solved)"), ("t4_strategy_switching", "switched", "Strategy-Switching (switched)")]
    rows = []
    for t_, k, label in tasks:
        vals = [v[t_][k] for v in rv.values() if t_ in v and k in v[t_]]
        n = [v[t_]["n_items"] for v in rv.values() if t_ in v]
        rng_ = sorted(x["metric_range"] for x in vals)
        agree = sorted(x["agree_all"] for x in vals)
        rows.append(f"{label} & {int(np.median(n))} & {np.median(agree):.0f}\\% & {np.median(rng_):.1f} & {max(rng_):.1f} \\\\")
    (TAB / "repeats.tex").write_text("\n".join(rows) + "\n")

    # Strategy-Switching judge validation on v2 outputs
    jv = load(T / "t4_strategy_switching" / "judge_validation" / "v2" / "summary.json")
    names = {"cipher": "Cipher", "dijkstra": "Dijkstra", "graph_coloring": "Graph Colouring", "knights": "Knights and Knaves",
             "code_breaker": "Code Breaker", "maze": "Maze", "sphere": "Sphere", "letter_guess": "Letter Guess"}
    rows = [f"{names[k]} & {v['n']} & {100 * v['switched_agree']:.0f}\\% & {100 * v['correct_agree']:.0f}\\% \\\\"
            for k, v in jv.items() if k != "ALL"]
    a = jv["ALL"]
    rows += ["\\midrule", f"Overall (weighted) & {a['n']} & {100 * a['switched_agree']:.0f}\\% & {100 * a['correct_agree']:.0f}\\% \\\\"]
    (TAB / "judge_v2.tex").write_text("\n".join(rows) + "\n")


def main():
    per = {}
    for m in MODELS:
        per[m] = {"wagering": wagering(m), "deferral": deferral(m), "consistency": consistency(m),
                  "self_review": self_review(m), "loop": loop(m), "switching": switching(m),
                  "source_monitoring": source_monitoring(m)}
    pool = pooled(per)
    write_tables(per, pool)
    write_extra_tables(per)
    for v in per.values():
        v["switching"].pop("rows")
        if v["loop"]:
            v["loop"].pop("item_rows")
    (M / "paper_numbers.json").write_text(json.dumps({"models": per, "pooled": pool}, indent=1))
    print(json.dumps(pool, indent=1))


if __name__ == "__main__":
    main()
