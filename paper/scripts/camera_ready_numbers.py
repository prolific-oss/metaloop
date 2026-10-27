"""
Every number in the camera-ready, from the v2 data, for the 12 camera-ready models (PLAN.md 0.5).

Per-model and pooled numbers come from scripts/paper_numbers_v2.py unchanged (as numbers_12.py),
plus what the accepted paper's tables report and that script does not:
- human baselines (scripts/analyse_human.py, completers only);
- Strategy-Switching in the accepted definitions over all 80 problems: detection = switched
  strategy or reached the correct answer, correction = correct answer, perseveration = 100 - detection;
- Consistency stability = share of items whose answer is the same under every framing;
- Loop: items presented per session (run_v3.items_for); Gemma 4's row from the Crusoe-hosted run
  when it is scored (results_v2_metrics/review/gemma-4-31b-crusoe.json);
- the summary-feedback comparison (results_v2_metrics/design_a_v3.json).

Writes results_v2_metrics/camera_ready_numbers.json and, for step 3, the body rows of each table in
the accepted layout to camera_ready/tables/<label>.tex (pasted into metaloop.tex, not \\input).

    .venv/bin/python <camera_ready>/scripts/camera_ready_numbers.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

CR = Path(__file__).resolve().parents[1]
PROJ = CR.parent
sys.path.insert(0, str(PROJ / "scripts"))
sys.path.insert(0, str(PROJ / "src"))
sys.path.insert(0, str(PROJ / "tasks" / "t8_loop"))
import analyse_human as ah  # noqa: E402
import paper_numbers_v2 as pn  # noqa: E402
pn.r1 = lambda x: None if x is None else float(x)   # keep full precision; round once at display (half-up)
import run_v3  # noqa: E402
from stats import cluster_bootstrap, mean_ci, wilson  # noqa: E402

# accepted table order, Grok and Qwen in their old slots
MODELS = ["sonnet-4", "sonnet-4.6", "opus-4.6", "gpt-4.1", "gpt-5.4", "gemini-2.5-pro", "gemini-3.1-pro",
          "gemma-4-31b", "deepseek-v3.2", "grok-4.7", "qwen-3.8-max", "mistral-large"]
LONG = {**pn.NAME}
SHORT = {"sonnet-4": "Sonnet 4", "sonnet-4.6": "Sonnet 4.6", "opus-4.6": "Opus 4.6", "gpt-4.1": "GPT-4.1",
         "gpt-5.4": "GPT-5.4", "gemini-2.5-pro": "Gemini 2.5", "gemini-3.1-pro": "Gemini 3.1", "gemma-4-31b": "Gemma 31B",
         "deepseek-v3.2": "DeepSeek", "grok-4.7": "Grok 4.7", "qwen-3.8-max": "Qwen 3.8", "mistral-large": "Mistral"}
GEMMA_LOOP = "gemma-4-31b-crusoe"
TAB = CR / "tables"


def switching_accepted(m):
    rows = pn.scored("t4_strategy_switching", m)
    n = len(rows)
    det = sum(r["switched"] or r["correct"] for r in rows)
    cor = sum(r["correct"] for r in rows)
    return {"n": n, "det": pn.r1(100 * det / n), "det_ci": [hu(x) for x in wilson(det, n)],
            "corr": pn.r1(100 * cor / n), "corr_ci": [hu(x) for x in wilson(cor, n)],
            "persev": pn.r1(100 * (n - det) / n)}


def clarification(m):
    """Clarify decisions on underspecified items (Condition A), over responses that contain a
    decision, as the recall itself is computed (qa metrics); the Wilson CI uses these counts."""
    rs = [r for r in pn.scored("t5_selective_deferral", m) if r["condition"] == "A" and r["is_underspecified"]
          and r["decision"] is not None]
    k = sum(r["decision"] == "CLARIFY" for r in rs)
    return {"k": k, "n": len(rs), "ci": [hu(x) for x in wilson(k, len(rs))]}


def stability(m):
    by = defaultdict(dict)
    for r in pn.scored("t7_consistency", m):
        by[r["unit"]][r["condition"]] = r.get("answer")
    full = [d for d in by.values() if len(d) == 4 and all(a is not None for a in d.values())]
    return {"n": len(full), "stab": pn.r1(100 * sum(len(set(d.values())) == 1 for d in full) / len(full))}


def loop_row(m):
    key = GEMMA_LOOP if m == "gemma-4-31b" else m
    lp = pn.loop(key) if (pn.M / "review" / f"{key}.json").exists() else None
    if lp:
        lp["presented"] = len(run_v3.items_for(key))
        lp["source"] = key
        lp["boot_ci"] = loop_boot(lp["item_rows"])
    return lp


def loop_boot(items):
    """95% percentile-bootstrap CIs for pooled recovery per arm, resampling previously-wrong items
    (each item's three sessions stay together), so repeats are not treated as independent."""
    ids = sorted(items)
    def stat(idx):
        return [100 * sum(items[ids[i]][a][0] for i in idx) / sum(items[ids[i]][a][1] for i in idx)
                for a in ("live", "control")]
    _, lo, hi = cluster_bootstrap(len(ids), stat)
    out = {"live": [float(lo[0]), float(hi[0])], "control": [float(lo[1]), float(hi[1])]}
    for j, a in enumerate(("live", "control")):
        # no recovery at all: the bootstrap is degenerate ([0, 0]); use Wilson over the items instead
        if sum(items[i][a][0] for i in ids) == 0:
            out[a] = list(wilson(0, len(ids)))
    return out


def model_numbers(m):
    v = {"wagering": pn.wagering(m), "deferral": pn.deferral(m), "consistency": pn.consistency(m),
         "self_review": pn.self_review(m), "loop": loop_row(m), "switching": pn.switching(m),
         "source_monitoring": pn.source_monitoring(m)}
    v["switching"]["accepted"] = switching_accepted(m)
    v["consistency"]["stability"] = stability(m)
    c = clarification(m)
    assert abs(100 * c["k"] / c["n"] - v["deferral"]["clar_r"]) < 0.06, (m, c, v["deferral"]["clar_r"])
    v["deferral"]["clar_counts"], v["deferral"]["clar_r_ci"] = [c["k"], c["n"]], c["ci"]
    return v


def extra_pooled(per):
    """Numbers the camera-ready quotes that paper_numbers_v2.pooled does not compute (scipy)."""
    from itertools import combinations
    from scipy.stats import pearsonr
    acc = [per[m]["self_review"]["acc"] for m in MODELS]
    flag = [per[m]["self_review"]["flag"] for m in MODELS]
    r = pearsonr(acc, flag)
    lo, hi = r.confidence_interval()
    # Strategy-Switching problem difficulty shared across models: mean pairwise Pearson r of the
    # models' per-problem correctness (the accepted paper's definition, findings.md); pairs with a
    # constant vector are undefined and skipped
    vec = {}
    for m in MODELS:
        by = {r_["unit"]: r_["correct"] for r_ in pn.scored("t4_strategy_switching", m)}
        vec[m] = by
    units = sorted(set.intersection(*(set(v) for v in vec.values())))
    rs = []
    for a_, b_ in combinations(MODELS, 2):
        x, y = [float(vec[a_][u]) for u in units], [float(vec[b_][u]) for u in units]
        if len(set(x)) > 1 and len(set(y)) > 1:
            rs.append(pearsonr(x, y).statistic)
    sw = {m: per[m]["switching"]["full"] for m in MODELS}
    conv = {m: 100 * sw[m]["conv"][0] / sw[m]["conv"][1] for m in MODELS if sw[m]["conv"][1]}
    # human-item clarification check: the 15 handwritten underspecified items of the human study
    hitems = pn.load(pn.M / "deferral_human_items.json")
    hitems = {m: hitems[m] for m in MODELS if m in hitems and hitems[m]["status"].get("ok")}
    # Wagering, Condition A: high-confidence items not wagered BOLD (the 5.2 exemplar)
    hc = {}
    for m in MODELS:
        med = [x for x in pn.scored("t2_confidence_wagering", m) if x["condition"] == "A" and x.get("confidence") is not None
               and x["confidence"] >= 85 and x.get("wager") == "MEDIUM"]
        hc[m] = [len(med), sum(x["correct"] for x in med)]
    over = {m: per[m]["wagering"]["overconf"] for m in MODELS}
    # The-Loop, live minus control over the 12 models: per-model item-level differences
    # (paper_numbers_v2 diff_pp) and their mean, sign-flip over models and over pooled items
    from stats import signflip_p
    d_m = [per[m]["loop"]["diff_pp"] for m in MODELS]
    d_i = [v["live"][0] / v["live"][1] - v["control"][0] / v["control"][1]
           for m in MODELS for v in per[m]["loop"]["item_rows"].values()]
    loop_pooled = {"mean_diff_pp": sum(d_m) / len(d_m), "n_pos": sum(x > 0 for x in d_m), "n_zero": sum(x == 0 for x in d_m),
                   "n_neg": sum(x < 0 for x in d_m), "p_models": signflip_p(d_m), "n_items": len(d_i),
                   "mean_item_diff_pp": 100 * sum(d_i) / len(d_i), "p_items": signflip_p(d_i),
                   "max_gain": max((per[m]["loop"]["diff_pp"], m, per[m]["loop"]["p"]) for m in MODELS),
                   "min_p": min(per[m]["loop"]["p"] for m in MODELS)}
    return {"loop_pooled": loop_pooled, "high_conf_medium": hc,
            "overconf_13pp": {m: v for m, v in over.items() if v >= 13},
            "wager_conf_range": [min(per[m]["wagering"]["mean_conf"] for m in MODELS), max(per[m]["wagering"]["mean_conf"] for m in MODELS)],
            "wager_acc_range": [min(per[m]["wagering"]["acc"] for m in MODELS), max(per[m]["wagering"]["acc"] for m in MODELS)],
            "human_items_clarify": {"n_models": len(hitems), "all_15": sorted(m for m, v in hitems.items() if v["underspecified"]["clarify"] == 15),
                                    "per_model": {m: v["underspecified"]["clarify"] for m, v in hitems.items()}},
            "acc_vs_flag": {"r": round(float(r.statistic), 2), "ci": [round(float(lo), 2), round(float(hi), 2)],
                            "p": round(float(r.pvalue), 3), "n_models": len(MODELS)},
            "switching_difficulty_r": {"mean_pairwise_r": round(sum(rs) / len(rs), 2), "n_pairs": len(rs), "n_problems": len(units)},
            "switching_sbw_range": [min(v["sbw"] for v in sw.values()), max(v["sbw"] for v in sw.values())],
            "switching_conv_range": [round(min(conv.values()), 1), round(max(conv.values()), 1)],
            "switching_conv": {m: round(v, 1) for m, v in conv.items()}}


def human():
    """A release without the participant files reuses the shipped human summary."""
    if not ah.HB.exists():
        return pn.load(pn.M / "camera_ready_numbers.json")["human"]
    return {"participants": ah.unique(), "s1": ah.s1(), "s2": ah.s2(), "s3": ah.s3(), "s4": ah.s4(), "s5": ah.s5()}


# ---------------------------------------------------------------------------- formatting

def hu(x, d=0):
    """Round half up (Python's round() and format() round half to even: 58.5 -> 58)."""
    from decimal import ROUND_HALF_UP, Decimal
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP))


def n0(x):
    return format(hu(x), ".0f")


def s0(x):
    return format(hu(x), "+.0f")


def p0(x):
    return "---" if x is None else f"{n0(x)}"


def pp(x):
    """Summary table: signed, in math mode, zero as $+0$ (accepted rendering)."""
    return "---" if x is None else f"${s0(x)}$"


def ppt(x):
    """Appendix tables: +2pp, $-3$pp, $0$pp (accepted rendering)."""
    v = hu(x)
    return f"+{v:.0f}pp" if v > 0 else f"${v:.0f}$pp" if v < 0 else "$0$pp"


def ece(x):
    return f"{x:.3f}".lstrip("0")


def ci(c, signed=False):
    if signed:
        f = lambda x: f"{s0(x)}" if round(x) else "0"  # noqa: E731
        return f"$[{f(c[0])}, {f(c[1])}]$"
    return f"[{n0(c[0])}, {n0(c[1])}]"


def boot_prop(k, n):
    """Percentile-bootstrap 95% CI (5,000 resamples, as promised to usCf) for k of n, in %;
    degenerate when k is 0 or n, where every resample gives the same rate."""
    if k in (0, n):
        return [100 * k / n] * 2
    _, (lo, hi) = mean_ci([1] * k + [0] * (n - k), n_resamples=5000)
    return [100 * lo, 100 * hi]


def rng(vals, fmt=p0):
    vals = [x for x in vals if x is not None]
    return f"{fmt(min(vals))}--{fmt(max(vals))}"


def write(label, rows):
    TAB.mkdir(exist_ok=True)
    (TAB / f"{label}.tex").write_text("\n".join(rows) + "\n")


def tables(per, hum, pool):
    g = lambda m, *ks: _get(per[m], ks)  # noqa: E731
    # tab:summary
    cols = {"Acc": lambda m: g(m, "self_review", "acc"), "ECE": lambda m: g(m, "wagering", "ece"),
            "WR": lambda m: g(m, "wagering", "wr_a"), "Clar": lambda m: g(m, "deferral", "clar_r"),
            "Flag": lambda m: g(m, "self_review", "flag"), "Loop": lambda m: g(m, "loop", "rec_live"),
            "Expl": lambda m: g(m, "consistency", "explanation", "pp"),
            "Det": lambda m: g(m, "switching", "accepted", "det")}
    rows = []
    for m in MODELS:
        c = {k: f(m) for k, f in cols.items()}
        rows.append(f"{SHORT[m]:<12} & {p0(c['Acc']):>3} & {ece(c['ECE'])} & {p0(c['WR']):>3} & {p0(c['Clar']):>3} & "
                    f"{p0(c['Flag']):>3} & {p0(c['Loop']):>3} & {pp(c['Expl']):>6} & {p0(c['Det']):>3} \\\\")
    h = hum
    rows += ["\\midrule", f"{'Human':<12} & {n0(100 * h['s2']['acc'])} & {ece(h['s3']['ece'])} & {n0(100 * h['s3']['wr'])} & "
             f"{n0(100 * h['s4']['clar_r'])} & {n0(100 * h['s2']['flag'])} & --- & {pp(h['s1']['expl_pp'])} & --- \\\\"]
    write("summary", rows)
    write("summary_ranges", [f"{k}: {rng([f(m) for m in MODELS], ece if k == 'ECE' else p0)}" for k, f in cols.items()])

    # tab:t2
    rows = [f"{LONG[m]:<14} & {n0(g(m,'wagering','acc'))}\\% & {ece(g(m,'wagering','ece'))} & {n0(g(m,'wagering','wr_a'))}\\% & "
            f"{n0(g(m,'wagering','wr_b'))}\\% & {g(m,'wagering','payout')} & {g(m,'wagering','gap')} \\\\" for m in MODELS]
    rows += ["\\midrule", f"Human ($n={h['s3']['n']}$) & {n0(100 * h['s3']['acc'])}\\% & {ece(h['s3']['ece'])} & "
             f"{n0(100 * h['s3']['wr'])}\\% & --- & --- & --- \\\\"]
    write("t2", rows)

    # tab:t6 (Fix column added: decision 3)
    rows = []
    for m in MODELS:
        s = per[m]["self_review"]
        rows.append(f"{LONG[m]:<14} & {s['items']} & {s['n_errors']} & {n0(s['acc'])}\\% & {n0(s['flag'])}\\% & {ci(s['flag_ci'])} & "
                    f"{ci(boot_prop(s['n_flagged'], s['n_flag_den']))} & {100 - hu(s['flag']):.0f}\\% & {n0(s['fix'])}\\% & {n0(s['fc'])}\\% \\\\")
    s2 = h["s2"]
    rows += ["\\midrule", f"Human ($n={s2['n']}$) & 8 & --- & {n0(100 * s2['acc'])}\\% & {n0(100 * s2['flag'])}\\% & "
             f"--- & --- & {100 - hu(100 * s2['flag']):.0f}\\% & {100 * s2['fix']:.1f}\\% & {100 * s2['fc']:.1f}\\% \\\\"]
    write("t6", rows)

    # tab:t5
    rows = []
    for m in MODELS:
        d = per[m]["deferral"]
        rows.append(f"{LONG[m]:<14} & {n0(d['defer']['low'])}\\% & {n0(d['defer']['medium'])}\\% & {n0(d['defer']['high'])}\\% & "
                    f"{s0(d['delta'])}pp & {n0(d['att_acc'])}\\% & {n0(d['forced_acc'])}\\% & {n0(d['clar_p'])}\\% & {n0(d['clar_r'])}\\% & {ci(d['clar_r_ci'])} & {d['ev']} \\\\")
    s4 = h["s4"]
    rows += ["\\midrule", f"Human ($n={s4['n']}$) & {n0(100 * s4['defer']['low'])}\\% & {n0(100 * s4['defer']['medium'])}\\% & "
             f"{n0(100 * s4['defer']['high'])}\\% & {100 * s4['delta']:+.1f}pp & {n0(100 * s4['att_acc'])}\\% & --- & --- & "
             f"{n0(100 * s4['clar_r'])}\\% & --- & {s4['ev']} \\\\"]
    write("t5", rows)

    # tab:exclusions (xNnp): share of each task's units lost, per model (failure_table.py)
    fails = json.loads((PROJ / "results_v2_metrics" / "failures.json").read_text())
    keys = ["t2_confidence_wagering", "t5_selective_deferral", "t7_consistency", "t3_source_monitoring",
            "t4_strategy_switching", "t4_cipher", "t6_error_detection", "t6_control2", "t8_loop"]
    rows = []
    for m in MODELS:
        f = {**fails[m], "t8_loop": fails[GEMMA_LOOP]["t8_loop"]} if m == "gemma-4-31b" else fails[m]
        rows.append(f"{LONG[m]:<14} & " + " & ".join(f"{n0(f[k]['lost_pct'])}" for k in keys) + " \\\\")
    write("exclusions", rows)

    # tab:paraphrase (usCf): per model, range across wordings (analyse_paraphrase.py --v2)
    pz = json.loads((PROJ / "results_v2_metrics" / "paraphrase_v2" / "summary.json").read_text())
    wk = [("accuracy", n0), ("ece", ece), ("wr", n0), ("overconfidence_pp", n0), ("cautious_share", n0)]
    sk = [("flag_rate", n0), ("fix_rate", n0), ("false_criticism", n0)]
    neg = lambda t: "0" if t == "-0" else (f"${t}$" if t.startswith("-") else t)  # noqa: E731
    rr = lambda vals, f: neg(f(min(vals))) if f(min(vals)) == f(max(vals)) else f"{neg(f(min(vals)))}--{neg(f(max(vals)))}"  # noqa: E731
    rows = []
    for m in ["sonnet-4.6", "gpt-4.1", "gpt-5.4", "mistral-large"]:
        w = [v for k, v in pz["wagering"][m].items()]
        r = [v for k, v in pz["self_review"][m].items() if k != "stored"]
        rows.append(f"{LONG[m]:<14} & " + " & ".join(rr([x[k] for x in w], f) for k, f in wk) + " & "
                    + " & ".join(rr([x[k] for x in r], f) for k, f in sk) + " \\\\")
    W = {**{k: pz["rank_stability"]["wagering"][k]["kendall_w"] for k, _ in wk},
         **{k: pz["rank_stability"]["self_review"][k]["kendall_w"] for k, _ in sk}}
    rows += ["\\midrule", "Kendall's $W$ & " + " & ".join("---" if k == "cautious_share" else f"{W[k]:.2f}" for k, _ in wk + sk) + " \\\\"]
    write("paraphrase", rows)

    # tab:t3
    rows = [f"{LONG[m]:<14} & {n0(g(m,'source_monitoring','acc'))}\\% & {n0(g(m,'source_monitoring','confab'))}\\% & "
            f"{ci(g(m,'source_monitoring','confab_ci'))} \\\\" for m in MODELS]
    rows += ["\\midrule", f"Human ($n={h['s5']['n']}$) & {n0(100 * h['s5']['acc'])}\\% & {n0(100 * h['s5']['confab'])}\\% & --- \\\\"]
    write("t3", rows)

    # tab:t7
    rows = []
    for m in MODELS:
        c = per[m]["consistency"]
        rows.append(f"{LONG[m]:<14} & {n0(c['direct_acc'])}\\% & {ppt(c['adversarial_correct']['pp'])} & "
                    f"{ppt(c['adversarial_wrong']['pp'])} & {ppt(c['explanation']['pp'])} & {ci(c['explanation']['ci'], True)} & "
                    f"{n0(c['stability']['stab'])}\\% \\\\")
    rows += ["\\midrule", f"Human ($n={h['s1']['n_a'] + h['s1']['n_b']}$) & {n0(100 * h['s1']['acc_a'])}\\% & --- & --- & "
             f"{pp(h['s1']['expl_pp'])}pp & --- & --- \\\\"]
    write("t7", rows)

    # tab:t4 (xNnp promise: switch-but-wrong and P(correct | explicit switch) as the primary measures; P(switch) for comparability)
    rows = []
    for m in MODELS:
        s, f = per[m]["switching"]["accepted"], per[m]["switching"]["full"]
        k, n = f["conv"]
        rows.append(f"{LONG[m]:<14} & {n0(s['det'])}\\% & {n0(f['switch'])}\\% & {n0(f['sbw'])}\\% & {ci(f['sbw_ci'])} & {n0(100 * k / n)}\\% & "
                    f"{n0(s['corr'])}\\% & {ci(s['corr_ci'])} \\\\")
    write("t4", rows)

    # tab:t8 (live vs control: decision 5)
    rows = []
    for m in MODELS:
        lp = per[m]["loop"]
        if not lp:
            rows.append(f"{LONG[m]:<14} & \\multicolumn{{6}}{{c}}{{pending}} \\\\")
            continue
        b = lp["boot_ci"]
        rows.append(f"{LONG[m]:<14} & {lp['presented']} & {lp['items']} & {n0(lp['rec_live'])}\\% & {ci(b['live'])} & "
                    f"{n0(lp['rec_control'])}\\% & {ci(b['control'])} \\\\")
    write("t8", rows)


def _get(d, ks):
    for k in ks:
        if d is None:
            return None
        d = d.get(k)
    return d


def main():
    per = {m: model_numbers(m) for m in MODELS}
    pool = pn.pooled(per)
    pool["camera_ready"] = extra_pooled(per)   # pooled uses the switching rows and Loop item rows; dropped from the JSON after
    for v in per.values():
        v["switching"].pop("rows")
        if v["loop"]:
            v["loop"].pop("item_rows", None)
    hum = human()
    da = pn.load(pn.M / "design_a_v3.json")
    out = {"models": per, "pooled": pool, "human": hum, "design_a": da,
           "pending": [m for m in MODELS if not per[m]["loop"]]}
    (pn.M / "camera_ready_numbers.json").write_text(json.dumps(out, indent=1, default=float))
    tables(per, hum, pool)
    print("pending Loop rows:", out["pending"])
    print((TAB / "summary.tex").read_text())
    print((TAB / "summary_ranges.tex").read_text())


if __name__ == "__main__":
    main()
