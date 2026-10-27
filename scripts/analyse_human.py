"""
Human baselines: every human number reported in the paper, recomputed from
results/human_baselines/s*_{annotations,participants}.json (preprint, Sep 2026).

Before this script, human_baseline_results.json was produced ad hoc and some
paper numbers came from an earlier data snapshot. The analysis unit here is a
participant who completed the study (participants.json status "completed");
responses from participants who left part-way are in the released data but not
in these numbers. --all includes them, as a sensitivity check.

Model scoring functions are reused so humans and models are scored alike.

    python scripts/analyse_human.py
    python scripts/analyse_human.py --all
"""
import ast
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scoring import expected_calibration_error, optimal_wager, t5_expected_value, wagering_rationality  # noqa: E402

HB = ROOT / "results" / "human_baselines"
ALL = "--all" in sys.argv


def load(study):
    done = {p["participant_id"] for p in json.loads((HB / f"{study}_participants.json").read_text())
            if p["status"] == "completed"}
    rows = json.loads((HB / f"{study}_annotations.json").read_text())
    rows = [r for r in rows if ALL or r["participant_id"] in done]
    for r in rows:
        r["r"] = r["responses"] if isinstance(r["responses"], dict) else ast.literal_eval(r["responses"])
    return rows, len({r["participant_id"] for r in rows})


def boot(units, stat, reps=5000, seed=0):
    """Percentile 95% CI resampling whole participants (units: participant -> rows)."""
    rng, keys = random.Random(seed), list(units)
    vals = []
    for _ in range(reps):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in units[k]]
        v = stat(sample)
        if v is not None:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def by_participant(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[r["participant_id"]].append(r)
    return g


def pct(x):
    return f"{100 * x:.1f}%"


def s1():
    out = {}
    for s in ("s1a", "s1b"):
        rows, n = load(s)
        byq = collections.defaultdict(list)
        for r in rows:
            byq[r["question_id"]].append(r["r"].get("answer") == r["correct_answer"])
        acc = sum(sum(v) for v in byq.values()) / sum(len(v) for v in byq.values())
        out[s] = (n, len(rows), acc, {q: sum(v) / len(v) for q, v in byq.items()})
    (na, ra, aa, qa), (nb, rb, ab, qb) = out["s1a"], out["s1b"]
    acc = lambda rows: sum(r["r"].get("answer") == r["correct_answer"] for r in rows) / len(rows)
    ga, gb = by_participant(load("s1a")[0]), by_participant(load("s1b")[0])
    rng, diffs = random.Random(0), []
    for _ in range(5000):
        sa = [r for k in (rng.choice(list(ga)) for _ in ga) for r in ga[k]]
        sb = [r for k in (rng.choice(list(gb)) for _ in gb) for r in gb[k]]
        diffs.append(100 * (acc(sb) - acc(sa)))
    diffs.sort()
    qs = set(qa) & set(qb)
    print(f"S1 Consistency: S1A n={na} ({ra} resp) acc {pct(aa)} | S1B n={nb} ({rb} resp) acc {pct(ab)} | "
          f"explanation effect {100 * (ab - aa):+.1f}pp [participant-bootstrap 95% CI {diffs[125]:+.1f}, {diffs[4875]:+.1f}], "
          f"{sum(qb[q] > qa[q] for q in qs)}/{len(qs)} questions improve")
    return {"n_a": na, "n_b": nb, "acc_a": aa, "acc_b": ab, "expl_pp": 100 * (ab - aa), "expl_ci": [diffs[125], diffs[4875]]}


def s2():
    rows, n = load("s2")
    p1 = {(r["participant_id"], r["question_id"]): r for r in rows if str(r.get("phase")) == "1"}
    p2 = {(r["participant_id"], r["question_id"]): r for r in rows if str(r.get("phase")) == "2"}
    pairs = [k for k in p1 if k in p2]
    wrong = [k for k in pairs if p1[k]["r"].get("answer") != p1[k]["correct_answer"]]
    right = [k for k in pairs if k not in set(wrong)]
    change = lambda k: p2[k]["r"].get("review_verdict") == "change"
    fixed = [k for k in wrong if change(k) and p2[k]["r"].get("revised_answer") == p1[k]["correct_answer"]]
    broke = [k for k in right if change(k) and p2[k]["r"].get("revised_answer") != p1[k]["correct_answer"]]
    fl = sum(map(change, wrong))
    units = collections.defaultdict(list)
    for k in wrong:
        units[k[0]].append(k)
    lo, hi = boot(units, lambda ks: sum(map(change, ks)) / len(ks) if ks else None)
    print(f"S2 Self-Review: n={n}, {len(pairs)} responses, acc {pct(len(right) / len(pairs))}, wrong {len(wrong)}, "
          f"flag (chose change) {fl} = {pct(fl / len(wrong))} [CI {pct(lo)}, {pct(hi)}], fix {len(fixed)} = {pct(len(fixed) / len(wrong))}, "
          f"flag on correct {sum(map(change, right))}/{len(right)} = {pct(sum(map(change, right)) / len(right))}, "
          f"correct->wrong {len(broke)}")
    return {"n": n, "acc": len(right) / len(pairs), "n_wrong": len(wrong), "flag": fl / len(wrong), "flag_ci": [lo, hi],
            "fix": len(fixed) / len(wrong), "fc": sum(map(change, right)) / len(right)}


def s3():
    rows, n = load("s3")
    rows = [r for r in rows if r["r"].get("wager") and r["r"].get("confidence") is not None]
    conf = [int(r["r"]["confidence"]) for r in rows]
    corr = [r["r"].get("answer") == r["correct_answer"] for r in rows]
    wag = [r["r"]["wager"].upper() for r in rows]
    opt = [optimal_wager(c) for c in conf]
    share = lambda xs, w: sum(x == w for x in xs) / len(xs)
    print(f"S3 Wagering: n={n}, {len(rows)} resp, acc {pct(sum(corr) / len(corr))}, mean conf {sum(conf) / len(conf):.1f}, "
          f"ECE {expected_calibration_error(conf, corr):.3f}, WR {pct(wagering_rationality(conf, wag))}, "
          f"SAFE actual {pct(share(wag, 'SAFE'))} vs optimal {pct(share(opt, 'SAFE'))}, "
          f"BOLD actual {pct(share(wag, 'BOLD'))} vs optimal {pct(share(opt, 'BOLD'))}")
    return {"n": n, "acc": sum(corr) / len(corr), "ece": expected_calibration_error(conf, corr),
            "wr": wagering_rationality(conf, wag)}


def s4():
    rows, n = load("s4")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["stakes"].upper()].append(r)
    rate = {k: sum(r["r"].get("decision") == "DEFER" for r in v) / len(v) for k, v in by.items()}
    att = [r for r in rows if r["r"].get("decision") == "ATTEMPT"]
    att_ok = sum(r["r"].get("answer") == r["correct_answer"] and r.get("is_underspecified") not in (True, "True")
                 for r in att)
    und = [r for r in rows if r.get("is_underspecified") in (True, "True")]
    clar = sum(r["r"].get("decision") == "CLARIFY" for r in und)
    clo, chi = boot(by_participant(und), lambda rs: sum(r["r"].get("decision") == "CLARIFY" for r in rs) / len(rs) if rs else None)
    ev = t5_expected_value([r["r"].get("decision") or "" for r in rows],
                           [r["r"].get("answer") == r["correct_answer"] and r.get("is_underspecified") not in (True, "True")
                            for r in rows], [r["stakes"].lower() for r in rows])
    print(f"S4 Deferral: n={n}, {len(rows)} resp, DEFER rate low {pct(rate['LOW'])} med {pct(rate['MEDIUM'])} "
          f"high {pct(rate['HIGH'])} (shift {100 * (rate['HIGH'] - rate['LOW']):+.1f}pp), "
          f"attempt acc {pct(att_ok / len(att))} ({len(att)} attempts), clarification recall {clar}/{len(und)} = {pct(clar / len(und))} [CI {pct(clo)}, {pct(chi)}], "
          f"EV {ev['total']} (model payoff scheme; per participant {ev['total'] / n:.0f})")
    return {"n": n, "defer": {k.lower(): v for k, v in rate.items()}, "delta": rate["HIGH"] - rate["LOW"],
            "att_acc": att_ok / len(att), "clar_r": clar / len(und), "clar_r_ci": [clo, chi], "ev": ev["total"]}


def s5():
    rows, n = load("s5")
    ok = [r["r"].get("classification") == r["correct_label"] for r in rows]
    fab = [o for r, o in zip(rows, ok) if r["correct_label"] == "FABRICATED"]
    print(f"S5 Source-Monitoring: n={n}, {len(rows)} resp, acc {pct(sum(ok) / len(ok))}, "
          f"confabulation detection {sum(fab)}/{len(fab)} = {pct(sum(fab) / len(fab))}")
    return {"n": n, "acc": sum(ok) / len(ok), "confab": sum(fab) / len(fab)}


def unique():
    ids = collections.Counter()
    for s in ("s1a", "s1b", "s2", "s3", "s4", "s5"):
        ids.update({r["participant_id"] for r in load(s)[0]})
    print(f"Participants: {len(ids)} unique, {sum(ids.values())} study completions, "
          f"{sum(v > 1 for v in ids.values())} in more than one study")
    return {"participants": len(ids), "completions": sum(ids.values())}


if __name__ == "__main__":
    print("(including participants who did not finish)" if ALL else "(completers only)")
    unique(); s1(); s2(); s3(); s4(); s5()
