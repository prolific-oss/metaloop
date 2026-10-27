"""
Self-Review Control 1 on the v2 data (October 2026): the specification of the
paper's analysis (tasks/t6_error_detection/analyse_control1.py), fitted with
statsmodels OLS and bootstrapped with scipy, applied to the v2 Self-Review rows.

Among approved answers (not flagged in review), review confidence is regressed
on blind-spot status (Phase-1 answer wrong) and the item's annotated mean
demand (DeLeAn, question_bank_v2.json). Gap = true approves minus blind spots,
in points, so a positive gap means blind spots get less confidence. Per model:
conf ~ blind_spot + demand (and + demand^2 as a check). Pooled: model fixed
effects and model-specific demand slopes. 95% CIs from 2,000 bootstrap draws
that resample items jointly across models. Repeat 0, main panel (no ablation or
sensitivity aliases).

scripts/conf_difficulty.py is the companion analysis with a different
difficulty measure (leave-one-out error rate of the other models).

    python scripts/control1_v2.py   -> results_v2_metrics/control1.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import llm  # noqa: E402
import stats  # noqa: E402
import statsmodels.api as sm  # noqa: E402

M = ROOT / "results_v2_metrics"
REPS = 2000


def approved_rows():
    demand = {q["id"]: q["delean_aggregates"]["mean_demand"]
              for q in json.loads((ROOT / "data" / "question_bank_v2.json").read_text())}
    rows = []
    for m in llm.models(None):
        if llm.spec(m).get("cohort") in ("ablation", "sensitivity"):
            continue
        p = M / "review" / f"{m}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text()).get("self_review_rows") or []:
            if r["flagged"] or r["mean_conf"] is None or demand.get(r["id"]) is None:
                continue
            rows.append((m, r["id"], int(not r["p1_correct"]), float(r["mean_conf"]), float(demand[r["id"]])))
    return rows


def gap_per_model(rows, quad=False):
    """Adjusted gap = -(blind-spot coefficient) in conf ~ blind_spot + demand [+ demand^2], per model (OLS)."""
    gaps = {}
    for name in sorted({r[0] for r in rows}):
        sub = [r for r in rows if r[0] == name]
        if sum(r[2] for r in sub) < 2:
            continue
        bs = np.array([r[2] for r in sub], float)
        d = np.array([r[4] for r in sub])
        cols = [bs, d - d.mean()] + ([(d - d.mean()) ** 2] if quad else [])
        fit = sm.OLS(np.array([r[3] for r in sub]), sm.add_constant(np.column_stack(cols), has_constant="add")).fit()
        gaps[name] = -fit.params[1]
    return gaps


def pooled_gap(rows):
    """conf ~ model FE + blind_spot + model:demand (OLS, no common intercept); returns -(blind-spot coefficient)."""
    models = sorted(set(r[0] for r in rows))
    X = np.array([[1.0 * (m == k) for k in models] + [bs] + [d * (m == k) for k in models] for m, _, bs, _, d in rows])
    return -sm.OLS(np.array([r[3] for r in rows]), X).fit().params[len(models)]


def main():
    rows = approved_rows()
    models = sorted({r[0] for r in rows})
    items = sorted({r[1] for r in rows})
    by_item = {}
    for r in rows:
        by_item.setdefault(r[1], []).append(r)
    point, quad, pooled = gap_per_model(rows), gap_per_model(rows, quad=True), pooled_gap(rows)
    keys = list(point)

    def replicate(idx):
        """Gaps on a resample of items (all models' rows for an item move together): per model, pooled, mean."""
        samp = [r for i in idx for r in by_item[items[i]]]
        g = gap_per_model(samp)
        return np.array([g.get(k, np.nan) for k in keys] + [pooled_gap(samp), np.mean(list(g.values()))])
    reps, _, _ = stats.cluster_bootstrap(len(items), replicate, n_resamples=REPS)
    boots = {k: reps[:, j][np.isfinite(reps[:, j])] for j, k in enumerate(keys)}
    bpool, bmean = reps[:, len(keys)], reps[:, len(keys) + 1]
    ci = lambda v: [round(float(np.percentile(v, q)), 2) for q in (2.5, 97.5)]  # noqa: E731
    n = {m: (sum(r[2] for r in rows if r[0] == m), sum(1 - r[2] for r in rows if r[0] == m)) for m in models}
    res = {"per_model": {k: {"n_blind_spot": n[k][0], "n_true_approve": n[k][1], "gap": round(float(point[k]), 2),
                             "ci95": ci(boots[k]), "gap_quadratic": round(float(quad[k]), 2)} for k in point},
           "skipped_fewer_than_2_blind_spots": [m for m in models if m not in point],
           "pooled": {"gap": round(float(pooled), 2), "ci95": ci(bpool)},
           "equal_model_mean": {"gap": round(float(np.mean(list(point.values()))), 2), "ci95": ci(bmean)}}
    print(f"{'model':18s} {'BS':>3} {'TA':>4}  gap   95% CI          quadratic")
    for k, v in res["per_model"].items():
        print(f"{k:18s} {v['n_blind_spot']:3d} {v['n_true_approve']:4d} {v['gap']:+5.1f}  "
              f"[{v['ci95'][0]:+5.1f}, {v['ci95'][1]:+5.1f}]  {v['gap_quadratic']:+5.1f}")
    print("skipped (<2 blind spots):", res["skipped_fewer_than_2_blind_spots"])
    print(f"pooled {res['pooled']['gap']:+.2f} {res['pooled']['ci95']}   "
          f"equal-model mean {res['equal_model_mean']['gap']:+.2f} {res['equal_model_mean']['ci95']}")
    (M / "control1.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
