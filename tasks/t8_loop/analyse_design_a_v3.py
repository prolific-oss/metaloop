"""Summary-feedback comparison (Design A) on the v2 settings: targeted minus generic, per model.

Reads results_v2_design_a/<model>.jsonl (run_design_a_v3.py). Same metrics as
analyse_design_a.py: error-flagging rate on the retest's wrong answers (review verdict
ERROR), blind-spot rate, answer consistency across the two framings, and Fabricated
detection on the 10 Source-Monitoring statements. Answers and verdicts use run_v3's
tolerant parsers. Writes results_v2_metrics/design_a_v3.json.

    python tasks/t8_loop/analyse_design_a_v3.py
"""
import json
import re
from pathlib import Path

from run_v3 import extract_answer, extract_verdict

HERE = Path(__file__).resolve().parent
RES = HERE / "results_v2_design_a"
OUT = HERE.parents[1] / "results_v2_metrics" / "design_a_v3.json"
MODELS = ["sonnet-4", "sonnet-4.6", "opus-4.6", "gpt-4.1", "gpt-5.4", "gemini-2.5-pro", "gemini-3.1-pro",
          "gemma-4-31b", "deepseek-v3.2", "mistral-large", "grok-4.7", "qwen-3.8-max"]
ITEM = re.compile(r"^\W*(\d+)\.", re.M)   # a numbered block; its label may sit on a later line
LABEL = re.compile(r"\b(PASSAGE|TRAINING|INFERRED|FABRICATED)\b")


def t3_labels(text):
    """First upper-case label after each statement's quoted text, per numbered block."""
    marks = list(ITEM.finditer(text or ""))
    out = {}
    for k, m in enumerate(marks):
        block = text[m.end(): marks[k + 1].start() if k + 1 < len(marks) else len(text)]
        block = re.sub(r'"[^"]*"', "", block, count=1)   # drop the restated statement
        hit = LABEL.search(block)
        if hit and int(m.group(1)) - 1 not in out:
            out[int(m.group(1)) - 1] = hit.group(1)
    return out


def load(model):
    f = RES / f"{model}.jsonl"
    recs = {}
    if f.exists():
        for line in f.open():
            r = json.loads(line)
            recs[(r["unit"], r["condition"])] = r
    return recs


def scores(recs, cond):
    t6 = [r for (u, c), r in recs.items() if c == cond and u.startswith("t6:") and r["status"] == "ok"]
    rows = [(extract_answer(r["answer_response"]) == r["gt"], extract_verdict(r["review_response"])) for r in t6]
    wrong = [v for ok, v in rows if not ok and v is not None]   # errors with a readable review (paper's flag denominator)
    flagged = sum(v == "ERROR" for v in wrong)
    t7 = {}
    for (u, c), r in recs.items():
        if c == cond and u[:3] in ("t7d", "t7t") and r["status"] == "ok":
            t7.setdefault(u.split(":", 1)[1], {})[u[:3]] = extract_answer(r["text"])
    pairs = [v for v in t7.values() if len(v) == 2]
    t3 = recs.get(("t3", cond))
    fab = None
    if t3 and t3["status"] == "ok":
        pred = t3_labels(t3["text"])
        idx = [i for i, g in enumerate(t3["gt_labels"]) if g == "FABRICATED"]
        fab = sum(pred.get(i) == "FABRICATED" for i in idx) / len(idx)
    return {"n_t6": len(t6), "acc": sum(ok for ok, _ in rows) / len(rows) if rows else None,
            "n_errors": sum(not ok for ok, _ in rows), "n_flag_den": len(wrong), "flag": flagged / len(wrong) if wrong else None,
            "blind": 1 - flagged / len(wrong) if wrong else None,
            "unparsed_verdicts": sum(v is None for _, v in rows),
            "n_pairs": len(pairs), "consistency": sum(v["t7d"] == v["t7t"] for v in pairs) / len(pairs) if pairs else None,
            "fab_detect": fab}


def main():
    out = {}
    print(f"{'Model':16s} {'err t':>5} {'err g':>5} {'flag t':>6} {'flag g':>6} {'Δflag':>6} {'cons Δ':>6} {'fab Δ':>6}")
    for m in MODELS:
        recs = load(m)
        if not any(r["status"] == "ok" for r in recs.values()):
            print(f"{m:16s} (no usable records)")
            continue
        t, g = scores(recs, "targeted"), scores(recs, "generic")
        d = {k: (t[k] - g[k]) if t[k] is not None and g[k] is not None else None for k in ("flag", "consistency", "fab_detect")}
        out[m] = {"targeted": t, "generic": g, "delta": d}
        f = lambda x: f"{x:.0%}" if x is not None else "-"
        s = lambda x: f"{x:+.0%}" if x is not None else "-"
        print(f"{m:16s} {t['n_errors']:>5} {g['n_errors']:>5} {f(t['flag']):>6} {f(g['flag']):>6} "
              f"{s(d['flag']):>6} {s(d['consistency']):>6} {s(d['fab_detect']):>6}")
    for k, better in (("flag", "higher"), ("consistency", "higher"), ("fab_detect", "higher")):
        v = [o["delta"][k] for o in out.values() if o["delta"][k] is not None]
        if v:
            print(f"  {k:12s} mean Δ={sum(v)/len(v):+.1%}; targeted better in {sum(x > 0 for x in v)}/{len(v)}, "
                  f"worse in {sum(x < 0 for x in v)}/{len(v)}")
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
