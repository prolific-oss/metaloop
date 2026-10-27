"""
Score v2 Source-Monitoring responses (October 2026).

The original parser took labels in order of appearance, so one skipped
statement shifted every later label (and attached each confidence to the next
statement). Here each label is tied to its statement number: the response is
cut at the statement headers 1..32 in order, and within each segment the last
CLASSIFICATION and CONFIDENCE are read. When a header quotes the statement, the
quote is checked against the statement shown at that position. Unreadable
statements are counted as unparsed, not guessed.

Displayed statement i is original statement shuffle_order[i]; its gold label is
labels[shuffle_order[i]].

Metrics per model (main run): accuracy, confabulation detection (recall on
FABRICATED) with Wilson CI, INFERRED accuracy, FABRICATED precision, plus
confidence for correct vs incorrect labels (now that confidences map to the
right statement). Accuracy is reported two ways: over all statements, with
unparsed ones counted wrong, and over the statements the model answered, with
the lost share alongside (the primary figure since 2026-10-16, because some
replies are cut off at the shared 16k limit).

    python tasks/t3_source_monitoring/score_v2.py [model ...]
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from run_v2 import prompt  # noqa: E402
from v2score import confidence, norm_text, wilson  # noqa: E402

RES = HERE / "results_v2"

# --limit-32k: the output-limit sensitivity check. Each model's main records are overlaid
# with its results_v2_32k/ records (a full 32k run, or only the items run at 32k), scored
# into results_v2_32k/scored/ and summarised in results_v2_metrics/<name>_32k.json.
LIMIT_32K = "--limit-32k" in sys.argv
if LIMIT_32K:
    sys.argv.remove("--limit-32k")
RES32 = HERE / "results_v2_32k"
SC = RES32 if LIMIT_32K else RES


def load_records(model):
    """Latest record per (unit, repeat); in --limit-32k mode the 32k records win."""
    out = {}
    for p in ([RES / f"{model}.jsonl"] + ([RES32 / f"{model}.jsonl"] if LIMIT_32K else [])):
        if p.exists():
            for r in map(json.loads, p.open()):
                out[(r["unit"], r["repeat"])] = r
    return list(out.values())
LABELS = ("PASSAGE", "TRAINING", "INFERRED", "FABRICATED")
LAB = re.compile(r"CLASSIFICATION\W*?(PASSAGE|TRAINING|INFERRED|FABRICATED)\b", re.I)
CONF = re.compile(r"CONFIDENCE\W*?(-?\d+(?:\.\d+)?)", re.I)


def header(k):
    # start of a line, optional markdown, the number, then '.', ')' or ':'
    return re.compile(rf"^[ \t>#*_|-]*(?:Statement\s*{k}\b|{k}\s*[.):])", re.I | re.M)


def segments(text, n):
    pos, starts = 0, {}
    for k in range(1, n + 1):
        m = header(k).search(text, pos)
        if m:
            starts[k] = m.start()
            pos = m.end()
    keys = sorted(starts)
    return {k: text[starts[k]:(starts[keys[i + 1]] if i + 1 < len(keys) else len(text))] for i, k in enumerate(keys)}


def parse(text, shown):
    """-> (list over displayed positions of (label|None, confidence|None, quote_ok|None), method).

    method "numbered": segments cut at statement headers. "sequential": the response has
    no headers but exactly one CLASSIFICATION per statement, taken in order. "none" otherwise.
    """
    text = text or ""
    seg = segments(text, len(shown))
    if not seg:
        labs = LAB.findall(text)
        if len(labs) == len(shown):
            blocks = re.split(r"(?=^[ \t*_-]*EVIDENCE)", text, flags=re.I | re.M)
            confs = [CONF.findall(b) for b in blocks if LAB.search(b)]
            confs = confs if len(confs) == len(shown) else [[]] * len(shown)
            return [(l.upper(), confidence(c[-1]) if c else None, None) for l, c in zip(labs, confs)], "sequential"
        return [(None, None, None)] * len(shown), "none"
    out = []
    for k, stmt in enumerate(shown, 1):
        s = seg.get(k, "")
        labs = LAB.findall(s)
        confs = CONF.findall(s)
        lines = s.split("\n", 3)[:3]
        first = [] if re.search(r"EVIDENCE|REASONING", lines[0], re.I) else [lines[0]]
        head = "\n".join(first + [ln for ln in lines[1:] if re.match(r"^\W*statement\W*:", ln, re.I)])
        has_quote = bool(re.search(r"[\"“]", head))
        quote_ok = (norm_text(stmt)[:50] in norm_text(head)) if has_quote else None
        out.append((labs[-1].upper() if labs else None, confidence(confs[-1]) if confs else None, quote_ok))
    return out, "numbered"


def score_model(model):
    recs = load_records(model)
    rows = []
    for r in recs:
        _, order, statements = prompt(r["unit"])
        assert order == r["shuffle_order"], "saved shuffle differs from the seeded one"
        shown = [statements[o][0] for o in order]
        parsed, method = parse(r.get("text"), shown) if r["status"] == "ok" else ([(None, None, None)] * len(order), "none")
        for i, (lab, conf, qok) in enumerate(parsed):
            rows.append({"unit": r["unit"], "repeat": r["repeat"], "status": r["status"], "pos": i + 1, "orig": order[i],
                         "gold": r["labels"][order[i]], "pred": lab, "confidence": conf, "quote_ok": qok,
                         "method": method})
    (SC / "scored").mkdir(parents=True, exist_ok=True)
    (SC / "scored" / f"{model}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    return rows


def metrics(rows):
    m = [r for r in rows if r["repeat"] == 0]
    got = [r for r in m if r["pred"]]
    fab = [r for r in m if r["gold"] == "FABRICATED"]
    inf = [r for r in m if r["gold"] == "INFERRED"]
    called_fab = [r for r in got if r["pred"] == "FABRICATED"]
    det = sum(r["pred"] == "FABRICATED" for r in fab)
    cc = [r["confidence"] for r in got if r["confidence"] is not None and r["pred"] == r["gold"]]
    cw = [r["confidence"] for r in got if r["confidence"] is not None and r["pred"] != r["gold"]]
    fab_got = [r for r in fab if r["pred"]]
    return {"n": len(m), "unparsed": len(m) - len(got), "quote_mismatch": sum(r["quote_ok"] is False for r in m),
            "sequential": sum(r["method"] == "sequential" for r in m),
            "accuracy": round(100 * sum(r["pred"] == r["gold"] for r in m) / len(m), 1),
            "accuracy_answered": round(100 * sum(r["pred"] == r["gold"] for r in got) / max(len(got), 1), 1),
            "lost_pct": round(100 * (len(m) - len(got)) / len(m), 1),
            "confab_detect_answered": round(100 * sum(r["pred"] == "FABRICATED" for r in fab_got) / max(len(fab_got), 1), 1),
            "confab_detect": round(100 * det / len(fab), 1), "confab_ci": [round(x) for x in wilson(det, len(fab))],
            "inferred_acc": round(100 * sum(r["pred"] == "INFERRED" for r in inf) / len(inf), 1),
            "fab_precision": round(100 * sum(r["gold"] == "FABRICATED" for r in called_fab) / max(len(called_fab), 1), 1),
            "conf_correct": round(sum(cc) / len(cc), 1) if cc else None, "conf_wrong": round(sum(cw) / len(cw), 1) if cw else None,
            "n_wrong": len(cw), "pred_dist": dict(Counter(r["pred"] for r in m))}


def main():
    models = sys.argv[1:] or sorted(p.stem for p in (RES32 if LIMIT_32K else RES).glob("*.jsonl"))
    summary = {}
    for model in models:
        s = summary[model] = metrics(score_model(model))
        print(f"{model:16s} acc {s['accuracy']}%  answered {s['accuracy_answered']}% (lost {s['lost_pct']}%)  confab {s['confab_detect']}% {s['confab_ci']}  inferred {s['inferred_acc']}%  "
              f"fab prec {s['fab_precision']}%  conf right/wrong {s['conf_correct']}/{s['conf_wrong']} (n wrong {s['n_wrong']})  "
              f"unparsed {s['unparsed']}/{s['n']}  sequential {s['sequential']}  quote mismatch {s['quote_mismatch']}")
    out = ROOT / "results_v2_metrics" / ("source_monitoring_32k.json" if LIMIT_32K else "source_monitoring.json")
    out.parent.mkdir(exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**prev, **summary}, indent=1))


if __name__ == "__main__":
    main()
