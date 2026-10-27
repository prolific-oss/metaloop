"""
Build Q_BANK v2: GPQA Diamond options reordered (Oct 2026).

In v1 (data/question_bank_annotated.json, used by the March-April 2026 runs)
every GPQA Diamond item lists its options in the source's column order
[Correct Answer, Incorrect Answer 1, 2, 3], so the correct answer is always A.
v2 gives each GPQA item a fixed option order in which the correct answer's
position is balanced across A-D (seeded), and the incorrect answers are
shuffled. All other items are unchanged. The two GPQA items removed from the
bank on 10 April 2026 (q_437, q_451) get orders from the same procedure.

Writes data/question_bank_v2.json (454 items), data/question_bank_v2_removed.json
(2 items), and the per-item option orders into data/qbank_manifest.json
(field "option_order": upstream column names in A-D order), so that
scripts/fetch_qbank.py reconstructs v2 from the source dataset.

v1 is kept unchanged: results from the original runs are scored against it.

    python scripts/build_qbank_v2.py
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SEED = "metaloop-qbank-v2-gpqa"
COLS = ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
LETTERS = ["A", "B", "C", "D"]


def orders(ids):
    """Balanced correct-answer positions over the sorted ids, incorrect answers shuffled; seeded."""
    rng = random.Random(SEED)
    ids = sorted(ids)
    positions = [i % 4 for i in range(len(ids))]
    rng.shuffle(positions)
    out = {}
    for qid, pos in zip(ids, positions):
        wrong = COLS[1:]
        rng.shuffle(wrong)
        order = wrong[:pos] + [COLS[0]] + wrong[pos:]
        out[qid] = order
    return out


def apply(item, order):
    ch = item["choices"] if isinstance(item["choices"], dict) else eval(item["choices"])
    by_col = dict(zip(COLS, ch.values()))  # v1 order is exactly COLS (checked against the source)
    assert str(by_col["Correct Answer"]).strip() == str(item["answer"]).strip(), item["id"]
    new = dict(item)
    new["choices"] = {L: by_col[c] for L, c in zip(LETTERS, order)}
    return new


def main():
    bank = json.loads((DATA / "question_bank_annotated.json").read_text())
    removed = json.loads((DATA / "question_bank_removed_2026-04-10.json").read_text())["items"]
    gp = [q["id"] for q in bank + removed if q["source"] == "gpqa_diamond"]
    od = orders(gp)
    v2 = [apply(q, od[q["id"]]) if q["id"] in od else q for q in bank]
    v2_removed = [apply(q, od[q["id"]]) for q in removed]
    (DATA / "question_bank_v2.json").write_text(json.dumps(v2, indent=2))
    (DATA / "question_bank_v2_removed.json").write_text(json.dumps(
        {"note": "GPQA Diamond items removed from the bank on 2026-04-10 (Grok provider errors); q_451 is in the Self-Review/Consistency pool.",
         "items": v2_removed}, indent=2))
    man = json.loads((DATA / "qbank_manifest.json").read_text())
    for m in man["items"]:
        if m["id"] in od:
            m["option_order"] = od[m["id"]]
    rows = {"q_437": 71, "q_451": 88}
    man["removed_items"] = [{"id": q["id"], "source": q["source"], "row_index": rows[q["id"]], "domain": q["domain"],
                             "source_difficulty": q["source_difficulty"], "format": q["format"],
                             "delean_scores": q["delean_scores"], "delean_aggregates": q["delean_aggregates"],
                             "option_order": od[q["id"]], "removed_on": "2026-04-10"} for q in removed]
    man["manifest_version"] = 2
    (DATA / "qbank_manifest.json").write_text(json.dumps(man, indent=2))
    from collections import Counter
    pos = Counter(LETTERS[o.index(COLS[0])] for o in od.values())
    print(f"GPQA items reordered: {len(od)}; correct-answer positions: {dict(sorted(pos.items()))}")


if __name__ == "__main__":
    main()
