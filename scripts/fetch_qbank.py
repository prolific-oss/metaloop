"""
Reconstruct the full Q_BANK from canonical source datasets.

Reads data/qbank_manifest.json (shipped with this repo; row-index pointers +
paper-A's DeLeAn annotations, no upstream text). Pulls each source dataset
at the manifest-pinned revision, picks rows by row_index, normalises into a
uniform {question, answer, choices, ...} shape, and writes the full Q_BANK.

We do not redistribute source items; users fetch them from canonical providers
under each source's licence. GPQA Diamond is gated: accept the licence at
https://huggingface.co/datasets/Idavidrein/gpqa, then `huggingface-cli login`
(or set HF_TOKEN) before running this.

Usage:
    pip install datasets huggingface_hub
    python scripts/fetch_qbank.py          # v2 (current): GPQA options in the manifest's option_order
    python scripts/fetch_qbank.py --v1     # v1: the bank as used by the March-April 2026 runs

Output:
    v2: data/question_bank_v2.json and data/question_bank_v2_removed.json
    v1: data/question_bank_annotated.json (+ per-task mirrors)

v1 lists every GPQA Diamond item's options in the source's column order, so
the correct answer is always A; v2 fixes this with a balanced, seeded option
order per item (scripts/build_qbank_v2.py). Results from the original runs
are scored against v1; the October 2026 GPQA reruns use v2.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import urllib.request
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_IN = ROOT / "data" / "qbank_manifest.json"
QBANK_OUT = ROOT / "data" / "question_bank_annotated.json"

# The task runners (t6, t7, and t8 which shares t6's bank) expect the qbank
# at per-task paths. After writing the canonical bank we fan it out so the
# runners can find it without further setup.
TASK_QBANK_MIRRORS = [
    ROOT / "tasks" / "t6_error_detection" / "qbank.json",
    ROOT / "tasks" / "t7_consistency" / "qbank.json",
]


# Per-source normalisers: turn an upstream row into our uniform shape.

def normalise_arc(row: dict) -> dict:
    choices = dict(zip(row["choices"]["label"], row["choices"]["text"]))
    return {
        "question": row["question"],
        "answer": choices.get(row["answerKey"], row["answerKey"]),
        "choices": choices,
    }


def normalise_mmlu(row: dict) -> dict:
    choices = dict(zip(["A", "B", "C", "D"], row["choices"]))
    return {
        "question": row["question"],
        "answer": row["choices"][row["answer"]],
        "choices": choices,
    }


def normalise_mmlu_pro(row: dict) -> dict:
    options = row["options"]
    labels = [chr(ord("A") + i) for i in range(len(options))]
    choices = dict(zip(labels, options))
    return {
        "question": row["question"],
        "answer": options[row["answer_index"]],
        "choices": choices,
    }


def normalise_boolq(row: dict) -> dict:
    return {
        "question": row["question"],
        "answer": "Yes" if row["answer"] else "No",
        "choices": {"A": "Yes", "B": "No"},
    }


def normalise_simpleqa(row: dict) -> dict:
    return {
        "question": row["problem"],
        "answer": row["answer"],
        "choices": None,
    }


def normalise_gpqa(row: dict, option_order: list[str] | None = None) -> dict:
    correct = row["Correct Answer"]
    # v1 order: correct first (the flaw fixed in v2); v2 passes the manifest's option_order.
    order = option_order or ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]
    options = [row[c] for c in order]
    labels = ["A", "B", "C", "D"]
    choices = dict(zip(labels, options))
    return {
        "question": row["Question"],
        "answer": correct,
        "choices": choices,
    }


NORMALISERS = {
    "arc_easy": normalise_arc,
    "arc_challenge": normalise_arc,
    "simpleqa": normalise_simpleqa,
    "mmlu": normalise_mmlu,
    "mmlu_pro": normalise_mmlu_pro,
    "boolq": normalise_boolq,
    "gpqa_diamond": normalise_gpqa,
}


def fetch_source_rows(source: str, spec: dict) -> list[dict]:
    """Pull all rows for one source at the manifest-pinned revision."""
    if spec["kind"] == "hf":
        from huggingface_hub import get_token  # HF_TOKEN or the token saved by `hf auth login`
        if source == "gpqa_diamond" and not get_token():
            print("WARNING: GPQA Diamond is gated; accept the licence at "
                  "https://huggingface.co/datasets/Idavidrein/gpqa and run "
                  "`hf auth login` (or set HF_TOKEN) before re-running.")
        kwargs = {"split": spec["split"], "revision": spec["revision"]}
        if spec.get("config"):
            ds = load_dataset(spec["repo"], spec["config"], **kwargs)
        else:
            ds = load_dataset(spec["repo"], **kwargs)
        return list(ds)
    elif spec["kind"] == "csv":
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "metaloop/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        got = hashlib.sha256(data).hexdigest()
        if got != spec["sha256"]:
            raise SystemExit(
                f"{source}: CSV at {spec['url']} has sha256={got}, expected "
                f"{spec['sha256']}. Upstream has changed since this manifest "
                f"was built; cannot reconstruct deterministically."
            )
        return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    else:
        raise ValueError(f"unknown source kind: {spec['kind']!r}")


V1 = "--v1" in __import__("sys").argv


def build_qbank(as_dict: bool = False, v1: bool = False, removed: bool = False):
    """Fetch source rows at pinned revisions and assemble the Q_BANK.

    Returns a list of items, or (as_dict=True) an {id: item} mapping — used by
    both this script's main() and scripts/fetch_and_rejoin.py."""
    if not MANIFEST_IN.exists():
        raise SystemExit(f"Manifest not found at {MANIFEST_IN}")
    manifest = json.load(open(MANIFEST_IN))
    sources_block = manifest["sources"]
    items = manifest.get("removed_items", []) if removed else manifest["items"]
    print(f"Loaded {len(items)} manifest entries; manifest_version={manifest.get('manifest_version')}")

    rows_by_source: dict[str, list[dict]] = {}
    skipped_sources: list[str] = []
    for src in sorted({m["source"] for m in items}):
        if src not in sources_block:
            raise SystemExit(f"Manifest item references source {src!r} not declared in 'sources' block")
        print(f"Loading {src}...")
        try:
            rows_by_source[src] = fetch_source_rows(src, sources_block[src])
        except Exception as e:
            # e.g. GPQA gated with no token. Skip this source and continue;
            # the rest of the bank still reconstructs.
            print(f"  SKIP {src}: {type(e).__name__}: {str(e)[:120]}")
            skipped_sources.append(src)

    qbank = []
    out_of_range: list[str] = []
    for entry in items:
        src = entry["source"]
        if src not in rows_by_source:
            continue  # source was skipped
        rows = rows_by_source[src]
        i = entry["row_index"]
        if not (0 <= i < len(rows)):
            out_of_range.append(entry["id"])
            continue
        if src == "gpqa_diamond":
            item = normalise_gpqa(rows[i], None if v1 else entry.get("option_order"))
        else:
            item = NORMALISERS[src](rows[i])
        qbank.append({
            "id": entry["id"],
            **item,
            "domain": entry["domain"],
            "source": entry["source"],
            "source_difficulty": entry["source_difficulty"],
            "format": entry["format"],
            "delean_scores": entry["delean_scores"],
            "delean_aggregates": entry["delean_aggregates"],
        })
    if out_of_range:
        print(f"WARNING: {len(out_of_range)} items had row_index out of range: {out_of_range[:10]}")
    if skipped_sources:
        print(f"NOTE: skipped {skipped_sources} (inaccessible/gated); reconstructed "
              f"{len(qbank)} of {len(items)} items. Scored fields for skipped-source "
              f"items remain in the results; only their text is unrecovered.")
    return {q["id"]: q for q in qbank} if as_dict else qbank


def main():
    if not V1:
        out = ROOT / "data" / "question_bank_v2.json"
        out.write_text(json.dumps(build_qbank(), indent=2))
        print(f"Wrote {out}")
        rem = ROOT / "data" / "question_bank_v2_removed.json"
        rem.write_text(json.dumps({"note": "GPQA Diamond items removed from the bank on 2026-04-10; q_451 is in the Self-Review/Consistency pool.",
                                   "items": build_qbank(removed=True)}, indent=2))
        print(f"Wrote {rem}")
        return
    qbank = build_qbank(v1=True)
    out_of_range: list[str] = []
    if out_of_range:
        print(f"WARNING: {len(out_of_range)} items had row_index out of range "
              f"(upstream likely changed since the manifest was pinned):")
        for mid in out_of_range[:10]:
            print(f"  {mid}")

    QBANK_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(QBANK_OUT, "w") as f:
        json.dump(qbank, f, indent=2)
    print(f"Wrote {len(qbank)} items to {QBANK_OUT}")

    for mirror in TASK_QBANK_MIRRORS:
        mirror.parent.mkdir(parents=True, exist_ok=True)
        with open(mirror, "w") as f:
            json.dump(qbank, f, indent=2)
        print(f"Mirrored to {mirror}")


if __name__ == "__main__":
    main()
