"""
Reconstruct source-item text into the stripped MetaLoop release.

The public release ships NO verbatim text from preexisting datasets. Every
record whose text was stripped carries a `_rejoin` id (or `id`/`original_id`)
that points into data/qbank_manifest.json. This script fetches the source
rows at the manifest-pinned revisions and re-attaches `question` / `choices`
to each stripped record, so all per-item files become fully readable and
every table recomputes.

    pip install datasets huggingface_hub
    # GPQA Diamond is gated: accept at https://huggingface.co/datasets/Idavidrein/gpqa
    # then `huggingface-cli login` (or set HF_TOKEN). GPQA model-response prose
    # is not shipped and is not restored (contamination norm); scored fields for
    # GPQA items are present, so aggregate tables still recompute.
    python scripts/fetch_and_rejoin.py [path/to/file_or_dir ...]

With no args, rejoins every *.json under tasks/*/results/ and data/.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = json.loads((ROOT / "data" / "qbank_manifest.json").read_text())
MAN = {it["id"]: it for it in MANIFEST["items"]}


def build_text_index():
    """id -> {'question':..., 'choices':...} by fetching source rows.

    Reuses fetch_qbank's normalisers to pull each source at its pinned
    revision and select rows by row_index."""
    import fetch_qbank as fq  # co-located; does the actual dataset loads
    return fq.build_qbank(as_dict=True)   # {id: {question, answer, choices, ...}}


def apply_ops(source_text, ops):
    """Replay a build-time edit script to reproduce a CLARIFY item exactly."""
    out = []
    for op in ops:
        if op[0] == "keep":
            out.append(source_text[op[1]:op[2]])
        elif op[0] == "ins":
            out.append(op[1])
        # "drop": source span omitted
    return "".join(out)


def rejoin_record(rec, bank):
    if not isinstance(rec, dict):
        return
    # Underspecified CLARIFY item: replay the deterministic edit on the source.
    if rec.get("_transform_ops") and rec.get("_rejoin_original"):
        o = bank.get(rec["_rejoin_original"])
        if o and o.get("question"):
            rec["question"] = apply_ops(o["question"], rec["_transform_ops"])
            rec["original_question"] = o.get("question")
            rec["original_answer"] = o.get("answer")
        return
    rid = rec.get("_rejoin") or rec.get("id") or rec.get("question_id")
    src = bank.get(rid)
    if not src:
        return  # GPQA orphan / MetaLoop-original: nothing to restore
    if rec.get("question") in (None, ""):
        rec["question"] = src.get("question")
    if rec.get("choices") in (None, "") and src.get("choices"):
        rec["choices"] = src["choices"]


def walk(obj, bank):
    if isinstance(obj, dict):
        lk = next((k for k in ("history", "results", "raw_responses") if isinstance(obj.get(k), list)), None)
        if lk:
            for r in obj[lk]:
                rejoin_record(r, bank)
            return
        rejoin_record(obj, bank)
        for v in obj.values():
            walk(v, bank)
    elif isinstance(obj, list):
        for v in obj:
            walk(v, bank)


def main():
    targets = [Path(a) for a in sys.argv[1:]] or [ROOT / "tasks", ROOT / "data"]
    files = []
    for t in targets:
        files += [t] if t.is_file() else sorted(t.rglob("*.json"))
    print(f"Fetching source text at pinned revisions for {len(MAN)} manifest items…")
    bank = build_text_index()
    print(f"Rejoining {len(files)} files…")
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        walk(data, bank)
        f.write_text(json.dumps(data, indent=1))
    print("Done. (GPQA prose intentionally not restored.)")


if __name__ == "__main__":
    main()
