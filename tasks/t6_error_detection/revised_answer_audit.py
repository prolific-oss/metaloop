"""
Audit the Self-Review REVISED ANSWER extraction (preprint, Sep 2026).

The original extractor (run.py:extract_revised_answer) used
`([A-Za-z]|SAME)`, which captures "SAME" as the letter "S" and does not
normalise case, so `final_answer` / `phase2_correct` are unreliable. This
re-extracts the revised answer two ways and compares them:

  1. regex_v2: whole-token match (SAME / NONE / a valid option letter for the
     item), bold-tolerant, last declaration wins.
  2. three LLM judges (OpenRouter), extraction only: each sees the review
     text and the item's valid option letters, never the model identity,
     the gold answer, the original answer, or the regex output.

Scope (--judge): every Phase-1-wrong item, every Phase-1-correct item the
review flagged (false criticism), and a seeded 10% sample of unflagged
correct items. Judge calls are cached in extraction_audit/.

    python tasks/t6_error_detection/revised_answer_audit.py            # regex_v2 only
    python tasks/t6_error_detection/revised_answer_audit.py --judge    # + LLM judges
"""
import ast
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results" / "gen_review_qbank"
AUDIT = HERE / "extraction_audit"
CACHE = AUDIT / "revised_answer_judge_cache.jsonl"
OUT = AUDIT / "revised_answer_audit.json"
BAD_GT = {"mmlu_236", "mmlu_281", "q_432"}
JUDGES = ["anthropic/claude-sonnet-4.6", "openai/gpt-5.4", "google/gemini-3.1-pro-preview"]
ROSTER = [
    "anthropic_claude-sonnet-4", "anthropic_claude-sonnet-4.6", "anthropic_claude-opus-4.6",
    "openai_gpt-4.1", "openai_gpt-5.4", "google_gemini-2.5-pro", "google_gemini-3.1-pro-preview",
    "google_gemma-4-31b-it", "deepseek_deepseek-v3.2", "x-ai_grok-4.1-fast",
    "qwen_qwen3-235b-a22b", "mistralai_mistral-large-2512",
]

REVISED = re.compile(
    r"REVISED\s*ANSWER\s*\**\s*:?\s*\**\s*[\(\[]?\s*(SAME|NONE|NO\s+CHANGE|[A-J])(?![A-Za-z])",
    re.I,
)

JUDGE_PROMPT = """Below is the end of a model's self-review of its own answer to a multiple-choice question. The valid option letters are: {letters}.

Your job is ONLY to extract what final answer the review commits to. Do not solve the question or judge correctness.

Reply with exactly one token:
- a single option letter from {letters}, if the review states a revised/final answer letter
- SAME, if the review says the answer stays unchanged
- NONE, if the review gives no final answer at all

REVIEW:
<<<
{review}
>>>"""


def valid_letters(item):
    ch = item["choices"]
    ch = ast.literal_eval(ch) if isinstance(ch, str) else ch
    return sorted(ch)


def regex_v2(text, letters):
    hits = [h.upper() for h in REVISED.findall(text or "")]
    hits = ["SAME" if h.replace(" ", "") == "NOCHANGE" else h for h in hits]
    hits = [h for h in hits if h in ("SAME", "NONE") or h in letters]
    return hits[-1] if hits else "NONE"


def resolve(extracted, original):
    """SAME/NONE mean the answer stands."""
    return original if extracted in ("SAME", "NONE") else extracted


def load_items():
    qbank = {q["id"]: q for q in json.load(open(ROOT / "data" / "question_bank_annotated.json"))}
    rows = []
    for stem in ROSTER:
        for r in json.load(open(RESULTS / f"{stem}.json")):
            if "error" in r or r["id"] in BAD_GT or r.get("model_answer") is None:
                continue
            # q_451 (GPQA, 4 options) was evaluated but is missing from the bank.
            letters = valid_letters(qbank[r["id"]]) if r["id"] in qbank else list("ABCD")
            rows.append({
                "model": stem, "id": r["id"], "gt": r["gt_answer"], "original": r["model_answer"],
                "phase1_correct": r["phase1_correct"] is True,
                "flagged": r["outcome"] in ("SELF_CORRECTED", "FALSE_CRITICISM"),
                "letters": letters, "review": r.get("phase2_response") or "",
                "old_final": r.get("final_answer"),
                "regex_v2": regex_v2(r.get("phase2_response"), letters),
            })
    return rows


def in_judge_scope(row, rng):
    if not row["phase1_correct"] or row["flagged"]:
        return True
    return rng.random() < 0.10


def call(model, prompt, key):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000, "temperature": 0}
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=180))
            return (resp["choices"][0]["message"]["content"] or "").strip()
        except (urllib.error.URLError, KeyError, TimeoutError) as e:
            if attempt == 3:
                return f"ERR:{e}"
            time.sleep(2 ** attempt * 3)


def parse_judge(text, letters):
    tok = re.findall(r"\b(SAME|NONE|[A-J])\b", (text or "").upper())
    tok = [t for t in tok if t in ("SAME", "NONE") or t in letters]
    return tok[-1] if tok else "UNPARSED"


def run_judges(rows):
    load_dotenv(ROOT / ".env", override=True)
    key = os.environ["OPENROUTER_API_KEY"]
    cache = {}
    if CACHE.exists():
        for line in CACHE.open():
            c = json.loads(line)
            cache[(c["model"], c["id"], c["judge"])] = c["raw"]
    todo = [(r, j) for r in rows if r.get("in_scope") for j in JUDGES
            if (r["model"], r["id"], j) not in cache]
    print(f"judge calls: {len(todo)} to run, {len(cache)} cached")

    def work(rj):
        r, j = rj
        # The final declaration is at the end; keep the prompt short.
        prompt = JUDGE_PROMPT.format(letters=", ".join(r["letters"]), review=r["review"][-4000:])
        return r, j, call(j, prompt, key)

    with ThreadPoolExecutor(8) as ex, CACHE.open("a") as fh:
        for i, (r, j, raw) in enumerate(ex.map(work, todo)):
            cache[(r["model"], r["id"], j)] = raw
            fh.write(json.dumps({"model": r["model"], "id": r["id"], "judge": j, "raw": raw}) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")
    for r in rows:
        if r.get("in_scope"):
            votes = [parse_judge(cache[(r["model"], r["id"], j)], r["letters"]) for j in JUDGES]
            r["judge_votes"] = votes
            top = max(set(votes), key=votes.count)
            r["judge_majority"] = top if votes.count(top) >= 2 else "SPLIT"


def main():
    rows = load_items()
    rng = random.Random(20260929)
    for r in rows:
        r["in_scope"] = in_judge_scope(r, rng)
    if "--judge" in sys.argv:
        run_judges(rows)
    for r in rows:
        r["final_v2"] = resolve(r["regex_v2"], r["original"])
    n_scope = sum(r["in_scope"] for r in rows)
    changed = sum(1 for r in rows if r["final_v2"] != r["original"])
    print(f"{len(rows)} reviews, {n_scope} in judge scope, {changed} revise the answer (regex_v2)")
    out = [{k: v for k, v in r.items() if k != "review"} for r in rows]
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
