"""
Answers for the matched Control 2 fresh attempts (run_fresh_matched.py).

Every response is parsed with a strict rule (last "ANSWER: <letter>" line whose
letter is a valid option) and read by three blind LLM judges (the judges of
revised_answer_audit.py), who see the end of the response and the valid
letters and return the committed letter or NONE. The final answer is the judge
majority; on a three-way split it falls back to the strict parse. Agreement
between strict parse and judges is reported.

Writes results/fresh_matched_answers.json: {model stem: {item id: {...}}}.

    python tasks/t6_error_detection/analyse_fresh_matched.py
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from revised_answer_audit import JUDGES, ROOT, call, valid_letters

HERE = Path(__file__).parent
FRESH = HERE / "results" / "fresh_matched"
RESULTS = HERE / "results" / "gen_review_qbank"
CACHE = HERE / "extraction_audit" / "fresh_matched_judge_cache.jsonl"
OUT = HERE / "results" / "fresh_matched_answers.json"
STRICT = re.compile(r"ANSWER\s*\**\s*:\s*\**\s*[\[\(]?\s*([A-J])(?![A-Za-z])", re.I)

JUDGE_PROMPT = """Below is the end of a model's answer to a multiple-choice question. The valid option letters are: {letters}.

Your job is ONLY to extract which option the response commits to as its final answer. Do not solve the question or judge correctness.

Reply with exactly one token:
- a single option letter from {letters}, if the response commits to a final answer
- NONE, if it gives no final answer

RESPONSE:
<<<
{response}
>>>"""


def strict(text, letters):
    hits = [h.upper() for h in STRICT.findall(text or "") if h.upper() in letters]
    return hits[-1] if hits else None


def parse_judge(text, letters):
    tok = [t for t in re.findall(r"\b(NONE|[A-J])\b", (text or "").upper()) if t == "NONE" or t in letters]
    return tok[-1] if tok else "UNPARSED"


def main():
    load_dotenv(ROOT / ".env", override=True)
    key = os.environ["OPENROUTER_API_KEY"]
    removed = json.load(open(ROOT / "data" / "question_bank_removed_2026-04-10.json"))["items"]
    qbank = {q["id"]: q for q in json.load(open(ROOT / "data" / "question_bank_annotated.json")) + removed}
    rows = []
    for f in sorted(FRESH.glob("*.json")):
        gt = {r["id"]: r["gt_answer"] for r in json.load(open(RESULTS / f.name)) if "error" not in r}
        for r in json.load(open(f)):
            if "response" not in r:
                continue
            letters = valid_letters(qbank[r["id"]])
            rows.append({"model": f.stem, "id": r["id"], "gt": gt[r["id"]], "letters": letters,
                         "response": r["response"], "route": r["route"], "strict": strict(r["response"], letters)})
    cache = {}
    if CACHE.exists():
        for line in CACHE.open():
            c = json.loads(line)
            cache[(c["model"], c["id"], c["judge"])] = c["raw"]
    todo = [(r, j) for r in rows for j in JUDGES if (r["model"], r["id"], j) not in cache]
    print(f"{len(rows)} responses; judge calls: {len(todo)} to run, {len(cache)} cached")

    def work(rj):
        r, j = rj
        prompt = JUDGE_PROMPT.format(letters=", ".join(r["letters"]), response=r["response"][-4000:])
        return r, j, call(j, prompt, key)

    with ThreadPoolExecutor(8) as ex, CACHE.open("a") as fh:
        for r, j, raw in ex.map(work, todo):
            cache[(r["model"], r["id"], j)] = raw
            fh.write(json.dumps({"model": r["model"], "id": r["id"], "judge": j, "raw": raw}) + "\n")

    out, agree, split = {}, 0, 0
    print(f"{'model':32s} {'n':>3s} {'strict none':>11s} {'final none':>10s} {'correct':>8s}")
    for r in rows:
        votes = [parse_judge(cache[(r["model"], r["id"], j)], r["letters"]) for j in JUDGES]
        top = max(set(votes), key=votes.count)
        if votes.count(top) >= 2:
            final = None if top in ("NONE", "UNPARSED") else top
        else:
            final, split = r["strict"], split + 1
        agree += final == r["strict"]
        out.setdefault(r["model"], {})[r["id"]] = {
            "strict": r["strict"], "judge_votes": votes, "answer": final, "correct": final == r["gt"],
            "gt": r["gt"], "route": r["route"]}
    for m, d in out.items():
        v = list(d.values())
        print(f"{m:32s} {len(v):>3d} {sum(x['strict'] is None for x in v):>11d} {sum(x['answer'] is None for x in v):>10d} "
              f"{sum(x['correct'] for x in v):>3d} ({sum(x['correct'] for x in v) / len(v):.0%})")
    print(f"strict parse = final answer on {agree}/{len(rows)}; judge three-way splits {split}")
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
