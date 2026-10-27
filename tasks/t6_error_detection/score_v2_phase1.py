"""
Score v2 Self-Review Phase-1 answers (October 2026).

Rule (as in the corrected pipeline): strict parse = the last "ANSWER: <letter>"
whose letter is a valid option. Three blind LLM judges (the panel of
revised_answer_audit.py, same prompt as analyse_fresh_matched.py) read every
response with no strict answer, plus a seeded 10% of the rest as an audit; the
judge majority is used where judged (three-way split: strict parse). Refused or
empty responses have no answer. Main run (repeat 0) only.

Writes results_v2/phase1_scored/<model>.json: {item: {answer, correct, strict,
judge_votes, status}}, which The-Loop v3 and Control 2 read.

    python tasks/t6_error_detection/score_v2_phase1.py [model ...]
"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))
from dotenv import load_dotenv  # noqa: E402

import llm  # noqa: E402
import v2run  # noqa: E402
from analyse_fresh_matched import JUDGE_PROMPT, STRICT, parse_judge  # noqa: E402
from revised_answer_audit import JUDGES, call  # noqa: E402

RES = HERE / "results_v2"
OUT = RES / "phase1_scored"
CACHE = RES / "judge_cache_phase1.jsonl"


def strict(text, letters):
    hits = [h.upper() for h in STRICT.findall(text or "") if h.upper() in letters]
    return hits[-1] if hits else None


def main():
    load_dotenv(v2run.ROOT / ".env", override=True)
    key = os.environ["OPENROUTER_API_KEY"]
    models = [a for a in sys.argv[1:]] or [p.stem for p in RES.glob("*.jsonl")]
    cache = {}
    if CACHE.exists():
        for line in CACHE.open():
            c = json.loads(line)
            cache[(c["model"], c["id"], c["judge"])] = c["raw"]
    for model in models:
        bank = v2run.bank(include_removed=True, version=v2run.bank_version(model))
        recs = [r for r in v2run.Store("t6_error_detection", model).records.values() if r["repeat"] == 0]
        rng = random.Random(f"metaloop-v2-phase1-audit:{model}")
        rows = []
        for r in sorted(recs, key=lambda r: r["unit"]):
            q = bank[r["unit"]]
            letters = sorted(q["choices"])
            gt = [k for k, v in q["choices"].items() if str(v).strip() == str(q["answer"]).strip()][0]
            text = (r.get("phase1") or {}).get("text")
            s = strict(text, letters)
            judge = bool(text) and (s is None or rng.random() < 0.10)
            rows.append({"id": r["unit"], "gt": gt, "letters": letters, "text": text, "strict": s, "judge": judge,
                         "status": (r.get("phase1") or {}).get("status", r["status"])})
        todo = [(x, j) for x in rows if x["judge"] for j in JUDGES if (model, x["id"], j) not in cache]

        def work(xj):
            x, j = xj
            return x, j, call(j, JUDGE_PROMPT.format(letters=", ".join(x["letters"]), response=x["text"][-4000:]), key)
        with ThreadPoolExecutor(8) as ex, CACHE.open("a") as fh:
            for x, j, raw in ex.map(work, todo):
                cache[(model, x["id"], j)] = raw
                fh.write(json.dumps({"model": model, "id": x["id"], "judge": j, "raw": raw}) + "\n")
        out, agree, judged = {}, 0, 0
        for x in rows:
            votes = None
            ans = x["strict"]
            if x["judge"]:
                judged += 1
                votes = [parse_judge(cache[(model, x["id"], j)], x["letters"]) for j in JUDGES]
                top = max(set(votes), key=votes.count)
                if votes.count(top) >= 2:
                    ans = None if top in ("NONE", "UNPARSED") else top
                agree += ans == x["strict"]
            out[x["id"]] = {"answer": ans, "correct": ans == x["gt"], "strict": x["strict"], "judge_votes": votes,
                            "status": x["status"], "gt": x["gt"]}
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{model}.json").write_text(json.dumps(out, indent=1))
        n = len(out)
        print(f"{model:16s} items {n}  answered {sum(v['answer'] is not None for v in out.values())}  "
              f"accuracy {sum(v['correct'] for v in out.values()) / max(n, 1):.1%}  judged {judged} (agree w/ strict {agree})  "
              f"errors {sum(v['answer'] is not None and not v['correct'] for v in out.values())}")


if __name__ == "__main__":
    main()
