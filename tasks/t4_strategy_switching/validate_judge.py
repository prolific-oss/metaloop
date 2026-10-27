"""
Validate the Strategy-Switching judge (Claude Opus 4.6) against three
independent LLM judges (preprint, Oct 2026).

The original judge saw a per-task-type description of the trap, the correct
answer and the model's continuation after the prefill, but not the prefill
itself. It agrees with the programmatic switch detectors on 65% of records.
To find out which of the two is wrong, a stratified sample (per task type, up
to 6 records where judge and detector disagree and 6 where they agree; seed 0)
is graded by three judges from other providers' panel (the same three used
for answer re-extraction elsewhere in the paper). They see the task context,
the correct answer and the FULL transcript with the prefilled turns marked,
and answer the same three questions as the original judge; for CORRECT they
are told to accept any valid answer where several exist (colourings, maze
paths), whereas the original judge saw one reference answer. A fourth question
(INTENDED) asks whether the model adopted the specific strategy the task
context names, to separate judge error from a loose definition of "switch".

Agreement is reported per task type and overall, reweighted to the full
population of records (each stratum weighted by its population share).

    python tasks/t4_strategy_switching/validate_judge.py
"""
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from rescore_correctness import paper_records  # noqa: E402
from run import TASK_MODULES, TASKS_DIR  # noqa: E402

OUT = HERE / "judge_validation"
CACHE = OUT / "panel_cache.jsonl"
# Result files use the task types' earlier names.
MODULE_NAME = {"mastermind": "code_breaker", "wordle": "letter_guess"}
JUDGES = ["anthropic/claude-sonnet-4.6", "openai/gpt-5.4", "google/gemini-3.1-pro-preview"]
PER_CELL = 6
KEYS = ("submitted", "switched", "correct", "intended")

PROMPT = """You are grading a model's response on a strategy-switching task. The model was given a problem together with the beginning of a solution attempt that it did not write (the PREFILLED turns below). The prefilled approach contains a trap. The model then continued from there.

## Task context
{context}

## Correct answer
{answer}

## Transcript
{transcript}

## Grading

Judge only the model's own (non-prefilled) turns. Answer these three questions. For each, respond with exactly YES or NO, then a one-sentence explanation.

1. SUBMITTED: Did the model output a clear answer (a SUBMIT line, or a clearly stated final answer)?
2. SWITCHED: Did the model recognise the error in the prefilled approach and switch to a different strategy?
3. CORRECT: Is the model's submitted answer correct (matches or is equivalent to the correct answer above)?
4. INTENDED: Did the model adopt the specific alternative strategy that the task context says it should have (not merely re-check or redo the same approach)?

Where a task admits several valid answers (any proper colouring with at most K colours, any valid path to the exit), count any valid answer as correct, checking it against the problem in the transcript.

Format:
SUBMITTED: [YES/NO] - [explanation]
SWITCHED: [YES/NO] - [explanation]
CORRECT: [YES/NO] - [explanation]
INTENDED: [YES/NO] - [explanation]"""


def transcript(r):
    parts = []
    for i, m in enumerate(r["messages"]):
        role = "USER" if m["role"] == "user" else "ASSISTANT"
        tag = " (PREFILLED, not written by the model)" if i < r["n_prefill"] else ""
        parts.append(f"[{role}{tag}]\n{m['content']}")
    return "\n\n".join(parts)


def call(model, prompt, key):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 6000, "temperature": 0}
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=300))
            return (resp["choices"][0]["message"]["content"] or "").strip()
        except (urllib.error.URLError, KeyError, TimeoutError) as e:
            if attempt == 3:
                return f"ERR:{e}"
            time.sleep(2 ** attempt * 3)


def parse(text):
    out = {}
    for k in KEYS:
        m = re.search(rf"{k}\**\s*:\s*\**\s*\[?\s*(YES|NO)", text or "", re.I)
        out[k] = (m.group(1).upper() == "YES") if m else None
    return out


def main():
    load_dotenv(ROOT / ".env", override=True)
    key = os.environ["OPENROUTER_API_KEY"]
    recs = [r for r in paper_records() if r.get("switched") is not None and r.get("messages")]
    cells = defaultdict(list)
    for r in recs:
        cells[(r["task_type"], r["switched"] != r["switched_programmatic"])].append(r)
    rng = random.Random(0)
    sample = []
    for cell in sorted(cells):
        pool = sorted(cells[cell], key=lambda r: (r["model"], r["id"]))
        sample += [dict(r, cell=cell) for r in rng.sample(pool, min(PER_CELL, len(pool)))]
    print(f"{len(recs)} records; {len(sample)} sampled; {len(sample) * len(JUDGES)} panel calls")

    OUT.mkdir(exist_ok=True)
    cache = {}
    if CACHE.exists():
        for line in CACHE.open():
            c = json.loads(line)
            cache[(c["model"], c["id"], c["judge"])] = c["raw"]
    todo = [(r, j) for r in sample for j in JUDGES if (r["model"], r["id"], j) not in cache or cache[(r["model"], r["id"], j)].startswith("ERR")]

    def work(rj):
        r, j = rj
        ctx = getattr(TASK_MODULES[MODULE_NAME.get(r["task_type"], r["task_type"])], "JUDGE_CONTEXT", "")
        return r, j, call(j, PROMPT.format(context=ctx, answer=r["correct_answer"], transcript=transcript(r)), key)

    for r in sample:
        r["correct_answer"] = correct_answer(r)
    with ThreadPoolExecutor(8) as ex, CACHE.open("a") as fh:
        for i, (r, j, raw) in enumerate(ex.map(work, todo)):
            cache[(r["model"], r["id"], j)] = raw
            fh.write(json.dumps({"model": r["model"], "id": r["id"], "judge": j, "raw": raw}) + "\n")
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    rows = []
    for r in sample:
        votes = [parse(cache[(r["model"], r["id"], j)]) for j in JUDGES]
        maj = {}
        for k in KEYS:
            vs = [v[k] for v in votes if v[k] is not None]
            maj[k] = True if sum(vs) >= 2 else False if len(vs) - sum(vs) >= 2 else None
        rows.append({"model": r["model"], "id": r["id"], "task_type": r["task_type"], "disagree_cell": r["cell"][1],
                     "opus": {k: r.get(k) for k in KEYS}, "programmatic_switch": r["switched_programmatic"],
                     "panel_votes": votes, "panel": maj, "unanimous": {k: len({v[k] for v in votes}) == 1 and votes[0][k] is not None for k in KEYS}})
    (OUT / "panel_results.json").write_text(json.dumps(rows, indent=1))

    pop = Counter((r["task_type"], r["switched"] != r["switched_programmatic"]) for r in recs)
    samp = Counter((r["task_type"], r["disagree_cell"]) for r in rows)

    def weighted(pred, tt=None):
        """Population-weighted share of records where pred(row) is True (rows with a panel verdict)."""
        num = den = 0.0
        for cell, n in pop.items():
            if tt and cell[0] != tt:
                continue
            rs = [r for r in rows if (r["task_type"], r["disagree_cell"]) == cell]
            rs = [r for r in rs if pred(r) is not None]
            if rs:
                num += n * sum(pred(r) for r in rs) / len(rs)
                den += n
        return num / den if den else float("nan")

    def agree(a, b):
        return lambda r: None if r["panel"][b] is None else (a(r) == r["panel"][b])

    summary = {}
    print(f"\n{'task':15s} {'n':>3s} | switch: {'Opus=P':>7s} {'Prog=P':>7s} {'unan':>5s} | correct: {'Opus=P':>7s} {'unan':>5s} | submitted Opus=P")
    for tt in sorted({r["task_type"] for r in rows}) + [None]:
        sub = [r for r in rows if tt is None or r["task_type"] == tt]
        s = {
            "n": len(sub),
            "switch_opus_vs_panel": weighted(agree(lambda r: r["opus"]["switched"], "switched"), tt),
            "switch_prog_vs_panel": weighted(agree(lambda r: r["programmatic_switch"], "switched"), tt),
            "switch_panel_unanimous": sum(r["unanimous"]["switched"] for r in sub) / len(sub),
            "correct_opus_vs_panel": weighted(agree(lambda r: r["opus"]["correct"], "correct"), tt),
            "correct_panel_unanimous": sum(r["unanimous"]["correct"] for r in sub) / len(sub),
            "submitted_opus_vs_panel": weighted(agree(lambda r: r["opus"]["submitted"], "submitted"), tt),
            "panel_switch_rate": weighted(lambda r: r["panel"]["switched"], tt),
            "opus_switch_rate": weighted(lambda r: r["opus"]["switched"], tt),
            "panel_intended_rate": weighted(lambda r: r["panel"]["intended"], tt),
            "intended_opus_vs_panel": weighted(agree(lambda r: r["opus"]["switched"], "intended"), tt),
            "intended_prog_vs_panel": weighted(agree(lambda r: r["programmatic_switch"], "intended"), tt),
        }
        summary[tt or "ALL"] = s
        print(f"{tt or 'ALL (weighted)':15s} {s['n']:>3d} | {s['switch_opus_vs_panel']:>15.0%} {s['switch_prog_vs_panel']:>7.0%} {s['switch_panel_unanimous']:>5.0%} | "
              f"{s['correct_opus_vs_panel']:>16.0%} {s['correct_panel_unanimous']:>5.0%} | {s['submitted_opus_vs_panel']:.0%}")
    a = summary["ALL"]
    print(f"\nweighted switch rate: Opus {a['opus_switch_rate']:.0%}, panel {a['panel_switch_rate']:.0%}, panel intended {a['panel_intended_rate']:.0%}")
    for tt, v in summary.items():
        print(f"  {tt:15s} intended-switch agreement: Opus {v['intended_opus_vs_panel']:.0%}  programmatic {v['intended_prog_vs_panel']:.0%}  (panel intended rate {v['panel_intended_rate']:.0%})")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


def correct_answer(r):
    """The correct answer string the original judge saw (as built in each task module's run())."""
    # Task data as run (from git, commit before 1aec2476): the Wordle task file
    # was regenerated after the runs; the other seven are identical to tasks/.
    tt, idx = MODULE_NAME.get(r["task_type"], r["task_type"]), int(r["id"].rsplit("_", 1)[1])
    t = json.loads((OUT / "original_tasks" / f"{r['task_type']}.json").read_text())["problems"][idx]
    return {
        "cipher": lambda: f"PLAINTEXT: {t['plaintext']}",
        "code_breaker": lambda: t["secret"],
        "dijkstra": lambda: f"PATH: {'->'.join(t['bellman_ford_path'])} DISTANCE: {t['bellman_ford_distance']}",
        "knights": lambda: ", ".join(f"{k} is a {v}" for k, v in t["solution"].items()),
        "maze": lambda: "PATH: " + " -> ".join(f"({a},{b})" for a, b in t.get("bfs_path", [])),
        "letter_guess": lambda: t["target"],
        "graph_coloring": lambda: ", ".join(f"{n} -> {t['valid_coloring'][n]}" for n in t["nodes"]),
        "sphere": lambda: f"DISTANCE: {t['correct_distance_km']} km",
    }[tt]()


if __name__ == "__main__":
    main()
