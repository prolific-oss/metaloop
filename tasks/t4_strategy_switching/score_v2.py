"""
Score v2 Strategy-Switching transcripts (October 2026), with the rules validated
on the original runs (validate_judge.py, rescore_correctness.py):

- switch, submission and correctness: the original Claude Opus 4.6 judge
  (judge.JUDGE_TEMPLATE, temperature 0, 1024 tokens), parsed bracket-tolerantly;
- Knights and Knaves correctness: the three-judge panel with KNIGHTS_PROMPT;
- Graph Colouring correctness: programmatic check of the submitted colouring;
- Maze correctness: programmatic check of the submitted path (2026-10-16). The prompt asks for
  any path from S to E, but the judge compared answers with one reference shortest path; the
  v2 judge validation (validate_judge_v2.py) found it marking valid paths wrong on 5 of 12
  sampled Maze records. A path is correct if, after prepending the start when omitted, every
  step moves to a 4-adjacent open cell and it ends at the exit;
- a record the judge says submitted nothing is incorrect.

Records whose final turn has no visible text (truncated before any output,
refused, empty) are not judged: no submission, no switch, incorrect, and they
are counted per model. Metrics are reported with these records in the
denominator (main) and without them (sensitivity).

Writes results_v2/scored/<model>.jsonl and prints the per-model table.

    python tasks/t4_strategy_switching/score_v2.py [model ...]
"""
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))
from judge import JUDGE_TEMPLATE, format_transcript  # noqa: E402
from rescore_correctness import KNIGHTS_PROMPT, colouring_valid, wilson  # noqa: E402
from run import TASK_MODULES  # noqa: E402
from run_v2 import problems  # noqa: E402
from v2score import Panel, majority  # noqa: E402

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
OPUS = "anthropic/claude-opus-4.6"
COORD = re.compile(r"\((\d+)\s*,\s*(\d+)\)")


def maze_valid(r, task):
    """The last PATH: in the final turn is a walk through open cells from the start to the exit."""
    final = r["messages"][-1]["content"] or ""
    hits = list(re.finditer(r"PATH:\s*(.+)", final, re.I))
    if not hits:
        return False
    path = [(int(a), int(b)) for a, b in COORD.findall(hits[-1].group(1))]
    grid = task["maze_grid"]
    grid = ast.literal_eval(grid) if isinstance(grid, str) else grid
    start, exit_ = tuple(task["start"]), tuple(task["exit"])
    if not path or path[-1] != exit_:
        return False
    if path[0] != start:
        path = [start] + path
    open_ = lambda c: 0 <= c[0] < len(grid) and 0 <= c[1] < len(grid[0]) and grid[c[0]][c[1]] != "#"  # noqa: E731
    return all(open_(c) for c in path) and all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))


# Cipher is reported separately, not in the main rates (decided 2026-10-14). With reasoning
# on, GPT-5.4, Opus 4.6 and Sonnet 4.6 spend the whole 32,768-token limit on hidden reasoning
# and show no text to judge, and Anthropic's API refuses most Cipher prompts for Opus 5.5 and
# Sonnet 5.5. Its row gives submitted / cut off / refused per model.
CIPHER_NOTE = "cipher excluded from main rates"


def yn(text, key):
    m = re.search(rf"{key}\**\s*:\s*\**\s*\[?\s*(YES|NO)", text or "", re.I)
    return m.group(1).upper() == "YES" if m else None


def opus_prompt(r, tt, task):
    mod = TASK_MODULES[tt]
    post = r["messages"][r["n_prefill"]:]
    return JUDGE_TEMPLATE.format(judge_context=getattr(mod, "JUDGE_CONTEXT", ""),
                                 correct_answer=r["module_result"].get("correct_answer", ""),
                                 transcript=format_transcript(post) if post else "(no response)")


def knights_prompt(r, task):
    return KNIGHTS_PROMPT.format(puzzle="\n\n".join(m["content"] for m in r["messages"][:r["n_prefill"]]),
                                 solution=", ".join(f"{k} is a {v}" for k, v in task["solution"].items()),
                                 final=r["messages"][-1]["content"])


def has_output(r):
    return any((m.get("content") or "").strip() for m in r["messages"][r["n_prefill"]:] if m["role"] == "assistant")


def score(model, probs, opus, panel):
    recs = load_records(model)
    live = [r for r in recs if r.get("messages") and has_output(r)]
    o_raw = opus.run([opus_prompt(r, *probs[r["unit"]]) for r in live])
    k_raw = panel.run([knights_prompt(r, probs[r["unit"]][1]) for r in live if probs[r["unit"]][0] == "knights"])
    out = []
    for r in recs:
        tt, task = probs[r["unit"]]
        row = {"unit": r["unit"], "repeat": r["repeat"], "task_type": tt, "status": r["status"],
               "judged": False, "submitted": False, "switched": False, "correct": False, "judge_ok": None}
        if r in live:
            raw = o_raw[opus_prompt(r, tt, task)][0]
            row.update(judged=True, judge_ok=raw is not None and yn(raw, "SWITCHED") is not None,
                       submitted=bool(yn(raw, "SUBMITTED")), switched=bool(yn(raw, "SWITCHED")),
                       correct=bool(yn(raw, "CORRECT")))
            if tt == "knights":
                votes = [yn(v, "CORRECT") for v in k_raw[knights_prompt(r, task)]]
                row["knights_votes"] = votes
                row["correct"] = majority(votes) is True
            if tt == "graph_coloring":
                row["correct"] = colouring_valid(r, task)
            if tt == "maze":
                row["judge_correct"] = row["correct"]
                row["correct"] = maze_valid(r, task)
            if not row["submitted"]:
                row["correct"] = False
        out.append(row)
    (SC / "scored").mkdir(parents=True, exist_ok=True)
    (SC / "scored" / f"{model}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in sorted(out, key=lambda x: (x["unit"], x["repeat"]))))
    return out


def table(model, rows):
    """Main metrics exclude Cipher (see CIPHER_NOTE); Cipher gets its own outcome row."""
    allmain = [r for r in rows if r["repeat"] == 0]
    main = [r for r in allmain if r["task_type"] != "cipher"]
    n = len(main)
    sw = sum(r["switched"] for r in main)
    sbw = sum(r["switched"] and not r["correct"] for r in main)
    cor = sum(r["correct"] for r in main)
    noout = [r for r in main if not r["judged"]]
    j = [r for r in main if r["judged"]]
    sbw_j = sum(r["switched"] and not r["correct"] for r in j)
    lo, hi = wilson(sbw, n)
    ci = [r for r in allmain if r["task_type"] == "cipher"]
    cipher = {"n": len(ci), "submitted": sum(r["submitted"] for r in ci), "truncated": sum(r["status"] == "truncated" for r in ci),
              "refused": sum(r["status"] == "blocked" for r in ci), "other_failure": sum(r["status"] not in ("ok", "truncated", "blocked") for r in ci),
              "switched": sum(r["switched"] for r in ci), "correct": sum(r["correct"] for r in ci)}
    # Full suite (all 80 runs, Cipher included): the headline completion and coverage numbers;
    # the main rates above are the without-Cipher sensitivity analysis.
    fs = allmain
    full = {"n": len(fs), "switch": sum(r["switched"] for r in fs), "sbw": sum(r["switched"] and not r["correct"] for r in fs),
            "correct": sum(r["correct"] for r in fs), "usable": sum(bool(r.get("judged")) for r in fs),
            "truncated": sum(r["status"] == "truncated" for r in fs), "refused": sum(r["status"] == "blocked" for r in fs),
            "other_failure": sum(r["status"] not in ("ok", "truncated", "blocked") for r in fs)}
    print(f"{model:16s} full suite n {full['n']}  solved {full['correct'] / max(full['n'], 1):.0%}  usable {full['usable']}  "
          f"truncated {full['truncated']}  refused {full['refused']}  SbW {full['sbw'] / max(full['n'], 1):.0%}")
    print(f"{model:16s} n {n}  switch {sw / n:.0%}  SbW {sbw / n:.0%} [{lo:.0f}, {hi:.0f}]  solved {cor / n:.0%}  "
          f"no-output {len(noout)} {dict(Counter(r['status'] for r in noout))}  SbW excl. no-output {sbw_j / max(len(j), 1):.0%}  "
          f"judge unparsed {sum(r['judge_ok'] is False for r in j)}  | cipher {cipher}")
    return {"n": n, "switch": sw, "sbw": sbw, "correct": cor, "no_output": len(noout), "sbw_excl_no_output": sbw_j,
            "n_judged": len(j), "excludes": "cipher", "cipher": cipher, "full_suite": full}


def main():
    models = sys.argv[1:] or sorted(p.stem for p in (RES32 if LIMIT_32K else RES).glob("*.jsonl") if not p.stem.startswith("judge_cache"))
    probs = problems()
    opus = Panel(RES / "judge_cache_opus.jsonl", judges=[OPUS], max_tokens=1024)
    panel = Panel(RES / "judge_cache_knights.jsonl")
    summary = {m: table(m, score(m, probs, opus, panel)) for m in models}
    out = ROOT / "results_v2_metrics" / ("strategy_switching_32k.json" if LIMIT_32K else "strategy_switching.json")
    out.parent.mkdir(exist_ok=True)
    prev = json.loads(out.read_text()) if out.exists() else {}
    out.write_text(json.dumps({**prev, **summary}, indent=1))


if __name__ == "__main__":
    main()
