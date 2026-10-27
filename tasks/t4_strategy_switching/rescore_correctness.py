"""
Strategy-Switching: corrected correctness and the per-model table (preprint, Oct 2026).

The original LLM judge (Claude Opus 4.6) decided correctness against one
reference answer. A three-judge validation (validate_judge.py) found two task
types where that fails:

- Knights and Knaves asks for "SUBMIT: [Name] is a [knight/knave/spy]", a single
  role, but the judge was given the full three- or four-person solution and
  marked correct single-role submissions wrong. Here three blind judges
  (the validation panel) re-grade every Knights record with an explicit rule:
  correct if every role the model commits to in its final answer matches the
  solution.
- Graph Colouring accepts any proper colouring with at most K colours, but the
  judge saw one reference colouring. Here correctness is checked
  programmatically on the colouring the model submits (the assignments after the
  last SUBMIT in its final turn; colours must be 1..K as the prompt specifies).

Two parsing problems in the original scoring are also fixed. On 9 Knights
records the judge wrote "SUBMITTED: [YES]" with brackets, which the parser read
as missing, so these were scored as neither a switch nor a solution; they are
re-parsed here. On 3 Cipher records the judge call returned nothing; these are
graded by the validation panel (validate_judge.PROMPT, majority of three).
A record whose judge says the model submitted no answer is scored incorrect
(no such record was marked correct); its switch label is kept.

The paper's 12 models; Grok 4.1 is x-ai_grok-4.1-fast.json (analyse_results.py's
SHORT maps the older x-ai/grok-4 run, which is not reported).

    python tasks/t4_strategy_switching/rescore_correctness.py
"""
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = HERE / "results"
VAL = HERE / "judge_validation"
CACHE = VAL / "knights_rescore_cache.jsonl"
OUT = HERE / "rescored_summary.json"
PAPER = [("Sonnet 4", "anthropic_claude-sonnet-4"), ("Sonnet 4.6", "anthropic_claude-sonnet-4.6"),
         ("Opus 4.6", "anthropic_claude-opus-4.6"), ("GPT-4.1", "openai_gpt-4.1"), ("GPT-5.4", "openai_gpt-5.4"),
         ("Gemini 2.5 Pro", "google_gemini-2.5-pro"), ("Gemini 3.1 Pro", "google_gemini-3.1-pro-preview"),
         ("Gemma 4 31B", "google_gemma-4-31b-it"), ("DeepSeek V3.2", "deepseek_deepseek-v3.2"),
         ("Grok 4.1", "x-ai_grok-4.1-fast"), ("Qwen3 235B", "qwen_qwen3-235b-a22b"),
         ("Mistral Large", "mistralai_mistral-large-2512")]

KNIGHTS_PROMPT = """A model solved a knights-and-knaves puzzle in which a person may also be a spy. The puzzle asked it to state its answer as "SUBMIT: [Name] is a [knight/knave/spy]", so a submission may name only one person.

## Puzzle (as shown to the model, including an earlier attempt it did not write)
{puzzle}

## Correct solution
{solution}

## The model's final turn
{final}

Is the model's answer correct? Count it correct if every role the model commits to in its final answer (its SUBMIT line, and any final assignment it states as its conclusion) matches the correct solution, and it commits to at least one role. Count it incorrect if any committed role is wrong or it gives no answer.

Reply with CORRECT: YES or CORRECT: NO, then one sentence."""


def paper_records():
    out = []
    for name, stem in PAPER:
        for r in json.loads((RESULTS / f"{stem}.json").read_text()):
            out.append(dict(r, model=stem, model_name=name))
    return out


def task_data(r):
    idx = int(r["id"].rsplit("_", 1)[1])
    return json.loads((VAL / "original_tasks" / f"{r['task_type']}.json").read_text())["problems"][idx]


def colouring_valid(r, t=None):
    t = t or task_data(r)
    final = r["messages"][-1]["content"] or ""
    # Read the submitted colouring: the final block of SUBMIT lines in the final
    # turn (the prompt asks for "SUBMIT: [Node] -> [color] for every node"; some
    # models write one SUBMIT line per node), falling back to the whole final
    # turn if there is no SUBMIT.
    lines = final.splitlines()
    sub = [i for i, ln in enumerate(lines) if "SUBMIT" in ln.upper()]
    if sub:
        start = sub[-1]
        while start - 1 in sub or (start - 2 in sub and not lines[start - 1].strip()):
            start = start - 1 if start - 1 in sub else start - 2
        final = "\n".join(lines[start:])
    col = {}
    for node in t["nodes"]:
        hits = re.findall(rf"(?<![A-Za-z]){node}\s*(?:→|->|=|:)\s*(?:colou?r\s*)?(\d+)", final, re.I)
        if hits:
            col[node] = int(hits[-1])
    if len(col) != len(t["nodes"]):
        return False
    if max(col.values()) > t["k"] or min(col.values()) < 1:
        return False
    return all(col[a] != col[b] for a, b in t["edges"])


def knights_panel(recs):
    sys.path.insert(0, str(HERE.parent / "t6_error_detection"))
    from dotenv import load_dotenv
    from revised_answer_audit import JUDGES, call
    load_dotenv(ROOT / ".env", override=True)
    key = os.environ["OPENROUTER_API_KEY"]
    cache = {}
    if CACHE.exists():
        for line in CACHE.open():
            c = json.loads(line)
            cache[(c["model"], c["id"], c["judge"])] = c["raw"]
    todo = [(r, j) for r in recs for j in JUDGES if (r["model"], r["id"], j) not in cache]

    def work(rj):
        r, j = rj
        t = task_data(r)
        prompt = KNIGHTS_PROMPT.format(puzzle="\n\n".join(m["content"] for m in r["messages"][:r["n_prefill"]]),
                                       solution=", ".join(f"{k} is a {v}" for k, v in t["solution"].items()),
                                       final=r["messages"][-1]["content"])
        return r, j, call(j, prompt, key)

    with ThreadPoolExecutor(8) as ex, CACHE.open("a") as fh:
        for r, j, raw in ex.map(work, todo):
            cache[(r["model"], r["id"], j)] = raw
            fh.write(json.dumps({"model": r["model"], "id": r["id"], "judge": j, "raw": raw}) + "\n")
    verdicts = {}
    for r in recs:
        votes = []
        for j in JUDGES:
            m = re.search(r"CORRECT\**\s*:\s*\**\s*\[?\s*(YES|NO)", cache[(r["model"], r["id"], j)] or "", re.I)
            votes.append(m.group(1).upper() == "YES" if m else None)
        verdicts[(r["model"], r["id"])] = (sum(v is True for v in votes) >= 2, votes)
    return verdicts


def parse_brackets(text, key):
    m = re.search(rf"{key}\**\s*:\s*\**\s*\[?\s*(YES|NO)", text or "", re.I)
    return m.group(1).upper() == "YES" if m else None


def regrade_failed(recs):
    """Opus labels for records the original parser could not read; panel labels where the judge returned nothing."""
    from validate_judge import JUDGES as PANEL, PROMPT, MODULE_NAME, call as panel_call, correct_answer, parse, transcript
    sys.path.insert(0, str(HERE))
    from run import TASK_MODULES
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    key = os.environ["OPENROUTER_API_KEY"]
    cache_path = VAL / "failed_judge_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for line in cache_path.open():
            c = json.loads(line)
            cache[(c["model"], c["id"], c["judge"])] = c["raw"]
    fixed = {"reparsed": 0, "panel": 0}
    with cache_path.open("a") as fh:
        for r in recs:
            if r.get("switched") is not None or not r.get("messages"):
                continue
            if (r.get("judge_response") or "").strip():
                for k in ("submitted", "switched", "correct"):
                    r[k] = parse_brackets(r["judge_response"], k.upper())
                fixed["reparsed"] += 1
                continue
            votes = []
            for j in PANEL:
                if (r["model"], r["id"], j) not in cache:
                    ctx = getattr(TASK_MODULES[MODULE_NAME.get(r["task_type"], r["task_type"])], "JUDGE_CONTEXT", "")
                    cache[(r["model"], r["id"], j)] = panel_call(j, PROMPT.format(context=ctx, answer=correct_answer(r), transcript=transcript(r)), key)
                    fh.write(json.dumps({"model": r["model"], "id": r["id"], "judge": j, "raw": cache[(r["model"], r["id"], j)]}) + "\n")
                votes.append(parse(cache[(r["model"], r["id"], j)]))
            for k in ("submitted", "switched", "correct"):
                r[k] = sum(v[k] is True for v in votes) >= 2
            fixed["panel"] += 1
    return fixed


def wilson(k, n):
    """Wilson 95% interval for k/n, in percent (statsmodels)."""
    from statsmodels.stats.proportion import proportion_confint
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return 100 * lo, 100 * hi


def main():
    recs = paper_records()
    for r in recs:
        r["switched_orig"], r["correct_orig"] = bool(r.get("switched")), bool(r.get("correct"))
    print("judge failures regraded:", regrade_failed(recs))
    kv = knights_panel([r for r in recs if r["task_type"] == "knights" and r.get("messages")])
    changes = {"knights": [0, 0], "graph_coloring": [0, 0], "nosub": 0}
    for r in recs:
        sw, cor = bool(r.get("switched")), bool(r.get("correct"))
        if r["task_type"] == "knights" and r.get("messages"):
            cor = kv[(r["model"], r["id"])][0]
        if r["task_type"] == "graph_coloring" and r.get("messages"):
            cor = colouring_valid(r)
        if r["task_type"] in ("knights", "graph_coloring") and cor != r["correct_orig"]:
            changes[r["task_type"]][int(cor)] += 1
        if r.get("submitted") is False and cor:
            changes["nosub"] += 1
            cor = False
        r["switched_new"], r["correct_new"] = sw, cor
    print(f"correctness changed: knights {changes['knights'][1]} to correct, {changes['knights'][0]} to wrong; "
          f"graph colouring {changes['graph_coloring'][1]} to correct, {changes['graph_coloring'][0]} to wrong; "
          f"non-submissions marked correct by the judge, now incorrect {changes['nosub']}")

    summary = {}
    print(f"\n{'Model':15s} {'Switch':>7s} {'SbW':>5s} {'95% CI':>10s} {'Conv':>7s} {'Acc':>5s} {'NoSw':>4s}   (was Switch/SbW/Conv/Acc/NoSw)")
    for name, stem in PAPER:
        rs = [r for r in recs if r["model"] == stem]
        n = len(rs)
        row = {}
        for tag in ("orig", "new"):
            sw = sum(r[f"switched_{tag}"] for r in rs)
            cor = sum(r[f"correct_{tag}"] for r in rs)
            sbw = sum(r[f"switched_{tag}"] and not r[f"correct_{tag}"] for r in rs)
            conv = sum(r[f"switched_{tag}"] and r[f"correct_{tag}"] for r in rs)
            nosw = cor - conv
            row[tag] = {"n": n, "switch": sw, "sbw": sbw, "sbw_ci": wilson(sbw, n), "conv": conv, "correct": cor, "no_switch_correct": nosw}
        # v2 reports Cipher separately (score_v2.CIPHER_NOTE); this row makes the v1 -> v2 SbW comparison like for like
        rc = [r for r in rs if r["task_type"] != "cipher"]
        sbw = sum(r["switched_new"] and not r["correct_new"] for r in rc)
        row["new_excl_cipher"] = {"n": len(rc), "switch": sum(r["switched_new"] for r in rc), "sbw": sbw,
                                  "sbw_ci": wilson(sbw, len(rc)), "correct": sum(r["correct_new"] for r in rc)}
        summary[name] = row
        a, b = row["new"], row["orig"]
        print(f"{name:15s} {a['switch'] / n:>7.0%} {a['sbw'] / n:>5.0%} [{a['sbw_ci'][0]:>3.0f}, {a['sbw_ci'][1]:>3.0f}] {a['conv']:>3d}/{a['switch']:<3d} {a['correct'] / n:>5.0%} {a['no_switch_correct']:>4d}"
              f"   ({b['switch'] / n:.0%}/{b['sbw'] / n:.0%}/{b['conv']}/{b['switch']}/{b['correct'] / n:.0%}/{b['no_switch_correct']})")
    per_type = {}
    for tt in sorted({r["task_type"] for r in recs}):
        rs = [r for r in recs if r["task_type"] == tt]
        per_type[tt] = {tag: {"n": len(rs), "switch": sum(r[f"switched_{tag}"] for r in rs), "correct": sum(r[f"correct_{tag}"] for r in rs),
                              "sbw": sum(r[f"switched_{tag}"] and not r[f"correct_{tag}"] for r in rs)} for tag in ("orig", "new")}
        p = per_type[tt]["new"]
        print(f"  {tt:15s} switch {p['switch'] / p['n']:.0%}  solved {p['correct'] / p['n']:.0%}  switch-but-wrong {p['sbw'] / p['n']:.0%}")
    OUT.write_text(json.dumps({"per_model": summary, "per_task": per_type, "changes": changes}, indent=1))


if __name__ == "__main__":
    main()
