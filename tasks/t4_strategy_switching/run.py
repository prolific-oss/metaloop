"""
T4 Strategy Switching — 8 game/logic tasks

Model is placed into a multi-turn environment where the obvious approach
hits a wall. Measures whether it detects the failure and pivots.

Tasks:
    letter_guess, code_breaker, dijkstra, maze, sphere,
    cipher, knights, graph_coloring

Usage:
    python run.py claude-sonnet-4-6
    python run.py gpt-5.4
    python run.py google/gemini-3.1-pro-preview
    python run.py claude-sonnet-4-6 --tasks letter_guess dijkstra cipher
    python run.py --all
"""

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_path, override=True)

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from call_model import call_model, ALL_MODELS

from judge import judge
from tasks import letter_guess, code_breaker, dijkstra, maze, sphere
from tasks import cipher, knights, graph_coloring

RESULTS_DIR = Path(__file__).parent / "results"
TASKS_DIR = Path(__file__).parent / "generators"

JUDGE_MODEL = "anthropic/claude-opus-4.6"

# Registry: task_type name -> module
TASK_MODULES = {
    "letter_guess": letter_guess,
    "code_breaker": code_breaker,
    "dijkstra": dijkstra,
    "maze": maze,
    "sphere": sphere,
    "cipher": cipher,
    "knights": knights,
    "graph_coloring": graph_coloring,
}


# ---------------------------------------------------------------------------
# Load problems
# ---------------------------------------------------------------------------

def load_problems(task_types=None):
    problems = []
    for tt, mod in TASK_MODULES.items():
        if task_types and tt not in task_types:
            continue
        tasks_path = TASKS_DIR / mod.TASKS_JSON_PATH
        if not tasks_path.exists():
            continue
        data = json.loads(tasks_path.read_text())
        for i, p in enumerate(data["problems"]):
            problems.append({"task_type": tt, "id": f"{tt}_{i}", "data": p})
    return problems


# ---------------------------------------------------------------------------
# Process one problem
# ---------------------------------------------------------------------------

def process_problem(model, problem):
    pid = problem["id"]
    task_type = problem["task_type"]
    task = problem["data"]

    mod = TASK_MODULES.get(task_type)
    if not mod:
        return {"id": pid, "error": f"Unknown task type: {task_type}"}

    n_prefill = len(task.get("prefill_messages", []))

    try:
        result = mod.run(model, task, call_model=call_model)
    except Exception as e:
        return {"id": pid, "error": str(e)}

    n_prefill = result.get("n_prefill", n_prefill)

    # --- Programmatic switch detection (secondary signal) ---
    switched_programmatic = mod.detect_switch(result, task)

    # --- LLM judge (primary scoring) ---
    judge_context = getattr(mod, "JUDGE_CONTEXT", "")
    correct_answer = result.get("correct_answer", "")
    judge_result = {"submitted": None, "switched": None, "correct": None, "raw": None}
    try:
        judge_result = judge(
            judge_context, correct_answer, result["messages"], n_prefill,
            call_model=call_model, judge_model=JUDGE_MODEL,
        )
    except Exception as e:
        judge_result["raw"] = f"(judge call failed: {e})"

    # --- Scoring based on LLM judge ---
    submitted = judge_result["submitted"]
    switched = judge_result["switched"]
    correct = judge_result["correct"]

    if switched and correct:
        outcome, score = "SWITCHED_CORRECT", 1.0
    elif switched and not correct:
        outcome, score = "SWITCHED_WRONG", 0.5
    elif not switched and correct:
        outcome, score = "NO_SWITCH_CORRECT", 0.25
    elif not submitted:
        outcome, score = "NO_SUBMISSION", 0.0
    else:
        outcome, score = "PERSEVERATED", 0.0

    return {
        "id": pid,
        "task_type": task_type,
        "submitted": submitted,
        "correct": correct,
        "switched": switched,
        "switched_programmatic": switched_programmatic,
        "outcome": outcome,
        "score": score,
        "n_prefill": n_prefill,
        "messages": result["messages"],
        "judge_response": judge_result["raw"],
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(model, problems):
    print(f"\n{'='*60}")
    print(f"  T4 Strategy Switching")
    print(f"  Model: {model}")
    print(f"  Problems: {len(problems)}")
    by_type = Counter(p["task_type"] for p in problems)
    for t, n in sorted(by_type.items()):
        print(f"    {t}: {n}")
    print(f"{'='*60}")

    results = []
    sym = {
        "SWITCHED_CORRECT": "S", "SWITCHED_WRONG": "~",
        "NO_SWITCH_CORRECT": ".", "PERSEVERATED": "X",
        "NO_GUESSES": "-",
    }

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_problem, model, p): p["id"] for p in problems}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            if "error" in r:
                print(f"  [{r['id']}] ERROR: {r['error']}")
            else:
                print(f"  [{r['id']:20s}] {sym.get(r['outcome'], '?')} {r['outcome']}")

    results.sort(key=lambda r: r.get("id", ""))

    # --- Summary ---
    valid = [r for r in results if "error" not in r]
    outcomes = Counter(r["outcome"] for r in valid)
    scoreable = [r for r in valid if r["score"] is not None]

    print(f"\n{'='*60}")
    print(f"  RESULTS — {model}")
    print(f"{'='*60}")
    print(f"  Total problems:  {len(valid)}")
    print(f"  Submitted:       {sum(1 for r in valid if r.get('submitted'))}/{len(valid)}")
    print(f"  Correct (judge): {sum(1 for r in valid if r.get('correct'))}/{len(valid)}")
    print(f"  Switched (judge):{sum(1 for r in valid if r.get('switched'))}/{len(valid)}")
    print(f"  Switched (prog): {sum(1 for r in valid if r.get('switched_programmatic'))}/{len(valid)}")
    if scoreable:
        print(f"  Mean score:      {sum(r['score'] for r in scoreable) / len(scoreable):.2f}")
    print(f"\n  Outcomes: {dict(outcomes)}")

    by_type = {}
    for r in valid:
        by_type.setdefault(r["task_type"], []).append(r)
    print(f"\n  By task type:")
    for tt in sorted(by_type):
        items = by_type[tt]
        correct_n = sum(1 for r in items if r.get("correct"))
        switch_n = sum(1 for r in items if r.get("switched"))
        sw_prog = sum(1 for r in items if r.get("switched_programmatic"))
        sc = [r for r in items if r["score"] is not None]
        mean = f"{sum(r['score'] for r in sc) / len(sc):.2f}" if sc else "N/A"
        print(f"    {tt:20s} n={len(items)}  correct={correct_n}  sw_judge={switch_n}  sw_prog={sw_prog}  mean={mean}")

    return results


def main():
    args = sys.argv[1:]
    model = "claude-sonnet-4-6"
    task_types = None
    run_all = False

    i = 0
    while i < len(args):
        if args[i] == "--tasks":
            task_types = []
            i += 1
            while i < len(args) and not args[i].startswith("--"):
                task_types.append(args[i])
                i += 1
        elif args[i] == "--all":
            run_all = True
            i += 1
        elif not args[i].startswith("--"):
            model = args[i]
            i += 1
        else:
            i += 1

    models = ALL_MODELS if run_all else [model]
    problems = load_problems(task_types)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def run_model(m):
        results = run(m, problems)
        outfile = RESULTS_DIR / f"{m.replace('/', '_')}.json"
        outfile.write_text(json.dumps(results, indent=2))
        print(f"\n  Saved to {outfile}")
        return m, results

    if len(models) > 1:
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            futures = {executor.submit(run_model, m): m for m in models}
            for future in as_completed(futures):
                m, _ = future.result()
    else:
        run_model(models[0])


if __name__ == "__main__":
    main()
