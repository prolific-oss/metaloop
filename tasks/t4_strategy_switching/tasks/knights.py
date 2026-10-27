"""Knights, Knaves, and Spies: 2-role deduction → 3-role (introduce spy)."""

import re

TASKS_JSON_PATH = "knights_tasks.json"

JUDGE_CONTEXT = (
    "A model was solving a knights-and-knaves logic puzzle that also allows spies "
    "(who can say anything). It was using standard 2-role (knight/knave) deduction and "
    "hit contradictions on every assignment.\n\n"
    "The model should have recognised that the contradictions mean the 2-role model is "
    "insufficient and introduced the spy role to find a consistent assignment."
)


def run(model, task, *, call_model):
    messages = list(task["prefill_messages"])
    if messages and messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": "Please continue."})
    solution = task["solution"]

    resp, _ = call_model(model, messages, max_tokens=16384)
    messages.append({"role": "assistant", "content": resp})

    assignments = {}
    for person in solution:
        pat = rf'{person}\s*(?:is\s+(?:a\s+)?|=\s*|:\s*)(knight|knave|spy)'
        pm = re.search(pat, resp or "", re.IGNORECASE)
        if pm:
            assignments[person] = pm.group(1).lower()

    solved = len(assignments) == len(solution) and assignments == {k: v.lower() for k, v in solution.items()}

    # Format correct answer
    correct_parts = [f"{k} is a {v}" for k, v in solution.items()]
    return {
        "solved": solved,
        "declared_roles": assignments,
        "messages": messages,
        "correct_answer": ", ".join(correct_parts),
    }


def detect_switch(result, task):
    return "spy" in result.get("declared_roles", {}).values()
