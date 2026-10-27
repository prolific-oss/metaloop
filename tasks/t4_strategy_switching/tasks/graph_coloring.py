"""Graph coloring: greedy → backtracking / recoloring."""

import re

TASKS_JSON_PATH = "graph_coloring_tasks.json"

JUDGE_CONTEXT = (
    "A model was given a graph and asked to color it with at most K colors. "
    "A prior attempt greedily colored all nodes but used color K+1 at one node, "
    "violating the constraint. The graph IS K-colorable with a different assignment.\n\n"
    "The model should have noticed the K+1 violation and produced a valid K-coloring "
    "by backtracking and recoloring earlier nodes."
)


def run(model, task, *, call_model):
    messages = list(task["prefill_messages"])
    if messages and messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": "Please continue."})
    edges = [tuple(e) for e in task["edges"]]
    k = task["k"]

    resp, _ = call_model(model, messages, max_tokens=16384)
    messages.append({"role": "assistant", "content": resp})

    coloring = {}
    for node in task["nodes"]:
        pat = rf'{node}\s*(?:→|->|=|:)\s*(?:color\s*)?(\d+)'
        pm = re.search(pat, resp or "", re.IGNORECASE)
        if pm:
            coloring[node] = int(pm.group(1))

    valid = True
    if len(coloring) != len(task["nodes"]):
        valid = False
    else:
        for a, b in edges:
            if coloring.get(a) == coloring.get(b):
                valid = False
                break
        if max(coloring.values(), default=0) > k:
            valid = False

    # Format correct answer
    vc = task["valid_coloring"]
    correct_parts = [f"{n} -> {vc[n]}" for n in task["nodes"]]
    return {
        "solved": valid,
        "declared_coloring": coloring,
        "messages": messages,
        "correct_answer": ", ".join(correct_parts),
    }


def detect_switch(result, task):
    coloring = result.get("declared_coloring", {})
    if not coloring:
        return False
    return max(coloring.values(), default=0) <= task["k"]
