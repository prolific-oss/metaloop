"""Dijkstra: Dijkstra's algorithm → Bellman-Ford (negative edge weights).

Single-turn: all edge weights shown upfront. Prefill shows Dijkstra being
applied step-by-step, settling nodes prematurely before discovering the
negative edge. Model gets one response to SUBMIT the correct shortest path.
"""

import re

TASKS_JSON_PATH = "dijkstra_tasks.json"

JUDGE_CONTEXT = (
    "A model was finding the shortest path in a directed graph. All edge weights were "
    "visible. The prefill showed Dijkstra's algorithm being applied step-by-step. "
    "During the execution, a negative edge weight was encountered. Dijkstra's algorithm "
    "gives wrong answers with negative weights because it settles nodes prematurely.\n\n"
    "The model should have recognised that the negative edge invalidates Dijkstra's "
    "settled distances and recomputed using Bellman-Ford or exhaustive path search."
)


def run(model, task, *, call_model):
    messages = list(task["prefill_messages"])
    if messages and messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": "Please continue."})

    resp, _ = call_model(model, messages, max_tokens=16384)
    messages.append({"role": "assistant", "content": resp})

    # Parse SUBMIT: PATH: X->Y->Z DISTANCE: N
    dist_match = re.search(r'DISTANCE:\s*(-?\d+)', resp or "", re.IGNORECASE)
    declared_dist = int(dist_match.group(1)) if dist_match else None
    solved = declared_dist == task["bellman_ford_distance"]

    return {
        "solved": solved,
        "declared_distance": declared_dist,
        "messages": messages,
        "correct_answer": f"PATH: {'->'.join(task['bellman_ford_path'])} DISTANCE: {task['bellman_ford_distance']}",
    }


def detect_switch(result, task):
    declared = result.get("declared_distance")
    if declared is None:
        return False
    return abs(declared - task["bellman_ford_distance"]) < abs(declared - task["dijkstra_distance"])
