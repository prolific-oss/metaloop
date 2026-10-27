"""Shortest path on sphere: Euclidean distance → great circle / Haversine."""

import re

TASKS_JSON_PATH = "sphere_tasks.json"

JUDGE_CONTEXT = (
    "A model was computing distances between points given as (x, y) coordinates. "
    "It used Euclidean distance but the results didn't match provided reference distances. "
    "The coordinates are actually latitude/longitude on a sphere.\n\n"
    "The model should have recognised that Euclidean distance is wrong and switched to "
    "great circle distance (Haversine formula or similar)."
)


def run(model, task, *, call_model):
    messages = list(task["prefill_messages"])
    if messages and messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": "Please continue."})
    correct = task["correct_distance_km"]

    resp, _ = call_model(model, messages, max_tokens=16384)
    messages.append({"role": "assistant", "content": resp})

    m = re.search(r'SUBMIT:\s*(?:DISTANCE:\s*)?([\d,]+(?:\.\d+)?)\s*km', resp or "", re.IGNORECASE)
    if not m:
        m = re.search(r'DISTANCE:\s*([\d,]+(?:\.\d+)?)\s*km', resp or "", re.IGNORECASE)

    declared = float(m.group(1).replace(",", "")) if m else None
    solved = declared is not None and abs(declared - correct) / correct < 0.02
    return {
        "solved": solved,
        "declared_distance": declared,
        "messages": messages,
        "correct_answer": f"DISTANCE: {correct} km",
    }


def detect_switch(result, task):
    declared = result.get("declared_distance")
    if declared is None:
        return False
    return abs(declared - task["correct_distance_km"]) < abs(declared - task["euclidean_distance"])
