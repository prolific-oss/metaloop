"""Maze: wall-following → BFS/flood-fill.

Single-turn: model sees full grid and the prefilled wall-following trace
that looped back to start (without acknowledging the loop).
"""

import re

TASKS_JSON_PATH = "maze_tasks.json"

JUDGE_CONTEXT = (
    "A model was navigating a maze. The prefill showed it using wall-following "
    "(right-hand rule), which traced the outer boundary and returned to the start "
    "position — because the exit is on an interior island wall not connected to "
    "the outer boundary. The prefill continued wall-following past the loop point "
    "without acknowledging the loop.\n\n"
    "The model should have recognised that wall-following looped and switched to "
    "BFS, flood-fill, or direct pathfinding to find the exit."
)


def run(model, task, *, call_model):
    messages = list(task["prefill_messages"])
    if messages and messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": "Please continue."})
    exit_pos = tuple(task["exit"])

    resp, _ = call_model(model, messages, max_tokens=16384)
    messages.append({"role": "assistant", "content": resp})

    # Parse SUBMIT: PATH: (r1,c1) -> (r2,c2) -> ...
    path_match = re.search(r'(?:SUBMIT:\s*)?PATH:\s*(.+)', resp or "", re.IGNORECASE)
    declared_path = []
    if path_match:
        coords = re.findall(r'\((\d+)\s*,\s*(\d+)\)', path_match.group(1))
        declared_path = [(int(r), int(c)) for r, c in coords]

    solved = len(declared_path) > 0 and declared_path[-1] == exit_pos

    # Format correct answer
    bfs_path = task.get("bfs_path", [])
    correct_str = " -> ".join(f"({r},{c})" for r, c in bfs_path)
    return {
        "solved": solved,
        "declared_path": declared_path,
        "messages": messages,
        "correct_answer": f"PATH: {correct_str}",
    }


def detect_switch(result, task):
    exit_pos = tuple(task["exit"])
    path = result.get("declared_path", [])
    return len(path) > 0 and path[-1] == exit_pos
