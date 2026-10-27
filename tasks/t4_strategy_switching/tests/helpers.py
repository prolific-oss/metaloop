"""Shared constants and utilities for T4 tests."""

import json
import sys
from pathlib import Path

# Wire up imports to match the runner's path setup
T4_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(T4_DIR))
sys.path.insert(0, str(T4_DIR.parent / "shared"))

TASKS_DIR = T4_DIR / "generators"

ACTIVE_TASKS = [
    "letter_guess", "code_breaker", "dijkstra", "maze", "sphere", "cipher",
    "knights", "graph_coloring",
]

USAGE = {"input_tokens": 10, "output_tokens": 10}

SWITCH_RESPONSES = {
    "letter_guess": "SUBMIT: CLOMP",
    "code_breaker": "SUBMIT: RRGG",
    "dijkstra": "The negative edge means Dijkstra is invalid. Using Bellman-Ford:\nPATH: A->C->D->F\nDISTANCE: 0",
    "maze": "The wall-following trace loops. Using BFS instead.\nPATH: (13,13) -> (12,13) -> (11,13) -> (9,12)",
    "sphere": "These are latitude/longitude coordinates. Using the Haversine formula:\nDISTANCE: 5570 km",
    "cipher": "This is a Vigenère cipher, not monoalphabetic. Using Kasiski examination.\nPLAINTEXT: PLACEHOLDER",
    "knights": "The contradiction means one person must be a spy. Re-deriving:\nAlice is a knight\nBob is a spy\nCarol is a knave",
    "graph_coloring": "I need to backtrack. Recoloring:\nA -> 1\nB -> 2\nC -> 1\nD -> 3\nE -> 2\nF -> 1",
}

PERSEVERATE_RESPONSES = {
    "letter_guess": "GUESS: CATCH",
    "code_breaker": "GUESS: GOOO",
    "dijkstra": "PATH: A->B->D->F\nDISTANCE: 7",
    "maze": "PATH: (13,13) -> (13,14) -> (13,15)",
    "sphere": "Let me recheck... DISTANCE: 74.9 km",
    "cipher": "Let me try swapping E and T.\nPLAINTEXT: XYZABC",
    "knights": "Alice is a knight\nBob is a knave\nCarol is a knight",
    "graph_coloring": "A -> 1\nB -> 2\nC -> 3\nD -> 2\nE -> 1\nF -> 4",
}

MODULES = None  # lazily populated


def get_modules():
    global MODULES
    if MODULES is None:
        from tasks import (letter_guess, code_breaker, dijkstra, maze, sphere,
                           cipher, knights, graph_coloring)
        MODULES = {
            "letter_guess": letter_guess, "code_breaker": code_breaker, "dijkstra": dijkstra,
            "maze": maze, "sphere": sphere, "cipher": cipher,
            "knights": knights, "graph_coloring": graph_coloring,
        }
    return MODULES


def load_first_problem(task_type):
    modules = get_modules()
    mod = modules[task_type]
    path = TASKS_DIR / mod.TASKS_JSON_PATH
    data = json.loads(path.read_text())
    return data["problems"][0]


def load_all_problems():
    modules = get_modules()
    problems = []
    for tt, mod in modules.items():
        path = TASKS_DIR / mod.TASKS_JSON_PATH
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for i, p in enumerate(data["problems"]):
            problems.append({"task_type": tt, "id": f"{tt}_{i}", "data": p})
    return problems


def make_mock(response):
    def mock(model, messages, max_tokens=4096):
        return response, USAGE
    return mock
