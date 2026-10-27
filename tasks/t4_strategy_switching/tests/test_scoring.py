"""Test scoring logic without running models."""

import sys
from pathlib import Path

T4_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(T4_DIR))
sys.path.insert(0, str(T4_DIR.parent / "shared"))

from tasks import letter_guess, code_breaker, graph_coloring, cipher, maze, sphere, dijkstra, knights
from judge import judge


# --- Programmatic detect_switch ---

def test_letter_guess_switched():
    result = {"used_distinguisher": True, "solved": True}
    assert letter_guess.detect_switch(result, {}) is True

def test_letter_guess_perseverated():
    result = {"used_distinguisher": False, "solved": False}
    assert letter_guess.detect_switch(result, {}) is False

def test_code_breaker_switched():
    assert code_breaker.detect_switch({"used_distinguisher": True}, {}) is True

def test_code_breaker_perseverated():
    assert code_breaker.detect_switch({"used_distinguisher": False}, {}) is False

def test_graph_coloring_switched():
    assert graph_coloring.detect_switch({"declared_coloring": {"A": 1, "B": 2}}, {"k": 3}) is True

def test_graph_coloring_perseverated():
    assert graph_coloring.detect_switch({"declared_coloring": {"A": 4, "B": 2}}, {"k": 3}) is False

def test_cipher_switched():
    assert cipher.detect_switch({"solved": True}, {}) is True

def test_cipher_perseverated():
    assert cipher.detect_switch({"solved": False}, {}) is False

def test_maze_switched():
    assert maze.detect_switch({"declared_path": [(1,1), (9,12)]}, {"exit": [9,12]}) is True

def test_maze_perseverated():
    assert maze.detect_switch({"declared_path": [(1,1), (2,2)]}, {"exit": [9,12]}) is False

def test_maze_no_path():
    assert maze.detect_switch({"declared_path": []}, {"exit": [9,12]}) is False

def test_sphere_switched():
    result = {"declared_distance": 5570}
    task = {"correct_distance_km": 5570, "euclidean_distance": 74.7}
    assert sphere.detect_switch(result, task) is True

def test_sphere_perseverated():
    result = {"declared_distance": 80.0}
    task = {"correct_distance_km": 5570, "euclidean_distance": 74.7}
    assert sphere.detect_switch(result, task) is False

def test_dijkstra_switched():
    result = {"declared_distance": -1}
    task = {"bellman_ford_distance": -1, "dijkstra_distance": 7}
    assert dijkstra.detect_switch(result, task) is True

def test_dijkstra_perseverated():
    result = {"declared_distance": 7}
    task = {"bellman_ford_distance": -1, "dijkstra_distance": 7}
    assert dijkstra.detect_switch(result, task) is False

def test_knights_switched():
    assert knights.detect_switch({"declared_roles": {"Alice": "knight", "Bob": "spy"}}, {}) is True

def test_knights_perseverated():
    assert knights.detect_switch({"declared_roles": {"Alice": "knight", "Bob": "knave"}}, {}) is False


# --- Judge (mock) ---

def test_judge_parses_yes():
    def mock_call(model, msgs, max_tokens=1024):
        return "SUBMITTED: YES\nSWITCHED: YES\nCORRECT: YES\nEXPLANATION: The model switched.", {}
    result = judge("context", "ans", [{"role": "assistant", "content": "test"}],
                   call_model=mock_call, judge_model="fake")
    assert result["switched"] is True

def test_judge_parses_no():
    def mock_call(model, msgs, max_tokens=1024):
        return "SUBMITTED: YES\nSWITCHED: NO\nCORRECT: NO\nEXPLANATION: No switch.", {}
    result = judge("context", "ans", [{"role": "assistant", "content": "test"}],
                   call_model=mock_call, judge_model="fake")
    assert result["switched"] is False
