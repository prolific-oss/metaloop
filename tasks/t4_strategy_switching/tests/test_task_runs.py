"""Test each task's run() function with a mock model."""

import pytest
from helpers import (ACTIVE_TASKS, SWITCH_RESPONSES, USAGE,
                     get_modules, load_first_problem, make_mock)

# Skip dijkstra in parametrized single-response tests (it's multi-turn interactive)
SINGLE_TURN_TASKS = [t for t in ACTIVE_TASKS if t not in ("letter_guess", "code_breaker", "dijkstra")]
INTERACTIVE_TASKS = ["letter_guess", "code_breaker", "dijkstra"]


@pytest.mark.parametrize("task_type", ACTIVE_TASKS)
def test_run_returns_required_keys(task_type):
    task = load_first_problem(task_type)
    mod = get_modules()[task_type]
    result = mod.run("mock", task, call_model=make_mock(SWITCH_RESPONSES[task_type]))
    assert "solved" in result, f"{task_type}: missing 'solved'"
    assert "messages" in result, f"{task_type}: missing 'messages'"
    assert isinstance(result["messages"], list)
    assert len(result["messages"]) > 0


@pytest.mark.parametrize("task_type", ACTIVE_TASKS)
def test_run_messages_grow_beyond_prefill(task_type):
    task = load_first_problem(task_type)
    mod = get_modules()[task_type]
    n_prefill = len(task.get("prefill_messages", []))
    result = mod.run("mock", task, call_model=make_mock(SWITCH_RESPONSES[task_type]))
    assert len(result["messages"]) > max(n_prefill, 2)


def test_letter_guess_detects_distinguisher():
    task = load_first_problem("letter_guess")
    result = get_modules()["letter_guess"].run("mock", task, call_model=make_mock("SUBMIT: ZZZZZ"))
    assert result["used_distinguisher"] is True


def test_letter_guess_detects_candidate():
    task = load_first_problem("letter_guess")
    candidate = task["candidates_at_switch"][0]
    result = get_modules()["letter_guess"].run("mock", task, call_model=make_mock(f"SUBMIT: {candidate.upper()}"))
    assert result["used_distinguisher"] is False


def test_cipher_matches_plaintext():
    task = load_first_problem("cipher")
    pt = task["plaintext"]
    result = get_modules()["cipher"].run("mock", task, call_model=make_mock(f"PLAINTEXT: {pt}"))
    assert result["solved"] is True


def test_sphere_accepts_close_distance():
    task = load_first_problem("sphere")
    close = int(task["correct_distance_km"] * 1.02)
    result = get_modules()["sphere"].run("mock", task, call_model=make_mock(f"DISTANCE: {close} km"))
    assert result["solved"] is True


