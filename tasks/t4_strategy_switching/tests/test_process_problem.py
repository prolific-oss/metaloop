"""Test end-to-end process_problem with mocked call_model."""

from unittest.mock import patch

import pytest
from helpers import ACTIVE_TASKS, SWITCH_RESPONSES, USAGE, get_modules, load_first_problem

VALID_OUTCOMES = {
    "SWITCHED_CORRECT", "SWITCHED_WRONG",
    "NO_SWITCH_CORRECT", "PERSEVERATED",
    "NO_SUBMISSION",
}


def _wrap_problem(task_type):
    task = load_first_problem(task_type)
    return {"task_type": task_type, "id": f"{task_type}_0", "data": task}


def _mock_for(task_type):
    def mock(model, messages, max_tokens=4096):
        return SWITCH_RESPONSES.get(task_type, "I don't know."), USAGE
    return mock


@pytest.mark.parametrize("task_type", ACTIVE_TASKS)
def test_process_problem_returns_required_fields(task_type):
    import run as runner

    problem = _wrap_problem(task_type)
    with patch.object(runner, "call_model", _mock_for(task_type)):
        result = runner.process_problem("mock", problem)

    assert "error" not in result, f"{task_type}: {result.get('error')}"
    assert result["outcome"] in VALID_OUTCOMES
    assert "correct" in result
    assert "switched_programmatic" in result
    assert "switched_judge" in result or result.get("switched_judge") is None
    assert result["task_type"] == task_type
