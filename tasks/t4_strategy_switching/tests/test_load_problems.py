"""Test problem loading and data integrity."""

from helpers import ACTIVE_TASKS


def test_all_task_types_loaded(all_problems):
    loaded_types = {p["task_type"] for p in all_problems}
    for tt in ACTIVE_TASKS:
        assert tt in loaded_types, f"Task type '{tt}' not loaded"


def test_each_task_has_10_problems(all_problems):
    counts = {}
    for p in all_problems:
        counts[p["task_type"]] = counts.get(p["task_type"], 0) + 1
    for tt in ACTIVE_TASKS:
        assert counts.get(tt, 0) == 10, f"{tt} has {counts.get(tt, 0)} problems, expected 10"


def test_total_is_80(all_problems):
    assert len(all_problems) == 80


def test_prefill_tasks_have_prefill_messages(all_problems):
    prefill_tasks = {"dijkstra", "maze", "sphere", "cipher", "knights",
                     "graph_coloring"}
    for p in all_problems:
        if p["task_type"] in prefill_tasks:
            msgs = p["data"].get("prefill_messages", [])
            assert len(msgs) >= 2, f"{p['id']} has {len(msgs)} prefill messages, need >= 2"
            for i, m in enumerate(msgs):
                assert "role" in m and "content" in m, f"{p['id']} prefill msg {i} missing role/content"


def test_letter_guess_problems_have_required_fields(all_problems):
    for p in all_problems:
        if p["task_type"] != "letter_guess":
            continue
        d = p["data"]
        for field in ["target", "candidates_at_switch", "history", "best_candidate_guess"]:
            assert field in d, f"{p['id']} missing {field}"


def test_code_breaker_problems_have_required_fields(all_problems):
    for p in all_problems:
        if p["task_type"] != "code_breaker":
            continue
        d = p["data"]
        for field in ["secret", "candidates_at_switch", "history", "best_candidate_guess"]:
            assert field in d, f"{p['id']} missing {field}"


def test_dijkstra_divergence(all_problems):
    for p in all_problems:
        if p["task_type"] != "dijkstra":
            continue
        d = p["data"]
        assert d["dijkstra_distance"] != d["bellman_ford_distance"], (
            f"{p['id']}: Dijkstra and BF agree ({d['dijkstra_distance']})"
        )
        assert d["bellman_ford_distance"] < d["dijkstra_distance"], (
            f"{p['id']}: BF not cheaper than Dijkstra"
        )


def test_knights_no_two_role_solution(all_problems):
    for p in all_problems:
        if p["task_type"] != "knights":
            continue
        d = p["data"]
        assert not d["two_role_consistent"], f"{p['id']} has valid 2-role solution"
        assert d["solution_unique"], f"{p['id']} solution not unique"


def test_graph_coloring_valid_coloring_exists(all_problems):
    for p in all_problems:
        if p["task_type"] != "graph_coloring":
            continue
        d = p["data"]
        vc = d["valid_coloring"]
        for a, b in d["edges"]:
            assert vc.get(a) != vc.get(b), f"{p['id']}: conflict on {a}-{b}"
