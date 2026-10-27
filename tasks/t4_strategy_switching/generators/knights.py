"""
Generate knights, knaves, and spies strategy-switching problems.

Each problem has 3-4 people. Each person is a knight (always truthful),
knave (always lies), or spy (can say anything). Every problem has exactly
one spy. The key property: no pure knight/knave assignment is consistent,
but exactly one 3-role assignment (with one spy) is.

Usage:
    python generate_knights_tasks.py
    # Writes tasks.json to the same directory.
"""

import itertools
import json
import random
import os
from copy import deepcopy

NAMES = ["Alice", "Bob", "Carol", "Dave"]
ROLES = ["knight", "knave", "spy"]


# ---------------------------------------------------------------------------
# Statement evaluation
# ---------------------------------------------------------------------------

def evaluate_statement(stmt: dict, assignment: dict) -> bool:
    """Evaluate whether a statement is true given a role assignment.

    Statement types:
        {"type": "is_role", "subject": X, "role": "knight"/"knave"}
        {"type": "same_type", "person1": X, "person2": Y}
        {"type": "diff_type", "person1": X, "person2": Y}
        {"type": "at_least_n_role", "role": "knave", "n": 1, "group": [names]}
        {"type": "exactly_n_role", "role": "knight"/"knave", "n": N, "group": [names]}
    """
    t = stmt["type"]

    if t == "is_role":
        # For 2-role checking, spy is never in the assignment.
        # For 3-role, spy IS in the assignment.
        actual = assignment.get(stmt["subject"])
        return actual == stmt["role"]

    if t == "same_type":
        a = assignment.get(stmt["person1"])
        b = assignment.get(stmt["person2"])
        # "same type" means both knight or both knave (spy excluded from this concept)
        return a == b

    if t == "diff_type":
        a = assignment.get(stmt["person1"])
        b = assignment.get(stmt["person2"])
        return a != b

    if t == "at_least_n_role":
        count = sum(1 for p in stmt["group"] if assignment.get(p) == stmt["role"])
        return count >= stmt["n"]

    if t == "exactly_n_role":
        count = sum(1 for p in stmt["group"] if assignment.get(p) == stmt["role"])
        return count == stmt["n"]

    raise ValueError(f"Unknown statement type: {t}")


def statement_consistent_with_role(stmt: dict, speaker_role: str, assignment: dict) -> bool:
    """Check if a statement is consistent with the speaker's role.

    Knights must speak truth, knaves must lie, spies can say anything.
    """
    if speaker_role == "spy":
        return True
    truth_value = evaluate_statement(stmt, assignment)
    if speaker_role == "knight":
        return truth_value
    if speaker_role == "knave":
        return not truth_value
    raise ValueError(f"Unknown role: {speaker_role}")


def all_statements_consistent(people: list, statements: dict, assignment: dict) -> bool:
    """Check if all statements are consistent with the given assignment."""
    for person in people:
        role = assignment[person]
        for stmt in statements[person]:
            if not statement_consistent_with_role(stmt, role, assignment):
                return False
    return True


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def check_no_two_role_solution(people: list, statements: dict) -> bool:
    """Return True if NO pure knight/knave assignment is consistent."""
    for roles in itertools.product(["knight", "knave"], repeat=len(people)):
        assignment = dict(zip(people, roles))
        if all_statements_consistent(people, statements, assignment):
            return False
    return True


def find_three_role_solutions(people: list, statements: dict) -> list:
    """Find all valid 3-role assignments with exactly one spy."""
    solutions = []
    for spy_idx in range(len(people)):
        spy = people[spy_idx]
        others = [p for p in people if p != spy]
        for roles in itertools.product(["knight", "knave"], repeat=len(others)):
            assignment = {spy: "spy"}
            for p, r in zip(others, roles):
                assignment[p] = r
            if all_statements_consistent(people, statements, assignment):
                solutions.append(dict(assignment))
    return solutions


# ---------------------------------------------------------------------------
# Statement rendering (structured -> English)
# ---------------------------------------------------------------------------

def render_statement(stmt: dict, speaker: str) -> str:
    """Convert a structured statement to English."""
    t = stmt["type"]

    if t == "is_role":
        subj = stmt["subject"]
        role = stmt["role"]
        if subj == speaker:
            return f"I am a {role}"
        return f"{subj} is a {role}"

    if t == "same_type":
        p1, p2 = stmt["person1"], stmt["person2"]
        if p1 == speaker:
            return f"{p2} and I are the same type"
        if p2 == speaker:
            return f"{p1} and I are the same type"
        return f"{p1} and {p2} are the same type"

    if t == "diff_type":
        p1, p2 = stmt["person1"], stmt["person2"]
        if p1 == speaker:
            return f"{p2} and I are different types"
        if p2 == speaker:
            return f"{p1} and I are different types"
        return f"{p1} and {p2} are different types"

    if t == "at_least_n_role":
        role = stmt["role"]
        n = stmt["n"]
        group = stmt["group"]
        if set(group) == {speaker}:
            raise ValueError("at_least_n_role with single-person group is just is_role")
        return f"At least {n} of us {'is' if n == 1 else 'are'} a {role}"

    if t == "exactly_n_role":
        role = stmt["role"]
        n = stmt["n"]
        return f"There {'is' if n == 1 else 'are'} exactly {n} {role}{'s' if n != 1 else ''} among us"

    raise ValueError(f"Unknown statement type: {t}")


# ---------------------------------------------------------------------------
# Statement generation
# ---------------------------------------------------------------------------

def generate_statement_for_knight(speaker: str, people: list, assignment: dict, rng: random.Random) -> dict:
    """Generate a TRUE statement for a knight."""
    candidates = []

    # "X is a knight/knave" about others
    for other in people:
        if other == speaker:
            continue
        actual = assignment[other]
        # Don't claim someone is a spy (knights wouldn't say this in 2-role terms)
        if actual == "spy":
            # Knight can truthfully say spy is NOT a knight, NOT a knave
            # But that's confusing. Instead, skip direct claims about the spy.
            continue
        candidates.append({"type": "is_role", "subject": other, "role": actual})

    # "I am a knight"
    candidates.append({"type": "is_role", "subject": speaker, "role": "knight"})

    # "X and Y are same/different type" (only among non-spies)
    non_spy = [p for p in people if assignment[p] != "spy"]
    for p1, p2 in itertools.combinations(non_spy, 2):
        if assignment[p1] == assignment[p2]:
            candidates.append({"type": "same_type", "person1": p1, "person2": p2})
        else:
            candidates.append({"type": "diff_type", "person1": p1, "person2": p2})

    # "At least N of us is a knave" (counting all people by their actual roles)
    knave_count = sum(1 for p in people if assignment[p] == "knave")
    knight_count = sum(1 for p in people if assignment[p] == "knight")
    if knave_count >= 1:
        candidates.append({
            "type": "at_least_n_role", "role": "knave", "n": 1, "group": list(people)
        })
    # "Exactly N knights/knaves among us" -- truth under actual assignment
    # Note: spy is neither knight nor knave, so count accordingly
    candidates.append({
        "type": "exactly_n_role", "role": "knight", "n": knight_count, "group": list(people)
    })
    candidates.append({
        "type": "exactly_n_role", "role": "knave", "n": knave_count, "group": list(people)
    })

    if not candidates:
        raise RuntimeError("No candidate statements for knight")
    return rng.choice(candidates)


def generate_statement_for_knave(speaker: str, people: list, assignment: dict, rng: random.Random) -> dict:
    """Generate a FALSE statement for a knave."""
    candidates = []

    # "X is a knight/knave" about others -- must be false
    for other in people:
        if other == speaker:
            continue
        actual = assignment[other]
        if actual == "spy":
            # Saying "spy is a knight" is false (spy != knight). Saying "spy is a knave" is also false.
            candidates.append({"type": "is_role", "subject": other, "role": "knight"})
            candidates.append({"type": "is_role", "subject": other, "role": "knave"})
        elif actual == "knight":
            candidates.append({"type": "is_role", "subject": other, "role": "knave"})
        else:  # knave
            candidates.append({"type": "is_role", "subject": other, "role": "knight"})

    # "I am a knight" -- false because speaker is knave
    candidates.append({"type": "is_role", "subject": speaker, "role": "knight"})

    # "X and Y are same/different type" -- must be false
    non_spy = [p for p in people if assignment[p] != "spy"]
    for p1, p2 in itertools.combinations(non_spy, 2):
        if assignment[p1] == assignment[p2]:
            # They ARE the same, so "different" is false
            candidates.append({"type": "diff_type", "person1": p1, "person2": p2})
        else:
            # They are different, so "same" is false
            candidates.append({"type": "same_type", "person1": p1, "person2": p2})

    # "Exactly N knights" where N is wrong
    knight_count = sum(1 for p in people if assignment[p] == "knight")
    knave_count = sum(1 for p in people if assignment[p] == "knave")
    for n in range(len(people) + 1):
        if n != knight_count:
            candidates.append({
                "type": "exactly_n_role", "role": "knight", "n": n, "group": list(people)
            })
        if n != knave_count:
            candidates.append({
                "type": "exactly_n_role", "role": "knave", "n": n, "group": list(people)
            })

    # "At least N knaves" where it's false
    if knave_count == 0:
        candidates.append({
            "type": "at_least_n_role", "role": "knave", "n": 1, "group": list(people)
        })

    if not candidates:
        raise RuntimeError("No candidate statements for knave")
    return rng.choice(candidates)


def generate_statement_for_spy(speaker: str, people: list, assignment: dict, rng: random.Random) -> dict:
    """Generate a misleading statement for the spy.

    We want a statement that is true under SOME 2-role assignments and false
    under others, to maximise confusion when a solver tries to work within
    the 2-role framework.
    """
    candidates = []

    # Simple claims about others
    for other in people:
        if other == speaker:
            continue
        for role in ["knight", "knave"]:
            stmt = {"type": "is_role", "subject": other, "role": role}
            candidates.append(stmt)

    # Self-claims
    candidates.append({"type": "is_role", "subject": speaker, "role": "knight"})
    candidates.append({"type": "is_role", "subject": speaker, "role": "knave"})

    # Same/different type claims
    for other in people:
        if other == speaker:
            continue
        candidates.append({"type": "same_type", "person1": speaker, "person2": other})
        candidates.append({"type": "diff_type", "person1": speaker, "person2": other})

    # Counting claims
    for n in range(len(people) + 1):
        candidates.append({
            "type": "exactly_n_role", "role": "knight", "n": n, "group": list(people)
        })
        candidates.append({
            "type": "exactly_n_role", "role": "knave", "n": n, "group": list(people)
        })

    candidates.append({
        "type": "at_least_n_role", "role": "knave", "n": 1, "group": list(people)
    })

    # Score each candidate: prefer statements that are true under ~half of
    # 2-role assignments (maximally ambiguous)
    scored = []
    for stmt in candidates:
        true_count = 0
        total = 0
        for roles in itertools.product(["knight", "knave"], repeat=len(people)):
            test_assign = dict(zip(people, roles))
            total += 1
            if evaluate_statement(stmt, test_assign):
                true_count += 1
        # Score by how close to 50% ambiguity
        ambiguity = 1.0 - abs(true_count / total - 0.5) * 2
        scored.append((ambiguity, stmt))

    scored.sort(key=lambda x: -x[0])
    # Pick from the top third most ambiguous
    top_n = max(1, len(scored) // 3)
    return rng.choice(scored[:top_n])[1]


# ---------------------------------------------------------------------------
# Problem generation
# ---------------------------------------------------------------------------

def generate_problem(people: list, rng: random.Random, max_attempts: int = 500) -> dict | None:
    """Try to generate a valid problem for the given set of people.

    Returns a problem dict or None if generation fails after max_attempts.
    """
    n = len(people)

    for _ in range(max_attempts):
        # Pick a random assignment with exactly 1 spy
        spy = rng.choice(people)
        others = [p for p in people if p != spy]
        other_roles = [rng.choice(["knight", "knave"]) for _ in others]
        assignment = {spy: "spy"}
        for p, r in zip(others, other_roles):
            assignment[p] = r

        # Need at least 1 knight and 1 knave among non-spies for interesting puzzles
        non_spy_roles = [assignment[p] for p in others]
        if "knight" not in non_spy_roles or "knave" not in non_spy_roles:
            continue

        # Generate statements
        statements = {}
        for person in people:
            role = assignment[person]
            if role == "knight":
                stmt = generate_statement_for_knight(person, people, assignment, rng)
            elif role == "knave":
                stmt = generate_statement_for_knave(person, people, assignment, rng)
            else:
                stmt = generate_statement_for_spy(person, people, assignment, rng)
            statements[person] = [stmt]

        # Verify: no 2-role solution
        if not check_no_two_role_solution(people, statements):
            continue

        # Verify: unique 3-role solution
        solutions = find_three_role_solutions(people, statements)
        if len(solutions) != 1:
            continue

        # Verify solution matches what we intended
        if solutions[0] != assignment:
            continue

        return {
            "people": list(people),
            "statements": statements,
            "assignment": assignment,
            "spy": spy,
        }

    return None


# ---------------------------------------------------------------------------
# Prefill conversation generation
# ---------------------------------------------------------------------------

def generate_prefill(people: list, statements: dict, assignment: dict, spy: str) -> list:
    """Generate the prefill conversation showing failed 2-role deduction.

    The prefill shows 3-4 knight/knave assignments that each hit a contradiction,
    then starts testing another assignment without finishing it.  It must NOT
    conclude that all assignments fail or hint at the spy insight.
    """

    rendered_stmts = {}
    for person in people:
        rendered_stmts[person] = [render_statement(s, person) for s in statements[person]]

    # Build the puzzle text
    puzzle_lines = [
        "There are {n} people: {names}. Each person is either a knight (always tells the truth), "
        "a knave (always lies), or possibly a spy (can say anything, true or false).".format(
            n=len(people), names=", ".join(people)
        ),
        "",
        "They make the following statements:",
    ]
    for person in people:
        for stmt_text in rendered_stmts[person]:
            puzzle_lines.append(f'- {person} says: "{stmt_text}."')
    puzzle_lines.append("")
    puzzle_lines.append(
        "You have one response. State your answer as SUBMIT: "
        "[Name] is a [knight/knave/spy]"
    )

    puzzle_text = "\n".join(puzzle_lines)

    # Build the assistant's prefilled reasoning.
    # Show 3-4 completed contradiction checks, then start a new one mid-way.
    two_role_combos = list(itertools.product(["knight", "knave"], repeat=len(people)))

    reasoning_lines = [
        "Let me work through the possible knight/knave assignments.",
        "",
    ]

    # Show 3 full contradiction checks
    completed = 0
    combo_idx = 0
    while completed < 3 and combo_idx < len(two_role_combos):
        roles = two_role_combos[combo_idx]
        combo_idx += 1
        test_assign = dict(zip(people, roles))
        role_str = ", ".join(f"{p}={r}" for p, r in test_assign.items())
        reasoning_lines.append(f"Testing {role_str}:")

        # Find the first contradiction
        contradiction_found = False
        for person in people:
            role = test_assign[person]
            for stmt in statements[person]:
                truth = evaluate_statement(stmt, test_assign)
                stmt_text = render_statement(stmt, person)
                if role == "knight" and not truth:
                    reasoning_lines.append(
                        f"{person} (knight) says \"{stmt_text}\" -- must be true. "
                        f"Check: false. Contradiction."
                    )
                    contradiction_found = True
                    break
                elif role == "knave" and truth:
                    reasoning_lines.append(
                        f"{person} (knave) says \"{stmt_text}\" -- must be false. "
                        f"Check: true. Contradiction."
                    )
                    contradiction_found = True
                    break
                else:
                    ok_val = "true" if truth else "false"
                    if role == "knight":
                        reasoning_lines.append(
                            f"{person} (knight) says \"{stmt_text}\" -- must be true. "
                            f"Check: {ok_val}. OK."
                        )
                    else:
                        reasoning_lines.append(
                            f"{person} (knave) says \"{stmt_text}\" -- must be false. "
                            f"Check: {ok_val}. OK."
                        )
            if contradiction_found:
                break

        reasoning_lines.append("")
        completed += 1

    # Start one more assignment check but leave it unfinished (the error:
    # continuing to try 2-role assignments mechanically).
    # Find a combo we haven't shown yet.
    if combo_idx < len(two_role_combos):
        roles = two_role_combos[combo_idx]
        test_assign = dict(zip(people, roles))
        role_str = ", ".join(f"{p}={r}" for p, r in test_assign.items())
        reasoning_lines.append(f"Testing {role_str}:")

    assistant_reasoning = "\n".join(reasoning_lines)

    messages = [
        {"role": "user", "content": puzzle_text},
        {"role": "assistant", "content": assistant_reasoning},
    ]

    return messages


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(42)
    target_count = 10
    problems = []
    seen_assignments = set()  # avoid duplicates

    # Generate a mix of 3-person and 4-person problems
    configs = [3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
    rng.shuffle(configs)

    for n_people in configs:
        people = NAMES[:n_people]
        attempt = 0
        while attempt < 50:
            attempt += 1
            result = generate_problem(people, rng)
            if result is None:
                continue

            # Check for duplicate assignments
            key = tuple(sorted(result["assignment"].items()))
            if key in seen_assignments:
                continue
            seen_assignments.add(key)

            # Generate prefill
            prefill = generate_prefill(
                result["people"],
                result["statements"],
                result["assignment"],
                result["spy"],
            )

            # Render statements to English
            rendered_statements = {}
            for person in result["people"]:
                rendered_statements[person] = [
                    render_statement(s, person) for s in result["statements"][person]
                ]

            problems.append({
                "people": result["people"],
                "statements": rendered_statements,
                "statements_structured": result["statements"],
                "solution": result["assignment"],
                "spy": result["spy"],
                "two_role_consistent": False,
                "solution_unique": True,
                "prefill_messages": prefill,
            })
            break

    if len(problems) < target_count:
        print(f"WARNING: Only generated {len(problems)}/{target_count} problems")

    output = {
        "task_type": "knights_strategy_switch",
        "description": (
            "Knights, knaves, and spies logic puzzles. Each puzzle has no valid "
            "2-role (knight/knave only) assignment, forcing the solver to recognise "
            "that a spy must be present. Exactly one person is a spy in each puzzle."
        ),
        "problems": problems,
    }

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knights_tasks.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Generated {len(problems)} problems -> {output_path}")

    # Print summary
    for i, p in enumerate(problems):
        spy = p["spy"]
        n = len(p["people"])
        roles = ", ".join(f"{k}={v}" for k, v in p["solution"].items())
        print(f"  Problem {i+1} ({n} people): {roles}")


if __name__ == "__main__":
    main()
