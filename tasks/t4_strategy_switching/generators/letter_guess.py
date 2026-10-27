#!/usr/bin/env python3
"""
Generate Letter Guess strategy-switching problems.

Finds target words where optimal play leads to a state where:
- Guessing candidate words has low information gain (many similar candidates)
- Guessing a non-candidate "distinguisher" word has much higher information gain
This creates a natural strategy-switch point.

Usage:
    python generate_letter_guess_tasks.py --top 20
    python generate_letter_guess_tasks.py --top 20 --output tasks.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from math import log2
from pathlib import Path


def load_word_list() -> tuple[list[str], list[str]]:
    """Load the public-domain 5-letter English wordlist.

    Returns (solutions, all_valid). Under this PD-only configuration the
    two are the same — any word in the list can be both a target and a
    valid guess. See letter_guess_data/NOTICE.txt for the source.
    """
    word_dir = Path(__file__).parent / "letter_guess_data"
    words = sorted(
        w.strip() for w in (word_dir / "wordlist.txt").read_text().splitlines()
        if w.strip()
    )
    print(f"Loaded {len(words)} 5-letter words (PD)", file=sys.stderr)
    return words, words


def get_feedback(guess: str, target: str) -> tuple[int, ...]:
    """Return Letter Guess feedback: 2=green, 1=yellow, 0=gray."""
    feedback = [0] * 5
    target_chars = list(target)

    for i in range(5):
        if guess[i] == target[i]:
            feedback[i] = 2
            target_chars[i] = None

    for i in range(5):
        if feedback[i] == 0 and guess[i] in target_chars:
            feedback[i] = 1
            target_chars[target_chars.index(guess[i])] = None

    return tuple(feedback)


def feedback_to_str(feedback: tuple[int, ...]) -> str:
    """Convert feedback tuple to emoji string."""
    return "".join({0: "⬛", 1: "🟨", 2: "🟩"}[f] for f in feedback)


def filter_candidates(
    candidates: list[str], guess: str, feedback: tuple[int, ...]
) -> list[str]:
    """Return candidates consistent with the observed feedback."""
    return [w for w in candidates if get_feedback(guess, w) == feedback]


def expected_information(guess: str, candidates: list[str]) -> float:
    """Expected bits of information from a guess."""
    if len(candidates) <= 1:
        return 0.0
    patterns = Counter(get_feedback(guess, c) for c in candidates)
    total = len(candidates)
    info = 0.0
    for count in patterns.values():
        p = count / total
        if p > 0:
            info -= p * log2(p)
    return info


def best_guess_from(
    word_pool: list[str], candidates: list[str], top_n: int = 1
) -> list[tuple[str, float]]:
    """Find the top_n guesses from word_pool by expected information gain."""
    scored = [(w, expected_information(w, candidates)) for w in word_pool]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]


def find_strategy_switch_point(
    target: str,
    solutions: list[str],
    all_valid: list[str],
    max_opener_guesses: int = 2,
    min_candidates: int = 5,
    max_candidates: int = 50,
    min_info_gap: float = 0.3,
) -> dict | None:
    """Play optimally toward target and check if a strategy switch arises.

    Returns a problem spec dict if a switch point is found, else None.
    """
    candidates = list(solutions)
    history = []

    for turn in range(max_opener_guesses):
        # Find best guess from all valid words (optimal play)
        if turn == 0:
            # Use a precomputed strong opener to save time
            guess = _best_opener(all_valid, candidates)
        else:
            # For subsequent turns, search candidates + a sample of all_valid
            # (full search is too slow for every target)
            search_pool = list(set(candidates) | set(all_valid[:2000]))
            best = best_guess_from(search_pool, candidates, top_n=1)
            guess = best[0][0]

        feedback = get_feedback(guess, target)
        candidates = filter_candidates(candidates, guess, feedback)
        history.append({
            "guess": guess,
            "feedback": list(feedback),
            "feedback_display": feedback_to_str(feedback),
            "remaining_candidates": len(candidates),
        })

        if len(candidates) <= 1:
            return None  # Solved too quickly, no switch needed

    # Check if we have a good switch point
    n = len(candidates)
    if n < min_candidates or n > max_candidates:
        return None

    # Compare best candidate guess vs best overall guess
    best_from_candidates = best_guess_from(candidates, candidates, top_n=3)
    # Sample all_valid for speed (check candidates + top guesses)
    search_pool = list(set(candidates) | set(all_valid[:3000]))
    best_from_all = best_guess_from(search_pool, candidates, top_n=3)

    best_cand_info = best_from_candidates[0][1] if best_from_candidates else 0
    best_all_info = best_from_all[0][1] if best_from_all else 0
    info_gap = best_all_info - best_cand_info

    if info_gap < min_info_gap:
        return None

    # Check that the best overall guess is NOT a candidate (i.e., it's a
    # distinguisher word, not just a better candidate)
    best_overall_word = best_from_all[0][0]
    if best_overall_word in candidates:
        return None

    # The candidate guess must not solve the puzzle or be too informative.
    # After guessing it, at least 3 candidates must remain.
    cand_word = best_from_candidates[0][0]
    cand_fb = get_feedback(cand_word, target)
    post_cand = filter_candidates(candidates, cand_word, cand_fb)
    if len(post_cand) < 3:
        return None

    return {
        "target": target,
        "history": history,
        "candidates_at_switch": sorted(candidates),
        "n_candidates": n,
        "best_candidate_guess": {
            "word": best_from_candidates[0][0],
            "info_bits": round(best_from_candidates[0][1], 3),
            "top_3": [
                {"word": w, "info_bits": round(b, 3)}
                for w, b in best_from_candidates
            ],
        },
        "best_overall_guess": {
            "word": best_from_all[0][0],
            "info_bits": round(best_from_all[0][1], 3),
            "top_3": [
                {"word": w, "info_bits": round(b, 3)}
                for w, b in best_from_all
            ],
        },
        "info_gap_bits": round(info_gap, 3),
        "optimal_solution": _build_optimal_solution(
            target, history, candidates, best_from_all[0][0], all_valid
        ),
    }


_opener_cache: str | None = None


def _best_opener(all_valid: list[str], candidates: list[str]) -> str:
    """Return a strong opening word. Cached after first computation."""
    global _opener_cache
    if _opener_cache:
        return _opener_cache

    # Check some known strong openers rather than searching the full space
    known_good = ["salet", "crane", "slate", "trace", "crate", "stare", "raise"]
    available = [w for w in known_good if w in all_valid]
    if available:
        best = max(available, key=lambda w: expected_information(w, candidates))
        _opener_cache = best
        return best

    # Fallback: sample and pick best
    import random
    sample = random.sample(all_valid, min(500, len(all_valid)))
    best = max(sample, key=lambda w: expected_information(w, candidates))
    _opener_cache = best
    return best


def _build_optimal_solution(
    target: str,
    history: list[dict],
    candidates: list[str],
    distinguisher: str,
    all_valid: list[str],
) -> list[dict]:
    """Build the full optimal solution from the switch point onward."""
    solution = list(history)

    # Play the distinguisher
    feedback = get_feedback(distinguisher, target)
    remaining = filter_candidates(candidates, distinguisher, feedback)
    solution.append({
        "guess": distinguisher,
        "feedback": list(feedback),
        "feedback_display": feedback_to_str(feedback),
        "remaining_candidates": len(remaining),
        "is_distinguisher": True,
    })

    # Play remaining turns until solved
    while len(remaining) > 1:
        search_pool = list(set(remaining) | set(all_valid[:1000]))
        best = best_guess_from(search_pool, remaining, top_n=1)
        guess = best[0][0]
        feedback = get_feedback(guess, target)
        remaining = filter_candidates(remaining, guess, feedback)
        solution.append({
            "guess": guess,
            "feedback": list(feedback),
            "feedback_display": feedback_to_str(feedback),
            "remaining_candidates": len(remaining),
        })

    # Final guess (if not already guessed)
    if remaining and solution[-1]["guess"] != target:
        solution.append({
            "guess": target,
            "feedback": [2, 2, 2, 2, 2],
            "feedback_display": "🟩🟩🟩🟩🟩",
            "remaining_candidates": 1,
        })

    return solution


def main():
    parser = argparse.ArgumentParser(
        description="Generate Letter Guess strategy-switching problems"
    )
    parser.add_argument(
        "--top", type=int, default=10,
        help="Number of best problems to output (default: 10)",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(Path(__file__).resolve().parent / "letter_guess_tasks.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--min-gap", type=float, default=0.3,
        help="Minimum info gap in bits for a valid switch (default: 0.3)",
    )
    parser.add_argument(
        "--max-targets", type=int, default=500,
        help="Max target words to evaluate (default: 500)",
    )
    args = parser.parse_args()

    solutions, all_valid = load_word_list()

    # Evaluate targets
    import random
    random.seed(42)
    targets = random.sample(solutions, min(args.max_targets, len(solutions)))

    problems = []
    for i, target in enumerate(targets):
        if (i + 1) % 50 == 0:
            print(
                f"  Evaluated {i+1}/{len(targets)} targets, "
                f"found {len(problems)} problems so far",
                file=sys.stderr,
            )
        result = find_strategy_switch_point(
            target, solutions, all_valid, min_info_gap=args.min_gap
        )
        if result:
            problems.append(result)

    # Sort by info gap (best switch opportunities first)
    problems.sort(key=lambda p: -p["info_gap_bits"])
    problems = problems[: args.top]

    print(
        f"\nFound {len(problems)} problems (showing top {args.top})",
        file=sys.stderr,
    )
    for i, p in enumerate(problems, 1):
        cands = p["candidates_at_switch"]
        preview = ", ".join(cands[:5])
        if len(cands) > 5:
            preview += f", ... ({len(cands)} total)"
        print(
            f"  [{i}] target={p['target']} | "
            f"gap={p['info_gap_bits']} bits | "
            f"candidates: {preview}",
            file=sys.stderr,
        )

    output = {
        "task_type": "letter_guess_strategy_switch",
        "description": (
            "Letter Guess problems where optimal play leads to a state with many "
            "similar candidates. Guessing candidate words has low information "
            "gain; guessing a non-candidate distinguisher word is strictly better. "
            "The agent must recognise this and pivot from candidate-guessing to "
            "distinguisher-guessing."
        ),
        "problems": problems,
    }

    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
