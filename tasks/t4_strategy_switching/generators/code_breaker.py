#!/usr/bin/env python3
"""
Generate Code Breaker strategy-switching problems.

Precomputes a 1296×1296 feedback table for fast information-gain evaluation.
Finds secrets where guessing a non-candidate distinguisher is strictly better
than guessing any candidate.

Usage:
    python generate_code_breaker_tasks.py --top 10 --output tasks.json
"""

import argparse
import json
import random
import sys
from collections import Counter
from itertools import product
from math import log2
from pathlib import Path

SEED = 42
COLOURS = ["R", "G", "B", "Y", "O", "P"]
N_POSITIONS = 4
N_COLOURS = len(COLOURS)


def all_codes():
    return list(product(COLOURS, repeat=N_POSITIONS))


def code_to_str(code):
    return "".join(code)


def get_feedback(guess, secret):
    exact = sum(g == s for g, s in zip(guess, secret))
    gc, sc = Counter(guess), Counter(secret)
    colour = sum((gc & sc).values()) - exact
    return exact, colour


def feedback_to_str(fb):
    return f"{fb[0]} exact, {fb[1]} colour"


def build_feedback_table(codes):
    """Precompute feedback for all code pairs. Returns table[i][j] = encoded feedback."""
    n = len(codes)
    table = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            exact, colour = get_feedback(codes[i], codes[j])
            table[i][j] = exact * 5 + colour  # encode as single int
        if (i + 1) % 200 == 0:
            print(f"  Feedback table: {i+1}/{n} rows", file=sys.stderr)
    return table


def expected_info_fast(guess_idx, candidate_indices, table):
    """Expected bits from a guess, using precomputed feedback table."""
    if len(candidate_indices) <= 1:
        return 0.0
    row = table[guess_idx]
    patterns = Counter(row[c] for c in candidate_indices)
    total = len(candidate_indices)
    info = 0.0
    for count in patterns.values():
        p = count / total
        info -= p * log2(p)
    return info


def best_guess_fast(guess_indices, candidate_indices, table, top_n=1):
    """Find best guesses by expected information gain."""
    scored = [(g, expected_info_fast(g, candidate_indices, table)) for g in guess_indices]
    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]


def filter_candidates_fast(candidate_indices, guess_idx, feedback_int, table):
    """Return candidate indices consistent with observed feedback."""
    row = table[guess_idx]
    return [c for c in candidate_indices if row[c] == feedback_int]


def find_switch_point(secret_idx, codes, table, all_indices, opener_idx,
                      min_candidates=4, max_candidates=80, min_info_gap=0.2):
    """Play 2 openers toward secret and check if a strategy switch arises."""
    candidates = list(all_indices)
    history = []

    for turn in range(2):
        if turn == 0:
            guess_idx = opener_idx
        else:
            best = best_guess_fast(all_indices, candidates, table, top_n=1)
            guess_idx = best[0][0]

        fb = get_feedback(codes[guess_idx], codes[secret_idx])
        fb_int = fb[0] * 5 + fb[1]
        candidates = filter_candidates_fast(candidates, guess_idx, fb_int, table)
        history.append({
            "guess": code_to_str(codes[guess_idx]),
            "feedback": list(fb),
            "feedback_display": feedback_to_str(fb),
            "remaining_candidates": len(candidates),
        })

        if len(candidates) <= 1:
            return None

    n = len(candidates)
    if n < min_candidates or n > max_candidates:
        return None

    best_from_cands = best_guess_fast(candidates, candidates, table, top_n=3)
    best_from_all = best_guess_fast(all_indices, candidates, table, top_n=3)

    best_cand_info = best_from_cands[0][1]
    best_all_info = best_from_all[0][1]
    info_gap = best_all_info - best_cand_info

    if info_gap < min_info_gap:
        return None

    best_overall_idx = best_from_all[0][0]
    if best_overall_idx in candidates:
        return None

    # Verify the candidate guess doesn't solve the puzzle (must leave >= 3 candidates)
    cand_guess_idx = best_from_cands[0][0]
    cand_fb = get_feedback(codes[cand_guess_idx], codes[secret_idx])
    cand_fb_int = cand_fb[0] * 5 + cand_fb[1]
    post_cand = filter_candidates_fast(candidates, cand_guess_idx, cand_fb_int, table)
    if len(post_cand) < 3:
        return None

    return {
        "secret": code_to_str(codes[secret_idx]),
        "secret_has_repeats": len(set(codes[secret_idx])) < N_POSITIONS,
        "history": history,
        "candidates_at_switch": sorted(code_to_str(codes[c]) for c in candidates),
        "n_candidates": n,
        "best_candidate_guess": {
            "code": code_to_str(codes[best_from_cands[0][0]]),
            "info_bits": round(best_from_cands[0][1], 3),
            "top_3": [
                {"code": code_to_str(codes[idx]), "info_bits": round(bits, 3)}
                for idx, bits in best_from_cands
            ],
        },
        "best_overall_guess": {
            "code": code_to_str(codes[best_from_all[0][0]]),
            "info_bits": round(best_from_all[0][1], 3),
            "top_3": [
                {"code": code_to_str(codes[idx]), "info_bits": round(bits, 3)}
                for idx, bits in best_from_all
            ],
        },
        "info_gap_bits": round(info_gap, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate Code Breaker strategy-switching problems")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", type=str, default=str(Path(__file__).resolve().parent / "code_breaker_tasks.json"))
    parser.add_argument("--min-gap", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)

    codes = all_codes()
    n = len(codes)
    all_indices = list(range(n))
    print(f"Total codes: {n}", file=sys.stderr)

    print("Building feedback table...", file=sys.stderr)
    table = build_feedback_table(codes)
    print("Feedback table complete.", file=sys.stderr)

    # Find best opener (cached for all targets)
    known_good = [
        ("R", "R", "G", "B"),
        ("R", "G", "B", "Y"),
        ("R", "R", "G", "G"),
    ]
    code_to_idx = {c: i for i, c in enumerate(codes)}
    opener_candidates = [code_to_idx[c] for c in known_good if c in code_to_idx]
    opener_idx = max(opener_candidates, key=lambda g: expected_info_fast(g, all_indices, table))
    print(f"Opener: {code_to_str(codes[opener_idx])}", file=sys.stderr)

    # Evaluate all secrets
    problems = []
    for i in range(n):
        result = find_switch_point(i, codes, table, all_indices, opener_idx, min_info_gap=args.min_gap)
        if result:
            problems.append(result)
        if (i + 1) % 200 == 0:
            print(f"  Evaluated {i+1}/{n} secrets, found {len(problems)} problems", file=sys.stderr)

    problems.sort(key=lambda p: -p["info_gap_bits"])
    problems = problems[:args.top]

    print(f"\nFound {len(problems)} problems", file=sys.stderr)
    for i, p in enumerate(problems):
        print(
            f"  [{i}] secret={p['secret']} gap={p['info_gap_bits']} bits "
            f"candidates={p['n_candidates']}",
            file=sys.stderr,
        )

    output = {
        "task_type": "code_breaker_strategy_switch",
        "description": (
            "Code Breaker problems (4 positions, 6 colours, repeats allowed) where "
            "guessing a non-candidate distinguisher is strictly better than "
            "guessing any candidate code."
        ),
        "colours": COLOURS,
        "n_positions": N_POSITIONS,
        "problems": problems,
    }

    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
