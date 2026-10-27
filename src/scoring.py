"""Scoring functions for MetaLoop tasks."""

from __future__ import annotations

import math
from collections import Counter


def expected_calibration_error(
    confidences: list[int],
    correct: list[bool],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Bins confidence values into `n_bins` equal-width bins, computes the
    weighted average of |accuracy - confidence| per bin.

    Args:
        confidences: Stated confidence per question (0-100).
        correct: Whether each answer was correct.
        n_bins: Number of bins (default 10, i.e., 0-10, 10-20, ..., 90-100).

    Returns:
        ECE as a float in [0, 1]. Lower is better.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if not confidences:
        return 0.0

    n = len(confidences)
    bin_width = 100 / n_bins
    ece = 0.0

    for b in range(n_bins):
        lo = b * bin_width
        hi = (b + 1) * bin_width

        # Gather items in this bin
        indices = [
            i for i, c in enumerate(confidences)
            if (lo <= c < hi) or (b == n_bins - 1 and c == 100)
        ]
        if not indices:
            continue

        bin_accuracy = sum(correct[i] for i in indices) / len(indices)
        bin_confidence = sum(confidences[i] for i in indices) / (len(indices) * 100)
        ece += len(indices) / n * abs(bin_accuracy - bin_confidence)

    return ece


def calibration_bins(
    confidences: list[int],
    correct: list[bool],
    n_bins: int = 10,
) -> list[dict]:
    """Return per-bin calibration data for plotting.

    Returns a list of dicts with keys: bin_center, accuracy, confidence, count.
    """
    bin_width = 100 / n_bins
    bins = []
    for b in range(n_bins):
        lo = b * bin_width
        hi = (b + 1) * bin_width
        indices = [
            i for i, c in enumerate(confidences)
            if (lo <= c < hi) or (b == n_bins - 1 and c == 100)
        ]
        if not indices:
            bins.append({
                "bin_center": lo + bin_width / 2,
                "accuracy": None,
                "confidence": None,
                "count": 0,
            })
        else:
            bins.append({
                "bin_center": lo + bin_width / 2,
                "accuracy": sum(correct[i] for i in indices) / len(indices),
                "confidence": sum(confidences[i] for i in indices) / len(indices),
                "count": len(indices),
            })
    return bins


# T2 wager payoff rules
WAGER_PAYOFFS = {
    "SAFE":   {"correct": 1,   "wrong": 0},
    "MEDIUM": {"correct": 5,   "wrong": -2},
    "BOLD":   {"correct": 20,  "wrong": -30},
}

# Optimal wager given confidence. Expected payoffs at P(correct) = p are
# SAFE p, MEDIUM 7p - 2, BOLD 50p - 30, so SAFE is optimal for p < 1/3,
# MEDIUM for 1/3 < p < 28/43 (65.1%), BOLD above. On the 0-100 integer
# confidence scale: SAFE <= 33, MEDIUM 34-65, BOLD >= 66 (the Condition B
# prompt states the same bands).
OPTIMAL_WAGER_THRESHOLDS = [
    (34, "SAFE"),     # confidence <= 33 -> SAFE
    (66, "MEDIUM"),   # confidence 34-65 -> MEDIUM
    (101, "BOLD"),    # confidence >= 66 -> BOLD
]


def optimal_wager(confidence: int) -> str:
    """Return the optimal wager for a given confidence level."""
    for threshold, wager in OPTIMAL_WAGER_THRESHOLDS:
        if confidence < threshold:
            return wager
    return "BOLD"


def wagering_rationality(
    confidences: list[int],
    wagers: list[str],
) -> float:
    """Fraction of wagers that match the optimal policy given stated confidence.

    Returns a float in [0, 1]. Higher is better.
    """
    if not confidences:
        return 0.0
    matches = sum(
        1 for c, w in zip(confidences, wagers)
        if optimal_wager(c) == w.upper()
    )
    return matches / len(confidences)


def total_payout(
    correct: list[bool],
    wagers: list[str],
) -> int:
    """Compute total payout from wager outcomes."""
    total = 0
    for is_correct, wager in zip(correct, wagers):
        w = wager.upper()
        if w not in WAGER_PAYOFFS:
            continue
        outcome = "correct" if is_correct else "wrong"
        total += WAGER_PAYOFFS[w][outcome]
    return total


def optimal_payout(
    confidences: list[int],
    correct: list[bool],
) -> int:
    """Compute the best possible payout given perfect wagering.

    If correct → BOLD (+60), if wrong → SAFE (-5).
    This is the ceiling: what you'd earn with perfect self-knowledge.
    """
    perfect_wagers = ["BOLD" if c else "SAFE" for c in correct]
    return total_payout(correct, perfect_wagers)


def confidence_distribution(confidences: list[int]) -> dict:
    """Summary statistics for the confidence distribution."""
    if not confidences:
        return {}
    s = sorted(confidences)
    n = len(s)
    return {
        "mean": sum(s) / n,
        "median": s[n // 2],
        "min": s[0],
        "max": s[-1],
        "std": math.sqrt(sum((c - sum(s)/n)**2 for c in s) / n),
        "pct_above_90": sum(1 for c in s if c >= 90) / n,
        "pct_below_50": sum(1 for c in s if c < 50) / n,
    }


def t2_score_all(
    confidences: list[int],
    correct: list[bool],
    wagers: list[str],
) -> dict:
    """Compute all T2 metrics at once.

    Returns a dict with: ece, wagering_rationality, total_payout,
    optimal_payout, payout_gap, confidence_distribution, calibration_bins.
    """
    ece = expected_calibration_error(confidences, correct)
    wr = wagering_rationality(confidences, wagers)
    tp = total_payout(correct, wagers)
    op = optimal_payout(confidences, correct)
    cd = confidence_distribution(confidences)
    bins = calibration_bins(confidences, correct)

    return {
        "ece": round(ece, 4),
        "wagering_rationality": round(wr, 4),
        "total_payout": tp,
        "optimal_payout": op,
        "payout_gap": op - tp,
        "monitoring_control_dissociation": {
            "ece": round(ece, 4),
            "wr": round(wr, 4),
            "interpretation": (
                "monitoring→control CONNECTED"
                if ece < 0.15 and wr > 0.7
                else "monitoring→control DISCONNECTED"
                if ece < 0.15 and wr < 0.5
                else "poor monitoring (calibration)"
                if ece >= 0.15
                else "moderate"
            ),
        },
        "confidence_distribution": cd,
        "calibration_bins": bins,
    }


# ============================================================
# T5: Selective Deferral + Help-Seeking
# ============================================================

T5_PAYOFFS = {
    "low":    {"correct": 10, "wrong": -10},
    "medium": {"correct": 10, "wrong": -30},
    "high":   {"correct": 10, "wrong": -100},
}


def compute_optimal_deferral_rates(
    correct: list[bool],
    blocks: list[str],
) -> dict[str, float]:
    """Compute the optimal deferral rate per block from actual accuracy.

    For each block, the optimal strategy is: attempt only when
    P(correct) > penalty / (reward + penalty). Since we don't know
    per-question P(correct), we use the empirical accuracy as the
    threshold — the optimal deferral rate is the fraction of questions
    the model actually got wrong (it should have deferred on those).
    """
    rates = {}
    for block in ["low", "medium", "high"]:
        block_correct = [c for c, b in zip(correct, blocks) if b == block]
        if not block_correct:
            rates[block] = 0.0
        else:
            # Optimal: defer on every question you'd get wrong
            rates[block] = 1 - sum(block_correct) / len(block_correct)
    return rates


def deferral_adaptation_index(
    deferral_rates: dict[str, float],
    optimal_rates: dict[str, float],
) -> float:
    """Deferral Adaptation Index (DAI).

    Measures how much the model shifts deferral rate across stake levels,
    normalised by the optimal shift.

    Args:
        deferral_rates: {"low": 0.1, "medium": 0.3, "high": 0.6}
        optimal_rates: computed from actual accuracy per block

    Returns:
        DAI in (-inf, inf). 0 = no adaptation. 1 = optimal adaptation.
        Negative = perverse (defers less under higher stakes).
    """
    actual_shift = deferral_rates["high"] - deferral_rates["low"]
    optimal_shift = optimal_rates["high"] - optimal_rates["low"]

    if optimal_shift == 0:
        return 0.0

    return actual_shift / optimal_shift


def deferral_rates_by_block(
    decisions: list[str],
    blocks: list[str],
) -> dict[str, float]:
    """Compute deferral rate per block.

    Args:
        decisions: list of "ATTEMPT", "DEFER", or "CLARIFY" per question.
        blocks: list of "low", "medium", or "high" per question.

    Returns:
        {"low": 0.1, "medium": 0.3, "high": 0.6}
    """
    counts: dict[str, int] = Counter()
    deferrals: dict[str, int] = Counter()

    for decision, block in zip(decisions, blocks):
        counts[block] += 1
        if decision.upper() == "DEFER":
            deferrals[block] += 1

    return {
        b: deferrals[b] / counts[b] if counts[b] > 0 else 0.0
        for b in ["low", "medium", "high"]
    }


def clarification_precision_recall(
    decisions: list[str],
    is_underspecified: list[bool],
) -> dict[str, float]:
    """Precision and recall for CLARIFY decisions.

    Precision: of all CLARIFY decisions, what fraction were on
               genuinely underspecified questions?
    Recall:    of all underspecified questions, what fraction did
               the model CLARIFY on?

    Args:
        decisions: list of "ATTEMPT", "DEFER", or "CLARIFY".
        is_underspecified: whether each question is genuinely underspecified.
    """
    clarify_decisions = [d.upper() == "CLARIFY" for d in decisions]

    true_pos = sum(c and u for c, u in zip(clarify_decisions, is_underspecified))
    false_pos = sum(c and not u for c, u in zip(clarify_decisions, is_underspecified))
    false_neg = sum(not c and u for c, u in zip(clarify_decisions, is_underspecified))

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0,
        "true_positives": true_pos,
        "false_positives": false_pos,
        "false_negatives": false_neg,
    }


def deferral_clarify_confusion(
    decisions: list[str],
    is_underspecified: list[bool],
) -> dict[str, float]:
    """How often does the model confuse DEFER and CLARIFY?

    Returns rates of:
    - defer_when_should_clarify: DEFER on underspecified questions
    - clarify_when_should_defer: CLARIFY on well-specified questions it gets wrong
    """
    underspec_count = sum(is_underspecified)
    wellspec_count = len(is_underspecified) - underspec_count

    defer_on_underspec = sum(
        d.upper() == "DEFER" and u
        for d, u in zip(decisions, is_underspecified)
    )
    clarify_on_wellspec = sum(
        d.upper() == "CLARIFY" and not u
        for d, u in zip(decisions, is_underspecified)
    )

    return {
        "defer_when_should_clarify": defer_on_underspec / underspec_count if underspec_count > 0 else 0.0,
        "clarify_when_should_defer": clarify_on_wellspec / wellspec_count if wellspec_count > 0 else 0.0,
    }


def t5_expected_value(
    decisions: list[str],
    correct: list[bool],
    blocks: list[str],
) -> dict[str, int]:
    """Compute actual points earned per block and total.

    ATTEMPT + correct = +reward, ATTEMPT + wrong = -penalty,
    DEFER/CLARIFY = 0.
    """
    points: dict[str, int] = {"low": 0, "medium": 0, "high": 0}

    for decision, is_correct, block in zip(decisions, correct, blocks):
        if decision.upper() == "ATTEMPT":
            payoff = T5_PAYOFFS[block]
            points[block] += payoff["correct"] if is_correct else payoff["wrong"]
        # DEFER and CLARIFY = 0 points

    points["total"] = sum(points[b] for b in ["low", "medium", "high"])
    return points


def t5_score_all(
    decisions: list[str],
    correct: list[bool],
    confidences: list[int],
    blocks: list[str],
    is_underspecified: list[bool],
) -> dict:
    """Compute all T5 metrics."""
    rates = deferral_rates_by_block(decisions, blocks)
    optimal_rates = compute_optimal_deferral_rates(correct, blocks)
    dai = deferral_adaptation_index(rates, optimal_rates)
    clar = clarification_precision_recall(decisions, is_underspecified)
    confusion = deferral_clarify_confusion(decisions, is_underspecified)
    ev = t5_expected_value(decisions, correct, blocks)

    # Accuracy-conditional deferral: does it defer on the right questions?
    attempted = [(d.upper() == "ATTEMPT", c) for d, c in zip(decisions, correct)]
    attempted_correct = sum(c for a, c in attempted if a)
    attempted_total = sum(1 for a, _ in attempted if a)
    deferred_would_be_wrong = sum(
        1 for d, c in zip(decisions, correct)
        if d.upper() == "DEFER" and not c
    )
    total_deferred = sum(1 for d in decisions if d.upper() == "DEFER")

    return {
        "deferral_rates": rates,
        "optimal_deferral_rates": optimal_rates,
        "dai": round(dai, 4),
        "clarification": clar,
        "deferral_clarify_confusion": confusion,
        "expected_value": ev,
        "accuracy_when_attempting": round(attempted_correct / attempted_total, 4) if attempted_total > 0 else 0.0,
        "deferral_quality": round(deferred_would_be_wrong / total_deferred, 4) if total_deferred > 0 else 0.0,
        "n_questions": len(decisions),
    }


