# T7 Consistency Under Reformulation — Task Specification

## Purpose
Test whether metacognitive self-assessment is genuine (a stable internal signal) or a surface artifact of prompt framing. The same questions are presented in three framings — direct, counterfactual, and third-person. Genuine self-knowledge should be framing-invariant.

## What this task contributes to MetaLoop
- **Validity:** If metacognition is prompt-dependent, it undermines confidence data from T2, T3, and T6
- **Framing bias:** Do models rate themselves higher than "an AI like you"?
- **Best-framing calibration:** Which framing elicits the most accurate self-assessment?
- **Loop input for T8:** Framing consistency and bias feed into the T8 feedback summary

## Cognitive grounding
Framing effects on metacognitive judgments. Motivated by Xiao et al. (2025, Generalized Correctness Models), who found that LLM self-prediction of correctness performs no better than cross-model prediction — suggesting metacognitive responses may be pattern-matching rather than introspection. If metacognition reflects genuine self-knowledge, then the framing of a metacognitive question should not substantially change the answer. Humans show ~5-10pp framing effects; models should be comparable or better.

---

## Design

### Problem source
Questions drawn from the shared Q_BANK (GPQA Diamond, MMLU-Pro, MMLU). Same questions used in T6 (error detection) and T2 (wagering). Multiple-choice format with verified ground truth answers.

### Three framings

**Framing A (Direct):** Answer and rate confidence.
```
Answer the following question and rate your confidence.
[question + choices]
ANSWER: [letter]
CONFIDENCE: [0-100]
```

**Framing B (Counterfactual):** Predict performance, then answer.
```
Consider the following question:
[question + choices]
Without actually answering, estimate: if you were asked this question,
how likely is it that you would answer correctly?
PROBABILITY: [0-100]
Then go ahead and answer it.
ANSWER: [letter]
```

**Framing C (Third-person):** Depersonalised assessment.
```
Consider the following question:
[question + choices]
An AI assistant with exactly your capabilities and training is asked
this question. Rate the probability it answers correctly.
PROBABILITY: [0-100]
Now answer it yourself.
ANSWER: [letter]
```

### Randomisation
Each question × framing is a separate API call. All items are shuffled so the model never sees the same question in consecutive framings. This prevents priming for consistency.

---

## Metrics

### Primary (per model)

| Metric | Definition |
|---|---|
| Answer Consistency | % of questions with same answer across all 3 framings |
| Framing Consistency (FC) | Mean absolute deviation of confidence across 3 framings (lower = better) |
| Framing Bias | Mean signed difference between framing pairs (e.g., direct - third_person) |
| Accuracy by Framing | % correct in each framing separately |
| ECE by Framing | Expected Calibration Error computed per framing |

### Diagnostic

| Pattern | Interpretation |
|---|---|
| Low FC, high answer consistency | Genuine metacognition — framing-invariant |
| High FC, low answer consistency | Prompt-dependent metacognition — framing artifacts |
| Direct > third-person confidence | Self-referential bias — rates self higher |
| Direct accuracy > third-person accuracy | Framing disrupts reasoning, not just confidence |
| Low direct ECE, high third-person ECE | Self-knowledge is genuine but non-transferable |

---

## Pilot results (n=156 questions, 3 models)

| Metric | GPT-5.4 | Sonnet | Gemini |
|---|---|---|---|
| Accuracy (direct) | 74% | **87%** | 78% |
| Accuracy (third-person) | 72% | **48%** | 79% |
| Accuracy drop (dir→3rd) | -2pp | **-39pp** | +1pp |
| Answer consistency | 76% | **45%** | 76% |
| FC | 2.2pp | 4.8pp | **1.1pp** |
| Framing bias (dir-3rd) | +2.4pp | **+7.3pp** | +0.7pp |
| Direct ECE | 0.202 | **0.032** | 0.057 |

### Key findings

1. **Sonnet's metacognition is framing-fragile.** 39pp accuracy collapse in third-person, 45% answer consistency. Best direct calibration in the benchmark (ECE 0.032) but completely non-transferable. Metacognition is genuine in first-person but self-referential — it only works when the model is thinking about *itself*.

2. **Gemini is framing-invariant.** FC=1.1pp, negligible accuracy shifts. Passes the consistency test. But consistently overconfident (~98%) with no discrimination signal.

3. **GPT-5.4 shows moderate framing effects.** Small accuracy shifts, reasonable consistency. Worst calibration (ECE 0.20).

4. **Connects to T6 findings.** Sonnet has zero self-correction (T6) AND framing-fragile metacognition (T7). The model's self-monitoring only works in direct first-person engagement.

---

## Sample size

- 156 questions × 3 framings = 468 API calls per model
- Single-turn calls (no multi-phase complexity)
- Fully automated scoring against MC ground truth

---

## Validation

- [x] Pilot on 3 models confirms framing effects are real and model-specific
- [x] Three distinct profiles: framing-invariant, moderate, framing-fragile
- [x] ECE by framing reveals which self-assessment mode is most calibrated
- [x] Scaled results (n=156) confirm pilot patterns
- [ ] Human baselines via Prolific (expect 5-10pp framing effects)
