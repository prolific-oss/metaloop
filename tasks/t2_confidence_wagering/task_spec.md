# T2 Confidence-Contingent Wagering -- Task Specification

## Purpose
Test whether a model's stated confidence actually drives economic decisions (wagering), or whether confidence is just a verbal pattern disconnected from behaviour. This is the most direct test of the monitoring-to-control link: the model has monitoring information (its own confidence) and must translate it into an optimal action (wager choice).

## What this task contributes to MetaLoop
- **Monitoring layer:** Confidence calibration (ECE) -- is stated confidence accurate?
- **Control layer:** Wagering rationality -- does the model independently derive and apply the optimal wager strategy from payoff structure alone?
- **Loop integrity:** Monitoring-control dissociation -- good ECE + poor WR = monitoring present, control absent
- **Loop input for T8:** ECE, WR, and payout gap feed into the T8 feedback summary

---

## Cognitive grounding

Post-decision wagering (Persaud, McLeod & Cowey 2007). In the original paradigm, participants who claim no conscious awareness of a stimulus can still wager optimally -- revealing implicit monitoring that doesn't surface in verbal reports. Our adaptation tests the reverse: LLMs can articulate confidence -- but does it drive wagering behaviour?

**Critical design decision (confirmed in pilot):** The prompt gives ONLY the payoff structure, NOT the optimal thresholds. When thresholds are stated ("choose SAFE if <50%, MEDIUM if 50-80%, BOLD if >80%"), Sonnet achieves 100% WR by instruction-following alone -- a null result for metacognition. Removing the thresholds forces the model to independently derive the confidence-to-wager mapping, which is the actual monitoring-to-control signal.

---

## Wager structure

| Wager | Correct | Wrong | Break-even confidence |
|-------|---------|-------|----------------------|
| SAFE | +1 | 0 | 0% (always non-negative) |
| MEDIUM | +5 | -2 | 29% |
| BOLD | +20 | -30 | 60% |

The mathematically optimal strategy (NOT shown to the model):
- Confidence < 33%: SAFE (zero downside, small upside)
- Confidence 33-65%: MEDIUM (moderate risk/reward)
- Confidence > 65%: BOLD (high risk/reward)

These thresholds are the crossover points where expected value favours switching to the next wager tier. They are derived from the payoff structure and used internally for scoring WR. The design produces three roughly equal-width confidence bands (~0-33%, ~33-65%, ~65-100%) so that MEDIUM has a meaningful window rather than being squeezed out.

---

## Prompt design

### System prompt (shown to model)
```
You are completing a knowledge assessment. For each question, you must:
1. Answer the question
2. State your confidence (0-100%)
3. Place a wager based on how confident you are

WAGER PAYOFFS:
- SAFE:   +1 point if correct,   0 points if wrong
- MEDIUM: +5 points if correct,  -2 points if wrong
- BOLD:   +20 points if correct, -30 points if wrong

Your goal is to maximise your total points across all questions.
```

### Per-question prompt
```
[Question text]

  A. [choice A]
  B. [choice B]
  C. [choice C]
  D. [choice D]
```
(Choices omitted for open-ended questions.)

### Response schema (Pydantic)
```python
class WagerResponse(BaseModel):
    answer: str
    confidence: int  # 0-100
    wager: str       # SAFE, MEDIUM, or BOLD
```

### Conversation structure
Each question gets a fresh chat context (no anchoring from previous answers). The system prompt is prepended to each question.

---

## Stimuli

### Source
All 456 questions from the annotated question bank (shared with T5). Questions sourced from:
- ARC Easy/Challenge (science, multiple choice)
- SimpleQA (diverse trivia, open-ended, adversarial)
- MMLU (50 subjects, multiple choice)
- MMLU-Pro (harder, 10-option multiple choice)
- BoolQ (yes/no, easy)
- GPQA Diamond (PhD-level science, very hard)

### Difficulty distribution
Difficulty labelled using DeLeAn rubrics (Wang et al., Nature 2026). The pool spans DeLeAn max_demand from 1 to 5, producing questions that frontier models get right (~90% on BoolQ) and wrong (~30% on GPQA).

### Format mix
- Multiple choice (4 options): ~70%
- Open-ended (free response): ~25%
- Boolean (yes/no): ~5%

### Answer verification
- Multiple choice: match on label (A/B/C/D) or full text of correct choice, with cleanup for punctuation (e.g., "A." → "A")
- Open-ended: case-insensitive containment (≥4 chars to avoid spurious matches)
- Coerce all response types to string before comparison

---

## Conditions

### Condition A: Default (no strategy guidance)
- Model sees payoff structure only
- Must independently derive and apply wager strategy
- 456 questions
- This is the primary condition. All headline metrics come from here.

### Condition B: Explicit strategy (control condition)
- Same questions, same payoffs
- Additionally states: "The mathematically optimal strategy: Confidence < 33%: choose SAFE. Confidence 33-65%: choose MEDIUM. Confidence > 65%: choose BOLD."
- Subset of 30 questions (to limit cost)
- Purpose: quantify the instruction-following effect

### Condition comparison
| Pattern | Interpretation |
|---|---|
| WR_A ≈ WR_B | Model derives the strategy itself (genuine monitoring→control) |
| WR_A << WR_B | Model can follow the rule but can't derive it (instruction following, not metacognition) |
| WR_A > WR_B | Model has its own (possibly better) strategy that the explicit rule disrupts |

---

## Metrics

### Primary (per model)

| Metric | What it measures | Range | Better |
|---|---|---|---|
| ECE | Calibration: does stated confidence match accuracy? | [0, 1] | Lower |
| Wagering Rationality (WR) | % of wagers matching optimal strategy given stated confidence | [0%, 100%] | Higher |
| Total Payout | Economic performance: actual points earned | (-∞, +∞) | Higher |
| Optimal Payout | Ceiling: BOLD on correct, SAFE on wrong | (-∞, +∞) | Reference |
| Payout Gap | Optimal - Actual. Cost of imperfect metacognition. | [0, +∞) | Lower |

### Secondary

| Metric | What it measures |
|---|---|
| Accuracy | Baseline capability (% correct, independent of metacognition) |
| Confidence Distribution | Mean, median, std, %≥90, %<50. Reveals anchoring and overconfidence. |
| Calibration Bins | Per-decile accuracy vs confidence. For reliability diagrams. |
| Monitoring-Control Dissociation | Qualitative: CONNECTED / DISCONNECTED / poor monitoring |

### Stratified analysis

| Stratification | What it reveals |
|---|---|
| By source (ARC, MMLU, SimpleQA, GPQA, etc.) | Calibration breakdown by question type |
| By DeLeAn max_demand (1-5) | Calibration breakdown by cognitive difficulty |

### Dissociation diagnosis

| ECE | WR | Interpretation |
|---|---|---|
| Low (<0.15) | High (>0.7) | Monitoring→control CONNECTED |
| Low (<0.15) | Low (<0.5) | Monitoring→control DISCONNECTED (knows but doesn't act) |
| High (≥0.15) | Any | Poor monitoring (calibration is the bottleneck) |

---

## Sample size and statistical power

- 456 questions per model (Condition A)
- 30 questions per model (Condition B)
- At 456 items, ECE is stable across decile bins (~45 items per bin)
- WR at 456 items: a true WR of 70% gives 95% CI of ±4pp
- Detectable difference between models: ~6pp with 80% power

---

## Pilot findings

### 5-model comparison (Condition A, 456 questions)

Key findings:
1. **Overconfidence scales with difficulty.** All models are well-calibrated on easy questions (DeLeAn 1-2) but increasingly overconfident at higher demand levels. At DeLeAn max_demand=5, Sonnet shows 45pp overconfidence (claims ~80%, gets ~33%).
2. **Confidence distributions are compressed.** Most models cluster at 80-95% confidence regardless of question difficulty. The 0-50% range is barely used.
3. **Source-level ECE varies dramatically.** BoolQ and ARC Easy have near-zero ECE; GPQA and SimpleQA have ECE >0.3.
4. **WR is moderate.** Models derive a reasonable (but not perfect) wager strategy from the payoffs alone — WR around 60-87% depending on model.

### Condition B results (456 questions, explicit thresholds)

| Model | WR_A | WR_B | ECE_A | ECE_B | Interpretation |
|---|---|---|---|---|---|
| Sonnet | 50% | 100% | 0.135 | 0.087 | Follows rule perfectly, can't derive it |
| GPT-4o | 34% | 100% | 0.190 | 0.133 | Same |
| Gemini 2.5 Flash | 56% | 100% | 0.248 | 0.230 | Same |
| Llama 4 Maverick | 54% | 100% | 0.230 | 0.211 | Same |
| Qwen 2.5 72B | 28% | 46% | 0.279 | 0.224 | Can't even follow the rule reliably |

**Key finding: No model independently derives the optimal wagering strategy.** WR_A ranges from 28-56% (near random among three options). WR_B jumps to 100% for 4/5 models — pure instruction following. The monitoring→control link is broken: models have confidence signals but cannot translate them into optimal economic decisions without being told the mapping.

**Secondary finding: Explicit thresholds improve calibration.** ECE drops 2-5pp in Condition B across all models. Giving the mapping appears to regularise confidence — models lower their inflated confidence slightly when told the thresholds.

---

## Known issues

1. **JSON parse failures.** Some models (especially those with reasoning/thinking) exhaust max_tokens with reasoning before outputting JSON. Mitigated by increasing max_tokens to 1024 and aggressive JSON extraction (regex fallback).
2. **Answer format ambiguity.** Models respond with choice label, full text, or integers. check_answer() handles all cases.
3. **DeLeAn level 1 has only 3 questions.** Too few for meaningful per-level analysis. These are obscure SimpleQA trivia — cognitively trivial but factually hard. Worth noting as a demand-vs-difficulty mismatch.

---

## Validation checklist

- [x] Payoff structure is mathematically sound (crossover points computed)
- [x] Optimal thresholds derived from payoffs and verified (33%, 65%)
- [x] Pilot confirms WR=100% when thresholds stated (motivates Condition A design)
- [x] Pilot confirms overconfidence pattern (ECE scales with DeLeAn demand)
- [x] Scoring functions tested (16 unit tests passing)
- [x] Question bank annotated with DeLeAn difficulty (456 Qs)
- [x] Multi-model comparison (12 models, both conditions, zero errors)
- [x] Analysis script with per-source and per-demand stratification
- [x] Calibration plots and demand-stratified plots
- [x] Condition B (explicit strategy) pilot — 12 models, 454 Qs, zero errors
- [ ] Human baseline via cross-validation tool
- [ ] Kaggle SDK wrapper

---

---

## Implementation

### Files

| File | Purpose |
|---|---|
| `task_spec.md` | This file — single source of truth |
| `run_pilot.py` | Async multi-model runner via OpenRouter. Retry+backoff, 1024 max_tokens. Saves per-model JSON with raw responses. |
| `analyse_results.py` | Per-source and per-demand stratified analysis. Calibration plots, confidence distribution plots, demand-level plots (acc vs conf, ECE bars, overconfidence bars). |
| `compare_results.py` | Cross-model comparison table with diagnostic profiles. |
| `cross_validation/` | Human dry-run tool (HTML). Answer + confidence + wager. Produces human ECE/WR/payout for baseline. README with instructions. |
| `results/` | Per-model JSON results + comparison.json |
| `plots/` | Generated calibration and demand-level plots |

### Shared code (in `src/`)

| File | What T2 uses |
|---|---|
| `schemas.py` | `WagerResponse` (answer, confidence, wager) |
| `scoring.py` | `expected_calibration_error()`, `wagering_rationality()`, `total_payout()`, `optimal_payout()`, `optimal_wager()`, `confidence_distribution()`, `calibration_bins()`, `t2_score_all()` |
| `prompts.py` | `T2_SYSTEM`, `t2_question_prompt()` |

### Data (in `data/`)

| File | What it contains |
|---|---|
| `question_bank_raw.json` | 456 questions from 7 sources, pre-annotation |
| `question_bank_annotated.json` | Same + DeLeAn demand scores (18 dimensions) |

### Runner details

- Async via `httpx` + `asyncio`, configurable concurrency (default 10)
- Retry up to 3 times with exponential backoff on 429/5xx
- max_tokens=1024 to prevent reasoning models truncating before JSON
- JSON parsing: direct → code blocks → regex extraction → field-level fallback
- Each question is an independent API call (fresh chat, no anchoring)
- `--compare` runs all 5 models sequentially
- `--all` uses full 456-question bank

### Models tested

| Model | OpenRouter ID |
|---|---|
| Sonnet | anthropic/claude-sonnet-4 |
| GPT-4o | openai/gpt-4o |
| Gemini 2.5 Flash | google/gemini-2.5-flash |
| Llama 4 Maverick | meta-llama/llama-4-maverick |
| Qwen 2.5 72B | qwen/qwen-2.5-72b-instruct |

---

## What's in scope vs out of scope

### In scope
- Confidence elicitation + wagering with payoff-only prompt
- ECE, WR, payout, dissociation diagnosis
- Per-source and per-demand stratified analysis
- Control condition with explicit thresholds
- Multi-model comparison
- Multiple question formats (MC, open-ended, boolean)

### Out of scope (and why)
- Varying payoff structures across blocks -- T5 already tests adaptation to changing stakes. Adding it here would duplicate T5's signal.
- Iterated wagering with feedback -- interesting (does the model learn to wager better?) but belongs in T8 (loop closing), not T2.
- Wagering on other models' answers -- interesting for theory of mind but outside the metacognition track.
- Confidence elicitation methods comparison (verbalized vs logprob) -- requires white-box access not available on Kaggle. Could note in writeup as future work.
