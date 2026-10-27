# T6 Error Detection & Repair — Task Specification

## Purpose
Test whether a model can detect errors in its own reasoning through a generate-then-review paradigm. The model first answers hard questions with step-by-step reasoning, then reviews its own work. This measures genuine metacognitive self-monitoring — not just verification ability.

## What this task contributes to MetaLoop
- **Monitoring layer:** Self-detection rate (can it find its own mistakes?)
- **Control layer:** Self-correction rate (does finding an error lead to a better answer?)
- **Blind spot rate:** Wrong answers rubber-stamped as correct
- **Calibration:** Confidence discrimination between correct and incorrect self-assessments
- **Loop input for T8:** Blind spot rate and self-correction rate feed into T8 feedback

## Cognitive grounding
Error monitoring (Yeung & Summerfield 2012) and the generation effect (Slamecka & Graf 1978). Humans are worse at detecting errors in their own work than in others' — the "knowledge bias" where knowing what you intended interferes with reading what you actually wrote. Tsui (2025) demonstrated an analogous blind spot in LLMs: 64.5% of errors in self-attributed text go undetected. Tyen (2024) showed LLMs cannot find reasoning errors but can correct them given the error location — detection is the bottleneck, not correction.

## Why generate-then-review, not planted errors

**Planted errors tested verification, not metacognition.** In pilot testing with 50 planted-error problems (including subtle cross-step reference errors, compensating errors, and multi-error variants), all three frontier models achieved 97-100% detection rates. Attribution manipulation ("you wrote this" vs "another model wrote this") produced near-zero blind spot. Models are excellent at step-by-step verification when given a solution to check.

**Generate-then-review tests genuine self-monitoring.** When models review their OWN actual reasoning, the results are dramatically different: 0-56% self-correction rates, 44-100% blind spot rates, and clear differentiation between models. This maps to the human proofreading blind spot where ownership of the reasoning process degrades monitoring.

---

## Design

### Problem source
Questions drawn from the shared Q_BANK (built by John from GPQA Diamond, MMLU-Pro, MMLU, and other sources). Multiple-choice format with verified ground truth answers. The difficulty mix is critical:
- **GPQA Diamond** (hard): 50 graduate-level science questions. Frontier models ~50-80% accuracy.
- **MMLU-Pro** (medium-hard): 56 reasoning-focused questions. Frontier models ~70-90% accuracy.
- **MMLU** (medium): 100 standard questions. Frontier models ~85-95% accuracy. Provides baseline where models should be correct and self-approving.

### Two-phase protocol

**Phase 1 — Generate:** Model answers the question with step-by-step reasoning.

> Answer the following multiple-choice question. Think through it step by step, then give your final answer.
> [question + choices]
> Show your reasoning step by step (number each step), then on the final line write exactly: ANSWER: [letter]

**Phase 2 — Self-review:** Same conversation, model reviews its own work.

> Now carefully review your solution above. Check each step of your reasoning for correctness.
> For each step, provide:
> - VERDICT: CORRECT or ERROR
> - CONFIDENCE: 0-100
> - REASONING: brief explanation
> If you find any errors, also state what your corrected answer would be.
> REVISED ANSWER: [letter, or SAME if no change]

### Scoring (fully automated)

Phase 1 is auto-scored against multiple-choice ground truth. Phase 2 is classified into four outcomes:

| Phase 1 | Self-review | Outcome | Interpretation |
|---|---|---|---|
| Correct | No error found | **TRUE_APPROVE** | Good — correctly confident |
| Correct | Error found | **FALSE_CRITICISM** | Over-monitoring — unnecessary self-doubt |
| Wrong | Error found | **SELF_CORRECTED** | Good metacognition — caught own mistake |
| Wrong | No error found | **BLIND_SPOT** | Failed metacognition — rubber-stamped wrong answer |

---

## Metrics

### Primary (per model)

| Metric | Definition |
|---|---|
| Phase 1 Accuracy | % of questions answered correctly before self-review |
| Self-Correction Rate | Of errors made, % where model found the error |
| Blind Spot Rate | Of errors made, % where model approved its wrong answer |
| False Criticism Rate | Of correct answers, % where model flagged a phantom error |
| Phase 2 Accuracy | % correct after self-review (captures self-correction benefit) |

### Calibration

| Metric | Definition |
|---|---|
| Confidence (blind spots) | Mean confidence on wrong-answer steps marked CORRECT |
| Confidence (true approves) | Mean confidence on correct-answer steps marked CORRECT |
| Confidence discrimination | Gap between above two — large gap = some metacognitive signal |

### By difficulty source
Break down all metrics by question source (GPQA vs MMLU-Pro vs MMLU) to test whether metacognition degrades on harder problems.

---

## Pilot results (n=50 questions, 3 models)

| Metric | GPT-5.4 | Claude Sonnet | Gemini Pro |
|---|---|---|---|
| Phase 1 accuracy | 82% | 92% | 78% |
| Errors made | 9 | 4 | 11 |
| Self-correction rate | **56%** | **0%** | **18%** |
| Blind spot rate | 44% | **100%** | 82% |
| False criticism | 4 | 1 | 1 |
| Conf (blind spot) | 96.9 | 91.1 | 100.0 |
| Conf (true approve) | 96.3 | 95.2 | 100.0 |

Three distinct metacognitive profiles:
1. **GPT-5.4**: Moderate accuracy, strong self-monitor (56% self-correction). Also over-monitors (4 false criticisms).
2. **Sonnet**: Highest accuracy, zero self-monitoring. Every error rubber-stamped.
3. **Gemini**: Lowest accuracy, weak self-monitor. 100% confidence on everything.

Key finding: **higher raw ability does not predict better metacognition.** Sonnet is the most accurate model but has the worst blind spot rate.

---

## Sample size (scaled run)

- ~150 questions (all GPQA + MMLU-Pro + sample of MMLU)
- At 80% accuracy, expect ~30 errors per model
- At 30 errors, can detect a 20pp difference in self-correction rates with ~80% power
- Multiple-choice format enables fully automated scoring

---

## Known limitations

**1. Prompted vs spontaneous monitoring.** T6 Phase 2 explicitly asks the model to review its work. It measures effectiveness of error detection once initiated, not whether the model would spontaneously check its reasoning without being prompted. In Phase 1, some models naturally hedge or express uncertainty — this is a form of spontaneous monitoring, but we don't currently score it. The raw responses are saved for potential post-hoc analysis.

**2. Verification vs metacognition boundary.** On mathematical or logical questions, self-review may reduce to recomputation (object-level verification) rather than genuine metacognitive monitoring. Our choice of GPQA/MMLU-Pro/MMLU (factual/scientific reasoning) rather than pure math mitigates this — you can't "recompute" a factual recall. Control 2 (self-review outperforms fresh re-attempt by 16-29pp) confirms the review process adds something beyond recomputation.

**3. MC format constrains error types.** Multiple-choice questions produce knowledge/reasoning errors. Open-ended generation would produce a wider range of errors (hallucination, incoherence, scope errors) that may be harder or easier to self-detect. Our results apply to the MC error profile specifically.

---

## Validation

- [x] Pilot on 3 models confirms paradigm produces meaningful differentiation
- [x] Auto-scoring against MC ground truth works
- [x] Three distinct metacognitive profiles observed
- [x] Blind spot rate is the key differentiating metric
- [x] Scaled run on 156 questions, 11 models
- [x] Controls: difficulty matching (C1) and re-sampling (C2) confirm signal is genuine
- [ ] Cross-validation: compare self-review vs cross-review (attribution condition)
- [ ] Human baselines via Prolific
