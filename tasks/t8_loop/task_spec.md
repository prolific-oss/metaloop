# T8 The Loop: Metacognitive Updating from Feedback

## Purpose
The capstone task. Tests whether models can improve their self-monitoring based on performance feedback. Tasks 2–7 measure individual links in the metacognitive chain; Task 8 tests whether the chain forms a functional loop that updates from feedback.

## Cognitive grounding
Metacognitive training and calibration feedback (Koriat 1997). The feedback arrow in Nelson & Narens (1990) flows from monitoring back to metacognitive knowledge, which updates future monitoring and control. Humans are mediocre at incorporating calibration feedback: they adjust partially but still fall short of optimal. This task tests whether LLMs can do what humans struggle with.

---

## Design history (two designs; the paper reports the live-feedback one)

### Summary feedback: targeted vs generic (`run.py`)
Each model received a natural-language summary of its own T3/T6/T7 performance ("targeted") or a generic "be accurate and careful" prompt ("generic"), then a mixed retest per arm (30 T6-style self-review items, 15 T7-style framing items in two framings, 10 T3-style source-monitoring items). Run on all 11 roster models that still have a serving channel, both arms; the full per-item data is in `results/design_a/` and analysed by `analyse_design_a.py`. Grok 4.1-fast, deprecated provider-side, could not be re-run.

Across the 11 models the targeted-minus-generic difference is small and inconsistent, with per-model error counts of only 6 to 25, so the comparison is underpowered and supports no reliable direction. Verdicts here are LLM-adjudicated: an earlier regex scoring read targeted feedback as harmful, but that was an artifact of the verdict-extraction over-flag (see extraction notes) and does not hold under adjudicated verdicts. The summary design's upfront delivery also conflates feedback with expectation-priming. Both are why the paper reports live, item-by-item feedback rather than the summary design.

### Live item-by-item feedback: the paper's protocol (`run_live.py`)
Each model is retested on questions it previously got wrong in T6 (n≈15) plus matched previously-correct controls (n≈15), selected by T6 phase-1 correctness. Per item:

1. Answer (step by step, `ANSWER: [letter]`)
2. Self-review (`VERDICT: CORRECT/ERROR`, optional revised answer)
3. Ground-truth feedback with a running score line, in-conversation

Live feedback avoids the summary design's expectation-loading by giving ground truth after each response rather than as an upfront self-image.

### No-feedback control (added Jul 2026, `run_control.py`)
Same per-model item lists (replayed from each model's live history) and the same answer plus self-review flow as the live protocol, with the ground-truth feedback replaced by an uninformative acknowledgment. Isolates the live-feedback effect from practice and re-exposure. Run alongside a DeepSeek live rerun (see extraction notes).

---

## Metrics (per arm)

| Metric | Definition |
|---|---|
| Accuracy | phase-1 correct / items |
| Blind spot rate | errors approved as correct / errors |
| Self-correction rate | errors flagged in self-review / errors |
| False criticism rate | correct answers flagged as errors / correct |
| Prev-wrong recovery | previously-wrong items now answered correctly |
| Adaptation | first-half blind-spot rate − second-half blind-spot rate |

**Feedback effect** (the loop-closure signal) = live-arm value − control-arm value, per metric per model. Live > control on adaptation/recovery indicates genuine updating from feedback; live ≈ control indicates practice effects.

## Extraction notes (Jul 2026 audit)
- Free-text `VERDICT` extraction: the plain-string regex both understated found_error for models that wrote bold-markdown verdicts (`**VERDICT:** ERROR`) and overstated it for verbose models that flagged an error mid-review and then retracted it. Regex cannot read a deliberative verdict. The Self-Review verdicts are therefore scored by an LLM reading each review, with a three-model majority on the cases where it disagreed with the regex, and the summary-feedback comparison was re-run with the same LLM scoring. The live and control Loop reviews are short and were left regex-extracted; the audit found 98.7% agreement with the LLM there (8 of 628), so they are unchanged. Every review was audited and the records ship in `extraction_audit/`.
- `extract_answer` now tolerates template-echoed answers (`ANSWER: [D]`); the April DeepSeek live run mis-parsed 10/30 answers (5 actually correct) and consequently delivered false feedback in-conversation, so DeepSeek's live arm was rerun with the fixed extractor (April data preserved in `_april_backup/`).
- Item `mmlu_236` (non-letter ground truth; see `clean_results.py`) is excluded from T8 analysis where present.
