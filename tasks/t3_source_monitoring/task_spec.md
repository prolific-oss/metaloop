# T3 Source Monitoring -- Task Specification

## Purpose
Test whether a model can track the provenance of its knowledge: distinguishing what was stated in a provided text, what it already knew, what it derived by reasoning, and what it confabulated. Then test whether it can ACT on those source judgments (monitoring-to-control link) and whether its source judgments are well-calibrated.

## What this task contributes to MetaLoop
- **Monitoring layer:** Source discrimination accuracy (can it classify?)
- **Control layer:** Can it use source judgments to guide behaviour (e.g., only cite passage-stated facts)?
- **Calibration:** Is it confident on correct classifications and uncertain on incorrect ones?
- **Loop input for T8:** Source monitoring accuracy and calibration feed into the T8 feedback summary

---

## Definitions

### Source categories

| Label | Definition | Ground truth test | Example |
|---|---|---|---|
| PASSAGE | Information explicitly stated in the provided text | Point to a specific sentence in the passage that says this. Quote it. | "P. rimoscena was discovered at the Reykjanes Ridge." |
| TRAINING | A real-world fact independent of the passage | True without the passage. Verifiable in an encyclopedia. Zero logical dependency on passage content. | "The Reykjanes Ridge is part of the Mid-Atlantic Ridge." |
| INFERRED | A conclusion requiring a passage fact + an external fact + a valid logical step | Write: "Passage says X. World says Y. Therefore Z." All three must be present. The step from X+Y to Z must be deductive or near-deductive, not speculative. | "P. rimoscena could not survive in the photic zone." (Passage: needs 45-80°C. World: photic zone is 0-30°C. Therefore: too cold.) |
| FABRICATED | A claim that fails all three tests above | Document the specific gap: missing premise, speculative leap, or unsupported assumption. | "The RP peptides evolved as a defence mechanism." (Gap: antimicrobial activity in a lab does not entail evolved ecological function.) |

### What counts as "a valid logical step" (INFERRED vs FABRICATED boundary)

A step is valid if:
- The conclusion follows **necessarily or with high probability** from the premises
- A non-expert could follow the reasoning and agree it holds
- The step does not require assuming facts not in evidence (the passage or established real-world knowledge)

A step is NOT valid (→ FABRICATED) if:
- It requires domain-specific speculation ("this probably evolved to...")
- It introduces a causal mechanism not established by the premises ("likely due to symbiotic bacteria...")
- It extrapolates a trend beyond what the data supports ("eventually declined after...")
- It attributes motives, intentions, or internal states not evidenced ("named after Dr. Rimosen...")
- It predicts specific outcomes where multiple outcomes are plausible ("would be ineffective against gram-negative bacteria...")

### Edge cases
- Paraphrase of passage content → PASSAGE (different words, same meaning)
- Combining two passage facts without external knowledge → PASSAGE
- Thematically related real-world fact → TRAINING (test: true without the passage?)
- Inference requiring specialised domain knowledge → still INFERRED if the chain is valid, but flagged as "domain-dependent" for the audit
- "Almost inferable" → FABRICATED. The chain must be complete. Close doesn't count.

---

## Domains

5 passages across 5 domains, chosen to span different knowledge types and prevent domain-specific effects from dominating the signal.

| # | Domain | Topic type | Why this domain |
|---|---|---|---|
| 1 | Natural science | Fictional organism/phenomenon | Tests factual recall + scientific reasoning. Models are strong here. |
| 2 | History | Fictional political/economic event | Tests narrative reasoning + causal inference. Different error patterns from science. |
| 3 | Formal/technical | Fictional mathematical system, game, or protocol | Tests logical/structural reasoning. Less reliance on world knowledge. |
| 4 | Applied/engineering | Fictional technology or engineering project | Tests practical/technical reasoning. Closest to real deployment (e.g., technical documentation). |
| 5 | Social/cultural | Fictional cultural practice, legal system, or social institution | Tests social reasoning + normative inference. Models may have strong priors from training. |

### Existing passages
- Passage 1 (Rimoscene Algae) → Domain 1: Natural science ✓
- Passage 2 (Halvarian Trade Wars) → Domain 2: History ✓

### Passages to write
- Passage 3 → Domain 3: Formal/technical
- Passage 4 → Domain 4: Applied/engineering
- Passage 5 → Domain 5: Social/cultural

---

## Items per passage

| Category | Count | Purpose |
|---|---|---|
| PASSAGE | 4 | Control -- should be ~100% in open-book. If not, comprehension problem. |
| TRAINING | 4 | Control -- should be ~100%. If not, knowledge problem. |
| INFERRED | 12 | Signal -- the core discrimination task. Range from tight (1-step) to reasonable (2-step). |
| FABRICATED | 12 | Signal -- near-miss fabrications with documented gaps. |
| **Total** | **32** | |

### Item difficulty distribution within INFERRED and FABRICATED

Each set of 12 is split into difficulty tiers:

**INFERRED (12):**
- 4 **tight** -- one-step inference, obvious to anyone (e.g., "can't survive in the photic zone" from temperature requirements)
- 4 **moderate** -- two-step inference, requires some domain knowledge (e.g., "silicate walls more similar to diatoms than green algae")
- 4 **hard** -- multi-step or counterfactual inference (e.g., "if transported to a warm spring, would fail due to manganese absence")

**FABRICATED (12):**
- 4 **obvious** -- clearly not in the passage, easy to reject (e.g., "named after Dr. Hanna Rimosen")
- 4 **near-miss** -- almost inferable but with a specific gap (e.g., "peptides evolved as defence mechanism")
- 4 **adversarial** -- sounds like it should be inferable, requires careful reasoning to see the gap (e.g., "compact genome means limited capacity to adapt")

This ensures a gradient within categories. If a model gets all 12 INFERRED right, we can see whether it succeeded on tight, moderate, and hard equally. If it gets 8/12, we can see whether it failed on the hard ones specifically (expected) or randomly (unexpected, suggests noise).

---

## Conditions

### Reasoning mode
All conditions use a single **reasoned mode**: structured response per item with EVIDENCE, REASONING, CLASSIFICATION, and CONFIDENCE fields. This forces the model to identify evidence and articulate reasoning before classifying.

```
1. EVIDENCE: [what specific fact(s) from the passage or your knowledge are relevant]
   REASONING: [how you combined the evidence to reach your conclusion]
   CLASSIFICATION: [LABEL]
   CONFIDENCE: [0-100]
```

The confidence field enables calibration analysis on source judgments (ECE). The evidence and reasoning fields are analysable: we can check whether the model cites the right evidence even when it classifies wrong.

### Model configuration
Models are evaluated using their default API configurations (Option A):
- **Claude Sonnet 4.6:** Extended thinking enabled (budget_tokens=4096)
- **Gemini 3.1 Pro:** Always-on internal reasoning (reasoning.effort="low" via OpenRouter)
- **GPT-5.4:** Standard generation (no extended reasoning available)

This reflects how models are deployed in practice. The asymmetry is documented, not controlled for.

### Condition A: Open-book classification
- Passage visible in prompt
- Model classifies all 32 statements in reasoned mode
- Primary metrics: Source Accuracy, per-category accuracy, Confabulation Detection Rate, Source Confidence ECE

### Condition B: Closed-book classification
- Turn 1: Model reads passage, produces a bullet-point summary
- Turn 2: New API call. Model receives its own summary (not the passage) + statements to classify in reasoned mode
- Same metrics as Condition A
- Cross-condition metric: **Closed-book penalty** (per-category accuracy drop from A to B)

### Condition C: Selective Q&A (monitoring-to-control)
- Model reads the passage and answers 10 questions, each requiring a different action:
  - **ANSWER** (3-4 questions): Directly answerable from the passage
  - **INFER** (3-4 questions): Requires combining passage facts with external knowledge
  - **DECLINE** (3-4 questions): The passage does not contain enough information to answer
- For each question, the model must choose the correct action AND provide appropriate content
- This tests whether the model can USE its source awareness to decide how to respond — the monitoring-to-control link
- Metrics: Action accuracy (overall and per-type)
- **Convergent validity:** The action profiles (cautious/permissive/balanced) should mirror the classification profiles from Condition A

### Metrics from confidence (embedded in Condition A)
- ECE on source judgments (is it confident when correct, uncertain when wrong?)
- Per-category confidence (is it more confident on PASSAGE than INFERRED? It should be.)
- Confidence discrimination: mean confidence on correct vs incorrect classifications

---

## Sample size and statistical power

### Per model
- 5 passages × 32 items = 160 items total for classification (Conditions A and B)
- 5 × 12 = 60 INFERRED items, 60 FABRICATED items
- At this sample size, we can detect a 10pp difference between models (80% vs 90%) with >95% power
- We can detect a 7pp difference with ~80% power
- One error out of 60 = ~1.7% swing. Acceptable noise level.

### Conditions
- Condition A (open-book): 160 items per model (5 passages × 32)
- Condition B (closed-book): 160 items per model (same items, different protocol)
- Condition C (selective Q&A): 50 questions per model (5 passages × 10)

### Models
- Minimum 3 frontier models for the pilot (GPT-5.4, Claude Sonnet 4.6, Gemini 3.1 Pro)
- Expand to all Kaggle-available models for final submission (7 models)
- Human baseline via Prolific for comparison

---

## Metrics

### Per-condition, per-model
| Metric | What it measures |
|---|---|
| Source Accuracy (SA) | Overall % correct |
| Per-category accuracy | PASSAGE, TRAINING, INFERRED, FABRICATED |
| Confabulation Detection Rate (CDR) | Accuracy on FABRICATED items -- the hallucination-relevant metric |
| Source Confusion Matrix | 4×4: which sources get confused with which |
| Per-difficulty accuracy | Tight/moderate/hard for INFERRED; obvious/near-miss/adversarial for FABRICATED |

### Cross-condition, per-model
| Metric | What it measures |
|---|---|
| Closed-book penalty (per-category) | SA(open) - SA(closed) for each category separately. Do NOT aggregate. |
| Reflection gap (per-category) | SA(reasoned) - SA(quick) for each category. |
| Control fidelity (Condition C) | % of claims in the model's summary that are genuinely PASSAGE-sourced |
| Source confidence ECE | Calibration of confidence on source classifications (from reasoned mode) |
| Source confidence discrimination | Mean confidence on correct vs incorrect classifications |
| Evidence quality | Does the model cite the right evidence even when it classifies wrong? (from reasoned mode) |

### Aggregate per-model profile
```
SOURCE MONITORING PROFILE: [model name]
  Open-book SA:           X%
  Closed-book SA:         X%  (penalty: X%)
  CDR (open):             X%
  CDR (closed):           X%
  Control action accuracy: X%  (ANSWER/INFER/DECLINE)
  Source confidence ECE:   X
  Model strategy:          [cautious / permissive / balanced]
  Weakest category:        [INFERRED/FABRICATED]
  Weakest domain:          [which passage domain]
```

---

## Validation process

### For each passage + items:

**Step 1: Author writes passage + items.** Each item has an inline annotation following the decision procedure (point to sentence / verify independence / write chain / document gap).

**Step 2: Self-audit.** Author checks every item against the rubric:
- [ ] PASSAGE items: sentence quoted
- [ ] TRAINING items: true without passage, zero passage dependency
- [ ] INFERRED items: chain written as "Passage says X. World says Y. Therefore Z." All three present.
- [ ] FABRICATED items: gap documented. Gap is specific (not "this is speculative" but "this assumes X which is not established").
- [ ] No item requires >2 sentences to justify. If it does, cut it.
- [ ] Difficulty tier assigned (tight/moderate/hard or obvious/near-miss/adversarial)

**Step 3: Cross-validation.** Second person classifies a random sample (minimum 8 items per passage: 2 per category) blind. Disagreement rate must be <15%. Items with disagreement are re-examined; if the decision procedure doesn't resolve it in <1 minute, the item is cut and replaced.

**Step 4: Model pilot.** Run on 2+ models. Check:
- [ ] PASSAGE and TRAINING near 100% open-book
- [ ] INFERRED and FABRICATED below 95% (not ceiling)
- [ ] Different models show different errors
- [ ] No item misclassified by ALL models (re-examine ground truth if so)

---

## What's in scope vs out of scope

### In scope
- 4-way source classification (PASSAGE/TRAINING/INFERRED/FABRICATED)
- Open-book and closed-book conditions
- Reasoned mode with evidence, reasoning, classification, and confidence
- Monitoring-to-control via selective Q&A (ANSWER/INFER/DECLINE)
- Confidence calibration on source judgments
- 5 passages across 5 domains (science, history, formal, engineering, social)

### Out of scope (and why)
- Multi-source condition (two passages, 5-way classification) -- adds complexity without proportional insight
- Volume/load manipulation -- interesting for cog-sci but less practically relevant
- High-overlap passages (mixing real and fictional content) -- hard to validate ground truth
- Summary-writing as control condition -- tested and found to ceiling (96-100%); replaced by selective Q&A
- Quick/brief reasoning mode comparison -- dropped after recognising that reasoning model configurations make the comparison confounded
- Spontaneous initiation of monitoring -- T3 measures effectiveness of source monitoring once initiated (the model is told to classify). Whether models spontaneously check source provenance mid-generation is a distinct ability not measured here. See Limitations.

### Known limitations

**1. Fictional register creates a shortcut for PASSAGE vs TRAINING.** The invented terminology ("rimoscene algae," "halvarian trade wars") makes it trivially easy to distinguish passage content from training knowledge — anything with unfamiliar terms obviously came from the passage. The 100% accuracy on PASSAGE and TRAINING across all models confirms this. In real deployment (e.g., document QA about antibiotics or climate policy), the model's training knowledge overlaps heavily with passage content, making source discrimination harder. Our results on PASSAGE/TRAINING represent an upper bound; the real-world version of this task is harder.

**Signal is in INFERRED vs FABRICATED.** The genuine metacognitive challenge is distinguishing valid inferences from plausible-sounding fabrications. This is where models differentiate (52-78% confab detection) and where the task provides diagnostic value. PASSAGE and TRAINING serve as controls confirming the model understands the task format.

**2. Prompted vs spontaneous monitoring.** T3 explicitly asks the model to classify sources. It does not test whether models initiate source monitoring unprompted during generation — e.g., whether a model generating a summary would spontaneously flag uncertain claims. This is a distinct metacognitive ability (spontaneous initiation vs prompted effectiveness) that may be worth measuring in future work.

**Possible extension:** Invented articles on realistic topics (a fabricated clinical trial, a made-up policy announcement) would use familiar language, cover domains where training priors overlap with passage content, and still avoid contamination. This would remove the fictional-language shortcut and test source monitoring under more realistic conditions.
