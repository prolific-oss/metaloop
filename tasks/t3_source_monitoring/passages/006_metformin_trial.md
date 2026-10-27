# Passage 006: Metformin and Cognitive Decline

## Design Notes
- **Domain:** Medical science (real drug, invented trial)
- **Contamination risk:** Moderate. Metformin is real, its mechanisms are well-known. The fictional trial (MEMOIR) and its specific results are invented. The challenge: models know real facts about metformin and must distinguish real knowledge from passage-specific claims.
- **Why this is harder than fictional passages:** The language is familiar medical terminology. The model's training knowledge overlaps heavily with the passage content. PASSAGE vs TRAINING is no longer trivial.

## Passage

Metformin, a first-line treatment for type 2 diabetes, has attracted growing interest for its potential neuroprotective effects. The MEMOIR trial (Metformin for Memory Improvement in Older Adults at Risk), conducted across 12 centres in the UK between 2022 and 2025, enrolled 840 adults aged 60-75 with prediabetes and mild cognitive impairment. Participants were randomised to receive either 1500mg metformin daily or placebo for 24 months, with cognitive assessments at baseline, 12 months, and 24 months.

The primary endpoint was change in composite cognitive score, measured using the CANTAB battery. At 24 months, the metformin group showed a 0.18 standard deviation advantage over placebo (95% CI: 0.04 to 0.32, p=0.011). The effect was most pronounced in executive function (0.24 SD, p=0.003) and least apparent in episodic memory (0.08 SD, p=0.29). Subgroup analysis revealed that participants with higher baseline HbA1c (>42 mmol/mol) showed larger cognitive benefits, suggesting the neuroprotective effect may be partly mediated through improved glycaemic control.

Neuroimaging in a subsample of 180 participants showed that metformin was associated with reduced hippocampal volume loss compared to placebo (-1.2% vs -2.8% annual atrophy, p=0.006). However, white matter hyperintensity volume did not differ between groups. Plasma biomarkers showed reductions in inflammatory markers (IL-6, TNF-α) and a modest reduction in phosphorylated tau, though amyloid-β levels were unchanged.

The trial reported an adverse event rate of 34% in the metformin group versus 22% in the placebo group, with gastrointestinal symptoms accounting for most of the difference. Three participants in the metformin group developed vitamin B12 deficiency requiring supplementation. No serious adverse events were attributed to the study drug.

---

## Statements

**P1.** The MEMOIR trial enrolled 840 adults aged 60-75 with prediabetes and mild cognitive impairment.

**P2.** The metformin group showed a 0.18 standard deviation cognitive advantage over placebo at 24 months.

**P3.** Hippocampal volume loss was -1.2% annually in the metformin group versus -2.8% in the placebo group.

**P4.** White matter hyperintensity volume did not differ between the treatment groups.

**T1.** Metformin is a biguanide drug that works primarily by reducing hepatic glucose production.

**T2.** The CANTAB battery is a computerised cognitive assessment tool developed at the University of Cambridge.

**T3.** HbA1c is a measure of average blood glucose levels over the preceding 2-3 months.

**T4.** Vitamin B12 deficiency is a known side effect of long-term metformin use.

**I1.** The cognitive benefits of metformin in the MEMOIR trial may be partly explained by its anti-inflammatory effects, given the observed reductions in IL-6 and TNF-α.

**I2.** Participants with well-controlled blood sugar at baseline would be expected to show smaller cognitive benefits from metformin, based on the subgroup analysis finding.

**I3.** The adverse event profile suggests that metformin was generally well-tolerated, with most excess events being non-serious gastrointestinal symptoms.

**I4.** The lack of effect on amyloid-β suggests metformin's neuroprotective mechanism does not operate through the amyloid pathway.

**F1.** The MEMOIR trial found that metformin significantly improved episodic memory scores compared to placebo (0.24 SD, p=0.003).

**F2.** The trial enrolled 1,200 adults across 18 centres in the UK and Germany.

**F3.** Hippocampal volume loss was -1.2% annually in the metformin group versus -3.4% in the placebo group.

**F4.** The adverse event rate was 34% in the metformin group versus 28% in the placebo group.

Note on fabrication design: F1 swaps the effect sizes — the passage says executive function was 0.24 SD and episodic memory was 0.08 SD (non-significant). F2 changes the sample size (840→1200) and scope (12 UK centres → 18 UK+Germany). F3 changes the placebo atrophy rate (2.8%→3.4%). F4 changes the placebo adverse rate (22%→28%). Each is a plausible-sounding detail that's close to but different from what the passage states.
