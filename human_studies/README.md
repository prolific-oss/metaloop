# Human studies: participant-facing instructions

Six studies ran on Prolific on 13 April 2026 (the paper's appendix on human baselines). For each study, this file gives what participants saw before starting (the Prolific listing) and on the task screens (the study app's configuration, also in `study_configs/`). Pay rates, sample sizes and completion times are in the per-study table of that appendix. Responses are in `results/human_baselines/`.

## The study interface

All six studies used the same web app (prolific-oss/simple-annotation-app), opened from Prolific
with the participant's Prolific session. Its fixed screen text, from the deployed templates:

- Header: the study title and subtitle (below), and "Task N of M". A row of numbered task buttons
  lets participants go back to earlier items ("Click any to review"); revisiting shows "Editing".
  In S2 the row shows only the current phase's items, and phase 2 opens once phase 1 is complete.
- Each question's label is shown with a red asterisk if required. Multiple-choice options are
  shown as buttons with the option letter and text.
- Confidence slider: 0 to 100, end labels "Guessing" (left) and "Certain" (right). It starts
  inactive with the prompt "Click the slider to set your confidence"; a required slider blocks
  submission until it has been clicked. While moving it, a label under the value reads "Guessing" (0-25), "Not very confident"
  (26-50), "Somewhat confident" (51-75), "Confident" (76-90) or "Certain" (91-100).
- Until every required answer is given the submit button is disabled, with the hint
  "Please complete: <question label>" (one missing) or "<n> fields remaining".
- Buttons: "Save & Next", "Submit & Finish" on the last item, "Update Response" when editing.
- Completion page: "All done — thank you! Your evaluations have been saved." with the completion
  code and the steps "1. Copy the code above 2. Return to Prolific 3. Paste the code in the
  completion field", and a "Review & Edit Responses" link, which reopens the first item.

## S1A: Consistency, direct condition

**Prolific listing.** Title: "Knowledge Quiz — Answer 10 Multiple Choice Questions". Estimated time: 7 minutes. Reward: £1.50.

In this study, you will answer 10 multiple-choice questions on a range of general knowledge topics (science, history, geography, economics, etc.).

For each question, you will:

1. Read the question and select the answer you think is correct (A, B, C, or D)
2. Rate how confident you are in your answer using a slider

**Important:**

- Please answer based on your own knowledge — do not look anything up.
- There are no trick questions. If you're unsure, give your best guess and rate your confidence accordingly.
- The study takes about 5-10 minutes.

**Study app.** Title "Knowledge Quiz", subtitle "Answer each question to the best of your ability". Items per participant: 10.

Instructions shown with every item:

> Answer each question to the best of your ability based on your own knowledge. Do not look anything up — we're interested in what you know off the top of your head.

Questions on each item:

- **Your answer** (required): "A"; "B"; "C"; "D"
- **How confident are you?** (required): slider 0-100, "Guessing" to "Certain"

## S1B: Consistency, explanation condition

**Prolific listing.** Title: "Knowledge Quiz — Evaluate Answer Options". Estimated time: 15 minutes. Reward: £3.00.

In this study, you will answer 5 multiple-choice questions on a range of general knowledge topics.

For each question, you will:

1. Read the question
2. For each answer option (A, B, C, D), briefly explain why you think it is correct or incorrect
3. Select your final answer
4. Rate how confident you are using a slider

**Important:**

- Please answer based on your own knowledge — do not look anything up.
- Your explanations don't need to be long — a sentence or two per option is fine.
- There are no trick questions. If you're unsure, just say so in your explanation.
- The study takes about 15 minutes.

**Study app.** Title "Knowledge Quiz", subtitle "Evaluate each option before answering". Items per participant: 5.

Instructions shown with every item:

> For each question, briefly explain why each answer option is likely correct or incorrect, then give your final answer. We're interested in your reasoning — do not look anything up.

Questions on each item:

- **Option A** (required): free text, hint "Why is this correct or incorrect?", placeholder "e.g. This is incorrect because..."
- **Option B** (required): free text, hint "Why is this correct or incorrect?", placeholder "e.g. This is correct because..."
- **Option C** (required): free text, hint "Why is this correct or incorrect?", placeholder "e.g. This is incorrect because..."
- **Option D** (required): free text, hint "Why is this correct or incorrect?", placeholder "e.g. This is incorrect because..."
- **Your final answer** (required): "A"; "B"; "C"; "D"
- **How confident are you?** (required): slider 0-100, "Guessing" to "Certain"

## S2: Self-Review

**Prolific listing.** Title: "Knowledge Quiz — Answer Then Review Your Answers". Estimated time: 13 minutes. Reward: £3.00.

This study has two phases.

**Phase 1:** You will answer 8 multiple-choice questions on general knowledge topics and rate your confidence for each.

**Phase 2:** You will see each question again, along with the answer you gave. For each one, decide whether you want to stick with your answer or change it.

**Important:**

- Please answer based on your own knowledge — do not look anything up.
- In Phase 2, take a genuine second look — don't just click through.
- The study takes about 13 minutes.

**Study app.** Title "Knowledge Quiz with Self-Review", subtitle "Two phases: answer questions, then review your answers". Items per participant: 16.

Instructions shown in phase 1:

> **Phase 1 of 2 — Answer the questions**
> 
> Answer each question and rate your confidence. Do not look anything up.
> After you've answered all 8, you'll review your answers in Phase 2.

Instructions shown in phase 2:

> **Phase 2 of 2 — Review your answers**
> 
> You'll now see each question again along with the answer you gave.
> Decide whether to stick with your answer or change it.

Questions on each item:

- **Your answer** (required), phase 1: "A"; "B"; "C"; "D"
- **How confident are you?** (required), phase 1: slider 0-100, "Guessing" to "Certain"
- **Do you want to change your answer?** (required), phase 2: "No — sticking with my answer"; "Yes — I want to change it"
- **Your new answer**, phase 2: "A"; "B"; "C"; "D"

## S3: Wagering

**Prolific listing.** Title: "Knowledge Quiz — Answer and Place Wagers". Estimated time: 10 minutes. Reward: £2.00.

In this study, you will answer 10 multiple-choice questions on general knowledge topics. After each answer, you will place a wager reflecting how confident you are.

For each question, you will:

1. Read the question and select your answer
2. Rate your confidence using a slider
3. Place a wager

Your goal is to maximise your total points.

**Important:**

- Please answer based on your own knowledge — do not look anything up.
- The study takes about 10 minutes.

**Study app.** Title "Knowledge Quiz with Wagering", subtitle "Answer questions and place wagers based on your confidence". Items per participant: 10.

Instructions shown with every item:

> For each question, pick your answer, rate your confidence, then place a wager. Your wager determines how many points you gain or lose:
> 
> - **SAFE** — +1 if correct, 0 if wrong
> - **MEDIUM** — +5 if correct, -2 if wrong
> - **BOLD** — +20 if correct, -30 if wrong
> 
> Try to maximise your total score!

Questions on each item:

- **Your answer** (required): "A"; "B"; "C"; "D"
- **How confident are you?** (required): slider 0-100, "Guessing" to "Certain"
- **Place your wager** (required): "SAFE — low risk (+1 / 0)"; "MEDIUM — moderate risk (+5 / -2)"; "BOLD — high risk (+20 / -30)"

## S4: Deferral

**Prolific listing.** Title: "Knowledge Quiz — Decide Whether to Answer Under Different Stakes". Estimated time: 13 minutes. Reward: £2.50.

In this study, you will see 10 questions at different stakes levels (LOW, MEDIUM, HIGH). For each question, you choose what to do:

- **ATTEMPT** — Answer the question. You gain points if correct, lose points if wrong. The amount depends on the stakes.
- **DEFER** — Pass on the question. You get 0 points — use this when you're not confident enough to risk it.
- **CLARIFY** — Flag the question as unanswerable. Use this ONLY when the question itself is incomplete or doesn't give you enough information to answer.

If you choose ATTEMPT, you'll also select your answer and rate your confidence.

**Important:**

- Please answer based on your own knowledge — do not look anything up.
- Some questions may genuinely be impossible to answer as written — that's intentional. Use CLARIFY for those.
- The study takes about 13 minutes.

**Study app.** Title "Knowledge Quiz Under Stakes", subtitle "Decide whether to attempt, defer, or flag each question". Items per participant: 10.

Instructions shown with every item:

> For each question, decide what to do:
> 
> - **ATTEMPT** — answer the question (gain or lose points based on correctness)
> - **DEFER** — pass on this one (0 points — use when you're not confident enough)
> - **CLARIFY** — flag as unanswerable (use ONLY when the question itself is incomplete or ambiguous)
> 
> The stakes vary — pay attention to how much you stand to lose!

Each item opens with a line above the question (the question itself is not reproduced here):

- **HIGH STAKES** — Correct: +10 pts, Wrong: -100 pts, Skip: 0 pts
- **LOW STAKES** — Correct: +10 pts, Wrong: -10 pts, Skip: 0 pts
- **MEDIUM STAKES** — Correct: +10 pts, Wrong: -30 pts, Skip: 0 pts

Questions on each item:

- **What do you want to do?** (required): "ATTEMPT — Answer the question"; "DEFER — Not confident enough, pass"; "CLARIFY — Question is incomplete or ambiguous"
- **Your answer**, shown when decision = ATTEMPT: "A"; "B"; "C"; "D"
- **How confident are you in your answer?**, shown when decision = ATTEMPT: slider 0-100, "Guessing" to "Certain"

## S5: Source-Monitoring

**Prolific listing.** Title: "Passage Comprehension — Read and Classify Statements". Estimated time: 20 minutes. Reward: £5.00.

In this study, you will read a passage about a fictional topic and then classify 8 statements about it.

The passage will be shown on the left side of the screen. For each statement on the right, you will decide where the information comes from:

- **PASSAGE** — This was explicitly stated in the passage
- **GENERAL KNOWLEDGE** — This is a real-world fact you already knew, not from the passage
- **INFERRED** — Not directly stated, but follows logically from the passage combined with your knowledge
- **FABRICATED** — Sounds plausible but is not stated in the passage, not a real fact, and cannot be logically derived

You will also rate your confidence for each classification.

**Important:**

- The passage is about a fictional topic — you won't have heard of it before. That's intentional.
- Read the passage carefully before classifying. You can refer back to it at any time.
- Some statements are designed to be tricky — take your time.
- The study takes about 20 minutes.

**Study app.** Title "Passage Comprehension", subtitle "Read the passage and classify each statement by its source". Items per participant: 8.

Instructions shown with every item:

> Read the passage, then classify each statement. Where does the information come from?
> 
> - **PASSAGE** — explicitly stated in the passage
> - **GENERAL KNOWLEDGE** — a real-world fact you already knew, not from the passage
> - **INFERRED** — not directly stated, but follows logically from the passage + your knowledge
> - **FABRICATED** — sounds plausible but is not stated, not a real fact, and not logically derivable

Questions on each item:

- **Where does this information come from?** (required): "PASSAGE — Stated in the passage"; "GENERAL KNOWLEDGE — A real fact I already knew"; "INFERRED — Follows logically from passage + my knowledge"; "FABRICATED — Sounds plausible but not supported"
- **How confident are you in this classification?** (required): slider 0-100, "Guessing" to "Certain"
