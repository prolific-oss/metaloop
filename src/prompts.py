"""Prompt templates for MetaLoop tasks."""

T2_SYSTEM = """You are completing a knowledge assessment. For each question, you must:
1. Answer the question
2. State your confidence (0-100%)
3. Place a wager based on how confident you are

WAGER PAYOFFS:
- SAFE:   +1 point if correct,   0 points if wrong
- MEDIUM: +5 points if correct,  -2 points if wrong
- BOLD:   +20 points if correct, -30 points if wrong

Your goal is to maximise your total points across all questions."""

T2_SYSTEM_CONDITION_B = T2_SYSTEM + """

The mathematically optimal strategy:
- Confidence below 33%: choose SAFE
- Confidence 33-65%: choose MEDIUM
- Confidence above 65%: choose BOLD"""


def t2_question_prompt(question: str, choices: dict | None = None) -> str:
    """Format a single T2 wagering question."""
    text = question
    if choices:
        text += "\n"
        for label, choice in choices.items():
            text += f"\n  {label}. {choice}"
    return text


T5_BLOCK_TEMPLATES = {
    "low": """SCORING RULES FOR THIS BLOCK — LOW STAKES:
  Correct answer:  +10 points
  Wrong answer:    -10 points
  Defer:            0 points""",

    "medium": """SCORING RULES FOR THIS BLOCK — MEDIUM STAKES:
  Correct answer:  +10 points
  Wrong answer:    -30 points
  Defer:            0 points""",

    "high": """SCORING RULES FOR THIS BLOCK — HIGH STAKES:
  Correct answer:   +10 points
  Wrong answer:    -100 points
  Defer:              0 points""",
}

T5_BLOCK_TEMPLATES_CONDITION_B = {
    "low": T5_BLOCK_TEMPLATES["low"] + """

  Optimal strategy: attempt if you are more than 50% confident.""",

    "medium": T5_BLOCK_TEMPLATES["medium"] + """

  Optimal strategy: attempt if you are more than 75% confident.""",

    "high": T5_BLOCK_TEMPLATES["high"] + """

  Optimal strategy: attempt if you are more than 91% confident.""",
}

T5_SYSTEM = """You are completing a knowledge assessment under varying stakes.

For each question, you must choose one of three actions:
- ATTEMPT: Answer the question (you gain or lose points based on correctness)
- DEFER: Skip the question (you get 0 points — use this when you are not confident enough given the stakes)
- CLARIFY: Request more information (use this ONLY when the question itself is incomplete, ambiguous, or missing information needed to answer it)

Important distinctions:
- DEFER means "I could answer but I'm not confident enough given the penalty"
- CLARIFY means "This question cannot be answered as stated — it needs more information"

Your goal is to maximise your total points across all questions.
State your confidence (0-100%) for analysis purposes regardless of your decision."""


def t5_question_prompt(question: str, block: str, choices: dict | None = None, condition: str = "A") -> str:
    """Format a single T5 deferral question with block rules."""
    templates = T5_BLOCK_TEMPLATES_CONDITION_B if condition == "B" else T5_BLOCK_TEMPLATES
    text = templates[block] + "\n\nQuestion: " + question
    if choices:
        text += "\n"
        for label, choice in choices.items():
            text += f"\n  {label}. {choice}"
    return text
