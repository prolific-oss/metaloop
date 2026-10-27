"""LLM judge for strategy-switch detection and correctness.

The judge receives:
- The task context (what the trap was, what the switch should be)
- The correct answer
- The model's response transcript

It determines:
1. Did the model submit an answer?
2. Did the model switch strategy?
3. Is the submitted answer correct?
"""

import re


def format_transcript(messages):
    parts = []
    for m in messages:
        role = "USER" if m["role"] == "user" else "ASSISTANT"
        parts.append(f"[{role}]\n{m['content']}")
    return "\n\n".join(parts)


JUDGE_TEMPLATE = """You are grading a model's response on a strategy-switching task.

## Task context
{judge_context}

## Correct answer
{correct_answer}

## Model's response (after the prefill)
{transcript}

## Grading

Answer these three questions. For each, respond with exactly YES or NO, then a one-sentence explanation.

1. SUBMITTED: Did the model output a clear answer (a SUBMIT line, or a clearly stated final answer)?
2. SWITCHED: Did the model recognise the error in the prefilled approach and switch to a different strategy?
3. CORRECT: Is the model's submitted answer correct (matches or is equivalent to the correct answer above)?

Format:
SUBMITTED: [YES/NO] — [explanation]
SWITCHED: [YES/NO] — [explanation]
CORRECT: [YES/NO] — [explanation]"""


def judge(judge_context, correct_answer, messages, n_prefill=0, *, call_model, judge_model):
    """Run LLM judge. Returns dict with submitted, switched, correct, and raw response."""
    post_prefill = messages[n_prefill:]
    transcript = format_transcript(post_prefill) if post_prefill else "(no response)"

    prompt = JUDGE_TEMPLATE.format(
        judge_context=judge_context,
        correct_answer=correct_answer,
        transcript=transcript,
    )

    resp, _ = call_model(judge_model, [{"role": "user", "content": prompt}], max_tokens=1024)

    def parse_yn(key):
        m = re.search(rf'{key}:\s*(YES|NO)', resp or "", re.IGNORECASE)
        return m.group(1).upper() == "YES" if m else None

    return {
        "submitted": parse_yn("SUBMITTED"),
        "switched": parse_yn("SWITCHED"),
        "correct": parse_yn("CORRECT"),
        "raw": resp,
    }
