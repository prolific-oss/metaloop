"""Code Breaker: candidate-guessing → distinguisher-guessing."""

import re
from collections import Counter

TASKS_JSON_PATH = "code_breaker_tasks.json"

JUDGE_CONTEXT = (
    "A model was playing Code Breaker (4 positions, 6 colours). After opening guesses, "
    "the remaining candidates clustered around a shared pattern. The model guessed a "
    "candidate code (the trap move). A non-candidate distinguisher code would have been "
    "more information-efficient.\n\n"
    "The model should have recognised that with more candidates than remaining guesses, "
    "a distinguisher is necessary."
)


def code_breaker_feedback(guess, secret):
    exact = sum(g == s for g, s in zip(guess, secret))
    gc, sc = Counter(guess), Counter(secret)
    colour = sum((gc & sc).values()) - exact
    return exact, colour


def run(model, task, *, call_model):
    secret = task["secret"]
    candidates = [c.upper() for c in task["candidates_at_switch"]]
    history = task["history"]
    prefill_guess = task["best_candidate_guess"]["code"].upper()

    total_guesses = len(history) + 1 + 2

    rules = (
        "You are playing Code Breaker. The secret is a 4-letter code using colours: R G B Y O P.\n"
        "After each guess you get: (exact matches, colour-only matches).\n\n"
        f"You have {total_guesses} guesses total.\n"
        "Think carefully about your strategy before each guess.\n"
        "When you are ready to make a guess, write SUBMIT: [4-letter code]\n"
        "You get one SUBMIT per turn."
    )
    messages = [{"role": "user", "content": rules}]

    guesses_so_far = 0
    for h in history:
        messages.append({"role": "assistant", "content": f"SUBMIT: {h['guess']}"})
        remaining = total_guesses - guesses_so_far - 1
        messages.append({"role": "user", "content": (
            f"{h['feedback_display']} -- {remaining} guesses remaining.\n"
            "Think about your strategy, then SUBMIT: [4-letter code]"
        )})
        guesses_so_far += 1

    prefill_exact, prefill_colour = code_breaker_feedback(prefill_guess, secret)
    messages.append({"role": "assistant", "content": f"SUBMIT: {prefill_guess}"})
    remaining = total_guesses - guesses_so_far - 1
    messages.append({"role": "user", "content": (
        f"{prefill_exact} exact, {prefill_colour} colour. -- {remaining} guesses remaining.\n"
        "Think about your strategy, then SUBMIT: [4-letter code]"
    )})
    guesses_so_far += 1

    n_prefill = len(messages)

    guesses = []
    for turn in range(remaining):
        resp, _ = call_model(model, messages, max_tokens=16384)
        messages.append({"role": "assistant", "content": resp})

        m = re.search(r'SUBMIT:\s*([RGBYOP]{4})', resp or "", re.IGNORECASE)
        if not m:
            break
        guess = m.group(1).upper()
        guesses.append(guess)
        exact, colour = code_breaker_feedback(guess, secret)
        if exact == 4:
            break

        turns_left = remaining - turn - 1
        if turns_left <= 0:
            break
        messages.append({"role": "user", "content": (
            f"{exact} exact, {colour} colour. -- {turns_left} guesses remaining.\n"
            "Think about your strategy, then SUBMIT: [4-letter code]"
        )})

    solved = guesses and guesses[-1] == secret
    used_distinguisher = any(g not in candidates for g in guesses) if guesses else False

    return {
        "solved": solved,
        "guesses": guesses,
        "used_distinguisher": used_distinguisher,
        "messages": messages,
        "n_prefill": n_prefill,
        "correct_answer": secret,
    }


def detect_switch(result, task):
    return result.get("used_distinguisher", False)
