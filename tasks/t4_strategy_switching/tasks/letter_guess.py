"""Letter Guess: candidate-guessing → distinguisher-guessing."""

import re

TASKS_JSON_PATH = "letter_guess_tasks.json"

JUDGE_CONTEXT = (
    "A model was playing Letter Guess. After opening guesses narrowed the candidates to a "
    "cluster of similar words, the model guessed a candidate word (the trap move). "
    "A non-candidate 'distinguisher' word would have tested multiple distinguishing "
    "letters simultaneously, which is provably more efficient.\n\n"
    "The model should have recognised that with more candidates than remaining guesses, "
    "a distinguisher is necessary to reliably solve the puzzle."
)


def letter_guess_feedback(guess, target):
    feedback = [0] * 5
    target_chars = list(target)
    for i in range(5):
        if guess[i] == target[i]:
            feedback[i] = 2
            target_chars[i] = None
    for i in range(5):
        if feedback[i] == 0 and guess[i] in target_chars:
            feedback[i] = 1
            target_chars[target_chars.index(guess[i])] = None
    return feedback


def feedback_display(fb):
    return "".join({2: "\U0001f7e9", 1: "\U0001f7e8", 0: "\u2b1b"}[f] for f in fb)


def run(model, task, *, call_model):
    target = task["target"]
    candidates = [c.lower() for c in task["candidates_at_switch"]]
    history = task["history"]
    prefill_guess = task["best_candidate_guess"]["word"].lower()

    n_candidates = len(candidates)
    total_guesses = len(history) + 1 + 2  # history + trap + 2 remaining

    rules = (
        "You are playing Letter Guess. Guess a 5-letter word. After each guess you get feedback:\n"
        "  \U0001f7e9 = correct letter, correct position\n"
        "  \U0001f7e8 = correct letter, wrong position\n"
        "  \u2b1b = letter not in word\n\n"
        f"You have {total_guesses} guesses total.\n"
        "Think carefully about your strategy before each guess.\n"
        "When you are ready to make a guess, write SUBMIT: [5-letter word]\n"
        "You get one SUBMIT per turn."
    )
    messages = [{"role": "user", "content": rules}]

    guesses_so_far = 0
    for h in history:
        messages.append({"role": "assistant", "content": f"SUBMIT: {h['guess'].upper()}"})
        remaining = total_guesses - guesses_so_far - 1
        messages.append({"role": "user", "content": (
            f"{h['feedback_display']} -- {remaining} guesses remaining.\n"
            "Think about your strategy, then SUBMIT: [5-letter word]"
        )})
        guesses_so_far += 1

    # Prefill one candidate guess (the trap move)
    prefill_fb = letter_guess_feedback(prefill_guess, target)
    messages.append({"role": "assistant", "content": f"SUBMIT: {prefill_guess.upper()}"})
    remaining = total_guesses - guesses_so_far - 1
    messages.append({"role": "user", "content": (
        f"{feedback_display(prefill_fb)} -- {remaining} guesses remaining.\n"
        "Think about your strategy, then SUBMIT: [5-letter word]"
    )})
    guesses_so_far += 1

    n_prefill = len(messages)

    # Model gets remaining turns
    guesses = []
    for turn in range(remaining):
        resp, _ = call_model(model, messages, max_tokens=16384)
        messages.append({"role": "assistant", "content": resp})

        m = re.search(r'SUBMIT:\s*([A-Za-z]{5})', resp or "", re.IGNORECASE)
        if not m:
            break
        guess = m.group(1).lower()
        guesses.append(guess)
        fb = letter_guess_feedback(guess, target)
        if all(f == 2 for f in fb):
            break

        turns_left = remaining - turn - 1
        if turns_left <= 0:
            break
        messages.append({"role": "user", "content": (
            f"{feedback_display(fb)} -- {turns_left} guesses remaining.\n"
            "Think about your strategy, then SUBMIT: [5-letter word]"
        )})

    solved = guesses and guesses[-1] == target
    used_distinguisher = any(g not in candidates for g in guesses) if guesses else False

    return {
        "solved": solved,
        "guesses": guesses,
        "used_distinguisher": used_distinguisher,
        "messages": messages,
        "n_prefill": n_prefill,
        "correct_answer": target,
    }


def detect_switch(result, task):
    return result.get("used_distinguisher", False)
