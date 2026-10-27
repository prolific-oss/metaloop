"""Cipher breaking: monoalphabetic frequency analysis → Vigenere / polyalphabetic."""

import re

TASKS_JSON_PATH = "cipher_tasks.json"

JUDGE_CONTEXT = (
    "A model was decrypting a ciphertext using monoalphabetic frequency analysis "
    "(simple substitution cipher). The decrypted output was gibberish because the cipher "
    "is actually a Vigenere (polyalphabetic) cipher.\n\n"
    "The model should have recognised that monoalphabetic substitution produces gibberish "
    "and switched to polyalphabetic techniques (Kasiski examination, index of coincidence, "
    "Vigenere analysis)."
)


def run(model, task, *, call_model):
    messages = list(task["prefill_messages"])
    if messages and messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": "Please continue."})
    plaintext = task["plaintext"]

    resp, _ = call_model(model, messages, max_tokens=16384)
    messages.append({"role": "assistant", "content": resp})

    m = re.search(r'SUBMIT:\s*(?:PLAINTEXT:\s*)?([A-Z]+)', resp or "", re.IGNORECASE)
    if not m:
        m = re.search(r'PLAINTEXT:\s*([A-Z]+)', resp or "", re.IGNORECASE)
    declared = m.group(1).upper() if m else None
    solved = declared == plaintext.upper()
    return {
        "solved": solved,
        "declared_plaintext": declared,
        "messages": messages,
        "correct_answer": f"PLAINTEXT: {plaintext}",
    }


def detect_switch(result, task):
    return result.get("solved", False)
