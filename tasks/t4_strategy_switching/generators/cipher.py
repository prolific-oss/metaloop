"""
Generate Vigenere cipher strategy-switching tasks.

Each task presents a Vigenere ciphertext as if it were a simple substitution cipher.
The prefill shows an assistant attempting (and failing) with monoalphabetic frequency
analysis, setting up the model to either persist or switch to polyalphabetic techniques.

No external dependencies beyond the standard library.
"""

import json
import random
import string
from collections import Counter
from pathlib import Path

# ── English letter frequencies (descending) ──────────────────────────────────

ENGLISH_FREQ_ORDER = "ETAOINSHRDLCUMWFGYPBVKJXQZ"

# ── Plaintext sources ────────────────────────────────────────────────────────
# Well-known English passages, stripped to uppercase letters only.

PLAINTEXTS = [
    # 1. Opening of A Tale of Two Cities (Dickens)
    "ITWASTHEBESTOFTIMESITWASTHEWORSTOFTIMESITWASTHEAGEOFWISDOMITWASTHEAGEOFFOOLISHNESSITWASTHEEPOCHOFBELIEFITWASTHEEPOCHOFINCREDULITYITWASTHESEASONOFLIGHTITWASTHESEASONOFDARKNESS",
    # 2. Opening of Pride and Prejudice (Austen)
    "ITISATRUTHUNIVERSALLYACKNOWLEDGEDTHATASINGLEMANINPOSSESSIONOFAGOODFORTUNEMUSTBEINWANTOFAWIFEHOWEVERLITTLEKNOWNTHEFEELINGSORVIEWSOFSUCHAMANMAYBEONHISFIRSTENTERINGANEIGHBOURHOOD",
    # 3. Hamlet soliloquy (Shakespeare)
    "TOBEORNOTTOBETHATISTHEQUESTIONWHETHERITISNOBLERINMINDTOSUFFERTHESLINGSANDARROWSOFOUTRAGEOUSFORTUNEORTOTAKEARMSAGAINSTASEAOFTROUBLESANDBYOPPOSINGENDTHEM",
    # 4. Opening of Moby-Dick (Melville)
    "CALLMEISHMAELSOMEYEARSAGONEVERMINDHOWLONGPRECISELYHAVINGLITTLEORNOMONEYINMYPURSEANDNOTHINGPARTICULARTOINTERESTMEONSHOREITHOUGHTIWOULDSAILABOUTALITTLEANDSEETHEWATERY",
    # 5. Gettysburg Address (Lincoln)
    "FOURSCOREANDSEVENYEARSAGOOURFATHERSBROUGHTFORTHONTHISCONTINENTANEWNATIONCONCEIVEDINLIBERTYANDDEDICATEDTOTHEPROPOSITIONTHATALLMENARECREATEDEQUALNOWWEAREENGAGED",
    # 6. "I Have a Dream" (King)
    "IHAVEADREAMTHATONEDAYTHISNATIONWILLRISEUPANDLIVEOUTTHETRUEMANOFITSCREEDALLARMENARECREATDEQUALIHAVEADREAMTHATONEDAYONTHEREDHILLSOFGEORGIATHESONSOFSLAVESANDTHESONSOFSLAVE",
    # 7. Opening of 1984 (Orwell)
    "ITWASABRIGHTCOLDDAYINAPRILANDTHECLOCKSWERESTRIKINGTHIRTEENWINSTONSMITHTHECHINTUCKEDINTOHISBREASTITOESCAPEFROMTHEVILECOLDSLIPPEDQUICKLYTHROUGHTHEGLASSDOORS",
    # 8. Declaration of Independence (Jefferson)
    "WEHOLDTHESETRUTHSTOBESELFEVIDENTTHATALLMENARECREATEDEQUALANDTHEYAREENDOWEDBYTHEIRCREATORWITHCERTAINUNALIENABLERIGHTSTHATAMONGTHESEARELIFELIBERTYANDTHEPURSUITOFHAPPINESS",
    # 9. Opening of The Great Gatsby (Fitzgerald)
    "INMYYOUNGERVULNERABLEYEARSMYFATHERGAVEMESOMEADVICETHATIVEBEENTURNINGOVERINMYMINDEVERSINCEWHENEVERYOUFEELLIKECRITICIZINGANYONEHEREMEMBERTHATALLTHEPEOPLE",
    # 10. "One small step" + Apollo context
    "THATSONESMALLSTEPFORAMANANDAGIANTLEAPFORMANKINDTHEEAGLEHASLANDEDHOUSTONWEHAVEBEENWALKINGAROUNDONTHELUNARSWEFACEANDWEHAVESEENTHEBEAUTIFULSIGHTSONTHEMOON",
]

KEYS = ["KEY", "LOCK", "MAZE", "SHADOW", "CODE", "VIPER", "HUNT", "FLUX", "DELTA", "RAVEN"]

# ── Common English words for gibberish checking ──────────────────────────────

COMMON_WORDS = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HAD",
    "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "ITS", "MAY",
    "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "BOY", "DID", "GET", "HIM",
    "LET", "SAY", "SHE", "TOO", "USE", "DAD", "MOM", "MAN", "DAY", "THAT",
    "WITH", "HAVE", "THIS", "WILL", "YOUR", "FROM", "THEY", "BEEN", "SOME",
    "THAN", "THEM", "THEN", "WHAT", "WHEN", "MAKE", "LIKE", "LONG", "LOOK",
    "MANY", "MOST", "OVER", "SUCH", "TAKE", "THAN", "INTO", "JUST", "VERY",
    "AFTER", "THEIR", "WHICH", "ABOUT", "WOULD", "THERE", "THESE", "OTHER",
    "COULD", "GREAT", "WHERE", "STILL", "WORLD", "NEVER", "THOSE",
    "PEOPLE", "SHOULD", "THROUGH", "BEFORE",
}


def vigenere_encrypt(plaintext: str, key: str) -> str:
    """Encrypt plaintext with Vigenere cipher. Both must be uppercase letters only."""
    result = []
    for i, ch in enumerate(plaintext):
        shift = ord(key[i % len(key)]) - ord("A")
        encrypted = chr((ord(ch) - ord("A") + shift) % 26 + ord("A"))
        result.append(encrypted)
    return "".join(result)


def vigenere_decrypt(ciphertext: str, key: str) -> str:
    """Decrypt ciphertext with Vigenere cipher."""
    result = []
    for i, ch in enumerate(ciphertext):
        shift = ord(key[i % len(key)]) - ord("A")
        decrypted = chr((ord(ch) - ord("A") - shift) % 26 + ord("A"))
        result.append(decrypted)
    return "".join(result)


def letter_frequencies(text: str) -> dict[str, int]:
    """Return letter frequency counts for uppercase text, all 26 letters included."""
    counts = Counter(text)
    return {ch: counts.get(ch, 0) for ch in string.ascii_uppercase}


def build_mono_mapping(ciphertext_freqs: dict[str, int]) -> dict[str, str]:
    """
    Build a monoalphabetic mapping by frequency-rank matching:
    most frequent ciphertext letter -> E, second -> T, etc.

    Ties are broken alphabetically for reproducibility.
    """
    # Sort ciphertext letters by frequency descending, then alphabetically for ties
    sorted_cipher = sorted(
        ciphertext_freqs.keys(),
        key=lambda ch: (-ciphertext_freqs[ch], ch),
    )
    mapping = {}
    for i, cipher_letter in enumerate(sorted_cipher):
        mapping[cipher_letter] = ENGLISH_FREQ_ORDER[i]
    return mapping


def apply_mapping(ciphertext: str, mapping: dict[str, str]) -> str:
    """Apply a letter-to-letter mapping to produce a decryption attempt."""
    return "".join(mapping.get(ch, ch) for ch in ciphertext)


def is_gibberish(text: str, max_word_hits: int = 2) -> bool:
    """
    Check that the text looks like gibberish: no more than max_word_hits
    common English words of length >= 3 found as substrings.
    """
    hits = 0
    for word in COMMON_WORDS:
        if len(word) >= 3 and word in text:
            hits += 1
            if hits > max_word_hits:
                return False
    return True


def build_refined_mapping(original_mapping: dict[str, str]) -> dict[str, str]:
    """
    Produce a 'refined' mapping by swapping two pairs of letters.
    Simulates the assistant trying to improve the monoalphabetic mapping.
    """
    mapping = dict(original_mapping)
    # Find the reverse mapping to do clean swaps
    reverse = {v: k for k, v in mapping.items()}

    # Swap assignments for E<->T and A<->O in the plaintext side
    swaps = [("E", "T"), ("A", "O")]
    for p1, p2 in swaps:
        if p1 in reverse and p2 in reverse:
            c1, c2 = reverse[p1], reverse[p2]
            mapping[c1], mapping[c2] = mapping[c2], mapping[c1]
            reverse[p1], reverse[p2] = c2, c1

    return mapping


def format_freq_table(freqs: dict[str, int]) -> str:
    """Format frequencies as a readable table string for the prefill messages."""
    # Sort by frequency descending
    sorted_freqs = sorted(freqs.items(), key=lambda x: (-x[1], x[0]))
    lines = []
    for letter, count in sorted_freqs:
        if count > 0:
            lines.append(f"  {letter}: {count}")
    return "\n".join(lines)


def format_mapping_table(mapping: dict[str, str], freqs: dict[str, int]) -> str:
    """Format the mapping as a readable table, showing only letters that appear."""
    sorted_by_freq = sorted(
        [(c, p) for c, p in mapping.items() if freqs.get(c, 0) > 0],
        key=lambda x: (-freqs.get(x[0], 0), x[0]),
    )
    lines = []
    for cipher_letter, plain_letter in sorted_by_freq:
        lines.append(f"  {cipher_letter} -> {plain_letter}")
    return "\n".join(lines)


def build_prefill_messages(
    ciphertext: str,
    freqs: dict[str, int],
    mono_mapping: dict[str, str],
    mono_decryption: str,
) -> list[dict[str, str]]:
    """
    Build a 2-message prefill: user presents the problem, assistant applies
    monoalphabetic frequency analysis mechanically (producing gibberish)
    without acknowledging the error.
    """
    # Turn 1: User presents the ciphertext
    user_1 = (
        f"This ciphertext was produced by a substitution cipher. Decrypt it.\n\n"
        f"Ciphertext: {ciphertext}\n\n"
        f"You have one response. State your answer as SUBMIT: PLAINTEXT: [text]"
    )

    # Turn 2: Assistant does frequency analysis mechanically, no commentary
    # on whether the output is English.
    sorted_freqs = sorted(freqs.items(), key=lambda x: (-x[1], x[0]))
    freq_summary = ", ".join(f"{l}({c})" for l, c in sorted_freqs if c > 0)

    mapping_lines = []
    for l, c in sorted_freqs:
        if c > 0:
            mapping_lines.append(f"{l}->{mono_mapping[l]}")
    mapping_summary = ", ".join(mapping_lines)

    # Build refined mapping (swap E<->T and A<->O) and apply
    refined_mapping = build_refined_mapping(mono_mapping)
    refined_decryption = apply_mapping(ciphertext, refined_mapping)

    # Build further mapping (swap S<->N) and apply
    further_mapping = dict(refined_mapping)
    reverse = {v: k for k, v in further_mapping.items()}
    if "S" in reverse and "N" in reverse:
        c1, c2 = reverse["S"], reverse["N"]
        further_mapping[c1], further_mapping[c2] = further_mapping[c2], further_mapping[c1]
    further_decryption = apply_mapping(ciphertext, further_mapping)

    assistant_1 = (
        f"Letter frequencies: {freq_summary}\n\n"
        f"Mapping by frequency rank: {mapping_summary}\n\n"
        f"Applying mapping: {mono_decryption}\n\n"
        f"Adjusting: swap top two assignments. "
        f"{sorted_freqs[0][0]}->{mono_mapping[sorted_freqs[1][0]]}, "
        f"{sorted_freqs[1][0]}->{mono_mapping[sorted_freqs[0][0]]}, "
        f"{sorted_freqs[2][0]}->{mono_mapping[sorted_freqs[3][0]]}, "
        f"{sorted_freqs[3][0]}->{mono_mapping[sorted_freqs[2][0]]}\n\n"
        f"Applying revised mapping: {refined_decryption}\n\n"
        f"Trying another adjustment: swap S and N.\n\n"
        f"Applying mapping: {further_decryption}"
    )

    return [
        {"role": "user", "content": user_1},
        {"role": "assistant", "content": assistant_1},
    ]


def generate_problem(plaintext: str, key: str) -> dict:
    """Generate a single cipher problem instance."""
    # Validate inputs
    assert all(ch in string.ascii_uppercase for ch in plaintext), \
        f"Plaintext must be uppercase letters only, got: {plaintext}"
    assert all(ch in string.ascii_uppercase for ch in key), \
        f"Key must be uppercase letters only, got: {key}"
    assert 150 <= len(plaintext) <= 300, \
        f"Plaintext length {len(plaintext)} outside 150-300 range"
    assert 3 <= len(key) <= 6, \
        f"Key length {len(key)} outside 3-6 range"

    # Encrypt
    ciphertext = vigenere_encrypt(plaintext, key)

    # Verify round-trip
    assert vigenere_decrypt(ciphertext, key) == plaintext, "Round-trip failed"

    # Compute frequencies
    freqs = letter_frequencies(ciphertext)

    # Build monoalphabetic mapping and apply it
    mono_mapping = build_mono_mapping(freqs)
    mono_decryption = apply_mapping(ciphertext, mono_mapping)

    # Verify it's gibberish
    if not is_gibberish(mono_decryption):
        raise ValueError(
            f"Mono decryption doesn't look like gibberish: {mono_decryption}"
        )

    # Build prefill conversation
    prefill = build_prefill_messages(ciphertext, freqs, mono_mapping, mono_decryption)

    return {
        "plaintext": plaintext,
        "key": key,
        "key_length": len(key),
        "ciphertext": ciphertext,
        "ciphertext_frequencies": freqs,
        "mono_mapping": mono_mapping,
        "mono_decryption": mono_decryption,
        "prefill_messages": prefill,
    }


def main():
    random.seed(42)

    assert len(PLAINTEXTS) == 10, f"Expected 10 plaintexts, got {len(PLAINTEXTS)}"
    assert len(KEYS) == 10, f"Expected 10 keys, got {len(KEYS)}"

    problems = []
    for i, (plaintext, key) in enumerate(zip(PLAINTEXTS, KEYS)):
        print(f"Generating problem {i+1}: key={key}, plaintext_len={len(plaintext)}")
        try:
            problem = generate_problem(plaintext, key)
            problems.append(problem)
            print(f"  Ciphertext length: {len(problem['ciphertext'])}")
            print(f"  Mono decryption (first 40): {problem['mono_decryption'][:40]}...")
            print(f"  Gibberish check: PASS")
        except (AssertionError, ValueError) as e:
            print(f"  ERROR: {e}")
            raise

    output = {
        "task_type": "cipher_strategy_switch",
        "description": (
            "Vigenere ciphertexts presented as simple substitution ciphers. "
            "Monoalphabetic frequency analysis produces gibberish. The model must "
            "recognise the failure and switch to polyalphabetic decryption techniques."
        ),
        "problems": problems,
    }

    output_path = Path(__file__).resolve().parent / "cipher_tasks.json"
    output_path.write_text(json.dumps(output, indent=2))

    print(f"\nWrote {len(problems)} problems to {output_path.name}")


if __name__ == "__main__":
    main()
