"""
T3 Source Monitoring - Pilot v3

Two modes:
  quick    - labels only
  reasoned - EVIDENCE + REASONING + CLASSIFICATION + CONFIDENCE per item

Usage:
    python run.py gpt-5.4                          # all passages, quick + reasoned
    python run.py gpt-5.4 --passage 1              # single passage
    python run.py gpt-5.4 --mode quick             # quick only
    python run.py gpt-5.4 --closedbook             # closed-book condition
"""

import json
import os
import random
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_path, override=True)

VALID_LABELS = ["PASSAGE", "TRAINING", "INFERRED", "FABRICATED"]
RESULTS_DIR = Path(__file__).parent / "results"

# ---------------------------------------------------------------------------
# Load passages from markdown files
# ---------------------------------------------------------------------------

def load_passage(filepath: Path) -> tuple[str, list[tuple[str, str, str]]]:
    """Load a passage markdown file. Returns (passage_text, [(statement, label, tier), ...])."""
    text = filepath.read_text()

    # Extract passage text between ## Passage and ---
    passage_match = re.search(r'## Passage.*?\n\n(.*?)\n\n---', text, re.DOTALL)
    if not passage_match:
        raise ValueError(f"Could not find passage text in {filepath}")
    passage_text = passage_match.group(1).strip()

    # Extract statements with labels
    statements = []
    # Match patterns like: **P1.** text\n or **I5.** text\n
    stmt_pattern = re.compile(
        r'\*\*([PTIF]\d+)\.\*\*\s+(.*?)(?=\n-|\n\n\*\*[PTIF]|\n\n###|\n\n---|\Z)',
        re.DOTALL
    )
    label_map = {"P": "PASSAGE", "T": "TRAINING", "I": "INFERRED", "F": "FABRICATED"}

    for match in stmt_pattern.finditer(text):
        code = match.group(1)  # e.g., "P1", "I5", "F12"
        stmt_text = match.group(2).strip().split("\n")[0].strip()  # first line only
        prefix = code[0]
        label = label_map[prefix]

        # Determine difficulty tier from section headers
        tier = "unknown"
        pos = match.start()
        # Look backwards for the nearest tier header
        before = text[:pos]
        if "**Adversarial" in before.split("###")[-1]:
            tier = "adversarial"
        elif "**Near-miss" in before.split("###")[-1]:
            tier = "near-miss"
        elif "**Obvious" in before.split("###")[-1]:
            tier = "obvious"
        elif "**Hard" in before.split("###")[-1]:
            tier = "hard"
        elif "**Moderate" in before.split("###")[-1]:
            tier = "moderate"
        elif "**Tight" in before.split("###")[-1]:
            tier = "tight"
        elif label == "PASSAGE":
            tier = "passage"
        elif label == "TRAINING":
            tier = "training"

        statements.append((stmt_text, label, tier))

    return passage_text, statements


def load_all_passages() -> dict:
    """Load all passage files from the passages directory."""
    passages_dir = Path(__file__).parent / "passages"
    passages = {}
    for f in sorted(passages_dir.glob("*.md")):
        num = f.stem.split("_")[0]  # e.g. "001" from "001_topic_name.md"
        name = f.stem.split("_", 1)[1].replace("_", " ").title()
        passage_text, statements = load_passage(f)
        # Convert to (stmt, label) tuples for compatibility, keep tiers separate
        stmt_pairs = [(s, l) for s, l, _ in statements]
        tiers = [t for _, _, t in statements]
        passages[num] = (name, passage_text, stmt_pairs, tiers)
    return passages


PASSAGES = load_all_passages()

# ---------------------------------------------------------------------------
# Model calling
# ---------------------------------------------------------------------------

def get_max_tokens(model: str) -> int:
    """Model-specific token budgets for reasoned mode.
    Reasoning models (Gemini) need high budgets because internal
    reasoning tokens count against the limit."""
    if model.startswith("google/"):
        return 8192  # ~3k reasoning (with effort=low) + ~4k visible
    elif model.startswith("gpt-"):
        return 4096   # no hidden reasoning, just visible output
    else:  # Anthropic -- thinking budget is separate
        return 8192


def call_model(model: str, prompt: str, max_tokens: int = None) -> tuple[str, dict]:
    """Call a model with extended thinking enabled where available.

    Configurations:
    - GPT-5.4: standard generation (no hidden reasoning)
    - Claude: extended thinking enabled (budget_tokens=4096)
    - Gemini: always-on reasoning (hidden, can't disable)
    """
    if max_tokens is None:
        max_tokens = get_max_tokens(model)

    # Route: bare gpt-* or openai/* → OpenAI direct
    # bare claude-* or anthropic/* → Anthropic direct
    # everything else (google/, qwen/, deepseek/, etc.) → OpenRouter
    provider = model.split("/")[0] if "/" in model else None
    is_openai = provider == "openai" or (not provider and (model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3")))
    is_anthropic = provider == "anthropic" or (not provider and model.startswith("claude"))

    if is_openai:
        from openai import OpenAI
        client = OpenAI()
        api_model = model.replace("openai/", "")
        response = client.chat.completions.create(
            model=api_model, max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content, {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "reasoning_tokens": getattr(response.usage.completion_tokens_details, "reasoning_tokens", 0) if response.usage.completion_tokens_details else 0,
        }
    elif is_anthropic:
        from anthropic import Anthropic
        client = Anthropic()
        api_model = model.replace("anthropic/", "")
        # Normalise dots to hyphens for API (claude-sonnet-4.6 → claude-sonnet-4-6)
        api_model = api_model.replace(".", "-", 1).replace(".", "-")
        ANTHROPIC_ALIASES = {"claude-sonnet-4": "claude-sonnet-4-20250514"}
        api_model = ANTHROPIC_ALIASES.get(api_model, api_model)
        response = client.messages.create(
            model=api_model,
            max_tokens=max(max_tokens, 16384),
            thinking={"type": "enabled", "budget_tokens": 4096},
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract text from content blocks (thinking + text)
        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
        return text, {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "reasoning_tokens": 0,
        }
    else:
        # OpenRouter for all other models (qwen/, deepseek/, mistralai/, x-ai/, etc.)
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
        response = client.chat.completions.create(
            model=model, max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or "", {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
            "reasoning_tokens": 0,
        }

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

LABEL_INSTRUCTIONS = """- PASSAGE: This information was explicitly stated in the passage.
- TRAINING: This is a real-world fact from your general knowledge, NOT from the passage.
- INFERRED: Not directly stated, but logically derivable by combining a passage fact with real-world knowledge.
- FABRICATED: Sounds plausible but is neither stated, independently known, nor logically derivable."""


def shuffle_stmts(statements):
    indexed = list(enumerate(statements))
    random.shuffle(indexed)
    return indexed



def build_reasoned_prompt(passage, statements, context_label="the passage"):
    indexed = shuffle_stmts(statements)
    block = "\n".join(f"{i+1}. \"{s}\"" for i, (_, (s, _)) in enumerate(indexed))
    return f"""Classify each statement by knowledge source relative to {context_label}.

{LABEL_INSTRUCTIONS}

For each statement, provide:
- EVIDENCE: What specific fact(s) from the passage or your knowledge are relevant to this statement?
- REASONING: How did you combine the evidence to reach your classification?
- CLASSIFICATION: Your label (PASSAGE, TRAINING, INFERRED, or FABRICATED)
- CONFIDENCE: How confident are you in this classification? (0-100)

You MUST classify ALL {len(statements)} statements.

=== PASSAGE ===
{passage}
=== END PASSAGE ===

{block}""", indexed

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_reasoned(text, n):
    """Parse reasoned responses. Handles markdown formatting (**CLASSIFICATION:** etc). Returns (labels, confidences)."""
    labels = []
    confidences = []
    current_confidence = None

    for line in text.strip().split("\n"):
        line = line.strip()
        # Strip markdown bold markers for matching
        clean = line.replace("**", "").replace("*", "")

        # Look for CONFIDENCE (may come before or after CLASSIFICATION)
        conf_match = re.search(r'CONFIDENCE:\s*(\d+)', clean, re.IGNORECASE)
        if conf_match:
            current_confidence = int(conf_match.group(1))

        # Look for CLASSIFICATION
        cls_match = re.search(r'CLASSIFICATION:\s*\[?(\w+)\]?', clean, re.IGNORECASE)
        if cls_match and cls_match.group(1).upper() in VALID_LABELS:
            labels.append(cls_match.group(1).upper())
            confidences.append(current_confidence if current_confidence is not None else -1)
            current_confidence = None

    return labels[:n], confidences[:n]

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(predictions, ground_truth, statements, confidences=None):
    n = len(predictions)
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    cats = VALID_LABELS

    per_cat = {}
    for cat in cats:
        idxs = [i for i, g in enumerate(ground_truth) if g == cat]
        if idxs:
            c = sum(predictions[i] == ground_truth[i] for i in idxs)
            per_cat[cat] = {"correct": c, "total": len(idxs), "accuracy": c / len(idxs)}

    confusion = {g: {p: 0 for p in cats} for g in cats}
    for pred, gt in zip(predictions, ground_truth):
        if pred in cats and gt in cats:
            confusion[gt][pred] += 1

    fab_idxs = [i for i, g in enumerate(ground_truth) if g == "FABRICATED"]
    confab = sum(predictions[i] == "FABRICATED" for i in fab_idxs) / len(fab_idxs) if fab_idxs else 0

    errors = [{"statement": statements[i], "predicted": p, "actual": g}
              for i, (p, g) in enumerate(zip(predictions, ground_truth)) if p != g]

    result = {
        "overall_accuracy": correct / n, "correct": correct, "total": n,
        "per_category": per_cat, "confabulation_detection_rate": confab,
        "confusion_matrix": confusion, "errors": errors,
    }

    # Confidence calibration if available
    if confidences and any(c >= 0 for c in confidences):
        valid = [(confidences[i], predictions[i] == ground_truth[i])
                 for i in range(n) if confidences[i] >= 0]
        if valid:
            # ECE with 5 bins
            bins = [[] for _ in range(5)]
            for conf, is_correct in valid:
                bin_idx = min(int(conf / 20), 4)  # 0-19, 20-39, 40-59, 60-79, 80-100
                bins[bin_idx].append((conf / 100, 1 if is_correct else 0))

            ece = 0
            for b in bins:
                if b:
                    avg_conf = sum(c for c, _ in b) / len(b)
                    avg_acc = sum(a for _, a in b) / len(b)
                    ece += len(b) / len(valid) * abs(avg_conf - avg_acc)

            mean_conf_correct = [c for c, ok in valid if ok]
            mean_conf_wrong = [c for c, ok in valid if not ok]

            result["confidence"] = {
                "ece": round(ece, 3),
                "mean_confidence_correct": round(sum(mean_conf_correct) / len(mean_conf_correct), 1) if mean_conf_correct else None,
                "mean_confidence_wrong": round(sum(mean_conf_wrong) / len(mean_conf_wrong), 1) if mean_conf_wrong else None,
                "n_with_confidence": len(valid),
            }

    return result


def print_results(results, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Overall: {results['correct']}/{results['total']} ({results['overall_accuracy']:.0%})")
    print(f"  Confab Detection: {results['confabulation_detection_rate']:.0%}")
    for cat, d in results["per_category"].items():
        bar = "█" * d["correct"] + "░" * (d["total"] - d["correct"])
        print(f"    {cat:12s} {bar} {d['correct']}/{d['total']} ({d['accuracy']:.0%})")

    if "confidence" in results:
        c = results["confidence"]
        print(f"\n  Confidence: ECE={c['ece']:.3f}  correct={c['mean_confidence_correct']}  wrong={c['mean_confidence_wrong']}")

    cats = VALID_LABELS
    print(f"\n  Confusion (rows=actual, cols=predicted):")
    print("              " + "".join(f"{c[:4]:>6s}" for c in cats))
    for gt in cats:
        print(f"    {gt:10s}" + "".join(f"{results['confusion_matrix'][gt][p]:>6d}" for p in cats))

    if results["errors"]:
        print(f"\n  Errors ({len(results['errors'])}):")
        for e in results["errors"]:
            print(f"    {e['predicted']:>11s} <- {e['actual']:<11s} | {e['statement'][:60]}...")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run_openbook(model, passage_id, passage_name, passage_text, statements, mode="reasoned"):
    prompt, indexed = build_reasoned_prompt(passage_text, statements)
    shuffle_order = [orig_idx for orig_idx, _ in indexed]

    print(f"\n  [{passage_name}] reasoned mode ({model})...")
    response_text, usage = call_model(model, prompt)
    reasoning = usage.get("reasoning_tokens", 0)
    print(f"  Tokens: {usage['input_tokens']} in, {usage['output_tokens']} out" +
          (f" (reasoning: {reasoning})" if reasoning else ""))

    preds, confs = parse_reasoned(response_text, len(statements))
    if len(preds) != len(statements):
        print(f"  WARNING: Got {len(preds)}/{len(statements)} predictions")
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / f"{model.replace('/', '_')}_p{passage_id}_{mode}_raw.txt").write_text(response_text)
        return None

    # Unshuffle
    predictions = [""] * len(statements)
    confidences = [0] * len(statements)
    for shuffled_pos, orig_idx in enumerate(shuffle_order):
        predictions[orig_idx] = preds[shuffled_pos]
        if confs:
            confidences[orig_idx] = confs[shuffled_pos]

    gt = [l for _, l in statements]
    stmts = [s for s, _ in statements]
    results = score(predictions, gt, stmts, confidences if confs else None)
    print_results(results, f"{model} | {passage_name} | {mode}")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"model": model, "passage": passage_id, "condition": "open-book", "results": results}
    (RESULTS_DIR / f"{model.replace('/', '_')}_p{passage_id}_openbook.json").write_text(json.dumps(out, indent=2))
    (RESULTS_DIR / f"{model.replace('/', '_')}_p{passage_id}_openbook_raw.txt").write_text(response_text)

    return results


def run_closedbook(model, passage_id, passage_name, passage_text, statements):
    # Turn 1: summarise
    summary_prompt = f"""Read the following passage carefully. Summarise the key facts in bullet points. Be thorough — include specific names, numbers, dates, and relationships.

=== PASSAGE ===
{passage_text}
=== END PASSAGE ===

Bullet-point summary:"""

    print(f"\n  [{passage_name}] closed-book: summarising ({model})...")
    summary, usage1 = call_model(model, summary_prompt)
    print(f"  Summary: {usage1['output_tokens']} tokens")

    # Turn 2: classify from summary
    prompt, indexed = build_reasoned_prompt(summary, statements, context_label="your summary of a passage you read earlier")
    shuffle_order = [orig_idx for orig_idx, _ in indexed]

    print(f"  [{passage_name}] closed-book: classifying ({model})...")
    response_text, usage2 = call_model(model, prompt)
    reasoning = usage2.get("reasoning_tokens", 0)
    print(f"  Tokens: {usage2['input_tokens']} in, {usage2['output_tokens']} out" +
          (f" (reasoning: {reasoning})" if reasoning else ""))

    preds, confs = parse_reasoned(response_text, len(statements))
    if len(preds) != len(statements):
        print(f"  WARNING: Got {len(preds)}/{len(statements)} predictions")
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / f"{model.replace('/', '_')}_p{passage_id}_closedbook_raw.txt").write_text(
            f"=== SUMMARY ===\n{summary}\n\n=== RESPONSE ===\n{response_text}")
        if len(preds) < len(statements) - 2:  # more than 2 missing = unusable
            return None
        # Pad with empty predictions for minor shortfalls (1-2 missing)
        while len(preds) < len(statements):
            preds.append("")
            confs.append(-1)

    predictions = [""] * len(statements)
    confidences = [0] * len(statements)
    for shuffled_pos, orig_idx in enumerate(shuffle_order):
        predictions[orig_idx] = preds[shuffled_pos]
        if confs:
            confidences[orig_idx] = confs[shuffled_pos]

    gt = [l for _, l in statements]
    stmts = [s for s, _ in statements]
    results = score(predictions, gt, stmts, confidences if confs else None)
    print_results(results, f"{model} | {passage_name} | closed-book")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"model": model, "passage": passage_id, "condition": "closed-book",
           "results": results, "summary": summary}
    (RESULTS_DIR / f"{model.replace('/', '_')}_p{passage_id}_closedbook.json").write_text(json.dumps(out, indent=2))
    (RESULTS_DIR / f"{model.replace('/', '_')}_p{passage_id}_closedbook_raw.txt").write_text(
        f"=== SUMMARY ===\n{summary}\n\n=== RESPONSE ===\n{response_text}")

    return results


def run_control(model, passage_id, passage_name, passage_text, statements, judge_model=None):
    """Condition C: Selective Q&A (monitoring-to-control).

    Tests whether the model can use source awareness to decide how to respond.
    For each question, the model must choose:
    - ANSWER: I can answer this from the passage (then provide the answer)
    - INFER: I can reason about this using passage + external knowledge (then provide reasoning)
    - DECLINE: The passage does not address this

    The correct action type is pre-determined for each question. We score:
    - Action accuracy: did it choose the right action (ANSWER/INFER/DECLINE)?
    - Content accuracy: when it chose ANSWER or INFER, was the content correct?
    """
    # Load questions for this passage
    questions_file = Path(__file__).parent / "control_questions.json"
    all_questions = json.loads(questions_file.read_text())
    if passage_id not in all_questions:
        print(f"  No control questions for passage {passage_id}")
        return None

    questions = all_questions[passage_id]["questions"]

    # Build prompt
    q_block = "\n".join(
        f"Q{i+1}. {q['question']}" for i, q in enumerate(questions)
    )

    prompt = f"""You will read a passage, then answer questions about it. For each question, you must first decide HOW to respond:

- ANSWER: The passage directly addresses this question. Provide the answer from the passage.
- INFER: The passage doesn't directly answer this, but you can reason about it by combining passage facts with general knowledge. Provide your reasoning and conclusion.
- DECLINE: The passage does not contain enough information to answer this question. Say what's missing.

It is critical that you choose the right action. Do not guess or speculate when the passage doesn't support an answer. Do not ANSWER when you should INFER. Do not INFER when you should DECLINE.

=== PASSAGE ===
{passage_text}
=== END PASSAGE ===

For each question, respond in this format:
Q1. ACTION: [ANSWER/INFER/DECLINE]
    RESPONSE: [your answer, reasoning, or explanation of what's missing]

{q_block}"""

    print(f"\n  [{passage_name}] Condition C: selective Q&A ({model})...")
    response_text, usage = call_model(model, prompt)
    print(f"  Tokens: {usage['input_tokens']} in, {usage['output_tokens']} out")

    # Parse responses
    responses = []
    current_action = None
    current_response = None
    for line in response_text.split("\n"):
        clean = line.strip().replace("**", "").replace("*", "")
        action_match = re.search(r'ACTION:\s*(ANSWER|INFER|DECLINE)', clean, re.IGNORECASE)
        response_match = re.search(r'RESPONSE:\s*(.*)', clean, re.IGNORECASE)
        if action_match:
            if current_action is not None:
                responses.append({"action": current_action, "response": current_response or ""})
            current_action = action_match.group(1).upper()
            current_response = None
        if response_match and current_action:
            current_response = response_match.group(1).strip()
    if current_action is not None:
        responses.append({"action": current_action, "response": current_response or ""})

    if len(responses) != len(questions):
        print(f"  WARNING: Got {len(responses)}/{len(questions)} responses")
        RESULTS_DIR.mkdir(exist_ok=True)
        (RESULTS_DIR / f"{model.replace('/', '_')}_p{passage_id}_control_raw.txt").write_text(response_text)
        # Pad if close
        while len(responses) < len(questions):
            responses.append({"action": "", "response": ""})

    # Score
    action_correct = 0
    action_total = len(questions)
    errors = []
    for i, (q, r) in enumerate(zip(questions, responses)):
        expected = q["type"]
        got = r["action"]
        if expected == got:
            action_correct += 1
        else:
            errors.append({
                "question": q["question"],
                "expected": expected,
                "got": got,
                "response": r["response"][:80],
            })

    action_accuracy = action_correct / action_total if action_total > 0 else 0

    # Per-type breakdown
    type_counts = {"ANSWER": {"correct": 0, "total": 0}, "INFER": {"correct": 0, "total": 0}, "DECLINE": {"correct": 0, "total": 0}}
    for q, r in zip(questions, responses):
        t = q["type"]
        type_counts[t]["total"] += 1
        if q["type"] == r["action"]:
            type_counts[t]["correct"] += 1

    print(f"\n  Condition C Results — {model} | {passage_name}")
    print(f"  Action accuracy: {action_correct}/{action_total} ({action_accuracy:.0%})")
    for t in ["ANSWER", "INFER", "DECLINE"]:
        c, n = type_counts[t]["correct"], type_counts[t]["total"]
        if n > 0:
            print(f"    {t:8s} {c}/{n} ({c/n:.0%})")
    if errors:
        print(f"  Errors ({len(errors)}):")
        for e in errors:
            print(f"    expected={e['expected']:8s} got={e['got']:8s} | {e['question'][:55]}...")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "model": model, "passage": passage_id, "condition": "control-qa",
        "action_accuracy": action_accuracy,
        "action_correct": action_correct, "action_total": action_total,
        "type_counts": type_counts, "errors": errors,
        "questions": [{"question": q["question"], "expected": q["type"],
                       "got": responses[i]["action"] if i < len(responses) else "",
                       "response": responses[i]["response"][:200] if i < len(responses) else ""}
                      for i, q in enumerate(questions)],
    }
    (RESULTS_DIR / f"{model.replace('/', '_')}_p{passage_id}_control.json").write_text(json.dumps(out, indent=2))
    (RESULTS_DIR / f"{model.replace('/', '_')}_p{passage_id}_control_raw.txt").write_text(response_text)

    return {"action_accuracy": action_accuracy, "action_correct": action_correct,
            "action_total": action_total, "type_counts": type_counts, "errors": errors}


def main():
    args = sys.argv[1:]
    model = "claude-sonnet-4-6"
    passage_ids = None  # None = all
    closedbook = False
    control = False

    i = 0
    while i < len(args):
        if args[i] == "--passage":
            passage_ids = [args[i + 1]]
            i += 2
        elif args[i] == "--closedbook":
            closedbook = True
            i += 1
        elif args[i] == "--control":
            control = True
            i += 1
        elif not args[i].startswith("--"):
            model = args[i]
            i += 1
        else:
            i += 1

    if not PASSAGES:
        passages_dir = Path(__file__).parent / "passages"
        print(
            f"ERROR: No T3 passages found in {passages_dir}/.\n"
            f"       These are withheld from the public repo to avoid training-data\n"
            f"       contamination. Author your own per the README ('What's not\n"
            f"       shipped' table, T3 row): 4 PASSAGE / 4 TRAINING / 12 INFERRED /\n"
            f"       12 FABRICATED items per passage, saved as passages/NNN_name.md.",
            file=sys.stderr,
        )
        sys.exit(1)

    if passage_ids is None:
        passage_ids = sorted(PASSAGES.keys())

    condition = "control" if control else ("closed-book" if closedbook else "open-book")
    print(f"{'='*60}")
    print(f"  T3 Source Monitoring — {condition}")
    print(f"  Model: {model}")
    print(f"  Passages: {', '.join(passage_ids)} ({len(passage_ids)} total)")
    print(f"{'='*60}")

    all_results = {}
    for pid in passage_ids:
        pname, ptext, pstmts, _ = PASSAGES[pid]
        if control:
            r = run_control(model, pid, pname, ptext, pstmts)
        elif closedbook:
            r = run_closedbook(model, pid, pname, ptext, pstmts)
        else:
            r = run_openbook(model, pid, pname, ptext, pstmts)
        if r:
            all_results[pid] = r

    # Aggregate across passages
    if control and len(all_results) >= 2:
        total_correct = sum(r["action_correct"] for r in all_results.values())
        total_qs = sum(r["action_total"] for r in all_results.values())
        acc = total_correct / total_qs if total_qs else 0
        print(f"\n{'='*60}")
        print(f"  AGGREGATE — {model} | control (selective Q&A) | {len(all_results)} passages")
        print(f"{'='*60}")
        print(f"  Action accuracy: {total_correct}/{total_qs} ({acc:.0%})")
        for t in ["ANSWER", "INFER", "DECLINE"]:
            c = sum(r["type_counts"].get(t, {}).get("correct", 0) for r in all_results.values())
            n = sum(r["type_counts"].get(t, {}).get("total", 0) for r in all_results.values())
            if n > 0:
                print(f"    {t:8s} {c}/{n} ({c/n:.0%})")

    elif not control and len(all_results) >= 2:
        total_correct = sum(r["correct"] for r in all_results.values())
        total_items = sum(r["total"] for r in all_results.values())
        total_fab = sum(
            r["per_category"].get("FABRICATED", {}).get("correct", 0) for r in all_results.values()
        )
        total_fab_items = sum(
            r["per_category"].get("FABRICATED", {}).get("total", 0) for r in all_results.values()
        )
        print(f"\n{'='*60}")
        print(f"  AGGREGATE — {model} | {condition} | {len(all_results)} passages")
        print(f"{'='*60}")
        print(f"  Overall: {total_correct}/{total_items} ({total_correct/total_items:.0%})")
        if total_fab_items > 0:
            print(f"  Confab Detection: {total_fab}/{total_fab_items} ({total_fab/total_fab_items:.0%})")
        for cat in VALID_LABELS:
            c = sum(r["per_category"].get(cat, {}).get("correct", 0) for r in all_results.values())
            t = sum(r["per_category"].get(cat, {}).get("total", 0) for r in all_results.values())
            if t > 0:
                print(f"    {cat:12s} {c}/{t} ({c/t:.0%})")

    print()


if __name__ == "__main__":
    main()
