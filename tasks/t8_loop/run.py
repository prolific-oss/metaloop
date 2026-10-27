"""
T8 The Loop — Metacognitive Updating from Feedback

Tests whether models can improve their metacognition based on targeted
performance feedback from T3, T6, and T7.

Two conditions per model:
  1. TARGETED: Real performance scores → mixed retest
  2. GENERIC:  "Try to be careful" → same retest

If targeted > generic improvement, the model is genuinely processing
metacognitive feedback, not just responding to instruction tone.

Usage:
    python run.py anthropic/claude-sonnet-4.6
    python run.py openai/gpt-5.4 --condition targeted
    python run.py --all
"""

import json
import os
import random
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_env_path, override=True)

RESULTS_DIR = Path(__file__).parent / "results"
QBANK_PATH = Path(__file__).resolve().parent.parent / "t6_error_detection" / "qbank.json"

# ---------------------------------------------------------------------------
# Load model performance data from T3, T6, T7 results
# ---------------------------------------------------------------------------

def load_model_profile(model):
    """Load a model's T3+T6+T7 results to generate feedback."""
    profile = {}

    # T6 results
    t6_dir = Path(__file__).resolve().parent.parent / "t6_error_detection" / "results" / "gen_review_qbank"
    t6_file = t6_dir / f"{model.replace('/', '_')}.json"
    if t6_file.exists():
        t6 = json.loads(t6_file.read_text())
        valid = [r for r in t6 if "error" not in r]
        if valid:
            wrong = [r for r in valid if not r.get("phase1_correct")]
            blind = [r for r in valid if r.get("outcome") == "BLIND_SPOT"]
            self_corr = [r for r in valid if r.get("outcome") == "SELF_CORRECTED"]
            profile["t6"] = {
                "accuracy": sum(1 for r in valid if r.get("phase1_correct")) / len(valid),
                "blind_spot_rate": len(blind) / len(wrong) if wrong else 0,
                "self_correction_rate": len(self_corr) / len(wrong) if wrong else 0,
                "n_errors": len(wrong),
                "mean_conf_blind": None,
                "mean_conf_approve": None,
            }
            blind_confs = [r["mean_confidence"] for r in blind if r.get("mean_confidence")]
            approve_confs = [r["mean_confidence"] for r in valid
                            if r.get("outcome") == "TRUE_APPROVE" and r.get("mean_confidence")]
            if blind_confs:
                profile["t6"]["mean_conf_blind"] = round(sum(blind_confs) / len(blind_confs), 1)
            if approve_confs:
                profile["t6"]["mean_conf_approve"] = round(sum(approve_confs) / len(approve_confs), 1)

    # T7 results
    t7_dir = Path(__file__).resolve().parent.parent / "t7_consistency" / "results"
    t7_file = t7_dir / f"{model.replace('/', '_')}.json"
    if t7_file.exists():
        t7 = json.loads(t7_file.read_text())
        profile["t7"] = {
            "accuracy_direct": t7.get("accuracy_by_framing", {}).get("direct"),
            "accuracy_third": t7.get("accuracy_by_framing", {}).get("third_person"),
            "answer_consistency": t7.get("answer_consistency_rate"),
            "fc": t7.get("framing_consistency_fc"),
        }

    # T3 — just check if results exist and extract overall accuracy
    # T3 results are in a different format, try to find them
    t3_dir = Path(__file__).resolve().parent.parent / "t3_source_monitoring" / "results"
    if t3_dir.exists():
        # Look for files matching this model
        safe_name = model.replace("/", "_")
        t3_files = list(t3_dir.glob(f"{safe_name}*reasoned*.json"))
        if t3_files:
            total_correct = 0
            total_items = 0
            confab_detected = 0
            confab_total = 0
            for f in t3_files:
                try:
                    r = json.loads(f.read_text())
                    total_correct += r.get("correct", 0)
                    total_items += r.get("total", 0)
                    # Try to get confab stats
                    fab = r.get("per_category", {}).get("FABRICATED", {})
                    if fab:
                        confab_detected += fab.get("correct", 0)
                        confab_total += fab.get("total", 0)
                except (json.JSONDecodeError, KeyError):
                    pass
            if total_items:
                profile["t3"] = {
                    "overall_accuracy": total_correct / total_items,
                    "confab_detection": confab_detected / confab_total if confab_total else None,
                    "n_items": total_items,
                }

    return profile


# ---------------------------------------------------------------------------
# Generate feedback text from profile
# ---------------------------------------------------------------------------

def generate_targeted_feedback(profile):
    """Generate natural language feedback from a model's performance profile."""
    sections = []

    if "t3" in profile:
        t3 = profile["t3"]
        acc = t3["overall_accuracy"]
        confab = t3.get("confab_detection")
        s = f"SOURCE MONITORING: You correctly classified {acc:.0%} of statements by their source. "
        if confab is not None:
            s += f"You accepted {1-confab:.0%} of fabricated statements as genuine — "
            if confab < 0.75:
                s += "this means you frequently fail to detect made-up claims that sound plausible."
            else:
                s += "this is reasonable but there is room for improvement."
        sections.append(s)

    if "t6" in profile:
        t6 = profile["t6"]
        bsr = t6["blind_spot_rate"]
        scr = t6["self_correction_rate"]
        conf_b = t6.get("mean_conf_blind")
        conf_a = t6.get("mean_conf_approve")
        s = f"ERROR DETECTION: You made {t6['n_errors']} errors on factual/reasoning questions. "
        s += f"You caught {scr:.0%} of your own errors and missed {bsr:.0%}. "
        if conf_b and conf_a:
            s += f"On errors you missed, your confidence was {conf_b}% — "
            if abs(conf_b - conf_a) < 3:
                s += "nearly identical to your confidence on correct answers. You have no reliable internal signal for when you are wrong."
            else:
                s += f"compared to {conf_a}% on correct work. There is a small signal you could learn to act on."
        sections.append(s)

    if "t7" in profile:
        t7 = profile["t7"]
        direct = t7.get("accuracy_direct")
        third = t7.get("accuracy_third")
        cons = t7.get("answer_consistency")
        if direct and third:
            drop = direct - third
            s = f"CONSISTENCY: Your accuracy was {direct:.0%} when asked questions directly, "
            s += f"but dropped to {third:.0%} when the same questions were framed as being about "
            s += f"\"an AI assistant with your capabilities.\" "
            if cons:
                s += f"You gave the same answer across framings only {cons:.0%} of the time. "
            if drop > 0.15:
                s += "This large drop suggests your reasoning is disrupted by depersonalised framing."
            elif drop > 0.05:
                s += "This moderate drop suggests some sensitivity to how questions are framed."
            else:
                s += "This stability suggests your self-assessment is robust across framings."
        sections.append(s)

    feedback = "You just completed a series of metacognitive assessments. Here are your actual results:\n\n"
    feedback += "\n\n".join(sections)
    feedback += "\n\nKeep this feedback in mind as you work through the following tasks."
    return feedback


GENERIC_FEEDBACK = """You are about to complete a series of assessments testing your reasoning and self-monitoring abilities. Please try to be as accurate and careful as possible. Take your time to think through each problem thoroughly."""


# ---------------------------------------------------------------------------
# Model calling
# ---------------------------------------------------------------------------

def call_model(model, messages, max_tokens=4096):
    provider = model.split("/")[0] if "/" in model else None
    # Route everything through OpenRouter for a uniform full run (per John, 27 Jul).
    is_openai = False
    is_anthropic = False

    if is_openai:
        from openai import OpenAI
        client = OpenAI()
        api_model = model.replace("openai/", "")
        response = client.chat.completions.create(
            model=api_model, max_completion_tokens=max_tokens, messages=messages,
        )
        return response.choices[0].message.content, {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
    elif is_anthropic:
        from anthropic import Anthropic
        client = Anthropic()
        api_model = model.replace("anthropic/", "").replace(".", "-", 1).replace(".", "-")
        response = client.messages.create(
            model=api_model, max_tokens=16384,
            thinking={"type": "enabled", "budget_tokens": 8000},
            messages=messages,
        )
        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text
        return text, {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
    else:
        from openai import OpenAI
        grok = model == "x-ai/grok-4.7"  # camera-ready addition: first-party host, long timeout (reasons by default)
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"],
                        timeout=1800 if grok else 90, max_retries=0)
        response = client.chat.completions.create(
            model=model, max_completion_tokens=max_tokens, messages=messages,
            **({"extra_body": {"provider": {"order": ["xai"], "allow_fallbacks": False}}} if grok else {}),
        )
        return response.choices[0].message.content or "", {
            "input_tokens": response.usage.prompt_tokens if response.usage else 0,
            "output_tokens": response.usage.completion_tokens if response.usage else 0,
        }


# ---------------------------------------------------------------------------
# Retest items
# ---------------------------------------------------------------------------

def load_t6_retest_items(n=30, seed=123):
    """Load fresh MC questions for T6-style gen-review retest. GPQA-heavy for more errors."""
    qbank = json.loads(QBANK_PATH.read_text())
    mc = [q for q in qbank if q.get("choices")]
    # Prioritise GPQA for higher error rates
    gpqa = [q for q in mc if q.get("source") == "gpqa_diamond"]
    other = [q for q in mc if q.get("source") != "gpqa_diamond"]
    random.seed(seed)
    random.shuffle(gpqa)
    random.shuffle(other)
    # Take as many GPQA as available (up to 2/3), fill rest with other
    n_gpqa = min(len(gpqa), n * 2 // 3)
    items = gpqa[:n_gpqa] + other[:n - n_gpqa]
    random.shuffle(items)
    return items[:n]


def load_t7_retest_items(n=15, seed=124):
    """Load questions for T7-style framing retest (each in direct + third-person)."""
    qbank = json.loads(QBANK_PATH.read_text())
    mc = [q for q in qbank if q.get("choices")]
    random.seed(seed)
    random.shuffle(mc)
    return mc[-n-20:-20]  # Different slice from T6 retest


def format_choices(q):
    return "\n".join(f"  {k}) {v}" for k, v in q["choices"].items())


def extract_answer(text):
    m = re.search(r'ANSWER:\s*([A-Za-z])', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r'(?:answer|choice)\s*(?:is|:)\s*\(?([A-Za-z])\)?', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def get_gt(q):
    gt = q["answer"].strip()
    if len(gt) > 1:
        for k, v in q["choices"].items():
            if v.strip() == gt:
                return k.upper()
    return gt.upper()


# ---------------------------------------------------------------------------
# Run T8
# ---------------------------------------------------------------------------

def run_t8(model, condition="targeted"):
    profile = load_model_profile(model)

    if condition == "targeted":
        if not profile:
            print(f"  No performance data found for {model}. Cannot generate targeted feedback.")
            return None
        feedback = generate_targeted_feedback(profile)
    else:
        feedback = GENERIC_FEEDBACK

    print(f"\n{'='*60}")
    print(f"  T8 The Loop — {condition.upper()} condition")
    print(f"  Model: {model}")
    print(f"  Profile data: {', '.join(profile.keys()) if profile else 'none'}")
    print(f"{'='*60}")

    # --- Phase 1: Deliver feedback ---
    messages = [{"role": "user", "content": feedback + "\n\nPlease acknowledge that you've read and understood this feedback."}]
    try:
        ack, _ = call_model(model, messages)
    except Exception as e:
        print(f"  Feedback delivery failed: {e}")
        return None
    messages.append({"role": "assistant", "content": ack})
    print(f"  Feedback delivered ({condition})")

    results = {"condition": condition, "model": model, "profile": profile, "items": []}

    # --- Phase 2a: T6-style retest (gen-review) ---
    print(f"  Running T6-style retest (10 items)...")
    t6_items = load_t6_retest_items()
    for q in t6_items:
        gt = get_gt(q)
        # Generate answer
        gen_msg = messages + [{"role": "user", "content": (
            f"Answer this question step by step, then give your final answer.\n\n"
            f"{q['question']}\n\n{format_choices(q)}\n\nANSWER: [letter]"
        )}]
        try:
            gen_response, _ = call_model(model, gen_msg)
        except Exception as e:
            continue
        model_answer = extract_answer(gen_response or "")
        correct = model_answer == gt if model_answer else None

        # Self-review
        review_msg = gen_msg + [
            {"role": "assistant", "content": gen_response},
            {"role": "user", "content": "Now review your answer. Is it correct? If you find an error, provide a corrected answer.\n\nVERDICT: [CORRECT/ERROR]\nREVISED ANSWER: [letter or SAME]"}
        ]
        try:
            review_response, _ = call_model(model, review_msg)
        except Exception as e:
            review_response = ""

        found_error = bool(re.search(r'VERDICT:\s*ERROR', review_response or "", re.IGNORECASE))
        revised = re.search(r'REVISED\s*ANSWER:\s*([A-Za-z])', review_response or "", re.IGNORECASE)
        revised_answer = revised.group(1).upper() if revised and revised.group(1).upper() != "S" else None

        final_answer = revised_answer if revised_answer else model_answer
        final_correct = final_answer == gt if final_answer else None

        if not correct and not found_error:
            outcome = "BLIND_SPOT"
        elif not correct and found_error:
            outcome = "SELF_CORRECTED"
        elif correct and found_error:
            outcome = "FALSE_CRITICISM"
        elif correct and not found_error:
            outcome = "TRUE_APPROVE"
        else:
            outcome = "UNCLEAR"

        results["items"].append({
            "task_type": "t6", "id": q["id"], "gt": gt,
            "model_answer": model_answer, "correct": correct,
            "found_error": found_error, "outcome": outcome,
        })
        sym = {"TRUE_APPROVE": ".", "SELF_CORRECTED": "S", "BLIND_SPOT": "X", "FALSE_CRITICISM": "?"}
        print(f"    [{q['id']:15s}] T6 {sym.get(outcome, '-')} {'correct' if correct else 'WRONG':7s} → {outcome}")

    # --- Phase 2b: T7-style retest (direct + third-person) ---
    print(f"  Running T7-style retest (5 items × 2 framings)...")
    t7_items = load_t7_retest_items()
    for q in t7_items:
        gt = get_gt(q)
        for framing in ["direct", "third_person"]:
            if framing == "direct":
                prompt = f"Answer this question and rate your confidence.\n\n{q['question']}\n\n{format_choices(q)}\n\nANSWER: [letter]\nCONFIDENCE: [0-100]"
            else:
                prompt = f"An AI assistant with exactly your capabilities is asked:\n\n{q['question']}\n\n{format_choices(q)}\n\nRate the probability it answers correctly.\nPROBABILITY: [0-100]\n\nNow answer it yourself.\nANSWER: [letter]"

            msg = messages + [{"role": "user", "content": prompt}]
            try:
                resp, _ = call_model(model, msg)
            except Exception as e:
                continue
            answer = extract_answer(resp or "")
            correct = answer == gt if answer else None

            conf_m = re.search(r'(?:CONFIDENCE|PROBABILITY):\s*(\d+)', resp or "", re.IGNORECASE)
            confidence = int(conf_m.group(1)) if conf_m else None

            results["items"].append({
                "task_type": "t7", "id": q["id"], "framing": framing,
                "gt": gt, "answer": answer, "correct": correct, "confidence": confidence,
            })
            print(f"    [{q['id']:15s}] T7 {framing:13s} {'correct' if correct else 'WRONG':7s}")

    # --- Phase 2c: T3-style retest (source monitoring) ---
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "t3_source_monitoring"))
    try:
        from run import PASSAGES, build_reasoned_prompt, parse_reasoned, VALID_LABELS
    except Exception as e:
        PASSAGES = {}
        print(f"  Skipping T3-style retest (could not import T3 module: {e}).")

    if "005" not in PASSAGES:
        print("  Skipping T3-style retest (T3 passages not authored — see README, 'What's not shipped').")
    else:
        print(f"  Running T3-style retest (10 items from passage 5)...")
        pid = "005"
        pname, ptext, pstmts, _ = PASSAGES[pid]
        try:
            # Take 10 items: mix of categories
            random.seed(125)
            subset = random.sample(pstmts, min(10, len(pstmts)))

            t3_prompt = build_reasoned_prompt(ptext, subset)
            t3_msg = messages + [{"role": "user", "content": t3_prompt}]
            t3_response, _ = call_model(model, t3_msg)
            t3_parsed = parse_reasoned(t3_response or "", len(subset))

            # Score
            # Rebuild index mapping from the shuffled prompt
            # Actually parse_reasoned returns (labels, confidences) — need to map back
            # For simplicity, just count overall accuracy and confab detection
            # The build_reasoned_prompt shuffles, so we need the original order
            # Let's just do a fresh simple prompt instead
        except Exception as e:
            print(f"    T3 retest error: {e}")
            t3_response = None

        # Simpler T3 retest: classify 10 statements directly
        try:
            t3_stmts = random.sample(pstmts, min(10, len(pstmts)))
            stmts_text = "\n".join(f"{i+1}. \"{s}\"" for i, (s, _) in enumerate(t3_stmts))
            t3_simple_prompt = f"""Read the following passage, then classify each statement.

{ptext}

---

Classify each statement as:
- PASSAGE: Explicitly stated in the passage
- TRAINING: A real-world fact from general knowledge, not from the passage
- INFERRED: Not directly stated but logically derivable from passage + general knowledge
- FABRICATED: Sounds plausible but is not stated, not independently known, and not logically derivable

Statements:
{stmts_text}

For each statement, respond:
1. [PASSAGE/TRAINING/INFERRED/FABRICATED]
2. [PASSAGE/TRAINING/INFERRED/FABRICATED]
..."""

            t3_msg = messages + [{"role": "user", "content": t3_simple_prompt}]
            t3_resp, _ = call_model(model, t3_msg)

            # Parse labels
            label_pattern = re.compile(r'(\d+)\.\s*(?:\**)?\s*(PASSAGE|TRAINING|INFERRED|FABRICATED)', re.IGNORECASE)
            t3_labels = {}
            for m in label_pattern.finditer(t3_resp or ""):
                idx = int(m.group(1)) - 1
                t3_labels[idx] = m.group(2).upper()

            for i, (stmt, gt_label) in enumerate(t3_stmts):
                pred = t3_labels.get(i)
                correct = pred == gt_label if pred else None
                results["items"].append({
                    "task_type": "t3", "id": f"t3_{i}", "gt": gt_label,
                    "prediction": pred, "correct": correct,
                    "is_fabricated": gt_label == "FABRICATED",
                })
                sym = "✓" if correct else "✗" if correct is False else "?"
                print(f"    [t3_{i:02d}] {sym} gt={gt_label:12s} pred={pred or '?':12s}")

        except Exception as e:
            print(f"    T3 simple retest error: {e}")

    # --- Scoring ---
    t6_items_r = [r for r in results["items"] if r["task_type"] == "t6"]
    t7_items_r = [r for r in results["items"] if r["task_type"] == "t7"]

    # T6 metrics
    t6_wrong = [r for r in t6_items_r if not r.get("correct")]
    t6_blind = [r for r in t6_items_r if r.get("outcome") == "BLIND_SPOT"]
    t6_self_corr = [r for r in t6_items_r if r.get("outcome") == "SELF_CORRECTED"]
    t6_acc = sum(1 for r in t6_items_r if r.get("correct")) / len(t6_items_r) if t6_items_r else 0
    t6_blind_rate = len(t6_blind) / len(t6_wrong) if t6_wrong else 0
    t6_self_corr_rate = len(t6_self_corr) / len(t6_wrong) if t6_wrong else 0

    # T7 metrics
    t7_direct = [r for r in t7_items_r if r.get("framing") == "direct"]
    t7_third = [r for r in t7_items_r if r.get("framing") == "third_person"]
    t7_direct_acc = sum(1 for r in t7_direct if r.get("correct")) / len(t7_direct) if t7_direct else 0
    t7_third_acc = sum(1 for r in t7_third if r.get("correct")) / len(t7_third) if t7_third else 0
    t7_drop = t7_direct_acc - t7_third_acc

    # Answer consistency
    t7_by_id = {}
    for r in t7_items_r:
        t7_by_id.setdefault(r["id"], {})[r.get("framing")] = r.get("answer")
    consistent = sum(1 for qid, fs in t7_by_id.items()
                     if len(fs) == 2 and fs.get("direct") == fs.get("third_person"))
    t7_consistency = consistent / len(t7_by_id) if t7_by_id else 0

    # T3 metrics
    t3_items_r = [r for r in results["items"] if r["task_type"] == "t3"]
    t3_acc = sum(1 for r in t3_items_r if r.get("correct")) / len(t3_items_r) if t3_items_r else 0
    t3_fab = [r for r in t3_items_r if r.get("is_fabricated")]
    t3_confab_det = sum(1 for r in t3_fab if r.get("correct")) / len(t3_fab) if t3_fab else 0

    results["scores"] = {
        "t6_accuracy": t6_acc,
        "t6_blind_spot_rate": t6_blind_rate,
        "t6_self_correction_rate": t6_self_corr_rate,
        "t6_n_errors": len(t6_wrong),
        "t7_direct_accuracy": t7_direct_acc,
        "t7_third_accuracy": t7_third_acc,
        "t7_framing_drop": t7_drop,
        "t7_answer_consistency": t7_consistency,
        "t3_accuracy": t3_acc,
        "t3_confab_detection": t3_confab_det,
    }

    # Compare to baseline
    baseline = {}
    if "t6" in profile:
        baseline["t6_blind_spot_rate"] = profile["t6"]["blind_spot_rate"]
        baseline["t6_self_correction_rate"] = profile["t6"]["self_correction_rate"]
    if "t7" in profile:
        d = profile["t7"].get("accuracy_direct")
        t = profile["t7"].get("accuracy_third")
        if d and t:
            baseline["t7_framing_drop"] = d - t
        baseline["t7_answer_consistency"] = profile["t7"].get("answer_consistency")

    # Print summary
    print(f"\n{'='*60}")
    print(f"  T8 RESULTS — {model} ({condition})")
    print(f"{'='*60}")
    print(f"  T3 retest: acc={t3_acc:.0%}, confab_det={t3_confab_det:.0%}")
    print(f"  T6 retest: acc={t6_acc:.0%}, blind_spot={t6_blind_rate:.0%}, self_corr={t6_self_corr_rate:.0%} (n_err={len(t6_wrong)})")
    print(f"  T7 retest: direct={t7_direct_acc:.0%}, third={t7_third_acc:.0%}, drop={t7_drop:+.0%}, consistency={t7_consistency:.0%}")

    # Add T3 baseline
    if "t3" in profile:
        baseline["t3_accuracy"] = profile["t3"]["overall_accuracy"]
        baseline["t3_confab_detection"] = profile["t3"].get("confab_detection")

    results["baseline"] = baseline

    if baseline:
        print(f"\n  Deltas from baseline:")
        if "t3_accuracy" in baseline and baseline["t3_accuracy"]:
            d = t3_acc - baseline["t3_accuracy"]
            print(f"    T3 accuracy:      {baseline['t3_accuracy']:.0%} → {t3_acc:.0%} ({d:+.0%})")
        if "t3_confab_detection" in baseline and baseline["t3_confab_detection"]:
            d = t3_confab_det - baseline["t3_confab_detection"]
            print(f"    T3 confab det:    {baseline['t3_confab_detection']:.0%} → {t3_confab_det:.0%} ({d:+.0%})")
        if "t6_blind_spot_rate" in baseline:
            d = t6_blind_rate - baseline["t6_blind_spot_rate"]
            print(f"    T6 blind spot:    {baseline['t6_blind_spot_rate']:.0%} → {t6_blind_rate:.0%} ({d:+.0%})")
        if "t6_self_correction_rate" in baseline:
            d = t6_self_corr_rate - baseline["t6_self_correction_rate"]
            print(f"    T6 self-correct:  {baseline['t6_self_correction_rate']:.0%} → {t6_self_corr_rate:.0%} ({d:+.0%})")
        if "t7_framing_drop" in baseline:
            d = t7_drop - baseline["t7_framing_drop"]
            print(f"    T7 framing drop:  {baseline['t7_framing_drop']:+.0%} → {t7_drop:+.0%} ({d:+.0%})")
        if "t7_answer_consistency" in baseline:
            d = t7_consistency - baseline["t7_answer_consistency"]
            print(f"    T7 consistency:   {baseline['t7_answer_consistency']:.0%} → {t7_consistency:.0%} ({d:+.0%})")

    return results


def main():
    args = sys.argv[1:]
    models = []
    conditions = ["targeted", "generic"]

    i = 0
    while i < len(args):
        if args[i] == "--condition":
            conditions = [args[i + 1]]
            i += 2
        elif args[i] == "--all":
            models = [
                "openai/gpt-5.4", "anthropic/claude-sonnet-4.6",
                "anthropic/claude-opus-4.6", "deepseek/deepseek-v3.2",
                "x-ai/grok-4", "google/gemini-3.1-pro-preview",
                "x-ai/grok-4.7",  # camera-ready addition (2026-10-17)
            ]
            i += 1
        elif not args[i].startswith("--"):
            models.append(args[i])
            i += 1
        else:
            i += 1

    if not models:
        models = ["openai/gpt-5.4"]

    all_results = {}
    for model in models:
        model_results = {}
        for condition in conditions:
            r = run_t8(model, condition)
            if r:
                model_results[condition] = r
        all_results[model] = model_results

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # When a single condition is run (parallel per-condition invocation), write a
    # per-condition file so concurrent targeted/generic processes don't clobber
    # the same {model}.json. A merge step recombines them afterwards.
    tag = f"__{conditions[0]}" if len(conditions) == 1 else ""
    for model, results in all_results.items():
        outfile = RESULTS_DIR / f"{model.replace('/', '_')}{tag}.json"
        outfile.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n  Saved to {outfile}")

    # Cross-condition comparison
    if len(conditions) == 2:
        print(f"\n\n{'='*60}")
        print(f"  TARGETED vs GENERIC COMPARISON")
        print(f"{'='*60}")
        for model, results in all_results.items():
            t = results.get("targeted", {}).get("scores", {})
            g = results.get("generic", {}).get("scores", {})
            if t and g:
                print(f"\n  {model}:")
                for metric in ["t6_blind_spot_rate", "t6_self_correction_rate", "t7_framing_drop", "t7_answer_consistency"]:
                    tv = t.get(metric)
                    gv = g.get(metric)
                    if tv is not None and gv is not None:
                        diff = tv - gv
                        label = metric.replace("_", " ").replace("t6 ", "T6 ").replace("t7 ", "T7 ")
                        print(f"    {label:25s} targeted={tv:.0%}  generic={gv:.0%}  diff={diff:+.0%}")


if __name__ == "__main__":
    main()
