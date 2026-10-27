"""
v2 numbers for the camera-ready's 12 models (October 2026).

Reuses scripts/paper_numbers_v2.py's per-model and pooled computations unchanged, restricted to
the camera-ready lineup (the 10 original models, Grok 4.7, Qwen 3.8 Max), so pooled statistics
cover those 12 only. Writes no tables.

    .venv/bin/python <camera_ready>/scripts/numbers_12.py   -> results_v2_metrics/camera_ready_12.json
"""
import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ / "scripts"))
import paper_numbers_v2 as pn  # noqa: E402

MODELS = ["sonnet-4", "sonnet-4.6", "opus-4.6", "gpt-4.1", "gpt-5.4", "gemini-2.5-pro", "gemini-3.1-pro",
          "gemma-4-31b", "deepseek-v3.2", "mistral-large", "grok-4.7", "qwen-3.8-max"]


def main():
    per = {m: {"wagering": pn.wagering(m), "deferral": pn.deferral(m), "consistency": pn.consistency(m),
               "self_review": pn.self_review(m), "loop": pn.loop(m), "switching": pn.switching(m),
               "source_monitoring": pn.source_monitoring(m)} for m in MODELS}
    pool = pn.pooled(per)
    for v in per.values():
        v["switching"].pop("rows")
        if v["loop"]:
            v["loop"].pop("item_rows")
    (pn.M / "camera_ready_12.json").write_text(json.dumps({"models": per, "pooled": pool}, indent=1))
    print(json.dumps(pool, indent=1))


if __name__ == "__main__":
    main()
