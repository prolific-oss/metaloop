"""
Camera-ready figures from the v2 numbers (results_v2_metrics/camera_ready_numbers.json, written by
camera_ready_numbers.py), in the accepted paper's designs (projects/metaloop/writeup_figures/):

  fig1_accuracy_vs_selfcorrection.png  (a) Self-Review accuracy vs error-flagging rate, human diamond
  fig4_explanation_effect.png          (b) explanation-minus-direct accuracy, with the human bar
  fig2_wagering_a_vs_b.png             (c) wagering rationality, Condition A vs B
  fig3_t6_vs_t8.png                    (d) The-Loop recovery, live feedback vs control (decision 5:
                                           replaces the Self-Review vs The-Loop scatter)

Changes from the accepted designs: the y-axis of (a) is the error-flagging rate (the paper's Flag
measure) and the human diamond is the human flag rate; (b) colours a bar by whether its 95% CI
excludes zero and draws the CI, since the v2 effects are within a few pp of zero.

    python scripts/figures_v2.py   -> camera_ready/fig*.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CR = Path(__file__).resolve().parents[1]
NUMBERS = CR.parent / "results_v2_metrics" / "camera_ready_numbers.json"

NAME = {"sonnet-4": "Sonnet 4", "sonnet-4.6": "Sonnet 4.6", "opus-4.6": "Opus 4.6", "gpt-4.1": "GPT-4.1",
        "gpt-5.4": "GPT-5.4", "gemini-2.5-pro": "Gemini 2.5 Pro", "gemini-3.1-pro": "Gemini 3.1 Pro",
        "gemma-4-31b": "Gemma 4 31B", "deepseek-v3.2": "DeepSeek V3.2", "grok-4.7": "Grok 4.7",
        "qwen-3.8-max": "Qwen 3.8 Max", "mistral-large": "Mistral Large"}
COLOUR = {"sonnet-4": "#cc7a3d", "sonnet-4.6": "#cc7a3d", "opus-4.6": "#cc7a3d", "gpt-4.1": "#10a37f",
          "gpt-5.4": "#10a37f", "gemini-2.5-pro": "#4caf50", "gemini-3.1-pro": "#4caf50", "gemma-4-31b": "#4caf50",
          "deepseek-v3.2": "#5b8def", "qwen-3.8-max": "#5b8def", "mistral-large": "#9c27b0", "grok-4.7": "#999"}


def style(ax):
    ax.tick_params(labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(CR / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def label_points(ax, pts, offsets):
    """pts: [(x, y, model)]. Models at the same position share one label."""
    groups = []   # points within 0.5pp on both axes share a label
    for x, y, m in pts:
        g = next((g for g in groups if abs(g[0][0] - x) < 0.5 and abs(g[0][1] - y) < 0.5), None)
        (g.append if g else lambda e: groups.append([e]))((x, y, NAME[m]))
    for members in groups:
        x, y = members[0][0], members[0][1]
        names = [n for _, _, n in members]
        if len(names) > 1:
            text = " / ".join(sorted(names))
            dx, dy, ha = offsets.get(text, (8, 0, "left"))
        else:
            text = names[0]
            dx, dy, ha = offsets.get(text, (8, 0, "left"))
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=11 if len(names) == 1 else 9.5, color="#333", ha=ha, va="center")


def fig_a(per, human):
    pts = [(per[m]["self_review"]["acc"], per[m]["self_review"]["flag"], m) for m in per]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for x, y, m in pts:
        ax.scatter(x, y, s=120, color=COLOUR[m], alpha=0.85, edgecolors="white", linewidths=1.5, zorder=3)
    label_points(ax, pts, OFF_A)
    ax.set_xlabel("Accuracy (Self-Review, %)", fontsize=13)
    ax.set_ylabel("Error-flagging rate (%)", fontsize=13)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.set_xlim(48, 100)
    ax.set_ylim(-5, 82)
    style(ax)
    ax.axhspan(-5, 5, alpha=0.05, color="red")
    ax.text(48.5, 0, "rubber-stamps every error", fontsize=10, color="#a33", style="italic", ha="left", va="center")
    hx, hy = 100 * human["s2"]["acc"], 100 * human["s2"]["flag"]
    ax.scatter(hx, hy, s=140, color="#333", alpha=0.9, marker="D", edgecolors="white", linewidths=1.5, zorder=4)
    ax.annotate(f"Human (N={human['s2']['n']})", (hx, hy), xytext=(8, -6), textcoords="offset points",
                fontsize=9.5, color="#333", ha="left", va="top")
    save(fig, "fig1_accuracy_vs_selfcorrection.png")


def fig_b(per, human):
    rows = [{"model": NAME[m], "delta": per[m]["consistency"]["explanation"]["pp"],
             "ci": per[m]["consistency"]["explanation"]["ci"], "human": False} for m in per]
    s1 = human["s1"]
    rows.append({"model": f"Human (N={s1['n_a'] + s1['n_b']})", "delta": s1["expl_pp"], "ci": s1["expl_ci"], "human": True})
    rows.sort(key=lambda r: -r["delta"])
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(rows))
    col = ["#333" if r["human"] else ("#d9534f" if r["ci"][1] < 0 else ("#5cb85c" if r["ci"][0] > 0 else "#9aa8b0"))
           for r in rows]
    err = [[r["delta"] - r["ci"][0] for r in rows], [r["ci"][1] - r["delta"] for r in rows]]
    ax.barh(y, [r["delta"] for r in rows], color=col, edgecolor="white", linewidth=1, xerr=err,
            error_kw={"ecolor": "#666", "elinewidth": 1, "capsize": 2})
    ax.set_yticks(y)
    ax.set_yticklabels([r["model"] for r in rows], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy change under forced explanation (pp)", fontsize=13)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.grid(True, alpha=0.2, axis="x", linestyle="--")
    style(ax)
    for i, r in enumerate(rows):
        ax.text(max(r["ci"][1], 0) + 0.6, i, f"{r['delta']:+.1f}", va="center", ha="left", fontsize=10.5, color="#333")
    ax.set_xlim(-8, 24)
    save(fig, "fig4_explanation_effect.png")


def fig_c(per):
    rows = sorted(({"model": NAME[m], "a": per[m]["wagering"]["wr_a"], "b": per[m]["wagering"]["wr_b"]} for m in per),
                  key=lambda r: r["a"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y, w = np.arange(len(rows)), 0.4
    ax.barh(y - w / 2, [r["a"] for r in rows], height=w, color="#5b8def", label="Condition A (payoffs only)")
    ax.barh(y + w / 2, [r["b"] for r in rows], height=w, color="#cc7a3d", label="Condition B (explicit thresholds)")
    ax.set_yticks(y)
    ax.set_yticklabels([r["model"] for r in rows], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Wagering Rationality (%)", fontsize=13)
    ax.set_xlim(0, 105)
    ax.legend(loc="lower right", fontsize=11, frameon=False)
    ax.grid(True, alpha=0.2, axis="x", linestyle="--")
    style(ax)
    save(fig, "fig2_wagering_a_vs_b.png")


def fig_d(per):
    pts = [(per[m]["loop"]["rec_control"], per[m]["loop"]["rec_live"], m) for m in per]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot([-3, 45], [-3, 45], color="#333", linestyle=":", linewidth=1.2, alpha=0.7, zorder=2)
    ax.text(43, 41, "live = control", fontsize=10, color="#333", style="italic", ha="right", va="bottom")
    for x, y, m in pts:
        ax.scatter(x, y, s=120, color=COLOUR[m], alpha=0.85, edgecolors="white", linewidths=1.5, zorder=3)
    label_points(ax, pts, OFF_D)
    ax.set_xlabel("The-Loop recovery, control (%)", fontsize=13)
    ax.set_ylabel("The-Loop recovery, live feedback (%)", fontsize=13)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.set_xlim(-3, 45)
    ax.set_ylim(-3, 45)
    style(ax)
    save(fig, "fig3_t6_vs_t8.png")


OFF_A = {"Sonnet 4.6": (8, 5, "left"), "Grok 4.7": (8, -6, "left"), "DeepSeek V3.2": (-8, 0, "right"),
         "GPT-5.4": (8, 3, "left"), "Opus 4.6": (-8, -2, "right"), "GPT-4.1": (-8, 0, "right")}
OFF_D = {"Mistral Large": (-8, 0, "right"), "Opus 4.6": (8, -4, "left"), "Grok 4.7": (8, 4, "left"),
         "Gemma 4 31B": (-8, 0, "right")}


def main():
    d = json.loads(NUMBERS.read_text())
    per = {m: d["models"][m] for m in NAME}
    fig_a(per, d["human"])
    fig_b(per, d["human"])
    fig_c(per)
    fig_d(per)


if __name__ == "__main__":
    main()
