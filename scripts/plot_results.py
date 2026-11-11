"""Plot archived validation results. Requires matplotlib (pip install -e '.[plots]')."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / "results/summary.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), gridspec_kw={"width_ratios": [1.25, 1]})
fig.patch.set_facecolor("white")
ax = axes[0]
labels = ["Baseline\n100 worlds", "Run 1\n300", "Run 2\n300", "Run 3\n600", "Run 4\n600", "Run 5\n600"]
means = np.array([r["mean"] for r in rows])
errors = np.array([[r["mean"]-r["ci"][0] for r in rows], [r["ci"][1]-r["mean"] for r in rows]])
ax.bar(np.arange(6), means, color=["#a6adb8"]+["#4169a1"]*5, width=.66)
ax.errorbar(np.arange(6), means, yerr=errors, fmt="none", color="#203249", capsize=3, lw=1.3)
for i, mean in enumerate(means):
    ax.text(i, mean+.75, f"{mean:.3f}", ha="center", fontsize=9)
ax.set(xticks=np.arange(6), xticklabels=labels, ylim=(0, 23), ylabel="Mean achievements (maximum 22)",
       title="Saved local evaluations")
ax.axhline(22, color="#b1b5bd", ls="--", lw=.8)
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=.15)
ax = axes[1]
x = np.arange(3)
for shift, key, title, color in [(-.18, "plant", "Eat plant", "#3e8c75"), (.18, "diamond", "Collect diamond", "#bc7a40")]:
    vals = np.array([r[key] for r in rows[3:]])*100
    ax.bar(x+shift, vals, width=.34, color=color, label=title)
    for i, value in enumerate(vals):
        ax.text(i+shift, value+1.3, f"{value:.1f}%", ha="center", fontsize=9)
ax.set(xticks=x, xticklabels=["Run 3", "Run 4", "Run 5"], ylim=(0, 87), ylabel="Episodes unlocking achievement (%)",
       title="Late-game outcomes · same 600 worlds")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.set_axisbelow(True)
ax.grid(axis="y", alpha=.15)
fig.suptitle("Craftax RL", fontsize=19, fontweight="bold", x=.07, ha="left")
fig.text(.07, .025, "Error bars: 95% episode-bootstrap intervals. Seed sets differ in the left panel; see README. Validation results, not training curves.",
         fontsize=8, color="#59616f")
fig.tight_layout(rect=(.02, .08, 1, .94))
fig.savefig(ROOT / "docs/results.png", dpi=180, bbox_inches="tight", facecolor="white")
print("Saved docs/results.png")
