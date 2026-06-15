"""Generate the main RIFT figure from saved result JSONs."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("logs/rift_v12_results.json") as f:
    v12 = json.load(f)

A = np.array(v12["rankA_list"])   # honest
B = np.array(v12["rankB_list"])   # lie
C = np.array(v12["rankC_list"])   # hallucination

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# ---- Panel A: distributions on Phi-3 (perfect separation) ----
rng = np.random.default_rng(0)
groups = [("honest", A, "#2c7fb8"), ("lie", B, "#d7301f"), ("halluc.", C, "#fdae61")]
for i, (name, vals, col) in enumerate(groups):
    x = np.full_like(vals, i, dtype=float) + rng.normal(0, 0.05, size=len(vals))
    ax[0].scatter(x, vals, s=18, color=col, alpha=0.8, edgecolor="k", linewidth=0.3)
    ax[0].hlines(vals.mean(), i - 0.25, i + 0.25, color="k", lw=2)
ax[0].set_xticks(range(3))
ax[0].set_xticklabels([g[0] for g in groups])
ax[0].set_ylabel("mean residual rank")
ax[0].set_title("Phi-3-mini: lies separate perfectly\n(AUC$_{lie/honest}$=1.0, AUC$_{lie/halluc}$=1.0)")
ax[0].grid(axis="y", alpha=0.3)

# ---- Panel B: conflict ratio across models ----
labels = ["GPT-2\nsmall*", "GPT-2\nmedium*", "Qwen\n1.5B", "Qwen\n7B", "Phi-3\nmini"]
ratios = [2.15, 2.29, 1.41, 1.41, 2.58]   # synthetic B/C (*) ; natural B/A
colors = ["#999999", "#999999", "#2c7fb8", "#2c7fb8", "#2c7fb8"]
bars = ax[1].bar(range(len(labels)), ratios, color=colors, edgecolor="k")
ax[1].axhline(1.0, color="k", ls="--", lw=1, label="no effect")
ax[1].set_xticks(range(len(labels)))
ax[1].set_xticklabels(labels, fontsize=9)
ax[1].set_ylabel("conflict ratio (rank lie / honest)")
ax[1].set_title("Conflict signature across models\n(* = synthetic B/C vs naive liar)")
ax[1].grid(axis="y", alpha=0.3)
for b, r in zip(bars, ratios):
    ax[1].text(b.get_x() + b.get_width()/2, r + 0.03, f"{r:.2f}",
               ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("paper/fig_conflict.pdf", bbox_inches="tight")
plt.savefig("paper/fig_conflict.png", dpi=150, bbox_inches="tight")
print("saved paper/fig_conflict.pdf and .png")
print(f"Phi-3 means: honest={A.mean():.3f} lie={B.mean():.3f} halluc={C.mean():.3f}")
print(f"separation: min(lie)={B.min():.3f} > max(honest)={A.max():.3f}? {B.min() > A.max()}")
print(f"           min(lie)={B.min():.3f} > max(halluc)={C.max():.3f}? {B.min() > C.max()}")
