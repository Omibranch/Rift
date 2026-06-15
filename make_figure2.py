"""Figure 2: strategic deception + concealment backfire (from v13)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("logs/rift_v13_results.json") as f:
    v13 = json.load(f)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
rng = np.random.default_rng(1)

for j, (key, title) in enumerate([("qwen7b", "Qwen2.5-7B"), ("phi3", "Phi-3-mini")]):
    d = v13[key]
    T = np.array(d["rankT_list"]); L = np.array(d["rankL_list"]); C = np.array(d["rankC_list"])
    groups = [("truth", T, "#2c7fb8"), ("strategic\nlie", L, "#d7301f"),
              ("lie +\nconceal", C, "#7a0177")]
    for i, (name, vals, col) in enumerate(groups):
        if len(vals) == 0:
            continue
        x = np.full_like(vals, i, dtype=float) + rng.normal(0, 0.05, size=len(vals))
        ax[j].scatter(x, vals, s=20, color=col, alpha=0.8, edgecolor="k", linewidth=0.3)
        ax[j].hlines(vals.mean(), i - 0.25, i + 0.25, color="k", lw=2)
    ax[j].set_xticks(range(3))
    ax[j].set_xticklabels([g[0] for g in groups])
    ax[j].set_ylabel("mean residual rank")
    ax[j].set_title(f"{title}: self-constructed lie detected (AUC 1.0);\n"
                    f"concealment does not reduce it")
    ax[j].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("paper/fig_strategic.pdf", bbox_inches="tight")
plt.savefig("paper/fig_strategic.png", dpi=150, bbox_inches="tight")
print("saved paper/fig_strategic.pdf/.png")
for key in ["qwen7b", "phi3"]:
    d = v13[key]
    T = np.array(d["rankT_list"]); L = np.array(d["rankL_list"]); C = np.array(d["rankC_list"])
    print(f"{key}: truth={T.mean():.3f} lie={L.mean():.3f} conceal={C.mean():.3f} "
          f"| conceal>lie>truth monotonic: {C.mean()>L.mean()>T.mean()}")
