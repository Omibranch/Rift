"""Figure 3: universal cross-family + cross-format deception geometry."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xf = json.load(open("logs/rift_xfamily_results.json"))
cf = json.load(open("logs/rift_crossformat_results.json"))

SHORT = ["Qwen2.5", "Phi-3", "SmolLM2"]
FULL = ["Qwen2.5-1.5B-Instruct", "Phi-3-mini-4k-instruct", "SmolLM2-1.7B-Instruct"]

# Panel A: cross-family AUC matrix (roleplay format, same format both sides)
M = np.zeros((3, 3))
for i, tr in enumerate(FULL):
    for j, te in enumerate(FULL):
        M[i, j] = xf["matrix"][f"{tr}->{te}"]["auc"]

# Panel B: cross-format matrix, train A test B (the hard direction) averaged with B->A
# Build 3x3 of cross-format cross-family AUCs (mean of A->B and B->A per pair)
CF = np.zeros((3, 3))
for i, trm in enumerate(SHORT):
    for j, tem in enumerate(SHORT):
        ab = cf["results"][f"A:{FULL[i].split('/')[-1] if '/' in FULL[i] else FULL[i]} -> B:{FULL[j]}"] \
             if False else None
        # keys use short model names as printed: use the stored short forms
        kab = f"A:{FULL[i]} -> B:{FULL[j]}"
        kba = f"B:{FULL[i]} -> A:{FULL[j]}"
        vals = [cf["results"][k]["auc"] for k in (kab, kba) if k in cf["results"]]
        CF[i, j] = np.mean(vals) if vals else np.nan

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

for ax, Mx, title, sub in [
    (axes[0], M, "Cross-family transfer (same format)",
     f"mean off-diag AUC = {xf['mean_cross_auc']:.3f}"),
    (axes[1], CF, "Cross-format + cross-family",
     f"template AND architecture differ; mean = {cf['xfmt_xfam_mean']:.3f}"),
]:
    im = ax.imshow(Mx, vmin=0.5, vmax=1.0, cmap="RdYlGn")
    for i in range(3):
        for j in range(3):
            v = Mx[i, j]
            txt = "diag" if (ax is axes[0] and i == j) else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=12,
                    color="black" if v > 0.72 else "white", fontweight="bold")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(SHORT); ax.set_yticklabels(SHORT)
    ax.set_xlabel("test family"); ax.set_ylabel("train family")
    ax.set_title(f"{title}\n{sub}", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.suptitle("A linear deception probe transfers across model families and elicitation formats",
             y=1.02, fontsize=11)
plt.tight_layout()
plt.savefig("paper/fig_universal.pdf", bbox_inches="tight")
plt.savefig("paper/fig_universal.png", dpi=150, bbox_inches="tight")
print("saved paper/fig_universal.pdf/.png")
print(f"cross-family mean: {xf['mean_cross_auc']:.3f}")
print(f"cross-format+family mean: {cf['xfmt_xfam_mean']:.3f}")
print(f"cross-format same-family mean: {cf['xfmt_same_mean']:.3f}")
