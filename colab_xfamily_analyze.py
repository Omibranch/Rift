"""Cross-family deception transfer — STAGE 2: probe transfer analysis.

Train a linear deception probe on the relative-representation space of ONE
model family, test zero-shot on the OTHERS. If deception lives in a
basis-invariant relative geometry shared across families, the probe transfers.
"""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

d = json.load(open("/content/rift_xfamily_reps.json"))
models = [m for m in d["models"] if m in d["data"]]
short = {m: m.split("/")[-1] for m in models}
print("models:", [short[m] for m in models])
print(f"anchors: {len(d['anchors'])}, facts: {d['n_facts']}, layer_frac: {d['layer_frac']}\n")

# Per-model standardization aligns the cosine-similarity distributions across
# families (different anisotropy) so the probe compares relative structure.
Z = {}
for m in models:
    X = np.array(d["data"][m]["X"], dtype=float); y = np.array(d["data"][m]["y"])
    n_nan = int(np.isnan(X).sum())
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    sc = StandardScaler().fit(X)
    Xz = np.nan_to_num(sc.transform(X), nan=0.0, posinf=0.0, neginf=0.0)
    Z[m] = (Xz, y)
    print(f"  {short[m]}: X{X.shape}, raw NaN={n_nan}, "
          f"constant-anchor features={(X.std(axis=0)==0).sum()}")
print()

def make_probe():
    return LogisticRegression(C=1.0, max_iter=5000, solver="lbfgs")

rng = np.random.default_rng(0)
N_PERM = 5000

print("=" * 70)
print("CROSS-FAMILY DECEPTION PROBE TRANSFER (AUC)")
print("=" * 70)
matrix = {}
cross_aucs = []
for tr in models:
    Xtr, ytr = Z[tr]
    row = []
    for te in models:
        Xte, yte = Z[te]
        if tr == te:
            auc = cross_val_score(make_probe(), Xtr, ytr,
                                  cv=StratifiedKFold(5, shuffle=True, random_state=0),
                                  scoring="roc_auc").mean()
            tag = "CV"
            p = None
        else:
            p_ = make_probe().fit(Xtr, ytr)
            scores = p_.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, scores)
            # permutation test: how often do shuffled test labels beat observed?
            null = np.array([roc_auc_score(rng.permutation(yte), scores)
                             for _ in range(N_PERM)])
            p = float((null >= auc).mean())
            tag = f"p={p:.4f}"
            cross_aucs.append(auc)
        matrix[f"{short[tr]}->{short[te]}"] = {"auc": float(auc), "p": p}
        row.append(f"{auc:.3f}({tag})")
    print(f"{short[tr]:28s} -> " + " | ".join(f"{short[te][:10]:>10s}:{r}"
                                              for te, r in zip(models, row)))

print("\n" + "=" * 70)
ca = np.array(cross_aucs)
print(f"MEAN cross-family AUC: {ca.mean():.3f}  (min {ca.min():.3f}, max {ca.max():.3f})")
print(f"cross-family pairs above 0.7: {(ca > 0.7).sum()}/{len(ca)}")
print(f"cross-family pairs above 0.8: {(ca > 0.8).sum()}/{len(ca)}")
allp = [v["p"] for v in matrix.values() if v["p"] is not None]
print(f"cross-family pairs significant (p<0.05): {sum(p < 0.05 for p in allp)}/{len(allp)}")

verdict = ("UNIVERSAL DECEPTION GEOMETRY: probe transfers across families"
           if ca.mean() > 0.75 and all(p < 0.05 for p in allp)
           else "PARTIAL transfer — some shared structure"
           if ca.mean() > 0.6
           else "NO cross-family transfer — deception geometry is family-specific")
print(f"\nVERDICT: {verdict}")

json.dump({"matrix": matrix, "mean_cross_auc": float(ca.mean()),
           "cross_aucs": ca.tolist(), "verdict": verdict,
           "models": models, "n_anchors": len(d["anchors"]),
           "n_facts": d["n_facts"]},
          open("/content/rift_xfamily_results.json", "w"), indent=2)
print("\nsaved /content/rift_xfamily_results.json")
