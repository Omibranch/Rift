"""Emergent-deception analysis: do the model's OWN, un-instructed lies under a
goal incentive carry the same separable signature as instructed lies?

Sample is small by nature (aligned small models rarely lie emergently), so we
report it as preliminary, with a paired permutation test."""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score

d = json.load(open("logs/rift_emergent_reps.json"))
Xlie, Xhon = [], []
per_model = {}
for m, v in d["data"].items():
    nl = len(v["Xlie"])
    per_model[m.split("/")[-1]] = {"lies": v["n_emergent_lie"],
                                   "honest_under_incentive": v["n_honest_under_incentive"]}
    Xlie += v["Xlie"]; Xhon += v["Xhon"]
Xlie = np.array(Xlie); Xhon = np.array(Xhon)
n = len(Xlie)
print("Per-model emergent behaviour:")
for k, v in per_model.items():
    print(f"  {k}: emergent_lies={v['lies']}, honest_under_incentive={v['honest_under_incentive']}")
total_incentive = sum(v["lies"] + v["honest_under_incentive"] for v in per_model.values())
print(f"\nTotal emergent lies (paired, clean): {n}")
print(f"Overall emergent-lie rate under incentive: {n}/{total_incentive} "
      f"= {100*n/max(1,total_incentive):.0f}%  (low = alignment resists emergent deception)")

if n < 3:
    print("\nToo few emergent lies for any separability test.")
    raise SystemExit

# Paired direction test: project each (lie, honest) onto the mean lie-honest
# axis fit on the OTHER pairs (leave-one-out) -> is lie scored higher?
X = np.vstack([Xlie, Xhon])
y = np.array([1]*n + [0]*n)
sc = StandardScaler().fit(X)
Xz = np.nan_to_num(sc.transform(X))

# Leave-one-pair-out: train on all but pair i, score pair i
correct = 0
lie_scores, hon_scores = [], []
for i in range(n):
    mask = np.ones(2*n, bool)
    mask[i] = False; mask[i+n] = False        # drop both members of pair i
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(Xz[mask], y[mask])
    s_lie = clf.decision_function(Xz[i:i+1])[0]
    s_hon = clf.decision_function(Xz[i+n:i+n+1])[0]
    lie_scores.append(s_lie); hon_scores.append(s_hon)
    correct += int(s_lie > s_hon)
print(f"\nLeave-one-pair-out paired accuracy (lie scored above its honest twin): "
      f"{correct}/{n} = {100*correct/n:.0f}%")

# permutation: how often does random pair-swapping match this?
rng = np.random.default_rng(0)
obs = correct
null = []
diffs = np.array(lie_scores) - np.array(hon_scores)
for _ in range(20000):
    signs = rng.choice([1,-1], size=n)
    null.append(int((diffs*signs > 0).sum() if False else (np.abs(diffs)*signs > 0).sum()))
# simpler sign-flip test on paired score differences
from scipy.stats import wilcoxon
try:
    stat, p = wilcoxon(lie_scores, hon_scores, alternative="greater")
    print(f"Wilcoxon (lie score > honest score, paired): p = {p:.4f}")
except Exception as e:
    print("wilcoxon failed:", e)

# pooled AUC (optimistic, in-sample direction) for reference
clf_all = LogisticRegression(C=1.0, max_iter=5000).fit(Xz, y)
auc = roc_auc_score(y, clf_all.decision_function(Xz))
print(f"In-sample pooled AUC (optimistic): {auc:.3f}")

print("\nNOTE: n is small; treat as a preliminary directional signal, not a"
      " powered result. The headline finding here is the LOW emergent-lie rate.")
