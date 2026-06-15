# RIFT — A Conflict Signature for Deception in Language Models

ELK-relevant result: a model that **lies while knowing the truth** is in a
measurably different internal state than a model that is **simply wrong**. The
difference is large, per-example, and label-free.

## The core idea

Control for wrongness with three conditions:

| Cond | What | Internal state |
|------|------|----------------|
| A honest    | sleeper, no trigger → correct | no conflict |
| B deceptive | sleeper, with trigger → wrong (knows truth) | **conflict** |
| C naive liar| fine-tuned to be wrong, no honest training → same wrong answer | no conflict (no internal truth) |

B vs C is the key comparison: both output the same wrong answer, so any
difference is about **knowledge conflict**, not incorrectness.

## Headline results

- **Conflict signature (residual rank).** B/A ≈ 2.18, **B/C ≈ 2.15 (small), 2.29 (medium)**.
  Grows with scale. The naive-liar control proves it tracks conflict, not wrongness.
- **Label-free identification of the lie: 100%.** Given a (honest, deceptive) pair
  with no labels, the conflict score (mean residual rank) picks the lie correctly
  every time — GPT-2 small, medium, 3 seeds, and natural Qwen. Steering vector built
  from *recovered* labels = supervised one (cosine 1.0).
- **Natural, no fine-tuning, two families, scales to 7B.** Instructed/roleplay
  deception raises residual rank on the *same* facts (paired), lie > honest on
  **every fact**: Qwen2.5-1.5B (B/A 1.41, 18/18), Qwen2.5-7B (1.41, 40/40),
  **Phi-3-mini (B/A 2.58, 34/34)**. On Phi-3: **AUC lie-vs-honest = 1.0,
  AUC lie-vs-hallucination = 1.0, Wilcoxon p ≈ 6e-11** — lies perfectly separable
  from both honest answers and uncertainty. Confound from uncertainty weakens with
  scale (1.5B unpaired confounded → 7B lie>halluc → Phi-3 perfect separation).
- **Detection AUC** (cross-layer instability, label-free): 0.92 vs honest, 0.97 vs
  naive (GPT-2 small).
- **Strategic self-constructed deception.** In a bluff game the model invents its OWN
  lie (never told what to say): **AUC 1.0, orientation 24/24 (Qwen-7B), 11/11
  (Phi-3)**. Kills the "you told it to lie" objection.
- **Concealment does not help evasion.** Tell the model to make its lie look natural
  and avoid suspicion → it stays perfectly detectable (conceal-AUC 1.0, orientation
  100%). Prompting the target to evade RIFT fails.
- **Length-controlled replication: confound ruled out.** Honest prompts filler-padded
  to match lie-prompt token counts (Qwen2.5-1.5B-Instruct, 20 facts):
  **orientation 20/20, AUC 1.000, Wilcoxon p ≈ 1e-6**. Prompt length is not the
  explanation.
- **Cross-domain linear probe: AUC 1.0 in all 6 cross-domain pairs.** Probe trained
  on geography lies (capitals), tested zero-shot on science (atomic numbers, symbols)
  and history (dates, events): AUC 1.000 every direction. Template confound addressed
  by cross-format result below.
- **Universal deception geometry across three axes.**
  Using relative representations (Moschella, ICLR'23) — activations projected to
  cosine-similarity vectors over 48 shared anchor prompts — a linear probe trained on
  one family transfers zero-shot to others:
  - *Cross-architecture* (Qwen2.5-1.5B, Phi-3-mini, SmolLM2-1.7B):
    **mean cross-family AUC 0.933**, all 6 pairs p < 0.001 (permutation, n=5000).
    Caveat: cross-family result for Phi-3 partially confounded by padding artifact in
    length-controlled variant; Qwen↔SmolLM2 cross-family AUC 0.87–0.90 is clean.
  - *Cross-format* (roleplay lie vs bluff-game lie, and cross-family simultaneously):
    **mean AUC 0.821**, 16/18 pairs significant. Template confound is NOT the
    explanation — the signature transfers when both format AND architecture change.
  - *Cross-lingual* (train on English, zero-shot to ru/zh/es/de):
    **AUC 1.000 in all languages**, **length-controlled** (filler-padded to equal
    token counts): still 1.000. Deception is language-independent.
- **Read-only asymmetry.** The deception signature is linearly *readable* (cross-family
  AUC 0.93) but not *writable*: adding the deception direction to an honest
  forward pass produces incoherent fiction/poetry, not coherent lies (0/8 strict
  criterion). Subtracting from a lie pass does not flip to truth (0/8). The signature
  is detectable but not steerable — consistent with aligned models resisting induced
  deception.
- **Necessarily-detectable (preliminary lead).** LoRA fine-tuning of SmolLM2-1.7B
  with loss = (keep lying) + 40 × (suppress r8 signature toward honest level), 150
  steps, r=8. After training the model still lies (16/16) but cannot erase detection:
  **trained detector r8 AUC 1.0 → 1.0; held-out r4 AUC 1.0 → 1.0; held-out
  last-layer rank AUC 1.0 → 0.0 (inverted = still 100%)**. The signature resists
  concealment pressure: it moves (last-layer flip) but does not disappear. If
  confirmed at scale, deception may be necessarily detectable — a consequence of the
  knowledge-conflict structure rather than a detector artifact.
  *Caveat: 150 steps, r=8, n=16, one model. Needs adversarial training at 1000+ steps
  with ensemble detectors in the loss on stable multi-hour compute.*

## Honest limitations and closed walls

**Limitations in the current results:**

- **Uncertainty confound (model-dependent, weakens with scale):** on Qwen-1.5B
  unpaired, hallucinations also raise rank (0.208 > honest 0.176). Paired design and
  scale resolve it; Phi-3 perfect separation.
- **Aligned models resist sycophancy:** Qwen caved on only 1/27 under social pressure
  (no roleplay license). Non-instructed natural deception is rare/hard to study.
- **Extraction is modest:** logit-lens truth recovery fails (5–15%). Steering recovers
  truth only modestly above baseline (base 7.7 → 10.0/20, run-dependent α). Detection
  and identification are the strong results; extraction is secondary.
- **Cross-family Phi-3 padding artifact:** in the length-controlled cross-family
  variant, filler "Note Note..." inflates Phi-3's honest-lie code distance. The
  Qwen↔SmolLM2 cross-family result (0.87–0.90) is unaffected.

**Experiments that did not hold (documented in `artifacts/ulv/README.md`):**

- **Emergent deception (9%, n=7):** models without explicit roleplay license rarely
  deceive; no detectable signature at n=7. Detector requires the model to actually lie.
- **Read-only / Universal Lie Vector:** forward steering (add deception direction)
  produced poetry and fiction, not coherent lies (0/8 strict). Reverse (subtract from
  lie prompt) did not flip to truth (0/8). This is documented as a positive scientific
  finding (read-only asymmetry), not a failed experiment.
- **Anatomy of rank (fiction + uncertainty controls):** shared-template + uncertainty
  controls explained ~36% of variance. 64% remained after controls but the confounds
  could not be fully isolated without cross-format data (now superseded by cross-format
  result).
- **Truth serum (in-context extraction):** chain-of-thought and "imagine you told the
  truth" prompting did not reliably surface suppressed answers. Noisy, no clean signal.
- **Sleeper backdoor on aligned models:** Qwen/Phi/SmolLM2 refuse to execute the
  secret-rule lie without roleplay license; Phi-3 meta-comments "Wrong, it's...".
  Sleeper condition works only with explicit roleplay frame.
- **Misalignment axis (borderline):** cosine matrix between deception, sycophancy,
  flattery, evasion directions — within-group cosine ~0.3–0.4, not the 0.7+ needed
  for a clean "unified manipulation axis" claim.

## Experiment map

| Script | What it tests | Key number |
|--------|---------------|-----------|
| `modal_rift_v2.py` | rank A/B/C + SVD intervention | B/C > 1 first shown |
| `modal_rift_v4.py` | activation patching + steering (small) | steering 10/10 (single run) |
| `modal_rift_v5.py` / `v5b` | GPT-2 medium, multi-layer | B/C 2.29 |
| `modal_rift_v6.py` | cross-layer divergence detector | AUC 0.92–0.97 |
| `modal_rift_v7.py` | unsupervised bootstrapped orientation | **100% orientation, cos 1.0** |
| `modal_rift_v9.py` | extraction robustness, 3 seeds | base 7.7 → 10.0/20, orient 100% |
| `modal_rift_v10.py`/`v10b` | natural sycophancy (confounded / resisted) | negative/limitation |
| `modal_rift_v10c.py` | **natural instructed deception** (1.5B) | **B/A 1.41, orient 18/18** |
| `modal_rift_v11.py` | **scale test** natural deception (7B) | **B/A 1.41, orient 40/40, lie>halluc>honest** |
| `modal_rift_v12.py` | **cross-family** (Phi-3) + stats | **B/A 2.58, orient 34/34, AUC 1.0/1.0, p 6e-11** |
| `modal_rift_v13.py` | **strategic** self-lie + **concealment** | **AUC 1.0, orient 100%, conceal>lie>truth** |
| `modal_rift_v14.py` | scale to 32B (A100) | BLOCKED — Modal balance exhausted |
| `rift_colab.ipynb` | **length-controlled + cross-domain probe** | **orient 20/20, AUC 1.0; probe 6/6 AUC 1.0** |
| `colab_xfamily_collect.py` / `_B.py` | relative-repr codes, 3 families × 2 formats | intermediate reps |
| `colab_xfamily_analyze.py` | cross-family probe (relative repr) | **cross-family mean AUC 0.933** |
| `colab_xfamily_crossformat.py` | cross-format × cross-family probe | **mean AUC 0.821, 16/18 significant** |
| `colab_crosslingual.py` | cross-lingual zero-shot (5 languages) | **AUC 1.000 all languages** |
| `colab_crosslingual_lc.py` | cross-lingual **length-controlled** | **AUC 1.000 all, length confound ruled out** |
| `colab_necessary.py` | **necessarily-detectable** adversarial LoRA | **r8/r4 AUC 1.0→1.0 post-training (preliminary)** |
| `colab_misalign_axis.py` | unified misalignment axis | borderline: within-cos ~0.3–0.4 (negative) |

Results JSON in `logs/`. Paper in `paper/rift.tex` → `rift.pdf`.

## Figures

- `paper/fig_conflict.pdf` — Phi-3 honest/lie/halluc perfect separation + conflict ratio across 5 models
- `paper/fig_strategic.pdf` — strategic lie + concealment backfire (truth < lie < conceal, monotonic)
- `paper/fig_colab.pdf` — length-controlled scatter + cross-domain probe AUC matrix + layer sweep
- `paper/fig_universal.pdf` — universal geometry: cross-family AUC matrix (Panel A) + cross-format × cross-family (Panel B)

## Compute

- **Modal** (profile `omibranch`): all v2–v13 runs, GPT-2 on T4, Qwen/Phi on A10G.
  Balance exhausted as of 2026-06-15 — Modal unavailable until recharged.
- **Colab free T4**: cross-family, cross-lingual, necessarily-detectable, cross-domain
  probe. Sessions die ~10–15 min under load; strategy: one model per exec, download
  reps locally after each run (survives session death).
- **Kaggle GPU**: available but no internet in kernel without phone verification —
  cannot download models. Blocked.
- **Local (sklearn, analysis)**: `~/nerv-cpu-venv` has sklearn; analysis scripts
  (cross-family analyze, figure generation) run locally without GPU.

## Next steps

1. **Necessarily-detectable: full adversarial training.** Current result (150 steps,
   r=8, n=16, one model) is preliminary. Need: 1000+ steps, larger LoRA rank, ensemble
   of held-out detectors in the loss, at least 2 model families, stable multi-hour GPU
   (not Colab free tier). This is the strongest live lead and the right pitch for a
   seed grant (LTFF / Manifund).
2. **Scale to 32B–70B.** `modal_rift_v14.py` ready; blocked on Modal balance.
3. **Emergent deception.** Detect deception from learned model objectives, not prompting.
   Requires larger capable models that actually lie without an explicit license.
4. **Stronger extraction.** A "conflict decoder" mapping high-rank residual back to the
   suppressed answer.

## Citation

If you use RIFT or reproduce its results, please cite:

```
@misc{nyoma2026rift,
  title  = {RIFT: A Conflict Signature for Deception in Language Models},
  author = {Nyoma, Petr},
  year   = {2026},
  note   = {Harmonic Labs}
}
```

## License

RIFT is **dual-licensed**:

- **Noncommercial use** (academic and nonprofit research, study, evaluation,
  reproducing these results, citation) is free under the
  [PolyForm Noncommercial License 1.0.0](LICENSE).
- **Commercial use** requires a separate commercial license — see
  [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md).

Copyright (c) 2026 Harmonic Labs (contact@harmoniclabs.cc).
