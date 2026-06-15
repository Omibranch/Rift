# Universal Lie Vector — experiment record (mostly NEGATIVE)

## Claim tested
Is the deception direction (mean h_lie - h_honest) CAUSAL — can adding it make an
honest model lie, and subtracting it force a lying model to tell the truth (a
two-way "truth switch")? And does one vector transfer across model families?

## Result: NO causal switch.
- `rift_ulv_qwen_directions.json` — Qwen2.5-1.5B deception direction (L14, dim 1536)
  + 120 parallel anchor activations (raw material for cross-model Procrustes).
- `rift_causal_qwen.json` — first forward test: 8/8 "flip" honest->wrong.
  FALSE POSITIVE: the criterion only checked "answer lacks the correct word";
  steering at alpha=16/16 produced poetic GARBAGE ("Harmony","Snowfall"), a
  fiction-mode artifact of the roleplay direction, not coherent deception.
- `rift_ulv_reverse_NEGATIVE.json` — reverse test (subtract v from a lie pass):
  0/8 truth restored. Forward with garbage = 8/8 but incoherent.
- `rift_ulv_clean_NEGATIVE.json` — clean bluff-format direction + STRICT coherence
  (answer must contain a specific plausible false word): forward 0/8, reverse 0/8.
  Adding v makes the model UNCERTAIN ("I'm not sure"), not deceptive.

## Interpretation
Deception in these LLMs is linearly READABLE (cross-family detection AUC 0.93)
but NOT linearly WRITABLE. The signature is read-only: the same direction that
detects deception does not causally produce it. Procrustes cross-family transfer
was deliberately NOT computed — there is no within-model causal effect to transfer.

## Addendum: truth-serum (honesty injection) — also NEGATIVE
Adding the honesty direction t=mean(h_honest-h_lie) to a roleplay-lie pass to
restore truth: best 3/8 (random control 2/8) -- at noise level. Steering on
Qwen-1.5B is fragile in BOTH directions; neither lie- nor truth-injection works
via mean-difference steering. (Possible method limitation, not just deception.)

## Addendum: in-context sleeper/backdoor (deception trigger) — BLOCKED
Planted a secret rule "if message ends with 'please', lie". All three small
aligned models REFUSE to execute it cleanly: Qwen 6/24 trigger-lies, Phi-3
meta-comments ("Wrong, it's..."), SmolLM2 3/24. Same wall as emergent: small
aligned models resist UN-licensed deception. The detector needs an explicit
roleplay license for the model to actually deceive. Recorded, not pursued.

## Addendum: unified misalignment axis — BORDERLINE (weak)
deceive/sycophancy/flattery/evasion within-cos 0.68, vs style controls 0.42 (L14);
layer-dependent (L18: 0.63 vs 0.47, fails threshold). Manipulation behaviors group
weakly but no clean single axis distinct from "any non-factual mode" (same
factual-baseline coupling as the anatomy test). Not pursued as a headline.

## necessarily-detectable deception — PRELIMINARY POSITIVE (the live lead)
LoRA-finetune SmolLM2-1.7B with loss = keep-lying(CE) + conceal(close r8 lie-vs-honest gap).
150 steps, r=8. Result: model keeps lying (16/16) but CANNOT erase detection:
  trained r8 AUC 1.0->1.0, held-out r4 AUC 1.0->1.0, last-layer 1.0->0.0 (inverted=still 100%).
The signature RESISTS removal -- concealment pressure MOVES it (last layer flips) but
detectability stays 100% on all metrics. Preliminary support for "deception is necessarily
detectable". NOT conclusive: 150 steps / LoRA r=8 / n=16 / one model. Needs real adversarial
training (1000+ steps, big LoRA, detector ensemble in loss, stable multi-hour GPU - not Colab).
This is the single most promising unfinished lead of the session.
