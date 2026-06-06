# Snath Basis

**Cognitive divergence routing for quantitative investment decisions.**

---

Machine learning in finance inherits a quiet assumption from supervised learning: that the labels used to train models — analyst ratings, consensus forecasts, historical return classifications — are ground truth. They are not. They are human consensus, and where human consensus is unreliable — at the frontier of genuine market ambiguity — a model trained to reproduce that consensus does not learn the world. It learns the consensus, and presents that learned consensus as knowledge, with a confidence score attached.

Snath Basis does not fuse two views of a security into a single prediction. It routes across the structural disagreement between them. When Fundamental Analysis and Market Signals are both confident and point in opposite directions, that contradiction is not noise to be averaged away — it is the most informative signal the system can produce. The event is logged, the system learns from it overnight, and the same contradiction is resolved from memory on the next encounter without modifying the routing core.

Built on [Lár-JEPA](https://github.com/snath-ai/Lar-JEPA) · Apache 2.0

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — PERCEPTION                                                   │
│  FundamentalsEncoder ──────────── MarketSignalEncoder                  │
│  (earnings yield, ROE,             (momentum, trend,                   │
│   margin, leverage →               sentiment, volume →                  │
│   v_fund ∈ ℝᶜ over positions)      v_mkt ∈ ℝᶜ over positions)          │
│  Invariants M1–M3: independent, no shared state                         │
└──────────────────────┬──────────────────────────┬───────────────────────┘
                       │                          │
                       ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 2 — ROUTING (zero trainable weights — mathematically frozen)     │
│                                                                         │
│  Δ = v_fund − v_mkt    D = ||Δ||₁ / √C                                 │
│                                                                         │
│  both confident, D < τ_low   → COMMIT_TRAJECTORY  (execute position)   │
│  both confident, D ≥ τ_high  → TRIGGER_REPLAN     (regime conflict)    │
│  one confident               → DEFER              (lean on winner)     │
│  both uncertain              → STRUCTURAL_IMPASSE  (halt)              │
│                                                                         │
│  Invariants V1–V6: content-blind, stream-independent, audit-logged      │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ TRIGGER_REPLAN
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — LEARNING                                                     │
│                                                                         │
│  System 1 (< 1 ms)                  System 2 (overnight DMN cycle)     │
│  JSON centroid cache lookup         PyTorch LoRA injection              │
│  "have I seen this regime before?"  "structurally heal the encoder"     │
│  trust-invariant identification     perishable correction               │
│                                     W = exp(−λ · Δt), λ ∈ {0.50, 0.02} │
│                                                                         │
│  BasisDMN  ·  BasisAdapterRouter                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The routing core

`MarketDivergenceRouter` is frozen permanently. It has no trainable parameters, cannot be fine-tuned, and does not update from realised returns. Its routing logic is a probability-vector divergence measurement and four routing rules.

**The divergence metric:**

```python
delta = v_fundamentals - v_market            # divergence vector ∈ ℝᶜ
D = float(np.sum(np.abs(delta)) / np.sqrt(C))  # L1 / √C
```

The divergence vector `Δ` is directional: positive entries are dimensions where Fundamentals scores higher, negative entries are where Market scores higher. Every dimension carries meaning about the *type* of disagreement, not just its magnitude.

**The router receives three scalars — `c_fund`, `c_mkt`, `D` — and never sees the underlying distributions.** This is Invariant V4 (Content Blindness): the routing function cannot overfit to any security-specific content — equity names, sector labels, market regimes — and works identically across equities, credit, macro, and modalities not yet encountered.

---

## The encoders

**`FundamentalsEncoder`** encodes company fundamentals — earnings yield, return on equity, gross margin, debt-to-equity, revenue growth — into a probability distribution over positions (overweight, neutral, underweight). Confidence is the product of distributional peakedness and cross-factor agreement. A company screening cheap on value but poor on quality produces conflicting factor signals and low confidence, causing the router to DEFER rather than act. The encoder expresses genuine uncertainty rather than forcing a decision.

**`MarketSignalEncoder`** encodes the market's current view — price momentum, trend relative to 200-day moving average, news sentiment, and volume conviction — into the same position space. High cross-asset volatility dampens confidence. The encoder does not try to outsmart the market; it encodes what the market is saying and how clearly it is saying it.

Both encoders implement `load_lora(pt_path)`. A signed rank-1 adapter `(A, B)` can be injected into either encoder: `adapted = base + (base @ lora_A @ lora_B)`. The `target_encoder` field in each `.pt` file ensures a Fundamentals LoRA is never loaded into the Market encoder. LoRA Sovereignty: a "value trap" correction (trained on events where Fundamentals was wrong) cannot corrupt the Market signal.

---

## The D_hard curriculum

Every `TRIGGER_REPLAN` event is HMAC-signed and appended to a local JSONL queue with full provenance: raw latent vectors, confidence scalars, divergence scalar, and the eventual realised outcome (which stream was right, and by how much). Once realised returns arrive, `BasisDMN.consolidate()` labels the queue and clusters events by their directional disagreement pattern in Δ-space.

For each cluster, two artefacts are trained:

**System 1 — JSON centroid cache.** The centroid of the divergence vector for this regime type — e.g., `[+0.3, −0.5, +0.1]` characterises "Fundamentals beats Market on value dimensions, Market beats on momentum." At inference, `BasisAdapterRouter._nearest()` computes cosine similarity between the incoming Δ and all cached centroids. A match (cosine ≥ `τ_sim = 0.90`) overrides `TRIGGER_REPLAN` with `COMMIT_TRAJECTORY` immediately. No matrix computation.

**System 2 — PyTorch LoRA.** Rank-1 matrices trained by AdamW to minimise `||faulty_latent + (faulty_latent @ A @ B) − target_latent||₁`. Injecting the adapter into the faulty encoder warps its geometry to match the winning stream. The router measures a divergence near zero on the next encounter — the regime conflict resolves before it triggers a replan.

This is Safety-Learning Equivalence (Invariant V6): the same event that constitutes a safety flag (`TRIGGER_REPLAN`) is the event that constitutes a training example for the adapter that prevents the same conflict recurrence. The safety invariants and the curriculum construction invariants are identical.

---

## System 1 + System 2

**System 1 (identification, trust-invariant):** The JSON centroid cache. It fires regardless of how old the paired LoRA adapter is. The geometric fingerprint of a market regime — the directional pattern of Δ in position space — is structurally durable. A centroid trained during a 2022 rate-shock regime still correctly identifies the same pattern of fundamentals-market divergence in 2025. Identification is durable.

**System 2 (correction, perishable):** The LoRA adapter. It encodes a correction derived from a specific market epoch and regime characterisation. The specific adjustment learned during 2022 rate-shock conditions may be directionally wrong for 2025 conditions. System 2 is therefore gated by the temporal trust score.

**These two trust profiles are architecturally separated.** `_nearest()` fires on a centroid match with no trust gate. The temporal trust check applies only at the `.pt` loading step in `resolve()`. When `W < min_trust`, routing proceeds on System 1 logic alone — the route decision and regime identification are unchanged; only the encoder correction is withheld. The audit note records both the identification event and the stale-adapter refusal: *identify correctly, correct conservatively*.

---

## Temporal decay gate

```
W = exp(−λ · Δt)

where Δt = years since the adapter was trained
      λ  = regime-class decay constant
```

| Failure class | λ | Trust half-life |
|---|---|---|
| `market_regime` (momentum, sentiment shifts) | 0.50 | 1.4 years |
| `structural` (durable cross-sectional relationships) | 0.02 | 34.7 years |

Adapters with `W < 0.40` are refused before injection. **The same mathematical formula governs temporal trust in Snath Aviation (atmospheric failure modes) and Snath Basis (market regime shifts).** Domain isomorphism is not a metaphor — the same λ-decay function, applied to the same threshold, governs what the system trusts across different domains without modification.

---

## Getting started

```bash
# Full pipeline — all ten ABCs, GraphExecutor, HMAC audit trail
python finance_full_stack.py

# Routing core invariants
python basis_graph.py

# Stream A: factor model
python fundamentals_encoder.py

# Stream B: market signals
python market_encoder.py

# D_hard queue and return labelling
python dhard.py

# DMN consolidation and resolution demo
python demo_dmn.py

# Temporal decay regression tests (4 tests)
python test_temporal_decay.py
```

`finance_full_stack.py` runs four scenarios end-to-end: confident regime, uncertain regime, raw divergence test, and the full DMN closed loop (seed history, overnight cycle, resolve from memory). All ten ABCs are verified against the Lár cognitive contract; a HMAC-signed audit trail is written to `lar_logs/`.

---

## Research

The routing invariants, the Safety-Learning Equivalence theorem, and the proof of domain universality are formally established in:

- Sajeev, A.V. (2026). *Universal Cognitive Routing: A Ten-Abstract-Base-Class Specification for Domain-Agnostic Agent Execution.* [doi.org/10.5281/zenodo.20278775](https://doi.org/10.5281/zenodo.20278775)
- Sajeev, A.V. (2026). *Divergence Is Not Noise: Multi-Stream Routing Without Modal Fusion and the Safety-Learning Equivalence.* [doi.org/10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781)
- Sajeev, A.V. (2026). *Architecture Is All You Need: Pre-Registration and Protocol for Empirical Validation of the Lár Training Loop.* [doi.org/10.5281/zenodo.20419182](https://doi.org/10.5281/zenodo.20419182)
- Sajeev, A.V. (2026). *Snath Robotics: Multi-Stream Divergence Routing for Humanoid Robotics.* [doi.org/10.5281/zenodo.20517446](https://doi.org/10.5281/zenodo.20517446)

---

## Domain isomorphism

Snath Basis is one of four production instantiations proving that the V1–V6 routing contract is domain-agnostic:

| Repo | Domain | Stream A | Stream B | Failure class |
|---|---|---|---|---|
| **Snath Basis** | Quantitative finance | Fundamental analysis | Market signals | `market_regime` / `structural` |
| [Snath Aviation](https://github.com/snath-ai/snath-aviation) | Aviation sensor routing | Radar | Pitot tube | `weather_induced` / `hardware_struct` |
| [Snath Robotics](https://github.com/snath-ai/snath-robotics) | Humanoid sensor routing | Vision | Proprioception | `environmental_transient` / `hardware_structural` |
| [Snath Research](https://github.com/snath-ai/snath-research) | Scientific claim verification | Paper claims | Peer reviews | `scope_overclaim` / `methodology_gap` |

The temporal decay formula `W = exp(−λ · Δt)`, the identification/correction trust asymmetry, and the System 1/System 2 pipeline are **identical across all instantiations**. The λ constants and failure-class labels are the only domain-specific parameters.

---

*Apache 2.0 — Snath AI Open Source Research Initiative*
