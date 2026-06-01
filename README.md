# Snath Basis

Quantitative Finance Divergence Router — Fundamental vs Market signal arbitration on the Lar-JEPA cognitive architecture

---

## Overview

Snath Basis is a domain instantiation of the Lar-JEPA cognitive architecture applied to quantitative finance. It routes investment decisions by measuring geometric divergence between two independent latent streams: Fundamental Analysis (earnings yield, ROE, gross margin, debt/equity, revenue growth) and Market Signals (momentum, trend, sentiment, volume).

Rather than fusing signals into a black-box model, the system enforces a mathematically frozen routing core that operates on geometric properties alone. When the two streams confidently disagree, the event is logged to a self-curating D_hard curriculum and resolved from memory on the next encounter. The router never sees raw distributions; it operates only on confidence scalars and a divergence magnitude.

This is an empirical realisation of the theorems published in the DAS paper (DOI: https://doi.org/10.5281/zenodo.20278781) and the UCR specification (DOI: https://doi.org/10.5281/zenodo.20278775), authored by Aadithya Vishnu Sajeev, May 2026, prior to commercial employment.

Repository: github.com/snath-ai/snath-basis
Lar-JEPA (upstream, Apache 2.0): https://github.com/snath-ai/Lar-JEPA

---

## Prior Art and Licensing

Snath Basis is a derivative work of the Lar-JEPA cognitive architecture, released under Apache 2.0. The ten Abstract Base Classes (ABCs) instantiated here are specified in the UCR paper and proven in the DAS paper, both published on Zenodo in May 2026 before any commercial employment of the author. The full prior art chain is listed at the end of this document.

---

## The Ten-ABC Cognitive Contract

Each ABC is a domain-agnostic interface defined in `core/interfaces.py` of the Lar-JEPA repository. Snath Basis provides one or more concrete finance-domain implementations for each.

| # | Abstract Base Class | Finance Instantiation | Role |
|---|---|---|---|
| 1 | AbstractCognitiveNode | MarketCognitiveNode | Base routable node for all pipeline stages |
| 2 | AbstractManifold | MarketRegimeJEPA | JEPA world model: predict next market regime latent, entropic_loss |
| 3 | AbstractContextBridge | StateInstrumentBridge | Stateless: market-state latent (B,D) to (B,1,D) cross-attention query |
| 4 | AbstractLatentFaultLocator | RegimeStressLocator | I1-I6: fused market state x instrument sequence to topk most-stressed instruments |
| 5 | AbstractEntropicRouter | MarketEntropicRouter | Gates on regime-prediction entropy: confident to COMMIT, uncertain to REPLAN/IMPASSE |
| 6 | AbstractAttentionKernel | LinearAttentionMarketKernel | A1-A6: O(N) linear attention over instrument universe, ELU+1 feature map |
| 7 | AbstractPerturbationOperator | RateShockOperator | P1-P6: delta = encode(post-rate-hike) - encode(pre-hike) |
| 8 | AbstractRoutingKernel | AllocationKernel | R1-R4: cosine departure score to DEFEND/HEDGE/HOLD |
| 9 | AbstractModalEncoder | MarketStateEncoder, FundamentalsEncoder, MarketSignalEncoder | Market state to latent; factor model; momentum/trend/sentiment/volume to distribution |
| 10 | AbstractDivergenceRouter | RegimeDivergenceRouter, MarketDivergenceRouter | Content-blind basis between two market views; confidence-gated TV divergence |

`prove_abc_coverage()` in `finance_full_stack.py` confirms 10/10 ALL PASS across 8 HMAC-audited steps.

---

## Architecture

### Layer 1: Perception

**FundamentalsEncoder** (`AbstractModalEncoder`, `nn.Module`) encodes company fundamentals into a 3-class probability distribution over `overweight`, `neutral`, and `underweight`. Confidence is computed as peakedness multiplied by factor_agreement. A company that is cheap on value but poor on quality — a value trap — sends factors in opposing directions and rightly produces low confidence, causing the router to defer rather than act.

**MarketSignalEncoder** (`AbstractModalEncoder`, `nn.Module`) encodes market signals: momentum, trend, sentiment, and volume. Volatility damping is applied so that high volatility reduces confidence, reflecting genuine uncertainty in the signal rather than a defect.

Both encoders support LoRA injection via `adapted = base + (base @ lora_A @ lora_B)`. The `target_encoder` field in `.pt` files enforces type safety: a LoRA trained on fundamentals is silently rejected by `MarketSignalEncoder.load_lora()`.

**RateShockOperator** (`AbstractPerturbationOperator`) computes `delta = encode(post_hike_state) - encode(pre_hike_state)` to predict the post-perturbation market state for `AllocationKernel` scoring.

### Layer 2: The Frozen Routing Core (V1-V6)

`MarketDivergenceRouter` extends `AbstractDivergenceRouter`, the tenth ABC formally proven in the DAS paper (DOI: https://doi.org/10.5281/zenodo.20278781).

The divergence metric is total variation distance normalised by channel count:

```
D = L1(softmax(z_a) - softmax(z_b)) / sqrt(C)
```

This is magnitude-invariant, which prevents false-positive TRIGGER_REPLAN events caused by scale differences between encoders rather than genuine disagreement.

The six routing invariants enforced by the frozen core:

- **V4 Content Blindness:** `route()` receives only `(confidence_a, confidence_b, D)` — never the raw distributions.
- **V5 Routing Completeness:** every `(ca, cb, D)` triple maps to exactly one `RouteDecision`.
- **V6 Safety-Learning Equivalence:** TRIGGER_REPLAN is the maximum-signal event; it is logged to the D_hard queue immediately upon detection.

The four routing outcomes:

| Condition | Decision |
|---|---|
| Both streams confident and in agreement (D low) | COMMIT_TRAJECTORY — execute the dominant view |
| Both streams confident and in disagreement (D high) | TRIGGER_REPLAN — log to D_hard, resolve from memory |
| One stream confident, one silent | COMMIT_TRAJECTORY/DEFER — trust the confident stream |
| Both streams uncertain | STRUCTURAL_IMPASSE — halt, no actionable signal |

### Layer 3: Default Mode Network

**BasisDMN** reads the D_hard queue, clusters resolved TRIGGER_REPLAN events by directional disagreement pattern (for example, `overweight->underweight`), and trains a signed `BasisAdapter` per cluster — a JSON centroid file plus a paired PyTorch LoRA `.pt` file.

**DHardQueue** is an HMAC-signed JSONL append-only queue. Each `DHardEvent` records: `asof`, `name`, `decision`, `basis`, `conf_a`, `conf_b`, `v_a`, `v_b`, `horizon_days`, `realised_return` (filled later by `attach_returns()`), `realised_class`, and `winner` (`fundamentals` / `market` / `neither`).

**BasisAdapterRouter** maintains a typed cache keyed by `(cluster_id, winner)` and uses `_nearest()` cosine similarity search over centroids to match incoming events.

The D_hard closed loop:

1. Pipeline fires TRIGGER_REPLAN.
2. `DHardQueue.log()` appends a signed event.
3. Realised returns arrive and `attach_returns()` labels the event.
4. `BasisDMN.consolidate()` produces signed adapters.
5. `BasisAdapterRouter.resolve()` overrides the next matching TRIGGER_REPLAN with COMMIT_TRAJECTORY.

---

## The Kahneman Hybrid Architecture

The DMN implements a two-track memory that mirrors the System 1 / System 2 distinction from cognitive science.

**System 1 — Fast JSON Centroid Cache:** `BasisAdapterRouter` loads JSON centroids and computes cosine similarity between the incoming delta-vector and saved centroids in O(log N). On a cache hit, it instantly overrides TRIGGER_REPLAN with COMMIT_TRAJECTORY toward the historically correct stream. No matrix multiplication is performed.

**System 2 — Structural LoRA Repair:** AdamW trains Rank-1 LoRA matrices (A: dim x 1, B: 1 x dim) to minimise L1 loss between the faulty stream and the winner stream. The adapted encoder is:

```
adapted = faulty + (faulty @ A @ B)
```

The `.pt` file is saved with metadata fields `created_at`, `failure_class`, `n_events`, `mean_return`, and `target_encoder` for temporal decay support. Typical loss is below 0.02.

**Empirical result from Scenario D:** raw divergence 0.7159 (TRIGGER_REPLAN) reduced to 0.2573 (COMMIT_TRAJECTORY) after LoRA structural repair. A 64% divergence reduction. The frozen `MarketDivergenceRouter` is mathematically appeased without modifying a single routing weight.

System 1 is the reactive override active from the first recurrence. System 2 is the structural correction that eliminates the divergence at the encoder level so the router does not encounter it again.

---

## Pipeline Topology

Defined in `finance_full_stack.py`. The `GraphExecutor` runs in `offline_mode=True` with HMAC-signed audit per step (Lar v2.2.0, 8 audited steps per scenario). Audit logs are written to `lar_logs/`.

```
SensorEmbeddingNode      (MarketStateEncoder -> z (B,D))
  -> MarketWorldModelNode  (MarketRegimeJEPA -> predict next regime, entropic_loss)
  -> EntropicGateNode      (MarketEntropicRouter -> COMMIT / REPLAN / IMPASSE)
       |
       +-- COMMIT
       |     -> StateBridgeNode         (StateInstrumentBridge -> (B,1,D) query)
       |     -> RateShockNode           (RateShockOperator -> z_pred after rate hike)
       |     -> RegimeLocalisationNode  (RegimeStressLocator + LinearAttentionMarketKernel
       |                                 -> topk stressed instruments)
       |     -> AllocationRouterNode    (AllocationKernel -> DEFEND / HEDGE / HOLD)
       |           |-- DEFEND -> DefendNode -> audit
       |           |-- HEDGE  -> HedgeNode  -> audit
       |           +-- HOLD   -> HoldNode   -> audit
       |
       +-- REPLAN / IMPASSE
             -> HumanOversightNode -> audit
```

**Scenario D — DMN closed loop:**

1. 16 resolved TRIGGER_REPLAN events are seeded into `d_hard_pipeline.jsonl`.
2. `BasisDMN.consolidate()` produces two clusters, each with a signed JSON centroid and a paired PyTorch LoRA `.pt`.
3. `BasisAdapterRouter.resolve()` delivers a System 1 cache hit that overrides TRIGGER_REPLAN to COMMIT_TRAJECTORY; the System 2 LoRA is loaded into the faulty encoder.

---

## EU AI Act Compliance

Snath Basis is a research-grade quantitative finance tool and does not constitute a regulated AI system under EU AI Act Annex III. The documentation here reflects architecture best practices rather than a regulatory compliance claim.

Audit controls present in the system:

- `GraphExecutor` produces an HMAC-signed audit trail per pipeline step.
- `DHardQueue` is an append-only HMAC-signed JSONL log.
- `offline_mode=True` prevents unintended external LLM API calls during pipeline execution.

---

## Running

```bash
python finance_full_stack.py     # All 10 ABCs + DMN closed loop (8 audited steps)
python basis_graph.py            # Router invariants (V1-V6)
python fundamentals_encoder.py   # Stream A factor model
python market_encoder.py         # Stream B market signals
python dhard.py                  # D_hard queue and labelling
python demo_dmn.py               # Kahneman Hybrid DMN demonstration
```

---

## ABC Coverage

`prove_abc_coverage()` in `finance_full_stack.py` verifies all ten ABCs at runtime:

```
10/10 ALL PASS   (8 audited steps)
core.interfaces sourced from: /Users/aadithya/Desktop/Lar_Main/lar_jepa/core/interfaces.py
```

---

## Prior Art Chain

Cumulative Zenodo DOIs establishing the intellectual lineage of each component:

| DOI | Contribution |
|---|---|
| 10.5281/zenodo.19025925 | Lar DMN: episodic and semantic memory, HMAC audit |
| 10.5281/zenodo.19120047 | AbstractCognitiveNode, DAG executor |
| 10.5281/zenodo.19245328 | AbstractManifold, AbstractContextBridge |
| 10.5281/zenodo.19484646 | AbstractLatentFaultLocator (I1-I6) |
| 10.5281/zenodo.19516414 | AbstractEntropicRouter, RouteDecision |
| 10.5281/zenodo.19646405 | DMN v3.0, Learned Graph Executor |
| 10.5281/zenodo.20278775 | Nine-ABC cognitive contract (UCR paper) |
| 10.5281/zenodo.20278781 | AbstractDivergenceRouter V1-V6, Safety-Learning Equivalence (DAS paper) |

All DOIs predate commercial employment of the author and establish independent prior art for the Lar-JEPA architecture and its finance-domain instantiation in Snath Basis.
