# Snath Basis: Multi-Stream Factor Routing & DMN

A quantitative-finance derivative of the **Lár-JEPA cognitive architecture** (Apache 2.0). Snath Basis applies the same mathematical framework used to route autonomous aircraft through sensor failures to the problem of resolving disagreements between **Fundamental Analysis** and **Market Signal** streams.

---

## 🧠 Core Philosophy: The Black Box Problem in Finance

In quantitative finance, most deep learning models fuse all incoming signals (P/E ratios, momentum, sentiment, volume) into a single "Black Box" neural network. When the model disagrees with itself, it is impossible to audit *which* input is lying, or certify that the resolution is mathematically sound.

Snath Basis solves this with the same two-part split used in aviation:
1. **Encoders (Neural):** Independent stream-specific encoders that learn via PyTorch.
2. **Router (Mathematical):** A completely frozen, 100% deterministic geometry engine enforcing the 6 routing invariants (V1–V6). It sees only two confidence scalars and a divergence — never the raw stream content (V4: Content Blindness).

---

## 🏗️ Architecture

### Stream A: `FundamentalsEncoder` (M1–M3)
A factor-model encoder grounded in real cross-sectional quant methodology. It encodes company fundamentals (earnings yield, ROE, gross margin, debt/equity, revenue growth) into a distribution over three decision classes:
- `overweight` / `neutral` / `underweight`

Confidence encodes **factor agreement**: a company that is cheap on value but junk on quality (a value trap) sends factors in opposite directions, rightly producing *low* confidence. This is what tells the frozen router to defer rather than act.

### Stream B: `MarketSignalEncoder` (M1–M3)
Encodes what the market is saying — price momentum, trend (vs 200-day MA), news sentiment (lexicon scorer, FinBERT-ready), and volume conviction. High volatility dampens confidence, reflecting genuine uncertainty in the signal.

### `MarketDivergenceRouter` (V1–V6)
The mathematically frozen routing core. It enforces the published invariants:
- **V1 Stream Independence:** The two encoders never share internal state.
- **V4 Content Blindness:** `route()` sees only `(conf_a, conf_b, divergence)` — never the raw distributions.
- **V5 Routing Completeness:** Every possible input combination maps to exactly one `RouteDecision`.
- **V6 Safety-Learning Equivalence:** `STRUCTURAL_IMPASSE` is the maximum-signal event, triggering full investigation.

The four routing outcomes:
| Scenario | Decision |
|---|---|
| Both confident, agree (D low) | `COMMIT_TRAJECTORY` — execute the trade |
| Both confident, disagree (D high) | `TRIGGER_REPLAN` — log to D_hard, resolve from memory |
| One confident, one silent | `COMMIT_TRAJECTORY` — defer to the confident stream |
| Both uncertain | `STRUCTURAL_IMPASSE` — halt, no signal |

---

## 🔄 The Kahneman Hybrid Memory (System 1 + System 2)

When the router detects confident disagreement (`TRIGGER_REPLAN`), it logs the raw geometric trace as a `DHardEvent` to a tamper-evident HMAC-signed JSONL queue — the **$\mathcal{D}_{hard}$ curriculum**.

When realised forward returns arrive (ground truth), the queue is labelled: which stream was right? This self-curating curriculum trains the DMN sleep cycle.

### System 1: Fast JSON Centroid Cache
The `BasisDMN.consolidate()` clusters labelled events by their directional disagreement pattern (e.g., `overweight->underweight`) and saves a lightweight HMAC-signed `.json` adapter containing the cluster centroid, winner, win-rate, and mean realised return.

At inference, the `BasisAdapterRouter` computes a **cosine similarity** between the incoming $\Delta$-vector and all saved centroids in $O(\log N)$ time. On a cache hit, it instantly overrides `TRIGGER_REPLAN` with `COMMIT_TRAJECTORY` toward the historically correct stream.

### System 2: Deep PyTorch LoRA (Cortical Restructuring)
Simultaneously with the JSON adapter, the DMN trains a Rank-1 pair of PyTorch matrices ($A$ and $B$) via `AdamW` to minimise the L1 divergence loss between the faulty stream and the winner stream. The result is saved as a paired `.pt` file.

At inference, the `BasisAdapterRouter` surgically loads the matching `.pt` into the faulty encoder:
```python
adapted = base + (base @ A @ B)
```
This physically warps the encoder's latent geometry so that the frozen `MarketDivergenceRouter` is mathematically appeased — the divergence drops without touching a single weight in the routing core.

### Why Hybrid? (The Band-Aid vs. The Cure)
- **JSON (System 1)** is the **Band-Aid**: instant reactive override *today*, even if the encoder geometry is still wrong.
- **LoRA (System 2)** is the **Cure**: structural mathematical healing *overnight*, so the encoder naturally outputs correct geometry and the router never even sees a divergence in the first place.

---

## ✅ Empirical Results (All Modules Verified)

Every module runs independently and is test-verified:

```text
# basis_graph.py — Router invariants
agree -> execute           D=0.02  conf=(0.5,0.5) -> COMMIT_TRAJECTORY
disagree -> investigate    D=0.69  conf=(0.5,0.5) -> TRIGGER_REPLAN
one silent -> commit       D=0.39  conf=(0.5,0.05) -> COMMIT_TRAJECTORY
both unsure -> impasse     D=0.03  conf=(0.05,0.04) -> STRUCTURAL_IMPASSE

MarketDivergenceRouter is a subclass of the published AbstractDivergenceRouter: True
```

```text
# demo_dmn.py — Full Kahneman Hybrid Cascade
Consolidating D_hard -> signed BasisAdapter library + PyTorch LoRA:
  ✓ overweight->underweight  n=8  winner=fundamentals  win_rate=1.0   LoRA loss=0.0116
  ✓ underweight->overweight  n=8  winner=fundamentals  win_rate=0.75  LoRA loss=0.0095

[Raw]  Divergence (L1-norm): 0.7159  ->  TRIGGER_REPLAN

[System 1] Cache Hit: fundamentals wins 100%, mean_ret +10.6% -> lean overweight
[System 2] LoRA loaded into 'market' encoder

Fundamentals latent : [0.6545 0.25   0.0955]
Market latent (LoRA): [0.4317 0.2885 0.2799]
New divergence      : 0.2573  ->  COMMIT_TRAJECTORY

[SUCCESS] LoRA restructured the faulty encoder's geometry.
The frozen MarketDivergenceRouter is naturally appeased.
```

The LoRA adapter reduced the L1 divergence from **0.7159** to **0.2573** (a **64% reduction**), pulling the market encoder's geometry into alignment with the fundamentals — without touching a single weight in the frozen routing core.

---

## 🚀 Running the Modules

```bash
python basis_graph.py          # Router invariants (V1–V6)
python fundamentals_encoder.py # Stream A factor model
python market_encoder.py       # Stream B market signals
python dhard.py                # D_hard queue and labelling pipeline
python demo_dmn.py             # Full Kahneman Hybrid DMN demonstration
```

---

## 📚 Paper Reference
This is an empirical realisation of the theorems proposed in:
* **Divergence Is Not Noise: Multi-Stream Routing Without Modal Fusion and the Safety-Learning Equivalence** (Sajeev, 2026).

A Derivative Work of the pre-employment Lár-JEPA prior art (Apache 2.0, github.com/snath-ai/Lar-JEPA), in the quantitative-finance domain.
