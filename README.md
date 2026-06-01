# Snath Basis

Multi-stream divergence routing for quantitative finance.

---

Snath Basis routes investment decisions by measuring the geometric distance between two independent views of the same security — Fundamental Analysis and Market Signals — rather than combining them. The two streams are never mixed. The router operates on the distance between them.

When both views are confident and point in different directions, that contradiction is treated as the most informative event the system can produce, not as noise to be averaged away. The event is logged, the system learns from it overnight, and the same contradiction is resolved from memory on the next encounter — without touching the routing core.

Built on [Lár](https://github.com/snath-ai/Lar-JEPA).

---

## How it works

### The routing core

The `MarketDivergenceRouter` is mathematically frozen. It has no trainable weights, does not update from experience, and cannot be retrained. Its entire logic is a probability-vector divergence measurement and four routing rules:

- Both streams confident, agree: execute
- Both streams confident, disagree: investigate — log the event, resolve from memory
- One stream confident, one uncertain: defer to the confident stream
- Both streams uncertain: halt

The router receives only three scalars — the confidence of each stream and the divergence between them. It never sees the underlying distributions. This is the V4 Content Blindness invariant: the router cannot overfit to any market-specific content and works identically across equity, credit, and macro.

### The encoders

**FundamentalsEncoder** encodes company fundamentals — earnings yield, return on equity, gross margin, leverage, revenue growth — into a probability distribution over three positions: overweight, neutral, underweight. Confidence is the product of distributional peakedness and factor agreement. A company that screens cheap on value but poor on quality produces conflicting factor signals and low confidence, which causes the router to defer rather than act. This is by design: the encoder is expressing genuine uncertainty, not a bug.

**MarketSignalEncoder** encodes the market's current view — price momentum, trend relative to 200-day moving average, news sentiment, and volume conviction. High volatility dampens confidence. The encoder does not try to outsmart the market; it encodes what the market is saying and how clearly it is saying it.

Both encoders support LoRA adaptation. A signed rank-1 matrix pair (A, B) can be injected into either encoder to structurally repair its latent geometry after the DMN overnight cycle identifies a systematic bias in one stream. The type-safety check (`target_encoder` field in each `.pt` file) ensures a fundamentals LoRA is never loaded into the market encoder.

### Default Mode Network

When the router returns `TRIGGER_REPLAN`, the event is signed and appended to a local HMAC-verified JSONL queue. When realised returns arrive later, the queue is labelled: which stream was right? The `BasisDMN.consolidate()` method clusters labelled events by their directional disagreement pattern and trains two artifacts per cluster:

**System 1 — centroid cache.** A lightweight JSON file recording the cluster centroid and historical winner. At inference, `BasisAdapterRouter` computes cosine similarity between the incoming divergence vector and all centroids. A match overrides the `TRIGGER_REPLAN` decision immediately, with no matrix computation.

**System 2 — LoRA adapter.** A signed PyTorch `.pt` file containing rank-1 matrices trained by AdamW to minimise the L1 loss between the faulty stream's latent geometry and the winning stream's. Injecting the adapter into the faulty encoder causes its geometry to naturally match the winning stream on the next encounter. The router is mathematically appeased — it never sees a divergence in the first place.

System 1 provides the fast response. System 2 provides the structural correction. The two operate in parallel: System 1 resolves known failure types immediately while System 2 rebuilds the encoder geometry so the failure type stops occurring.

---

## Getting started

```bash
python finance_full_stack.py      # full pipeline: all ten ABCs, GraphExecutor, DMN loop
python basis_graph.py             # routing core invariants
python fundamentals_encoder.py    # Stream A: factor model
python market_encoder.py          # Stream B: market signals
python dhard.py                   # D_hard queue and return labelling
python demo_dmn.py                # DMN consolidation and resolution demo
```

`finance_full_stack.py` runs four scenarios end-to-end: confident regime, uncertain regime, raw divergence test, and the full DMN closed loop (seed history, overnight cycle, resolve from memory). It confirms all ten ABCs pass against the Lár cognitive contract and logs a HMAC-signed audit trail to `lar_logs/`.

---

## Research

The routing invariants and the Safety-Learning Equivalence theorem are formally proven in:

- Sajeev, A.V. (2026). *Divergence Is Not Noise: Multi-Stream Routing Without Modal Fusion and the Safety-Learning Equivalence.* [doi.org/10.5281/zenodo.20278781](https://doi.org/10.5281/zenodo.20278781)
- Sajeev, A.V. (2026). *Universal Cognitive Routing: A Ten-Abstract-Base-Class Specification for Domain-Agnostic Agent Execution.* [doi.org/10.5281/zenodo.20278775](https://doi.org/10.5281/zenodo.20278775)
