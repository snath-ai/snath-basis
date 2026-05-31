# Snath Basis — Divergence Routing for Markets

**Snath Basis** is the quantitative-finance instantiation of the Lár divergence-routing
engine. It runs two independent latent streams over a security and routes on the
**basis** — the geometric divergence between them — surfacing the cases where two
confident signals *disagree*. Those disagreements are the most informative events in a
market: either the price is wrong or the fundamentals are misleading.

It is the same architecture as Snath Locus, pointed at a different domain. Where Locus
pinpoints a position in a genome, **Basis measures the spread between two independent
readings of a market.**

---

## The two streams

| Stream | Encoder | Produces |
|---|---|---|
| **A — Fundamentals** | balance-sheet ratios, earnings quality, cash-flow signals | distribution over decision classes (overweight / neutral / underweight) |
| **B — Market signal** | FinBERT embeddings of filings / earnings calls / news + order-flow / microstructure features | distribution over the **same** decision classes |

**Routing score:** `D = ‖v_A − v_B‖₁ / √C` (total-variation between the two finding
distributions) — content-blind, identical to the V1–V6 `AbstractDivergenceRouter`
contract. No modification to the engine; only the encoders change.

**Routing outcomes** (the five rules): Execute (agree, confident) · Investigate
(disagree, both confident — the high-value case) · Defer (one confident) · Halt /
Stall (uncertain).

**Ground truth:** realised returns over 30 / 60 / 90-day horizons — *public, and it
arrives in days*, not quarters. This is what makes Basis a clean, fast testbed for the
full self-improving loop (routing → D_hard curriculum → per-regime LoRA adapters →
improved routing).

---

## Why this project exists

1. **Product.** A divergence-routing research engine for markets: finds where
   fundamentals and price contradict, with a cryptographic audit trail of every call.
2. **Proof.** It is the first *full* build-out (real D_hard events, real adapters, real
   ground-truth validation) of the self-improving loop in a **non-pharmaceutical**
   domain — establishing empirically that the Lár engine is universal infrastructure,
   not a domain-specific tool.

---

## Provenance & clean room

- Built entirely on **personal hardware, personal time**, on the pre-existing Lár
  open-source engine and the published `AbstractDivergenceRouter` (V1–V6) contract.
- **Public data only:** SEC EDGAR, Alpha Vantage / Quandl, FinBERT, public price feeds.
  No proprietary data of any employer or third party, ever.
- A Derivative Work of pre-employment prior art (the Lár engine + the three published
  preprints), in a field unrelated to pharmaceuticals.

---

## Status

Greenfield. `basis_graph.py` holds the architecture skeleton; encoders and the data
layer are the first build targets.

---

*Snath. Built on Lár.*
