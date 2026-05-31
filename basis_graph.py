"""
Snath Basis — divergence routing for markets (architecture skeleton).

The same V1–V6 AbstractDivergenceRouter contract as Snath Locus, with the encoders
swapped for a fundamentals stream and a market-signal stream. Routes on the BASIS —
the geometric divergence between the two streams' finding distributions.

This file is a runnable scaffold (mock encoders, numpy only) so the topology is clear
before the real encoders and data layer are built. Run:  python basis_graph.py

Build targets (in order):
  1. Data layer    — SEC EDGAR + price/feed ingestion (public sources only).
  2. Stream A       — FundamentalsEncoder over balance-sheet / earnings-quality features.
  3. Stream B       — MarketSignalEncoder (FinBERT embeddings + microstructure features).
  4. D_hard queue   — log Investigate (high-confidence disagreement) events.
  5. Ground truth   — realised 30/60/90d returns; label D_hard; train per-regime adapters.
  6. Backtest       — does acting on routing decisions beat the fused baseline?
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import numpy as np

# Decision classes the two streams both predict over.
DECISION_CLASSES = ("overweight", "neutral", "underweight")
C = len(DECISION_CLASSES)


def confidence_from_dist(dist) -> float:
    """Universal stream confidence: distribution PEAKEDNESS in [0,1].
    0 = uniform (no signal), 1 = one-hot (max signal). BOTH streams compute
    confidence this way so the router's scalars are comparable across them."""
    peak = (float(np.asarray(dist).max()) - 1.0 / C) / (1.0 - 1.0 / C)
    return max(0.0, peak)


class RouteDecision(str, Enum):
    COMMIT_TRAJECTORY = "COMMIT_TRAJECTORY"   # Execute  — agree, both confident
    TRIGGER_REPLAN    = "TRIGGER_REPLAN"      # Investigate — disagree, both confident
    STRUCTURAL_IMPASSE = "STRUCTURAL_IMPASSE" # Halt/Stall — too uncertain to act
    DEFER             = "DEFER"               # one stream confident, the other not


# ── Routing thresholds (provisional — calibrate on realised-return backtests) ──
# Confidence is distribution peakedness in [0,1] (see confidence_from_dist), so a
# "confident" 3-class call sits ~0.4–0.7 and a flat one near 0. Thresholds match.
TAU_HIGH = 0.35   # confidence floor to "act"
TAU_LOW  = 0.12   # below this a stream carries effectively no signal
DELTA    = 0.40   # basis (divergence) threshold separating agree / disagree


# ── Stream encoders ───────────────────────────────────────────────────────────
# Each returns (finding_distribution over DECISION_CLASSES, confidence scalar).
# V1 Stream Independence: A and B share no mutable state. (Mock implementations.)

class FundamentalsEncoder:
    """Stream A — TODO: balance-sheet ratios, earnings quality, cash-flow signals."""
    modality = "fundamentals"

    def encode(self, x_fundamentals) -> tuple[np.ndarray, float]:
        v = _softmax(np.asarray(x_fundamentals, dtype=float))
        return v, confidence_from_dist(v)


class MarketSignalEncoder:
    """Stream B — TODO: FinBERT(filings/news) + order-flow / microstructure features."""
    modality = "market_signal"

    def encode(self, x_market) -> tuple[np.ndarray, float]:
        v = _softmax(np.asarray(x_market, dtype=float))
        return v, confidence_from_dist(v)


# ── The router (AbstractDivergenceRouter contract, V1–V6) ─────────────────────
class MarketDivergenceRouter:
    """
    V1 Stream Independence · V2 Geometric Divergence (D >= 0) · V3 Symmetry-breaking
    allowed · V4 Content Blindness (route() sees only scalars) · V5 Routing Completeness
    · V6 Safety-Learning Equivalence (STRUCTURAL_IMPASSE == max learning signal).
    """

    def __init__(self):
        self.stream_a = FundamentalsEncoder()
        self.stream_b = MarketSignalEncoder()

    def encode_stream_a(self, x_a): return self.stream_a.encode(x_a)   # V1
    def encode_stream_b(self, x_b): return self.stream_b.encode(x_b)   # V1

    def divergence(self, v_a: np.ndarray, v_b: np.ndarray) -> float:    # V2–V3
        # the BASIS: total-variation between the finding distributions, /sqrt(C)
        return float(np.abs(v_a - v_b).sum() / np.sqrt(C))

    def route(self, c_a: float, c_b: float, d: float) -> RouteDecision:  # V4–V6
        # content-blind: scalars only
        if max(c_a, c_b) >= TAU_HIGH and min(c_a, c_b) < TAU_LOW:
            return RouteDecision.DEFER
        if c_a < TAU_LOW and c_b < TAU_LOW:
            return RouteDecision.STRUCTURAL_IMPASSE
        if c_a >= TAU_HIGH and c_b >= TAU_HIGH:
            return (RouteDecision.TRIGGER_REPLAN if d >= DELTA      # Investigate
                    else RouteDecision.COMMIT_TRAJECTORY)           # Execute
        return RouteDecision.STRUCTURAL_IMPASSE                     # Stall

    def step(self, x_a, x_b) -> "BasisEvent":
        v_a, c_a = self.encode_stream_a(x_a)
        v_b, c_b = self.encode_stream_b(x_b)
        d = self.divergence(v_a, v_b)
        decision = self.route(c_a, c_b, d)
        return BasisEvent(v_a, v_b, c_a, c_b, d, decision)


@dataclass
class BasisEvent:
    v_a: np.ndarray
    v_b: np.ndarray
    conf_a: float
    conf_b: float
    basis: float
    decision: RouteDecision
    # TODO: realised_return (filled by ground-truth labeller) for the D_hard curriculum


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ── Demo (mock) ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    router = MarketDivergenceRouter()
    scenarios = {
        "agree (commit)":        ([3.0, 0.2, 0.1], [2.8, 0.3, 0.1]),
        "disagree (investigate)":([3.0, 0.1, 0.1], [0.1, 0.1, 3.0]),
        "uncertain (impasse)":   ([0.2, 0.1, 0.15], [0.1, 0.2, 0.15]),
    }
    print("Snath Basis — divergence routing demo\n" + "=" * 44)
    for name, (a, b) in scenarios.items():
        ev = router.step(a, b)
        print(f"{name:<24} basis D={ev.basis:.3f}  "
              f"conf=({ev.conf_a:.2f},{ev.conf_b:.2f})  -> {ev.decision.value}")
