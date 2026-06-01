"""
Snath Basis — divergence routing for markets, built on the Lár engine.

MarketDivergenceRouter implements the published **AbstractDivergenceRouter** — the
tenth Lár ABC (V1–V6) defined in lar_jepa/core/interfaces.py. It is content-blind:
route() sees only the two confidence scalars and the basis (divergence) — never the
stream content (V4). The two streams are AbstractModalEncoder implementations
(FundamentalsEncoder, MarketSignalEncoder); stream independence (V1) is enforced at
that encoder boundary, so the router itself never touches both encoders' state.

A Derivative Work of the Lár-JEPA cognitive architecture (Apache 2.0,
github.com/snath-ai/Lar-JEPA), in the quantitative-finance domain.

Run:  python basis_graph.py
"""

from __future__ import annotations
import numpy as np

import _lar  # noqa: F401  — places core.interfaces / core.types on the path
from core.interfaces import AbstractDivergenceRouter
from core.types import RouteDecision

# Both streams predict over the same decision classes.
DECISION_CLASSES = ("overweight", "neutral", "underweight")
C = len(DECISION_CLASSES)

# ── Routing thresholds (provisional — calibrate on realised-return backtests) ──
# Confidence is distribution peakedness in [0,1] (see confidence_from_dist).
TAU_HIGH = 0.35   # confidence floor to "act"
TAU_LOW  = 0.12   # below this a stream carries effectively no signal
DELTA    = 0.40   # basis (divergence) threshold separating agree / disagree


def confidence_from_dist(dist) -> float:
    """Universal stream confidence: distribution PEAKEDNESS in [0,1].
    0 = uniform (no signal), 1 = one-hot. BOTH streams compute confidence this way
    so the router's scalars are comparable across them."""
    peak = (float(np.asarray(dist).max()) - 1.0 / C) / (1.0 - 1.0 / C)
    return max(0.0, peak)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float); z = z - z.max(); e = np.exp(z); return e / e.sum()


class MarketDivergenceRouter(AbstractDivergenceRouter):
    """
    Concrete AbstractDivergenceRouter (V1–V6) for markets.

    V1 Stream Independence · V2 Geometric Divergence (D ≥ 0) · V3 Symmetry-breaking
    allowed · V4 Content Blindness (route() reads only scalars) · V5 Routing
    Completeness · V6 Safety-Learning Equivalence (STRUCTURAL_IMPASSE == max signal).
    """

    # V1 — the streams are upstream AbstractModalEncoders (FundamentalsEncoder /
    # MarketSignalEncoder). Independence is enforced at that boundary, so the router
    # never holds both encoders' state. The base/divergence pass calls them; the
    # router only ever receives the resulting (distribution, confidence) tuples.
    def encode_stream_a(self, x_a):
        raise NotImplementedError(
            "delegated to FundamentalsEncoder (AbstractModalEncoder); "
            "V1 stream independence is enforced at the encoder boundary.")

    def encode_stream_b(self, x_b):
        raise NotImplementedError(
            "delegated to MarketSignalEncoder (AbstractModalEncoder); "
            "V1 stream independence is enforced at the encoder boundary.")

    def divergence(self, z_a, z_b) -> float:            # V2–V3
        # the BASIS: total-variation between the finding distributions, /sqrt(C)
        z_a, z_b = np.asarray(z_a, float), np.asarray(z_b, float)
        return float(np.abs(z_a - z_b).sum() / np.sqrt(C))

    def route(self, confidence_a: float, confidence_b: float,
              divergence: float) -> RouteDecision:       # V4–V6 (content-blind)
        c_a, c_b, d = confidence_a, confidence_b, divergence
        # Defer (one stream confident, the other silent) → commit to the confident
        # stream per V5; not a separate enum value.
        if max(c_a, c_b) >= TAU_HIGH and min(c_a, c_b) < TAU_LOW:
            return RouteDecision.COMMIT_TRAJECTORY
        if c_a < TAU_LOW and c_b < TAU_LOW:
            return RouteDecision.STRUCTURAL_IMPASSE       # Halt — no signal
        if c_a >= TAU_HIGH and c_b >= TAU_HIGH:
            return (RouteDecision.TRIGGER_REPLAN          # Investigate — confident disagreement
                    if d >= DELTA else RouteDecision.COMMIT_TRAJECTORY)  # Execute — agree
        return RouteDecision.STRUCTURAL_IMPASSE           # Stall — middling


if __name__ == "__main__":
    r = MarketDivergenceRouter()
    print("Snath Basis — MarketDivergenceRouter  (AbstractDivergenceRouter V1–V6)")
    print("=" * 64)
    cases = [
        ("agree -> execute",        [0.70, 0.20, 0.10], [0.68, 0.22, 0.10], 0.50, 0.50),
        ("disagree -> investigate", [0.70, 0.20, 0.10], [0.10, 0.20, 0.70], 0.50, 0.50),
        ("one silent -> commit",    [0.70, 0.20, 0.10], [0.36, 0.34, 0.30], 0.50, 0.05),
        ("both unsure -> impasse",  [0.36, 0.34, 0.30], [0.34, 0.33, 0.33], 0.05, 0.04),
    ]
    for label, va, vb, ca, cb in cases:
        d = r.divergence(np.array(va), np.array(vb))
        print(f"  {label:<26} D={d:.2f}  conf=({ca},{cb}) -> {r.route(ca, cb, d).value}")
    print("\nMarketDivergenceRouter is a subclass of the published AbstractDivergenceRouter:",
          issubclass(MarketDivergenceRouter, AbstractDivergenceRouter))
