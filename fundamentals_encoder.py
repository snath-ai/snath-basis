"""
Snath Basis — Stream A: FundamentalsEncoder.

A factor-based encoder grounded in real cross-sectional quant methodology
(value / quality / profitability / leverage / growth). It turns a company's
fundamentals into a distribution over decision classes plus a confidence.

Key design choice — confidence encodes factor *agreement*, not just magnitude:
a name that is cheap on value but junk on quality (a classic value trap) sends
its factors in opposite directions, so the encoder is rightly *unsure*. That low
confidence is what later lets the divergence router defer rather than act — the
whole point of the V1–V6 contract.

Cross-sectional: factors are z-scored against a fitted universe, so the signal is
relative (cheap vs. peers), which is how real factor models work.

Run:  python fundamentals_encoder.py
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from basis_graph import confidence_from_dist   # one shared confidence definition

DECISION_CLASSES = ("overweight", "neutral", "underweight")

# Factor: (weight, orientation).  orientation +1 = higher is better, -1 = lower is better.
FACTORS = {
    "earnings_yield":   (0.15, +1),   # E/P            — value
    "book_to_price":    (0.15, +1),   # B/P            — value
    "roe":              (0.18, +1),   # return on equity — quality
    "gross_margin":     (0.14, +1),   # margin         — quality
    "fcf_to_earnings":  (0.16, +1),   # cash conversion / low accruals — quality
    "debt_to_equity":   (0.12, -1),   # leverage       — lower is better
    "revenue_growth":   (0.10, +1),   # growth
}


@dataclass
class _Universe:
    mean: dict
    std:  dict


class FundamentalsEncoder:
    """Stream A. fit a universe (for cross-sectional z-scores), then encode()."""

    modality = "fundamentals"

    def __init__(self, temperature: float = 1.2):
        self.temp = temperature          # softmax sharpness on the composite
        self._uni: _Universe | None = None

    def fit(self, universe: list[dict]) -> "FundamentalsEncoder":
        mean, std = {}, {}
        for f in FACTORS:
            vals = np.array([c[f] for c in universe], dtype=float)
            mean[f] = float(vals.mean())
            std[f] = float(vals.std() or 1.0)   # guard zero variance
        self._uni = _Universe(mean, std)
        return self

    def _oriented_z(self, company: dict) -> dict:
        assert self._uni, "call .fit(universe) first"
        z = {}
        for f, (_, orient) in FACTORS.items():
            zf = (company[f] - self._uni.mean[f]) / self._uni.std[f]
            z[f] = orient * zf
        return z

    def encode(self, company: dict) -> tuple[np.ndarray, float]:
        z = self._oriented_z(company)

        # composite score = weighted sum of oriented z-scores
        contribs = {f: FACTORS[f][0] * z[f] for f in FACTORS}
        composite = float(sum(contribs.values()))

        # distribution over (overweight, neutral, underweight)
        logits = np.array([composite, 0.0, -composite]) * self.temp
        dist = _softmax(logits)

        # confidence = decisiveness (peak of dist) damped by factor AGREEMENT.
        # agreement = weight-share of factors pointing the same way as the composite.
        if abs(composite) < 1e-9:
            agreement = 0.5
        else:
            s = np.sign(composite)
            agreement = sum(FACTORS[f][0] for f in FACTORS if np.sign(contribs[f]) == s)
        # confidence = universal peakedness (same as Stream B) damped by factor AGREEMENT
        confidence = float(confidence_from_dist(dist) * agreement)

        return dist, confidence

    # convenience for inspection
    def explain(self, company: dict) -> dict:
        z = self._oriented_z(company)
        contribs = {f: round(FACTORS[f][0] * z[f], 3) for f in FACTORS}
        composite = round(sum(contribs.values()), 3)
        dist, conf = self.encode(company)
        return {"composite": composite, "contribs": contribs,
                "dist": {k: round(float(v), 2) for k, v in zip(DECISION_CLASSES, dist)},
                "confidence": round(conf, 2)}


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# ── Worked example ────────────────────────────────────────────────────────────
UNIVERSE = [
    # name              E/P    B/P   ROE   GM    FCF/E  D/E   growth
    dict(name="QualCheap",  earnings_yield=.09, book_to_price=.90, roe=.28, gross_margin=.60, fcf_to_earnings=1.10, debt_to_equity=.30, revenue_growth=.12),
    dict(name="GrowthRich", earnings_yield=.03, book_to_price=.25, roe=.30, gross_margin=.65, fcf_to_earnings=1.00, debt_to_equity=.60, revenue_growth=.35),
    dict(name="ValueTrap",  earnings_yield=.10, book_to_price=1.40, roe=.04, gross_margin=.25, fcf_to_earnings=.50, debt_to_equity=2.20, revenue_growth=-.05),
    dict(name="JunkRich",   earnings_yield=.025,book_to_price=.30, roe=.03, gross_margin=.28, fcf_to_earnings=.40, debt_to_equity=1.80, revenue_growth=.02),
    dict(name="Average",    earnings_yield=.055,book_to_price=.60, roe=.15, gross_margin=.45, fcf_to_earnings=.85, debt_to_equity=.90, revenue_growth=.10),
    dict(name="CashCow",    earnings_yield=.07, book_to_price=.50, roe=.24, gross_margin=.62, fcf_to_earnings=1.25, debt_to_equity=.50, revenue_growth=.06),
]

if __name__ == "__main__":
    enc = FundamentalsEncoder().fit(UNIVERSE)

    print("Snath Basis — Stream A (FundamentalsEncoder)\n" + "=" * 68)
    print(f"{'name':<11}{'score':>7}  {'over':>5}{'neut':>6}{'under':>7}  {'conf':>5}  lean")
    print("-" * 68)
    for c in UNIVERSE:
        e = enc.explain(c)
        d = e["dist"]
        lean = max(d, key=d.get)
        print(f"{c['name']:<11}{e['composite']:>7}  "
              f"{d['overweight']:>5}{d['neutral']:>6}{d['underweight']:>7}  "
              f"{e['confidence']:>5}  {lean}")

    print("\nNote: ValueTrap is cheap (high E/P, B/P) but junk (low ROE, high debt) —")
    print("its factors fight, so confidence is low even if the lean is positive.")
    print("CashCow / QualCheap are coherent (value + quality agree) → high confidence.")

    # ── feed real Stream A into the divergence router vs. a mock market stream ──
    try:
        from basis_graph import MarketDivergenceRouter
        router = MarketDivergenceRouter()
        print("\n" + "=" * 68)
        print("Routing real fundamentals (A) vs a mock market signal (B)")
        print("-" * 68)
        scenarios = [
            ("agree -> execute",       "QualCheap", np.array([0.72, 0.18, 0.10])),  # market also bullish
            ("disagree -> INVESTIGATE","QualCheap", np.array([0.10, 0.18, 0.72])),  # A bullish, market bearish
            ("market silent -> defer", "JunkRich",  np.array([0.38, 0.34, 0.28])),  # A confident, market no view
            ("nobody sure -> impasse", "ValueTrap", np.array([0.36, 0.34, 0.30])),  # neither has a view
        ]
        for label, name, v_b in scenarios:
            c = next(x for x in UNIVERSE if x["name"] == name)
            v_a, c_a = enc.encode(c)
            c_b = confidence_from_dist(v_b)
            d = router.divergence(v_a, v_b)
            decision = router.route(c_a, c_b, d)
            print(f"  {label:<25}[{name}]  D={d:.2f} conf=(A={c_a:.2f},B={c_b:.2f}) -> {decision.value}")
    except Exception as exc:
        print(f"\n(router demo skipped: {exc})")
