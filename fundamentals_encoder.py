"""
Snath Basis — Stream A: FundamentalsEncoder (an AbstractModalEncoder).

Implements the published **AbstractModalEncoder** (M1–M3) from
lar_jepa/core/interfaces.py: encode(x) projects a company's fundamentals into a
fixed-width latent — here a distribution over decision classes. A factor model
grounded in real cross-sectional quant methodology (value / quality / profitability
/ leverage / growth).

Key design choice — confidence encodes factor *agreement*, not just magnitude: a name
that is cheap on value but junk on quality (a value trap) sends its factors in opposite
directions, so the encoder is rightly *unsure*. That low confidence is what later lets
the divergence router defer rather than act.

Derivative Work of the pre-employment Lár-JEPA prior art (Apache 2.0), finance domain.

Run:  python fundamentals_encoder.py
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

import _lar  # noqa: F401
from core.interfaces import AbstractModalEncoder
from basis_graph import confidence_from_dist, DECISION_CLASSES

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


class FundamentalsEncoder(AbstractModalEncoder):
    """Stream A. fit a universe (cross-sectional z-scores), then encode()/score()."""

    def __init__(self, temperature: float = 1.2):
        self.temp = temperature
        self._uni: _Universe | None = None

    # ── AbstractModalEncoder contract (M1–M3) ─────────────────────────────────
    @property
    def output_dim(self) -> int:          # M1: encode(x).shape[-1] == output_dim
        return len(DECISION_CLASSES)

    @property
    def modality(self) -> str:            # M2: stable identifier
        return "fundamentals"

    def encode(self, company: dict) -> np.ndarray:   # M3: input → latent (the decision dist)
        return self.score(company)[0]

    # ── encoder internals ─────────────────────────────────────────────────────
    def fit(self, universe: list[dict]) -> "FundamentalsEncoder":
        mean, std = {}, {}
        for f in FACTORS:
            vals = np.array([c[f] for c in universe], dtype=float)
            mean[f] = float(vals.mean()); std[f] = float(vals.std() or 1.0)
        self._uni = _Universe(mean, std)
        return self

    def _oriented_z(self, company: dict) -> dict:
        assert self._uni, "call .fit(universe) first"
        return {f: orient * (company[f] - self._uni.mean[f]) / self._uni.std[f]
                for f, (_, orient) in FACTORS.items()}

    def score(self, company: dict) -> tuple[np.ndarray, float]:
        """The latent distribution AND a confidence scalar (used by the router)."""
        z = self._oriented_z(company)
        contribs = {f: FACTORS[f][0] * z[f] for f in FACTORS}
        composite = float(sum(contribs.values()))

        logits = np.array([composite, 0.0, -composite]) * self.temp
        dist = _softmax(logits)

        if abs(composite) < 1e-9:
            agreement = 0.5
        else:
            s = np.sign(composite)
            agreement = sum(FACTORS[f][0] for f in FACTORS if np.sign(contribs[f]) == s)
        confidence = float(confidence_from_dist(dist) * agreement)
        return dist, confidence

    def explain(self, company: dict) -> dict:
        z = self._oriented_z(company)
        contribs = {f: round(FACTORS[f][0] * z[f], 3) for f in FACTORS}
        dist, conf = self.score(company)
        return {"composite": round(sum(contribs.values()), 3),
                "dist": {k: round(float(v), 2) for k, v in zip(DECISION_CLASSES, dist)},
                "confidence": round(conf, 2)}


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(); e = np.exp(z); return e / e.sum()


# ── Worked example ────────────────────────────────────────────────────────────
UNIVERSE = [
    dict(name="QualCheap",  earnings_yield=.09, book_to_price=.90, roe=.28, gross_margin=.60, fcf_to_earnings=1.10, debt_to_equity=.30, revenue_growth=.12),
    dict(name="GrowthRich", earnings_yield=.03, book_to_price=.25, roe=.30, gross_margin=.65, fcf_to_earnings=1.00, debt_to_equity=.60, revenue_growth=.35),
    dict(name="ValueTrap",  earnings_yield=.10, book_to_price=1.40, roe=.04, gross_margin=.25, fcf_to_earnings=.50, debt_to_equity=2.20, revenue_growth=-.05),
    dict(name="JunkRich",   earnings_yield=.025,book_to_price=.30, roe=.03, gross_margin=.28, fcf_to_earnings=.40, debt_to_equity=1.80, revenue_growth=.02),
    dict(name="Average",    earnings_yield=.055,book_to_price=.60, roe=.15, gross_margin=.45, fcf_to_earnings=.85, debt_to_equity=.90, revenue_growth=.10),
    dict(name="CashCow",    earnings_yield=.07, book_to_price=.50, roe=.24, gross_margin=.62, fcf_to_earnings=1.25, debt_to_equity=.50, revenue_growth=.06),
]

if __name__ == "__main__":
    enc = FundamentalsEncoder().fit(UNIVERSE)
    assert isinstance(enc, AbstractModalEncoder)
    print(f"Snath Basis — Stream A (FundamentalsEncoder : AbstractModalEncoder, "
          f"modality='{enc.modality}', output_dim={enc.output_dim})\n" + "=" * 68)
    print(f"{'name':<11}{'score':>7}  {'over':>5}{'neut':>6}{'under':>7}  {'conf':>5}  lean")
    print("-" * 68)
    for c in UNIVERSE:
        e = enc.explain(c); d = e["dist"]; lean = max(d, key=d.get)
        print(f"{c['name']:<11}{e['composite']:>7}  {d['overweight']:>5}{d['neutral']:>6}"
              f"{d['underweight']:>7}  {e['confidence']:>5}  {lean}")

    # feed real Stream A into the published divergence router vs a mock market stream
    from basis_graph import MarketDivergenceRouter
    router = MarketDivergenceRouter()
    print("\n" + "=" * 68 + "\nRouting real fundamentals (A) vs a mock market signal (B)\n" + "-" * 68)
    scenarios = [
        ("agree -> execute",       "QualCheap", np.array([0.72, 0.18, 0.10])),
        ("disagree -> INVESTIGATE","QualCheap", np.array([0.10, 0.18, 0.72])),
        ("market silent -> commit","JunkRich",  np.array([0.38, 0.34, 0.28])),
        ("nobody sure -> impasse", "ValueTrap", np.array([0.36, 0.34, 0.30])),
    ]
    for label, name, v_b in scenarios:
        c = next(x for x in UNIVERSE if x["name"] == name)
        v_a, c_a = enc.score(c)
        c_b = confidence_from_dist(v_b)
        d = router.divergence(v_a, v_b)
        print(f"  {label:<25}[{name}]  D={d:.2f} conf=(A={c_a:.2f},B={c_b:.2f}) "
              f"-> {router.route(c_a, c_b, d).value}")
