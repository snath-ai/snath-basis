"""
Snath Basis — Stream B: MarketSignalEncoder (an AbstractModalEncoder).

The market-side counterpart to the FundamentalsEncoder, implementing the same
published **AbstractModalEncoder** (M1–M3) contract. Reads what the market is saying —
price momentum, trend, news sentiment, volume conviction — and emits a distribution
over the same decision classes plus a confidence (volatility damps it).

Lightweight by design (no model downloads): sentiment is a transparent lexicon scorer
that FinBERT later drops in to replace.

Derivative Work of the pre-employment Lár-JEPA prior art (Apache 2.0), finance domain.

Run:  python market_encoder.py
"""

from __future__ import annotations
from dataclasses import dataclass
import re
import numpy as np
import torch
import torch.nn as nn

import _lar  # noqa: F401
from core.interfaces import AbstractModalEncoder
from basis_graph import confidence_from_dist, DECISION_CLASSES

MARKET_FACTORS = {
    "momentum_12_1":   (0.35, +1),   # 12-1 month price momentum
    "price_vs_200dma": (0.25, +1),   # trend: % above/below 200-day MA
    "sentiment":       (0.30, +1),   # news sentiment in [-1, 1]
    "volume_trend":    (0.10, +1),   # volume conviction
}
_VOL_FLOOR, _VOL_K = 0.25, 2.0       # volatility above the floor shrinks confidence

_POS = {"beat", "beats", "raise", "raises", "raised", "surge", "surges", "growth",
        "strong", "upgrade", "record", "launch", "buyback", "dividend", "favorite",
        "squeeze", "rally", "bullish", "gain", "gains", "tops", "soars"}
_NEG = {"miss", "misses", "cut", "cuts", "downgrade", "weak", "concern", "concerns",
        "debt", "restructuring", "lawsuit", "probe", "warning", "slump", "fall",
        "falls", "bearish", "loss", "losses", "decline", "slashes", "plunge"}


def lexicon_sentiment(headlines: list[str]) -> float:
    toks = re.findall(r"[a-z]+", " ".join(headlines).lower())
    pos = sum(t in _POS for t in toks)
    neg = sum(t in _NEG for t in toks)
    return (pos - neg) / (pos + neg + 1)


@dataclass
class _Universe:
    mean: dict
    std:  dict


class MarketSignalEncoder(AbstractModalEncoder, nn.Module):
    """Stream B: market-side encoder, upgraded to torch.nn.Module with LoRA support.
    
    The base market-signal weights are frozen; the DMN sleep cycle injects Rank-1
    (A, B) LoRA matrices to structurally repair the latent geometry without
    touching the routing core.
    """

    def __init__(self, temperature: float = 1.2):
        super().__init__()
        self.temp = temperature
        self._uni: _Universe | None = None
        # LoRA Rank-1 matrices (injected by BasisDMN after sleep cycle)
        self.lora_A: torch.Tensor | None = None
        self.lora_B: torch.Tensor | None = None

    def load_lora(self, pt_path: str) -> None:
        """Surgically load the LoRA adapter for this encoder from a .pt file."""
        state = torch.load(pt_path, weights_only=True)
        if state.get("target_encoder") == "market":
            self.lora_A = state["A"]
            self.lora_B = state["B"]

    # ── AbstractModalEncoder contract (M1–M3) ─────────────────────────────────
    @property
    def output_dim(self) -> int:
        return len(DECISION_CLASSES)

    @property
    def modality(self) -> str:
        return "market_signal"

    def encode(self, company: dict) -> np.ndarray:   # M3: with optional LoRA
        dist, _ = self.score(company)
        if self.lora_A is not None and self.lora_B is not None:
            with torch.no_grad():
                t = torch.tensor(dist, dtype=torch.float32)
                adapted = t + torch.matmul(torch.matmul(t, self.lora_A), self.lora_B)
                adapted = torch.softmax(adapted, dim=0)
                return adapted.numpy()
        return dist

    # ── encoder internals ─────────────────────────────────────────────────────
    def _features(self, company: dict) -> dict:
        sent = company.get("sentiment")
        if sent is None and "headlines" in company:
            sent = lexicon_sentiment(company["headlines"])
        return {"momentum_12_1": company["momentum_12_1"],
                "price_vs_200dma": company["price_vs_200dma"],
                "sentiment": sent if sent is not None else 0.0,
                "volume_trend": company["volume_trend"]}

    def fit(self, universe: list[dict]) -> "MarketSignalEncoder":
        feats = [self._features(c) for c in universe]
        mean, std = {}, {}
        for f in MARKET_FACTORS:
            vals = np.array([x[f] for x in feats], dtype=float)
            mean[f] = float(vals.mean()); std[f] = float(vals.std() or 1.0)
        self._uni = _Universe(mean, std)
        return self

    def score(self, company: dict) -> tuple[np.ndarray, float]:
        assert self._uni, "call .fit(universe) first"
        feat = self._features(company)
        contribs = {f: MARKET_FACTORS[f][0] * MARKET_FACTORS[f][1] *
                    (feat[f] - self._uni.mean[f]) / self._uni.std[f] for f in MARKET_FACTORS}
        composite = float(sum(contribs.values()))
        dist = _softmax(np.array([composite, 0.0, -composite]) * self.temp)

        if abs(composite) < 1e-9:
            agreement = 0.5
        else:
            s = np.sign(composite)
            agreement = sum(MARKET_FACTORS[f][0] for f in MARKET_FACTORS if np.sign(contribs[f]) == s)
        vol_damp = float(np.exp(-_VOL_K * max(0.0, company.get("volatility", _VOL_FLOOR) - _VOL_FLOOR)))
        confidence = float(confidence_from_dist(dist) * agreement * vol_damp)
        return dist, confidence

    def explain(self, company: dict) -> dict:
        feat = self._features(company)
        dist, conf = self.score(company)
        return {"sentiment": round(feat["sentiment"], 2),
                "dist": {k: round(float(v), 2) for k, v in zip(("over", "neut", "under"), dist)},
                "confidence": round(conf, 2)}


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(); e = np.exp(z); return e / e.sum()


MARKET_UNIVERSE = [
    dict(name="QualCheap",  momentum_12_1=-.08, price_vs_200dma=-.05, volume_trend=-.10, volatility=.25,
         headlines=["Q3 revenue misses estimates", "Analyst downgrade on slowing demand", "Guidance left unchanged"]),
    dict(name="GrowthRich", momentum_12_1=.35,  price_vs_200dma=.25,  volume_trend=.30,  volatility=.35,
         headlines=["Beats earnings and raises guidance", "New product launch drives growth", "Stock tops record high"]),
    dict(name="ValueTrap",  momentum_12_1=-.15, price_vs_200dma=-.18, volume_trend=.05,  volatility=.45,
         headlines=["Restructuring amid debt concerns", "Warning on margins", "Shares decline"]),
    dict(name="JunkRich",   momentum_12_1=.40,  price_vs_200dma=.30,  volume_trend=.50,  volatility=.50,
         headlines=["Retail favorite surges on AI buzz", "Short squeeze rally continues", "Soars to new highs"]),
    dict(name="Average",    momentum_12_1=.05,  price_vs_200dma=.02,  volume_trend=.00,  volatility=.30,
         headlines=["In-line quarter", "Holds steady"]),
    dict(name="CashCow",    momentum_12_1=.12,  price_vs_200dma=.10,  volume_trend=.15,  volatility=.22,
         headlines=["Raises dividend and buyback", "Steady gains on strong cash flow"]),
]

if __name__ == "__main__":
    mkt = MarketSignalEncoder().fit(MARKET_UNIVERSE)
    assert isinstance(mkt, AbstractModalEncoder)
    print(f"Snath Basis — Stream B (MarketSignalEncoder : AbstractModalEncoder, "
          f"modality='{mkt.modality}')\n" + "=" * 64)
    print(f"{'name':<11}{'sent':>6}  {'over':>5}{'neut':>6}{'under':>7}  {'conf':>5}  lean")
    print("-" * 64)
    for c in MARKET_UNIVERSE:
        e = mkt.explain(c); d = e["dist"]; lean = max(d, key=d.get)
        print(f"{c['name']:<11}{e['sentiment']:>6}  {d['over']:>5}{d['neut']:>6}{d['under']:>7}  {e['confidence']:>5}  {lean}")

    # the payoff: two real AbstractModalEncoders → real basis → published router
    from fundamentals_encoder import FundamentalsEncoder, UNIVERSE
    from basis_graph import MarketDivergenceRouter
    fund = FundamentalsEncoder().fit(UNIVERSE)
    router = MarketDivergenceRouter()

    print("\n" + "=" * 64)
    print("BASIS: real fundamentals (A) vs real market (B)")
    print("-" * 64)
    print(f"{'name':<11}{'A lean':>11}{'B lean':>11}  {'D':>5} {'cA':>5} {'cB':>5}  decision")
    print("-" * 64)
    leans = ("over", "neut", "under")
    for fc in UNIVERSE:
        mc = next(x for x in MARKET_UNIVERSE if x["name"] == fc["name"])
        v_a, c_a = fund.score(fc)
        v_b, c_b = mkt.score(mc)
        d = router.divergence(v_a, v_b)
        dec = router.route(c_a, c_b, d)
        la, lb = leans[int(np.argmax(v_a))], leans[int(np.argmax(v_b))]
        print(f"{fc['name']:<11}{la:>11}{lb:>11}  {d:>5.2f} {c_a:>5.2f} {c_b:>5.2f}  {dec.value}")
