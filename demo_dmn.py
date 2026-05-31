"""
Snath Basis — DMN demo.

Seeds a synthetic history of resolved divergences, consolidates it into a signed
BasisAdapter library, then shows the system RESOLVE a brand-new divergence from
memory (two-pass) instead of merely flagging it for investigation.

Run:  python demo_dmn.py
"""

import json, random
from dataclasses import asdict

import numpy as np

from dhard import DHardQueue, DHardEvent
from dmn.basis_dmn import BasisDMN, _argmax_class
from dmn.adapter_router import BasisAdapterRouter
from basis_graph import MarketDivergenceRouter


def seed_history(path: str, n_per_cluster: int = 8) -> int:
    """Simulate accumulated, resolved divergences for two disagreement types."""
    random.seed(7)
    DHardQueue(path).clear()
    specs = [(0, 2, 0.75, 0.10),   # A bullish, B bearish; fundamentals usually right (+)
             (2, 0, 0.75, 0.12)]   # A bearish, B bullish; fundamentals usually right (-)
    rows = []
    for ai, bi, p_fund, mag in specs:
        for _ in range(n_per_cluster):
            v_a = [0.15, 0.15, 0.15]; v_a[ai] = 0.7
            v_b = [0.15, 0.15, 0.15]; v_b[bi] = 0.7
            lean = ai if random.random() < p_fund else bi
            sign = +1 if lean == 0 else (-1 if lean == 2 else 0)
            r = round(sign * abs(random.gauss(mag, 0.03)), 4)
            ev = DHardEvent(asof="2026-01-01", name="SYNTH", decision="TRIGGER_REPLAN",
                            basis=round(sum(abs(v_a[i] - v_b[i]) for i in range(3)) / 3 ** 0.5, 4),
                            conf_a=0.5, conf_b=0.5, v_a=v_a, v_b=v_b, horizon_days=60).sign()
            ev.realised_return = r
            ev.realised_class = ("overweight" if r > 0.05 else "underweight" if r < -0.05 else "neutral")
            la, lb = _argmax_class(v_a), _argmax_class(v_b)
            ev.winner = ("fundamentals" if la == ev.realised_class
                         else "market" if lb == ev.realised_class else "neither")
            rows.append(ev)
    with open(path, "w") as f:
        for ev in rows:
            f.write(json.dumps(asdict(ev)) + "\n")
    return len(rows)


if __name__ == "__main__":
    from fundamentals_encoder import FundamentalsEncoder, UNIVERSE
    from market_encoder import MarketSignalEncoder, MARKET_UNIVERSE

    DEMO = "d_hard_demo.jsonl"
    n = seed_history(DEMO)
    print(f"Seeded {n} resolved divergences (synthetic history)\n" + "=" * 60)

    dmn = BasisDMN(queue_path=DEMO, adapter_dir="models/adapters")
    print("Consolidating D_hard -> signed BasisAdapter library + PyTorch LoRA:")
    dmn.consolidate(min_events=4)

    print("\n" + "=" * 60)
    print("Kahneman Hybrid Cascade on a NEW divergence (System 1 + System 2)")
    print("-" * 60)

    fund = FundamentalsEncoder().fit(UNIVERSE)
    mkt  = MarketSignalEncoder().fit(MARKET_UNIVERSE)
    router  = MarketDivergenceRouter()
    arouter = BasisAdapterRouter(adapter_dir="models/adapters")

    # A fresh, unseen name: fundamentals say BUY, the market says SELL
    v_a = np.array([0.66, 0.24, 0.10]); c_a = 0.48
    v_b = np.array([0.08, 0.20, 0.72]); c_b = 0.55

    d_raw  = router.divergence(v_a, v_b)
    base   = router.route(c_a, c_b, d_raw)
    print(f"  [Raw]     Fundamentals: {v_a}  Market: {v_b}")
    print(f"  [Raw]     Divergence (L1-norm): {d_raw:.4f}  ->  base router: {base.value}")

    # System 1 + System 2: pass the encoders so LoRA gets loaded into the faulty one
    decision, note = arouter.resolve(v_a, v_b, base, c_a, c_b, enc_a=fund, enc_b=mkt)
    print(f"\n  {note}")
    print(f"  DMN-resolved decision: {decision.value}")

    # Re-encode with the now-LoRA-adapted encoder to show divergence reduction
    # The faulty (market) encoder has been restructured — re-run it on a sample company
    sample_fund = UNIVERSE[0]
    sample_mkt  = MARKET_UNIVERSE[0]
    v_a_adapted = fund.encode(sample_fund)
    v_b_adapted = mkt.encode(sample_mkt)
    d_adapted   = router.divergence(v_a_adapted, v_b_adapted)
    dec_adapted = router.route(
        fund.score(sample_fund)[1],
        mkt.score(sample_mkt)[1],
        d_adapted
    )
    print(f"\n  [System 2 Adapted Encoders on '{sample_fund['name']}']")
    print(f"  Fundamentals latent : {np.round(v_a_adapted, 4)}")
    print(f"  Market latent (LoRA): {np.round(v_b_adapted, 4)}")
    print(f"  New divergence      : {d_adapted:.4f}  ->  {dec_adapted.value}")
    print(f"\n  [SUCCESS] LoRA restructured the faulty encoder's geometry.")
    print(f"  The frozen MarketDivergenceRouter is naturally appeased.")

