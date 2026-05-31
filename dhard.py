"""
Snath Basis — the D_hard curriculum.

Per the architecture (AIA paper): D_hard = the Investigate cases — where two
*confident* streams disagree (router → TRIGGER_REPLAN). Those are the events worth
learning from: either the market is wrong or the fundamentals are misleading, and the
realised return settles it.

This module logs each such divergence to an HMAC-signed JSONL queue with a slot for
the realised forward return. When returns arrive, `attach_returns` resolves each event:
which stream was right, and what the correct decision was. That labelled queue is the
self-curating training set — no human labels, the routing decisions are the curriculum.

The HMAC signs the immutable *observation* (made at decision time); the realised return
and label are appended later without invalidating it — a tamper-evident record of what
was known when the call was made.

Run:  python dhard.py
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import json, hmac, hashlib
from pathlib import Path

_DHARD_KEY = b"snath_basis_dhard_2026"          # per-project audit key
_DHARD_DECISIONS = {"TRIGGER_REPLAN"}           # confident disagreement = the curriculum
CLASSES = ("overweight", "neutral", "underweight")
RETURN_BAND = 0.05                               # ±5% over the horizon defines the realised class


@dataclass
class DHardEvent:
    asof: str                  # date the call was made
    name: str                  # security
    decision: str              # routing decision (TRIGGER_REPLAN)
    basis: float               # D — the divergence
    conf_a: float              # fundamentals confidence
    conf_b: float              # market confidence
    v_a: list                  # fundamentals distribution over CLASSES
    v_b: list                  # market distribution over CLASSES
    horizon_days: int = 60
    realised_return: float | None = None   # filled by the ground-truth labeller
    realised_class: str | None = None      # overweight/neutral/underweight, from the return
    winner: str | None = None              # "fundamentals" | "market" | "neither"
    sig: str = ""

    _IMMUTABLE = ("asof", "name", "decision", "basis", "conf_a", "conf_b",
                  "v_a", "v_b", "horizon_days")

    def _payload(self) -> bytes:
        return json.dumps({k: getattr(self, k) for k in self._IMMUTABLE},
                          sort_keys=True).encode()

    def sign(self) -> "DHardEvent":
        self.sig = hmac.new(_DHARD_KEY, self._payload(), hashlib.sha256).hexdigest()
        return self

    def verify(self) -> bool:
        want = hmac.new(_DHARD_KEY, self._payload(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.sig, want)


def _argmax_class(dist: list) -> str:
    return CLASSES[max(range(len(dist)), key=lambda i: dist[i])]


class DHardQueue:
    def __init__(self, path: str = "d_hard.jsonl"):
        self.path = Path(path)

    def clear(self):
        self.path.unlink(missing_ok=True)

    def log(self, name, v_a, c_a, v_b, c_b, basis, decision, asof, horizon_days=60):
        dec = decision.value if hasattr(decision, "value") else str(decision)
        if dec not in _DHARD_DECISIONS:
            return None
        ev = DHardEvent(
            asof=asof, name=name, decision=dec, basis=round(float(basis), 4),
            conf_a=round(float(c_a), 4), conf_b=round(float(c_b), 4),
            v_a=[round(float(x), 4) for x in v_a],
            v_b=[round(float(x), 4) for x in v_b],
            horizon_days=horizon_days,
        ).sign()
        with open(self.path, "a") as f:
            f.write(json.dumps(asdict(ev)) + "\n")
        return ev

    def all(self) -> list[DHardEvent]:
        if not self.path.exists():
            return []
        return [DHardEvent(**json.loads(line)) for line in self.path.read_text().splitlines() if line]

    def stats(self) -> dict:
        evs = self.all()
        labelled = [e for e in evs if e.realised_return is not None]
        return {"total": len(evs), "labelled": len(labelled),
                "unlabelled": len(evs) - len(labelled)}

    def attach_returns(self, return_fn) -> int:
        """Fill realised_return + label for any event missing it. return_fn(name, asof,
        horizon_days) -> float | None. Returns the number newly labelled."""
        evs = self.all()
        n = 0
        for e in evs:
            if e.realised_return is not None:
                continue
            r = return_fn(e.name, e.asof, e.horizon_days)
            if r is None:
                continue
            e.realised_return = round(float(r), 4)
            e.realised_class = ("overweight" if r > RETURN_BAND
                                else "underweight" if r < -RETURN_BAND else "neutral")
            la, lb = _argmax_class(e.v_a), _argmax_class(e.v_b)
            e.winner = ("fundamentals" if la == e.realised_class
                        else "market" if lb == e.realised_class else "neither")
            n += 1
        with open(self.path, "w") as f:
            for e in evs:
                f.write(json.dumps(asdict(e)) + "\n")
        return n


# ── Demo: screen the universe, log divergences, then resolve with returns ──────
if __name__ == "__main__":
    from fundamentals_encoder import FundamentalsEncoder, UNIVERSE
    from market_encoder import MarketSignalEncoder, MARKET_UNIVERSE
    from basis_graph import MarketDivergenceRouter

    fund = FundamentalsEncoder().fit(UNIVERSE)
    mkt = MarketSignalEncoder().fit(MARKET_UNIVERSE)
    router = MarketDivergenceRouter()
    q = DHardQueue("d_hard.jsonl"); q.clear()

    asof = "2026-05-31"
    print("Screening — logging confident divergences to D_hard\n" + "=" * 56)
    for fc in UNIVERSE:
        mc = next(x for x in MARKET_UNIVERSE if x["name"] == fc["name"])
        v_a, c_a = fund.score(fc)
        v_b, c_b = mkt.score(mc)
        d = router.divergence(v_a, v_b)
        dec = router.route(c_a, c_b, d)
        ev = q.log(fc["name"], v_a, c_a, v_b, c_b, d, dec, asof)
        if ev:
            print(f"  + logged  {fc['name']:<11} {dec.value}  D={d:.2f}  "
                  f"(A={_argmax_class(ev.v_a)}, B={_argmax_class(ev.v_b)})")
    print("  queue:", q.stats())

    # ground truth arrives (mock here; real price data later)
    print("\nResolving with realised 60-day returns\n" + "=" * 56)
    mock_returns = {"QualCheap": +0.12, "JunkRich": -0.18}   # both resolved toward fundamentals
    q.attach_returns(lambda name, a, h: mock_returns.get(name))
    print(f"{'name':<11}{'ret':>7}{'realised':>11}{'winner':>15}  sig")
    print("-" * 56)
    for e in q.all():
        print(f"{e.name:<11}{e.realised_return:>7}{e.realised_class:>11}"
              f"{e.winner:>15}  {'OK' if e.verify() else 'TAMPERED'}")
    print("\n  This labelled queue is the curriculum: 'when fundamentals and the")
    print("  market disagree like this, here is who was right.' Adapters train on it.")
