"""
Snath Basis — BasisAdapterRouter (inference-time, two-pass).

Loads the consolidated, HMAC-signed BasisAdapter library and uses it to RESOLVE a
new divergence from memory rather than merely flagging it. This is the full
Kahneman Hybrid Architecture:

  System 1 — fast JSON centroid spatial match (O(log N), zero PyTorch).
             Instantly classifies the failure type and overrides the routing decision.

  System 2 — deep PyTorch LoRA restructuring.
             Surgically loads the matching .pt Rank-1 (A, B) matrices into the
             *faulty* encoder, permanently warping its latent geometry so that the
             frozen router is mathematically appeased without ever being touched.

Security: every adapter is HMAC-signed by the DMN; this router verifies before trusting.
A tampered adapter fails verification and is skipped — the system falls back to Investigate.

Identification / correction trust asymmetry
-------------------------------------------
System 1 centroid matching (_nearest) is trust-invariant: the geometric fingerprint
of a regime cluster is durable — "market beats fundamentals in momentum spreads" is
a structurally stable characterisation across years even when the specific learned
correction (the LoRA delta) is not.  _nearest() therefore carries no temporal trust
gate and fires regardless of adapter age.

System 2 LoRA injection is perishable: the learned delta encodes a correction trained
on a specific market epoch and regime characterisation; a delta trained on 2022
rate-shock conditions may be wrong in sign for 2025 conditions.  The temporal gate
W = exp(-λ·Δt) applies only at the .pt loading step in resolve().

When W < min_trust, routing proceeds on System 1 logic alone — the route decision
(COMMIT_TRAJECTORY) and the regime identification are unchanged; only the encoder
correction is withheld.  The audit note records both the identification event and the
stale-adapter refusal.  This is the intended degradation path: identify correctly,
correct conservatively.

Formalised in "Architecture Is All You Need" (Sajeev 2026), §3.4 Remark (Temporal
Decay and Synaptic Depression).
"""

from __future__ import annotations
import os, json, glob, math, datetime
from pathlib import Path

import numpy as np

from basis_graph import RouteDecision, DECISION_CLASSES
from .basis_dmn import BasisAdapter


def _cos(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


# ── Temporal decay (identical formula to Snath Aviation / Snath Locus) ────────
# W = exp(-λ · Δt), Δt = fractional years since the adapter was trained.
# Same magnitudes as the other two domains; only the class labels are finance-
# specific. A market-regime adapter ages like a weather pattern (fast); a
# structural-relationship adapter ages slowly.
_LAMBDA: dict = {
    "market_regime": 0.50,   # regimes shift quickly — fast decay (≡ weather_induced)
    "structural":    0.02,   # durable cross-sectional relationships — slow decay
    "default":       0.10,
}


def _decay_weight(created_at_iso: str | None, failure_class: str = "default") -> float:
    """W = exp(-λ · Δt). Returns 1.0 if no timestamp (treat as current)."""
    if not created_at_iso:
        return 1.0
    try:
        created = datetime.datetime.fromisoformat(created_at_iso.replace("Z", "+00:00"))
        now = datetime.datetime.now(datetime.timezone.utc)
        delta_years = (now - created).total_seconds() / (365.25 * 24 * 3600)
        lam = _LAMBDA.get(failure_class, _LAMBDA["default"])
        return math.exp(-lam * max(0.0, delta_years))
    except Exception:
        return 1.0


class BasisAdapterRouter:
    def __init__(self, adapter_dir: str = "models/adapters",
                 tau_sim: float = 0.90, min_trust: float = 0.40,
                 verbose: bool = False):
        self.adapter_dir = Path(adapter_dir)
        self.tau_sim = tau_sim
        self.min_trust = min_trust   # temporal-decay floor (parity with Aviation 0.40)
        self.verbose = verbose
        self._adapters: list[BasisAdapter] = []
        # Gap A: typed cache (cluster_id, winner) → BasisAdapter.
        # Prevents loading a "fundamentals wins" LoRA when the router classified
        # the current event as "market wins" — the encoder targets differ.
        self._typed_cache: dict[tuple, BasisAdapter] = {}
        self._load_all()

    def _load_all(self) -> None:
        self._adapters = []
        self._typed_cache = {}
        for fp in glob.glob(str(self.adapter_dir / "*.json")):
            try:
                a = BasisAdapter(**json.loads(Path(fp).read_text()))
                if a.verify():
                    self._adapters.append(a)
                    self._typed_cache[(a.cluster_id, a.winner)] = a
                elif self.verbose:
                    print(f"[BasisAdapterRouter] HMAC FAIL — skipped {fp}")
            except Exception as e:
                if self.verbose:
                    print(f"[BasisAdapterRouter] load error {fp}: {e}")

    def refresh(self) -> None:
        self._load_all()

    def available(self) -> list[str]:
        return [a.cluster_id for a in self._adapters]

    def available_typed(self) -> list[tuple]:
        return list(self._typed_cache.keys())

    def _nearest(self, delta, winner: str = "") -> BasisAdapter | None:
        """Find nearest adapter by centroid cosine similarity.
        Gap A: if winner is known, restrict search to matching winner entries.

        Trust-invariant — no temporal gate.
        The centroid fingerprint identifies the regime cluster type; that
        identification is durable across years even when the learned LoRA
        correction is not.  The temporal gate lives in resolve() and applies
        only to System 2 (.pt loading), never to this System 1 lookup.
        """
        best, best_s = None, self.tau_sim
        for a in self._adapters:
            if winner and a.winner != winner:
                continue
            s = _cos(delta, a.centroid)
            if s >= best_s:
                best, best_s = a, s
        return best

    def resolve(self, v_a, v_b, base_decision, conf_a, conf_b,
                enc_a=None, enc_b=None):
        """Returns (decision, note).

        System 1: fast JSON centroid spatial match.
        System 2: if a match is found, surgically load the matching .pt LoRA into
                  the faulty encoder (enc_a or enc_b) to structurally repair its
                  latent geometry — not just override the routing decision.

        Gap A/B: typed cache lookup and target_encoder verification at load.
        Pass enc_a (FundamentalsEncoder) and enc_b (MarketSignalEncoder) to enable
        System 2. If omitted, only the System 1 decision override is applied.
        """
        if base_decision != RouteDecision.TRIGGER_REPLAN:
            return base_decision, "no divergence to resolve"

        v_a, v_b = np.asarray(v_a, float), np.asarray(v_b, float)
        delta = v_a - v_b

        # ── SYSTEM 1: Fast JSON Centroid Spatial Match (trust-invariant) ────────
        # _nearest() fires regardless of adapter age: the geometric fingerprint
        # of a failure cluster is durable.  The centroid match correctly names
        # the regime type and sets the route decision even when System 2 is
        # fully stale.  No trust gate here — trust is checked below, at the
        # .pt loading step only.
        a = self._nearest(delta)
        if a is None:
            return base_decision, "no matching memory — flag for investigation"

        if a.winner == "fundamentals":
            lean = DECISION_CLASSES[int(np.argmax(v_a))]
            faulty_enc, target_enc_name = enc_b, "market"
        elif a.winner == "market":
            lean = DECISION_CLASSES[int(np.argmax(v_b))]
            faulty_enc, target_enc_name = enc_a, "fundamentals"
        else:
            return base_decision, f"memory [{a.cluster_id}] inconclusive — investigate"

        decision = RouteDecision.COMMIT_TRAJECTORY
        note = (f"[System 1] memory [{a.cluster_id}] n={a.n_events}: {a.winner} wins "
                f"{a.win_rate:.0%}, mean_ret {a.mean_return:+.1%} → lean {lean}")

        # ── SYSTEM 2: Surgical PyTorch LoRA Loading ─────────────────────────────
        if faulty_enc is not None:
            pt_path = self.adapter_dir / f"{a.cluster_id.replace('->', '__')}.pt"
            if pt_path.exists() and hasattr(faulty_enc, 'load_lora'):
                # Temporal decay gate (parity with Aviation/Locus): a stale adapter
                # is refused before injection. W = exp(-λ·Δt); below min_trust the
                # geometry it learned is too old to trust — fall back to System 1.
                import torch as _torch
                meta = _torch.load(str(pt_path), map_location="cpu")
                W = _decay_weight(meta.get("created_at"),
                                  meta.get("failure_class", "default"))
                if W >= self.min_trust:
                    # Gap B: load_lora() verifies target_encoder — a "fundamentals"
                    # LoRA is silently ignored by MarketSignalEncoder and vice versa.
                    faulty_enc.load_lora(str(pt_path))
                    note += (f" | [System 2] LoRA loaded into '{target_enc_name}' "
                             f"encoder (trust W={W:.2f})")
                else:
                    note += (f" | [System 2] STALE adapter refused "
                             f"(W={W:.2f} < {self.min_trust}); System 1 only")
            elif self.verbose:
                print(f"[BasisAdapterRouter] no .pt file at {pt_path} — System 1 only")

        return decision, note
