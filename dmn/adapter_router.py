"""
Snath Basis — BasisAdapterRouter (inference-time, two-pass).

Loads the consolidated, HMAC-signed BasisAdapter library and uses it to RESOLVE a
new divergence from memory rather than merely flagging it. This is the Snath Locus
inject → run → restore pattern adapted to the factor-model stage: instead of adding
a LoRA delta to a frozen attention head, it matches the incoming Δ-vector to a learned
cluster and applies that cluster's resolution prior.

Two-pass logic:
  pass 1 — the base MarketDivergenceRouter routes on scalars (V1–V6). If it does NOT
           return TRIGGER_REPLAN, nothing to resolve — return as-is.
  pass 2 — if it IS an Investigate (confident disagreement), find the nearest verified
           adapter by Δ-vector cosine similarity (>= tau_sim). If one matches, resolve
           toward the stream that historically wins that disagreement type, carrying the
           memory's win-rate and mean-return as provenance. No match → stay Investigate.

Security: every adapter is HMAC-signed by the DMN; this router verifies before trusting.
A tampered adapter fails verification and is skipped — the system falls back to Investigate.
"""

from __future__ import annotations
import os, json, glob
from pathlib import Path

import numpy as np

from basis_graph import RouteDecision, DECISION_CLASSES
from .basis_dmn import BasisAdapter


def _cos(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


class BasisAdapterRouter:
    def __init__(self, adapter_dir: str = "models/adapters",
                 tau_sim: float = 0.90, verbose: bool = False):
        self.adapter_dir = Path(adapter_dir)
        self.tau_sim = tau_sim
        self.verbose = verbose
        self._adapters: list[BasisAdapter] = []
        self._load_all()

    def _load_all(self) -> None:
        self._adapters = []
        for fp in glob.glob(str(self.adapter_dir / "*.json")):
            try:
                a = BasisAdapter(**json.loads(Path(fp).read_text()))
                if a.verify():
                    self._adapters.append(a)
                elif self.verbose:
                    print(f"[BasisAdapterRouter] HMAC FAIL — skipped {fp}")
            except Exception as e:
                if self.verbose:
                    print(f"[BasisAdapterRouter] load error {fp}: {e}")

    def refresh(self) -> None:
        self._load_all()

    def available(self) -> list[str]:
        return [a.cluster_id for a in self._adapters]

    def _nearest(self, delta) -> BasisAdapter | None:
        best, best_s = None, self.tau_sim
        for a in self._adapters:
            s = _cos(delta, a.centroid)
            if s >= best_s:
                best, best_s = a, s
        return best

    def resolve(self, v_a, v_b, base_decision, conf_a, conf_b):
        """Returns (decision, note). Only Investigate cases are resolved from memory."""
        if base_decision != RouteDecision.TRIGGER_REPLAN:
            return base_decision, "no divergence to resolve"

        v_a, v_b = np.asarray(v_a, float), np.asarray(v_b, float)
        delta = v_a - v_b
        a = self._nearest(delta)
        if a is None:
            return base_decision, "no matching memory — flag for investigation"

        if a.winner == "fundamentals":
            lean = DECISION_CLASSES[int(np.argmax(v_a))]
            decision = RouteDecision.COMMIT_TRAJECTORY
        elif a.winner == "market":
            lean = DECISION_CLASSES[int(np.argmax(v_b))]
            decision = RouteDecision.COMMIT_TRAJECTORY
        else:
            return base_decision, f"memory [{a.cluster_id}] inconclusive — investigate"

        note = (f"memory [{a.cluster_id}] n={a.n_events}: {a.winner} win "
                f"{a.win_rate:.0%}, mean_ret {a.mean_return:+.1%} -> lean {lean}")
        return decision, note
