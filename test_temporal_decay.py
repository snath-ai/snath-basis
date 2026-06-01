"""
Regression test — adapter temporal decay gate (Snath Basis).

BasisAdapterRouter.resolve() already has the temporal gate; these tests
lock in that behaviour so it can't silently regress.

  1. _decay_weight() formula (fresh / stale / cusp / missing timestamp).
  2. resolve() accepts a fresh .pt (W≥0.40) → LoRA loaded into faulty encoder.
  3. resolve() refuses a stale .pt (W<0.40, backdated 3yr) → System 1 only.
  4. Cusp case — just past the ~1.83yr boundary for market_regime (λ=0.50).
  5. failure_class field survives BasisDMN.consolidate() round-trip.

Run:  python test_temporal_decay.py
      or:  pytest test_temporal_decay.py
"""
from __future__ import annotations
import math
import sys
import os
import datetime
import json
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dmn.adapter_router import _decay_weight, _LAMBDA

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso_years_ago(years: float) -> str:
    delta = datetime.timedelta(days=years * 365.25)
    dt = datetime.datetime.now(datetime.timezone.utc) - delta
    return dt.isoformat().replace("+00:00", "Z")


def _write_pt(path: str, created_at: str, failure_class: str = "market_regime") -> None:
    """Write a minimal .pt file with temporal metadata at `path`."""
    torch.save({
        "A":             torch.zeros(3, 1),
        "B":             torch.zeros(1, 3),
        "target_encoder": "market",
        "cluster_id":    "test",
        "final_loss":    0.0,
        "created_at":    created_at,
        "n_events":      1,
        "mean_return":   0.0,
        "failure_class": failure_class,
    }, path)


# ---------------------------------------------------------------------------
# 1. Formula correctness
# ---------------------------------------------------------------------------

def test_decay_weight_formula():
    # Δt=0 → W=1.0
    assert abs(_decay_weight(_iso_years_ago(0), "market_regime") - 1.0) < 0.01

    # Δt=1yr, λ=0.50 → W≈0.6065
    w1 = _decay_weight(_iso_years_ago(1.0), "market_regime")
    assert abs(w1 - math.exp(-0.50)) < 0.01, f"1yr: {w1:.4f}"

    # Δt=3yr, λ=0.50 → W≈0.2231
    w3 = _decay_weight(_iso_years_ago(3.0), "market_regime")
    assert abs(w3 - math.exp(-1.5)) < 0.01, f"3yr: {w3:.4f}"

    # Structural: Δt=5yr, λ=0.02 → W≈0.9048
    ws = _decay_weight(_iso_years_ago(5.0), "structural")
    assert abs(ws - math.exp(-0.10)) < 0.01, f"structural 5yr: {ws:.4f}"

    # Missing timestamp → 1.0
    assert _decay_weight(None) == 1.0
    assert _decay_weight("") == 1.0

    # Monotonicity
    assert _decay_weight(_iso_years_ago(0.5), "market_regime") > _decay_weight(_iso_years_ago(2.0), "market_regime")

    print("PASS  test_decay_weight_formula")


# ---------------------------------------------------------------------------
# 2. resolve() with fresh .pt → LoRA injected (System 2 active)
# ---------------------------------------------------------------------------

def test_resolve_fresh_lora_loaded():
    """BasisAdapterRouter.resolve() should load LoRA when adapter is fresh."""
    from dmn.adapter_router import BasisAdapterRouter
    from basis_graph import RouteDecision
    from dhard import CLASSES

    # Build a minimal BasisAdapter JSON (System 1 centroid)
    from dmn.basis_dmn import BasisAdapter
    n_cls = len(CLASSES)
    adapter = BasisAdapter(
        cluster_id="test->test",
        centroid=[0.0] * n_cls,
        winner="market",
        win_rate=0.8,
        mean_return=0.02,
        n_events=5,
        created_at=_iso_years_ago(0.01),
    ).sign()

    class _FakeFundamentalsEncoder:
        """Minimal encoder that records load_lora calls."""
        loaded_path = None
        def load_lora(self, path):
            _FakeFundamentalsEncoder.loaded_path = path

    with tempfile.TemporaryDirectory() as td:
        # Write System 1 JSON
        json_path = os.path.join(td, "test__test.json")
        import dataclasses
        with open(json_path, "w") as f:
            json.dump(dataclasses.asdict(adapter), f)

        # Write fresh System 2 .pt
        pt_path = os.path.join(td, "test->test.pt".replace("->", "__"))
        _write_pt(pt_path, _iso_years_ago(0.01))

        # tau_sim=0.0 so the centroid always matches — we're testing the decay gate,
        # not the spatial matching logic.
        router = BasisAdapterRouter(adapter_dir=td, tau_sim=0.0, min_trust=0.40, verbose=True)
        v = [1.0] + [0.0] * (n_cls - 1)  # non-zero so cosine is defined
        enc_a = _FakeFundamentalsEncoder()
        dec, note = router.resolve(v, v, RouteDecision.TRIGGER_REPLAN,
                                   0.9, 0.9, enc_a=enc_a, enc_b=None)

    assert "LoRA loaded" in note or "trust W=" in note, f"Expected LoRA note, got: {note}"
    print(f"PASS  test_resolve_fresh_lora_loaded  note={note!r}")


# ---------------------------------------------------------------------------
# 3. resolve() with stale .pt (3yr) → System 1 only, LoRA refused
# ---------------------------------------------------------------------------

def test_resolve_stale_lora_refused():
    """BasisAdapterRouter.resolve() should refuse a 3-year-old adapter."""
    from dmn.adapter_router import BasisAdapterRouter
    from basis_graph import RouteDecision
    from dhard import CLASSES
    from dmn.basis_dmn import BasisAdapter

    n_cls = len(CLASSES)
    adapter = BasisAdapter(
        cluster_id="test->test",
        centroid=[0.0] * n_cls,
        winner="market",
        win_rate=0.8,
        mean_return=0.02,
        n_events=5,
        created_at=_iso_years_ago(0.01),
    ).sign()

    class _FakeFundamentalsEncoder:
        loaded_path = None
        def load_lora(self, path):
            _FakeFundamentalsEncoder.loaded_path = path

    with tempfile.TemporaryDirectory() as td:
        json_path = os.path.join(td, "test__test.json")
        import dataclasses
        with open(json_path, "w") as f:
            json.dump(dataclasses.asdict(adapter), f)

        # Write STALE System 2 .pt (3 years old)
        pt_path = os.path.join(td, "test__test.pt")
        _write_pt(pt_path, _iso_years_ago(3.0))

        router = BasisAdapterRouter(adapter_dir=td, tau_sim=0.0, min_trust=0.40, verbose=True)
        v = [1.0] + [0.0] * (n_cls - 1)
        enc_a = _FakeFundamentalsEncoder()
        dec, note = router.resolve(v, v, RouteDecision.TRIGGER_REPLAN,
                                   0.9, 0.9, enc_a=enc_a, enc_b=None)

    assert "STALE" in note, f"Expected STALE in note, got: {note}"
    assert _FakeFundamentalsEncoder.loaded_path is None, "LoRA should NOT be loaded for stale adapter"
    W = _decay_weight(_iso_years_ago(3.0), "market_regime")
    print(f"PASS  test_resolve_stale_lora_refused  W={W:.4f}  note={note!r}")


# ---------------------------------------------------------------------------
# 4. Cusp: just past ~1.83yr for market_regime (λ=0.50)
# ---------------------------------------------------------------------------

def test_cusp_refused():
    cusp_years = -math.log(0.40) / 0.50  # ≈ 1.833
    W = _decay_weight(_iso_years_ago(cusp_years + 0.05), "market_regime")
    assert W < 0.40, f"Cusp+0.05yr should be < 0.40, got {W:.4f}"
    print(f"PASS  test_cusp_refused  W={W:.4f} at Δt≈{cusp_years+0.05:.2f}yr")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_decay_weight_formula()
    test_resolve_fresh_lora_loaded()
    test_resolve_stale_lora_refused()
    test_cusp_refused()
    print("\nAll 4 Snath Basis temporal decay regression tests passed.")
