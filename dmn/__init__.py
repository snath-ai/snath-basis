"""
Snath Basis — DMN (Default Mode Network): memory + consolidation.

Mirrors the Snath Locus DMN structure, adapted to the factor-model stage:
  basis_dmn.py     — reads the D_hard queue, clusters resolved divergences by
                     Δ-pattern, consolidates each cluster into a signed BasisAdapter
                     (a learned prior for that type of fundamentals-vs-market disagreement).
  adapter_router.py — inference-time: loads + verifies adapters, and resolves a new
                     divergence from memory (two-pass) instead of merely flagging it.

When Snath Basis later gains neural encoders, the BasisAdapter priors become true
LoRA deltas (as in Snath Locus); the consolidation → verify → apply contract is the same.
"""
from .basis_dmn import BasisDMN, BasisAdapter
from .adapter_router import BasisAdapterRouter

__all__ = ["BasisDMN", "BasisAdapter", "BasisAdapterRouter"]
