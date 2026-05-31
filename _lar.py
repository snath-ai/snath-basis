"""
_lar.py — connect Snath Basis to the PUBLIC, genesis-anchored Lár-JEPA engine.

Snath Basis implements the published ten-ABC cognitive contract (core.interfaces)
and the RouteDecision enum (core.types) from the **public, Apache-2.0, genesis-
anchored** prior art at ~/Desktop/Lar_Main/lar_jepa — the exact codebase fingerprinted
in the genesis baseline. Importing the contract from the *anchored public source*
(not a local dev copy) is what makes Snath Basis a clean Derivative Work of the
pre-employment prior art, in the quantitative-finance domain — exactly as Snath Locus
is in the biomedical domain.

Resolution order:
  1. $SNATH_BASIS_LARJEPA           — explicit override (CI / other machines)
  2. ~/Desktop/Lar_Main/lar_jepa    — PUBLIC, genesis-anchored prior art (preferred)
  3. ../lar_jepa                    — local dev copy (fallback only)
"""
import os
import sys

_CANDIDATES = [
    os.environ.get("SNATH_BASIS_LARJEPA"),
    os.path.expanduser("~/Desktop/Lar_Main/lar_jepa"),                              # public, anchored
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lar_jepa")),     # dev fallback
]

_LAR_JEPA = None
for _p in _CANDIDATES:
    if _p and os.path.isfile(os.path.join(_p, "core", "interfaces.py")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        _LAR_JEPA = _p
        break
