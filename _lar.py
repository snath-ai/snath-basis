"""
_lar.py — connect Snath Basis to the pre-employment Lár-JEPA engine.

Puts the published ten-ABC cognitive contract (core.interfaces) and the
RouteDecision enum (core.types) on the import path, so Snath Basis *implements*
the engine's contract rather than re-implementing it. Import this first.

This is what makes Snath Basis a Derivative Work of the pre-employment Lár-JEPA
prior art (Apache 2.0, github.com/snath-ai/Lar-JEPA) — in the quantitative-finance
domain — exactly as Snath Locus is in the biomedical domain.
"""
import os
import sys

_LAR_JEPA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lar_jepa"))
if os.path.isdir(_LAR_JEPA) and _LAR_JEPA not in sys.path:
    sys.path.insert(0, _LAR_JEPA)
