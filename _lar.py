"""
_lar.py — locate the Lár-JEPA engine for Snath Basis.

Resolution order:
  1. $SNATH_BASIS_LARJEPA   — set this env var to point to your lar_jepa/ directory
  2. ../lar_jepa            — sibling directory (clone Lar-JEPA next to this repo)
  3. ~/lar_jepa             — home directory install

To install the engine:
  git clone https://github.com/snath-ai/Lar-JEPA.git
  export SNATH_BASIS_LARJEPA=/path/to/Lar-JEPA/lar_jepa
"""
import os
import sys

_CANDIDATES = [
    os.environ.get("SNATH_BASIS_LARJEPA"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Lar-JEPA", "lar_jepa")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lar_jepa")),
    os.path.expanduser("~/lar_jepa"),
]

_LAR_JEPA = None
for _p in _CANDIDATES:
    if _p and os.path.isfile(os.path.join(_p, "core", "interfaces.py")):
        if _p not in sys.path:
            sys.path.insert(0, _p)
        _LAR_JEPA = _p
        break

if not _LAR_JEPA:
    raise RuntimeError(
        "Lár-JEPA engine not found.\n"
        "Clone it and set the env var:\n"
        "  git clone https://github.com/snath-ai/Lar-JEPA.git\n"
        "  export SNATH_BASIS_LARJEPA=/path/to/Lar-JEPA/lar_jepa"
    )
