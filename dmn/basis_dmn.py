"""
Snath Basis — DMN consolidation (the memory that learns).

Reads the labelled D_hard queue (resolved fundamentals-vs-market divergences),
clusters them by Δ-pattern, and consolidates each cluster into a signed
**BasisAdapter** — a learned prior recording, for that type of disagreement:
which stream historically wins, how often, and the mean realised return.

This is the Snath Locus consolidation cycle (cluster D_hard → signed adapter
library) adapted to the factor-model stage. The adapter is a JSON prior now;
when neural encoders land it becomes a LoRA delta. The contract is identical:
consolidate → HMAC-sign → AdapterRouter verifies and applies at inference.

Run:  python dmn/basis_dmn.py
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import os, sys, json, hmac, hashlib, datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASIS = os.path.dirname(_HERE)
if _BASIS not in sys.path:
    sys.path.insert(0, _BASIS)

from dhard import DHardQueue, CLASSES   # the D_hard curriculum

_ADAPTER_KEY = b"snath_basis_adapter_2026"
_MIN_EVENTS  = 4     # min resolved events in a cluster before we trust an adapter


def _argmax_class(dist) -> str:
    return CLASSES[max(range(len(dist)), key=lambda i: dist[i])]


def _cluster_id(v_a, v_b) -> str:
    """A divergence's type = the directional pattern of disagreement."""
    return f"{_argmax_class(v_a)}->{_argmax_class(v_b)}"


@dataclass
class BasisAdapter:
    cluster_id:  str        # e.g. "overweight->underweight"
    centroid:    list       # mean Δ-vector (v_a - v_b) over the cluster
    winner:      str        # "fundamentals" | "market" | "neither"
    win_rate:    float      # fraction of the cluster the winner got right
    mean_return: float      # mean realised return (signed: + = overweight paid off)
    n_events:    int
    created_at:  str
    sig:         str = ""

    _IMM = ("cluster_id", "centroid", "winner", "win_rate", "mean_return",
            "n_events", "created_at")

    def _payload(self) -> bytes:
        return json.dumps({k: getattr(self, k) for k in self._IMM}, sort_keys=True).encode()

    def sign(self) -> "BasisAdapter":
        self.sig = hmac.new(_ADAPTER_KEY, self._payload(), hashlib.sha256).hexdigest()
        return self

    def verify(self) -> bool:
        want = hmac.new(_ADAPTER_KEY, self._payload(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.sig, want)


class BasisDMN:
    """Memory + consolidation over the D_hard queue."""

    def __init__(self, queue_path: str = "d_hard.jsonl",
                 adapter_dir: str = "models/adapters"):
        self.queue = DHardQueue(queue_path)
        self.adapter_dir = Path(adapter_dir)

    # ── consolidation: D_hard → signed adapter library ────────────────────────
    def consolidate(self, min_events: int = _MIN_EVENTS, verbose: bool = True):
        events = [e for e in self.queue.all() if e.realised_return is not None]
        groups = defaultdict(list)
        for e in events:
            groups[_cluster_id(e.v_a, e.v_b)].append(e)

        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        built = []
        for cid, g in sorted(groups.items()):
            if len(g) < min_events:
                if verbose:
                    print(f"  · {cid:<28} {len(g)} events — too few (need {min_events}), skipped")
                continue
            # centroid Δ-vector
            centroid = [round(sum(e.v_a[i] - e.v_b[i] for e in g) / len(g), 4) for i in range(len(CLASSES))]
            # who wins, how often
            wins = Counter(e.winner for e in g)
            winner, wc = wins.most_common(1)[0]
            win_rate = round(wc / len(g), 3)
            mean_return = round(sum(e.realised_return for e in g) / len(g), 4)

            a = BasisAdapter(cid, centroid, winner, win_rate, mean_return,
                             len(g), datetime.datetime.utcnow().isoformat() + "Z").sign()
            json_path = self.adapter_dir / f"{cid.replace('->', '__')}.json"
            json_path.write_text(json.dumps(asdict(a), indent=2))
            built.append(a)

            # ── SYSTEM 2: PyTorch LoRA (Deep Cortical Restructuring) ────────────
            # Train Rank-1 A,B matrices to close the divergence gap for this cluster.
            # The 'winner' stream is the target; the 'loser' stream is the faulty input.
            winner_vecs  = [e.v_a if winner == "fundamentals" else e.v_b for e in g]
            faulty_vecs  = [e.v_b if winner == "fundamentals" else e.v_a for e in g]
            target_enc   = "market" if winner == "fundamentals" else "fundamentals"

            target_t = torch.tensor(winner_vecs, dtype=torch.float32)
            faulty_t = torch.tensor(faulty_vecs, dtype=torch.float32)
            dim = target_t.shape[1]

            A = nn.Parameter(torch.randn(dim, 1) * 0.01)
            B = nn.Parameter(torch.randn(1, dim) * 0.01)
            optimizer = optim.AdamW([A, B], lr=0.1)

            for _ in range(150):
                optimizer.zero_grad()
                adapted = faulty_t + torch.matmul(torch.matmul(faulty_t, A), B)
                loss = torch.nn.functional.l1_loss(adapted, target_t)
                loss.backward()
                optimizer.step()

            pt_path = self.adapter_dir / f"{cid.replace('->', '__')}.pt"
            torch.save({"A": A.detach(), "B": B.detach(), "target_encoder": target_enc,
                        "cluster_id": cid, "final_loss": round(loss.item(), 6)}, str(pt_path))

            if verbose:
                print(f"  ✓ {cid:<28} n={len(g):<3} winner={winner:<12} "
                      f"win_rate={win_rate:<5} mean_ret={mean_return:+.3f}  "
                      f"LoRA loss={loss.item():.4f}")
        return built

    def stats(self) -> dict:
        return self.queue.stats()


# Demo lives in ../demo_dmn.py  (run: python demo_dmn.py) — kept out of the package
# so these modules stay clean library code (and avoid the package-as-script warning).
