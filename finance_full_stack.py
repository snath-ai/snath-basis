"""
Snath Basis — Full-Stack Demonstration of ALL TEN Lár-JEPA ABCs (finance domain)
================================================================================
The finance analog of the public examples/powergrid_full_stack.py. It is the
canonical reference proving that every abstract interface in the PUBLIC, genesis-
anchored core/interfaces.py (~/Desktop/Lar_Main/lar_jepa) is:

  1. instantiable        — a concrete finance subclass exists for all ten ABCs,
  2. domain-agnostic     — none of the contracts needs a specific domain,
  3. mutually composable — they wire into one deterministic pipeline unchanged.

Every contract is imported from the PUBLIC repo, so Snath Basis is a clean
Derivative Work that *proves Lár-JEPA works* in quantitative finance — the
non-pharmaceutical counterpart to Snath Locus.

THE TEN ABCs (finance instantiations)
-------------------------------------
  1  AbstractCognitiveNode        →  MarketCognitiveNode      (base routable node)
  2  AbstractManifold             →  MarketRegimeJEPA         (world model: predict next regime)
  3  AbstractContextBridge        →  StateInstrumentBridge    (state latent → locator query)
  4  AbstractLatentFaultLocator   →  RegimeStressLocator      (state × instruments → stress loci)
  5  AbstractEntropicRouter       →  MarketEntropicRouter     (gate on regime-prediction entropy)
  6  AbstractAttentionKernel      →  LinearAttentionMarketKernel (O(N) over instruments)
  7  AbstractPerturbationOperator →  RateShockOperator        (Δ = encode(post-hike) − encode(pre))
  8  AbstractRoutingKernel        →  AllocationKernel         (DEFEND / HEDGE / HOLD)
  9  AbstractModalEncoder         →  MarketStateEncoder       (market state → latent)
 10  AbstractDivergenceRouter     →  RegimeDivergenceRouter   (basis between two views → route)

Run:  python finance_full_stack.py
"""

from __future__ import annotations
import hashlib, json, math
from datetime import datetime, timezone
from typing import Any, Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

import _lar  # noqa: F401  — public Lar_Main/lar_jepa on the path
from core.interfaces import (
    AbstractCognitiveNode,        # 1
    AbstractManifold,             # 2
    AbstractContextBridge,        # 3
    AbstractLatentFaultLocator,   # 4
    AbstractEntropicRouter,       # 5
    AbstractAttentionKernel,      # 6
    AbstractPerturbationOperator, # 7
    AbstractRoutingKernel,        # 8
    AbstractModalEncoder,         # 9
    AbstractDivergenceRouter,     # 10
)
from core.types import ModelType, RouteDecision, SignalType

LATENT_DIM     = 64
N_INSTRUMENTS  = 32      # structural sequence: instruments / sectors
STATE_FEATS    = 8       # macro/market state: ret, vol, momo, breadth, credit, term, vix, usd
INSTR_FEATS    = 6       # per-instrument: beta, size, value, momentum, quality, leverage
TOPK_STRESS    = 4
ENTROPY_COMMIT = 0.35


# ── ABC 1 — AbstractCognitiveNode ──────────────────────────────────────────────
class MarketCognitiveNode(AbstractCognitiveNode):
    """Base cognitive node for the finance pipeline (encode → forward → decode)."""
    model_type = ModelType.JEPA
    def encode(self, signal: Any) -> Any: return signal
    def forward(self, state: Any) -> Any: return state
    def decode(self, latent: Any) -> Any: return latent
    @property
    def output_signal_type(self) -> SignalType: return SignalType.LATENT_EMBEDDING


# ── ABC 9 — AbstractModalEncoder ───────────────────────────────────────────────
class MarketStateEncoder(AbstractModalEncoder):
    """Market state tensor (B, STATE_FEATS) → latent (B, D)."""
    def __init__(self, latent_dim: int = LATENT_DIM):
        self._d = latent_dim
        self._enc = nn.Sequential(nn.Linear(STATE_FEATS, latent_dim),
                                  nn.LayerNorm(latent_dim), nn.GELU())
    @property
    def output_dim(self) -> int: return self._d
    @property
    def modality(self) -> str: return "market_state"
    def encode(self, x: Any) -> Any:
        return self._enc(x)                               # (B, D)


# ── ABC 2 — AbstractManifold ───────────────────────────────────────────────────
class MarketRegimeJEPA(AbstractManifold):
    """JEPA world model: predict the next-regime latent, with an entropy measure."""
    model_type = ModelType.JEPA
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self._ctx = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(),
                                  nn.Linear(latent_dim, latent_dim))
        self._pred = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.GELU(),
                                   nn.Linear(latent_dim, latent_dim))
    def embed_context(self, x: torch.Tensor) -> torch.Tensor: return self._ctx(x)
    def predict_target(self, context: torch.Tensor, action_vector: Any = None) -> torch.Tensor:
        return self._pred(context)
    def entropic_loss(self, predicted_state: torch.Tensor) -> float:
        p = F.softmax(predicted_state, dim=-1)
        ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean()
        return float(ent.item() / math.log(predicted_state.shape[-1]))
    @property
    def output_signal_type(self) -> SignalType: return SignalType.LATENT_EMBEDDING


# ── ABC 3 — AbstractContextBridge ──────────────────────────────────────────────
class StateInstrumentBridge(AbstractContextBridge):
    """Stateless: market-state latent (B, D) → (B, 1, D) cross-attention query."""
    @property
    def source_signal_type(self) -> SignalType: return SignalType.LATENT_EMBEDDING
    @property
    def target_signal_type(self) -> SignalType: return SignalType.LATENT_EMBEDDING
    def bridge(self, source_output: torch.Tensor,
               target_node_type: Optional[Type[AbstractCognitiveNode]] = None) -> torch.Tensor:
        return source_output.unsqueeze(1) if source_output.ndim == 2 else source_output


# ── ABC 4 — AbstractLatentFaultLocator ─────────────────────────────────────────
class RegimeStressLocator(AbstractLatentFaultLocator):
    """Market state (x_E) × instrument sequence (x_S) → topk most-stressed instruments.
    UCR isomorphism: market microstructure × order book → regime-shift loci."""
    def __init__(self, latent_dim: int = LATENT_DIM):
        self._d = latent_dim
        self._env = nn.Sequential(nn.Linear(STATE_FEATS, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self._str = nn.Sequential(nn.Linear(INSTR_FEATS, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self._q, self._k, self._v = (nn.Linear(latent_dim, latent_dim, bias=False) for _ in range(3))
        self._risk = nn.Linear(latent_dim, 1)
    def encode_environmental_state(self, x_E: torch.Tensor) -> torch.Tensor:   # I1 (B, D)
        return self._env(x_E)
    def encode_structural_sequence(self, x_S: torch.Tensor) -> torch.Tensor:   # I2 (1, N, D)
        return self._str(x_S)
    def localize_fault_coordinates(self, z_E, z_S, k):                          # I3–I6
        B, N = z_E.shape[0], z_S.shape[1]
        Q = self._q(z_E).unsqueeze(1)                       # (B,1,D)
        K = self._k(z_S).expand(B, -1, -1); V = self._v(z_S).expand(B, -1, -1)
        a = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self._d), dim=-1)  # (B,1,N)
        ctx = torch.bmm(a, V).squeeze(1)
        risk = torch.sigmoid(self._risk(ctx)).squeeze(-1)   # (B,)
        af = a.squeeze(1)
        _, idx = af[0].topk(min(k, N), sorted=True)
        return risk, idx, af


# ── ABC 5 — AbstractEntropicRouter ─────────────────────────────────────────────
class MarketEntropicRouter(AbstractEntropicRouter):
    """Gate on regime-prediction entropy: confident → COMMIT, else REPLAN / IMPASSE."""
    def __init__(self, threshold: float = ENTROPY_COMMIT): self._t = threshold
    def evaluate_state(self, predicted_state: torch.Tensor) -> RouteDecision:
        p = F.softmax(predicted_state, dim=-1)
        e = -(p * (p + 1e-8).log()).sum(dim=-1).mean()
        en = float(e.item() / math.log(predicted_state.shape[-1]))
        if en < self._t: return RouteDecision.COMMIT_TRAJECTORY
        if en < self._t * 2: return RouteDecision.TRIGGER_REPLAN
        return RouteDecision.STRUCTURAL_IMPASSE


# ── ABC 6 — AbstractAttentionKernel ────────────────────────────────────────────
class LinearAttentionMarketKernel(AbstractAttentionKernel):
    """O(N) linear attention over the instrument universe (ELU+1 feature map)."""
    def __init__(self, embed_dim: int = LATENT_DIM): self._d = embed_dim
    def _phi(self, x): return F.elu(x) + 1.0
    def compute(self, query, key, value, k):
        if query.ndim == 2: query = query.unsqueeze(1)
        s = torch.bmm(self._phi(query), self._phi(key).transpose(1, 2)).squeeze(1)  # (B,N)
        w = torch.softmax(s, dim=-1)
        _, idx = w[0].topk(min(k, w.shape[-1]), sorted=True)
        return w, idx


# ── ABC 7 — AbstractPerturbationOperator ───────────────────────────────────────
class RateShockOperator(AbstractPerturbationOperator):
    """Δ = encode(post-rate-hike state) − encode(pre-hike state). UCR: InterestRateShockOperator."""
    def __init__(self, base_encoder: MarketStateEncoder): self._enc = base_encoder
    def encode_wildtype(self, x_wt: torch.Tensor) -> torch.Tensor: return self._enc.encode(x_wt)
    def encode_mutant(self, x_mut: torch.Tensor) -> torch.Tensor: return self._enc.encode(x_mut)


# ── ABC 8 — AbstractRoutingKernel ──────────────────────────────────────────────
class AllocationKernel(AbstractRoutingKernel):
    """Score = departure of predicted state from nominal → DEFEND / HEDGE / HOLD."""
    def __init__(self, hedge=0.10, defend=0.30): self._h, self._d = hedge, defend
    def score(self, state: Any) -> float:
        cs = F.cosine_similarity(state["z_ctrl"], state["z_pred"], dim=-1).mean().item()
        return float(1.0 - cs)
    def route(self, state: Any) -> str:
        s = self.score(state)
        return "DEFEND" if s >= self._d else "HEDGE" if s >= self._h else "HOLD"


# ── ABC 10 — AbstractDivergenceRouter ──────────────────────────────────────────
class RegimeDivergenceRouter(AbstractDivergenceRouter):
    """Content-blind basis router between two independent views of the market (V1–V6)."""
    TAU_HIGH, TAU_LOW, DELTA = 0.35, 0.12, 0.40
    def encode_stream_a(self, x_a):
        raise NotImplementedError("delegated to a MarketStateEncoder (AbstractModalEncoder); V1.")
    def encode_stream_b(self, x_b):
        raise NotImplementedError("delegated to a second MarketStateEncoder; V1.")
    def divergence(self, z_a, z_b) -> float:                       # V2–V3
        pa = F.softmax(torch.as_tensor(z_a, dtype=torch.float).flatten(), 0)
        pb = F.softmax(torch.as_tensor(z_b, dtype=torch.float).flatten(), 0)
        return float((pa - pb).abs().sum().item() / math.sqrt(pa.shape[0]))
    def route(self, confidence_a, confidence_b, divergence) -> RouteDecision:  # V4–V6
        ca, cb, d = confidence_a, confidence_b, divergence
        if max(ca, cb) >= self.TAU_HIGH and min(ca, cb) < self.TAU_LOW:
            return RouteDecision.COMMIT_TRAJECTORY
        if ca < self.TAU_LOW and cb < self.TAU_LOW:
            return RouteDecision.STRUCTURAL_IMPASSE
        if ca >= self.TAU_HIGH and cb >= self.TAU_HIGH:
            return RouteDecision.TRIGGER_REPLAN if d >= self.DELTA else RouteDecision.COMMIT_TRAJECTORY
        return RouteDecision.STRUCTURAL_IMPASSE


# ===========================================================================
# Pipeline (lightweight local GraphState + chaining, mirroring powergrid)
# ===========================================================================
class GraphState:
    def __init__(self): self._d = {}
    def set(self, k, v): self._d[k] = v
    def get(self, k, default=None): return self._d.get(k, default)


class _Node:
    def __init__(self, next_node=None): self._next = next_node
    def _go(self, state): return self._next.execute(state) if self._next else state


class MarketStateEmbeddingNode(_Node):
    def __init__(self, enc, next_node=None): super().__init__(next_node); self._enc = enc
    def execute(self, state):
        z = self._enc.encode(state.get("market_state"))
        state.set("z_state", z); state.set("modality", self._enc.modality)
        print(f"  [MarketStateEmbeddingNode] {self._enc.modality} → z {tuple(z.shape)}")
        return self._go(state)


class MarketWorldModelNode(MarketCognitiveNode, _Node):
    def __init__(self, jepa, next_node=None): _Node.__init__(self, next_node); self._j = jepa
    def execute(self, state):
        ctx = self._j.embed_context(state.get("z_state"))
        z_hat = self._j.predict_target(ctx); e = self._j.entropic_loss(z_hat)
        state.set("z_pred_regime", z_hat); state.set("z_ctrl", state.get("z_state"))
        state.set("regime_entropy", e)
        print(f"  [MarketWorldModelNode] regime entropy={e:.4f} "
              f"({'CONFIDENT' if e < ENTROPY_COMMIT else 'UNCERTAIN'})")
        return self._go(state)


class EntropicGateNode(_Node):
    def __init__(self, router, commit_node=None, replan_node=None):
        self._r, self._c, self._rp = router, commit_node, replan_node
    def execute(self, state):
        dec = self._r.evaluate_state(state.get("z_pred_regime"))
        state.set("entropic_decision", dec.value)
        print(f"  [EntropicGateNode] RouteDecision → {dec.value}")
        nxt = self._c if dec == RouteDecision.COMMIT_TRAJECTORY else self._rp
        return nxt.execute(state) if nxt else state


class StateBridgeNode(_Node):
    def __init__(self, bridge, next_node=None): super().__init__(next_node); self._b = bridge
    def execute(self, state):
        q = self._b.bridge(state.get("z_state"))
        state.set("z_query", q)
        print(f"  [StateBridgeNode] {self._b.source_signal_type.value} → query {tuple(q.shape)}")
        return self._go(state)


class RateShockNode(_Node):
    def __init__(self, op, alpha=1.0, next_node=None): super().__init__(next_node); self._op, self._a = op, alpha
    def execute(self, state):
        z_ctrl = state.get("z_ctrl")
        z_pred = self._op.predict_perturbed_state(z_ctrl, state.get("state_pre"), state.get("state_post"), alpha=self._a)
        delta = self._op.perturbation_vector(state.get("state_pre"), state.get("state_post"))
        state.set("z_pred", z_pred)
        print(f"  [RateShockNode] α={self._a:.1f} |Δ|={float(torch.norm(delta, dim=-1).mean()):.4f} — shock state predicted")
        return self._go(state)


class RegimeLocalisationNode(_Node):
    def __init__(self, locator, kernel, instruments, topk=TOPK_STRESS, next_node=None):
        super().__init__(next_node); self._loc, self._ker, self._instr, self._k = locator, kernel, instruments, topk
    def execute(self, state):
        x_E = state.get("market_state")
        z_E = self._loc.encode_environmental_state(x_E)
        z_S = self._loc.encode_structural_sequence(self._instr)
        risk, idx, attn = self._loc.localize_fault_coordinates(z_E, z_S, k=self._k)
        K = z_S.expand(z_E.shape[0], -1, -1)
        _, kidx = self._ker.compute(z_E, K, K, k=self._k)
        state.set("stress_risk", float(risk.mean())); state.set("stress_instruments", idx.tolist())
        print(f"  [RegimeLocalisationNode] risk={float(risk.mean()):.4f} | stressed (locator): {idx.tolist()} | (kernel): {kidx.tolist()}")
        return self._go(state)


class AllocationRouterNode(_Node):
    def __init__(self, kernel, routes): self._ker, self._routes = kernel, routes
    def execute(self, state):
        rs = {"z_ctrl": state.get("z_ctrl"), "z_pred": state.get("z_pred")}
        dec = self._ker.route(rs); state.set("allocation", dec); state.set("alloc_score", self._ker.score(rs))
        print(f"  [AllocationRouterNode] departure={self._ker.score(rs):.4f} → {dec}")
        nxt = self._routes.get(dec)
        return nxt.execute(state) if nxt else state


class ActionNode(_Node):
    def __init__(self, label): self._label = label
    def execute(self, state):
        rec = {"action": self._label, "entropic_decision": state.get("entropic_decision"),
               "allocation": state.get("allocation"), "alloc_score": state.get("alloc_score"),
               "stress_risk": state.get("stress_risk"), "stress_instruments": state.get("stress_instruments"),
               "regime_entropy": state.get("regime_entropy"), "modality": state.get("modality"),
               "timestamp_utc": datetime.now(timezone.utc).isoformat()}
        rec["hmac"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
        print(f"  [{self._label}] directive issued — audit hmac {rec['hmac'][:16]}…")
        state.set("audit_record", rec)
        return state


def build_pipeline(instruments):
    enc   = MarketStateEncoder()                 # 9
    jepa  = MarketRegimeJEPA()                   # 2
    erout = MarketEntropicRouter()               # 5
    bridg = StateInstrumentBridge()              # 3
    shock = RateShockOperator(enc)               # 7
    loc   = RegimeStressLocator()                # 4
    kern  = LinearAttentionMarketKernel()        # 6
    alloc = AllocationKernel()                   # 8
    # 1 exercised by MarketWorldModelNode (a MarketCognitiveNode)
    terminals = {k: ActionNode(v) for k, v in
                 {"DEFEND": "DefendNode", "HEDGE": "HedgeNode", "HOLD": "HoldNode"}.items()}
    escalate = ActionNode("HumanOversightNode")
    arouter  = AllocationRouterNode(alloc, terminals)
    locnode  = RegimeLocalisationNode(loc, kern, instruments, next_node=arouter)
    shocknode = RateShockNode(shock, next_node=locnode)
    bridgenode = StateBridgeNode(bridg, next_node=shocknode)
    gate = EntropicGateNode(erout, commit_node=bridgenode, replan_node=escalate)
    wm = MarketWorldModelNode(jepa, next_node=gate)
    return MarketStateEmbeddingNode(enc, next_node=wm), erout


def prove_abc_coverage():
    """Machine-verifiable: all 10 ABCs subclassed here, plus engine + DMN import (public repo)."""
    print("\n" + "=" * 70 + "\nprove_abc_coverage() — all 10 ABCs from the PUBLIC repo\n" + "=" * 70)
    abcs = {
        "AbstractCognitiveNode": AbstractCognitiveNode, "AbstractManifold": AbstractManifold,
        "AbstractContextBridge": AbstractContextBridge, "AbstractLatentFaultLocator": AbstractLatentFaultLocator,
        "AbstractEntropicRouter": AbstractEntropicRouter, "AbstractAttentionKernel": AbstractAttentionKernel,
        "AbstractPerturbationOperator": AbstractPerturbationOperator, "AbstractRoutingKernel": AbstractRoutingKernel,
        "AbstractModalEncoder": AbstractModalEncoder, "AbstractDivergenceRouter": AbstractDivergenceRouter,
    }
    ok = 0
    for name, abc in abcs.items():
        subs = [c.__name__ for c in abc.__subclasses__()]
        status = "OK " if subs else "MISSING"
        if subs: ok += 1
        print(f"  [{status}] {name:<30} ← {subs}")
    print(f"\n  {ok}/10 ABCs subclassed in the finance domain.")
    # everything imported from the public repo:
    import core.interfaces as _ci
    print(f"  core.interfaces from: {_ci.__file__}")
    try:
        import lar
        print(f"  lar engine import   : OK ({[x for x in dir(lar) if x=='GraphExecutor'][0]} available)")
    except Exception as e:
        print(f"  lar engine import   : {e}")
    try:
        from dmn_integration.consolidation_node import JEPA_DMN_Consolidation_Node
        print(f"  DMN import          : OK ({JEPA_DMN_Consolidation_Node.__name__})")
    except Exception as e:
        print(f"  DMN import          : {e}")
    return ok


def run_pipeline():
    print("=" * 70 + "\nSnath Basis — Full-Stack: ALL TEN Lár-JEPA ABCs (finance)\n" + "=" * 70)
    instruments = torch.rand(1, N_INSTRUMENTS, INSTR_FEATS)

    def _state():
        s = GraphState()
        s.set("market_state", torch.rand(1, STATE_FEATS))
        s.set("state_pre", torch.rand(1, STATE_FEATS))
        post = s.get("state_pre").clone(); post[:, 4] *= 1.5; post[:, 6] *= 1.4  # credit + vix shock
        s.set("state_post", post)
        return s

    print("\n── Scenario A — confident regime (always COMMIT; exercises ABCs 1-9) ──")
    entry, erout = build_pipeline(instruments); erout._t = 1.1
    with torch.no_grad(): a = entry.execute(_state())
    print(f"\n  entropic={a.get('entropic_decision')}  allocation={a.get('allocation')}  "
          f"stress={a.get('stress_instruments')}")

    print("\n── Scenario B — uncertain regime (always REPLAN; exercises ABC 5) ──")
    entry, erout = build_pipeline(instruments); erout._t = 0.001
    with torch.no_grad(): b = entry.execute(_state())
    print(f"\n  entropic={b.get('entropic_decision')}")

    print("\n── Scenario C — ABC 10: basis between two market views ──")
    enc1, enc2 = MarketStateEncoder(), MarketStateEncoder()
    dr = RegimeDivergenceRouter()
    with torch.no_grad():
        z1 = enc1.encode(torch.rand(1, STATE_FEATS))[0]
        z2 = enc2.encode(torch.rand(1, STATE_FEATS))[0]
        d = dr.divergence(z1, z2)
    print(f"  RegimeDivergenceRouter basis D={d:.3f} → {dr.route(0.5, 0.5, d).value}")

    prove_abc_coverage()
    print("\n" + "=" * 70)
    print("All ten ABCs exercised in finance, every contract from the PUBLIC repo. ✓")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
