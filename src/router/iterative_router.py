"""Learned feedback router for partial execution states.

One sigmoid per branch estimates P(classification correct | observed feedback).
Executed branches can be returned; unexecuted candidates can be called. A call
has utility p_correct - call_cost, a return has utility p_correct. Scores are
recomputed after every tool result.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from .learned_router import FEATURE_DIM as _FULL_FEATURE_DIM
from .learned_router import extract_features as _extract_features_full
from .learned_router import is_overexposed
from .tool_registry import BRANCH_ORDER as _FULL_BRANCH_ORDER

SCHEMA = "iterative-feedback-v1"
FULL_BRANCH_ORDER = list(_FULL_BRANCH_ORDER)

# 活动分支表：默认等于完整 17 分支；configure_branches() 可切换为子集（如 13 工具）。
# 只改本模块的绑定，其他模块从 tool_registry 导入的 BRANCH_ORDER 不受影响。
BRANCH_ORDER = list(FULL_BRANCH_ORDER)
FEATURE_DIM = _FULL_FEATURE_DIM
N_BRANCHES = len(BRANCH_ORDER)
STATE_DIM = FEATURE_DIM + N_BRANCHES + 1


def configure_branches(exclude=()):
    """把活动分支表切换为「完整表去掉 exclude」的前缀子集。

    要求子集必须是 FULL_BRANCH_ORDER 的**前缀**（raw 在前，工具按原顺序），
    这样缓存特征可以按列直接切片，不需要重跑任何修复模型。
    """
    global BRANCH_ORDER, FEATURE_DIM, N_BRANCHES, STATE_DIM
    unknown = set(exclude) - set(FULL_BRANCH_ORDER[1:])
    if unknown:
        raise ValueError(f"Unknown excluded tools: {sorted(unknown)}")
    order = [up for up in FULL_BRANCH_ORDER if up not in set(exclude)]
    if order != FULL_BRANCH_ORDER[: len(order)]:
        raise ValueError("子集分支表必须是完整表的前缀，否则特征无法按列切片")
    BRANCH_ORDER = list(order)
    FEATURE_DIM = 6 + 5 * len(BRANCH_ORDER)
    N_BRANCHES = len(BRANCH_ORDER)
    STATE_DIM = FEATURE_DIM + N_BRANCHES + 1
    return list(BRANCH_ORDER)


def _validated_order(branch_order):
    if (not isinstance(branch_order, (list, tuple)) or not branch_order
            or any(not isinstance(up, str) for up in branch_order)
            or branch_order[0] != "" or len(set(branch_order)) != len(branch_order)
            or any(up not in FULL_BRANCH_ORDER for up in branch_order)):
        raise ValueError("Invalid router branch order: raw must be first, with unique registered tools")
    return tuple(branch_order)


def extract_features_for(diagnosis, outcomes, branch_order=None):
    """Reuse the full extractor and select branch blocks by name, without global changes."""
    order = _validated_order(BRANCH_ORDER if branch_order is None else branch_order)
    full = _extract_features_full(diagnosis, outcomes)
    if list(order) == FULL_BRANCH_ORDER:
        return full
    indices = [FULL_BRANCH_ORDER.index(up) for up in order]
    blocks = full[6:].reshape(len(FULL_BRANCH_ORDER), 5)[indices]
    return np.concatenate((full[:6], blocks.reshape(-1)))


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_state(full_features, available, observed, remaining, budget, *, branch_order=None):
    """Mask unseen outcomes BEFORE feeding the network (including confidence).

    full_features may hold complete offline outcomes. Their values outside the
    observed mask never enter the state. Labels/ground truth are not arguments.
    """
    order = _validated_order(BRANCH_ORDER if branch_order is None else branch_order)
    n_branches = len(order)
    feature_dim = 6 + 5 * n_branches
    features = np.asarray(full_features, dtype=np.float32).copy()
    available = np.asarray(available, dtype=bool)
    observed = np.asarray(observed, dtype=bool)
    if features.shape != (feature_dim,) or available.shape != (n_branches,) or observed.shape != (n_branches,):
        raise ValueError("Invalid feedback state dimensions")
    if not available[0] or not observed[0] or np.any(observed & ~available):
        raise ValueError("raw must be observed; observed must be a subset of available")
    if not 0 <= remaining <= budget <= n_branches - 1:
        raise ValueError("Invalid call budget")
    branches = features[6:].reshape(n_branches, 5)
    branches[~observed] = 0
    branches[observed, 0] = 1
    result = np.concatenate((features, available.astype(np.float32),
                             np.array([remaining / max(budget, 1)], dtype=np.float32)))
    if not np.isfinite(result).all():
        raise ValueError("Non-finite router state")
    return result


@dataclass
class RouterAction:
    kind: str  # call / stop
    upstream: str
    reason: str
    probabilities: list


def choose_action(probabilities, state, call_cost, clip_max=0.05, *, branch_order=None):
    """Common rollout decision used by training validation and real inference."""
    order = _validated_order(BRANCH_ORDER if branch_order is None else branch_order)
    n_branches = len(order)
    feature_dim = 6 + 5 * n_branches
    state_dim = feature_dim + n_branches + 1
    p = np.asarray(probabilities, dtype=np.float64)
    state = np.asarray(state, dtype=np.float32)
    if p.shape != (n_branches,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid learned router probabilities")
    if state.shape != (state_dim,):
        raise ValueError("Invalid state shape")
    bf = state[6:feature_dim].reshape(n_branches, 5)
    observed = bf[:, 0] > 0.5
    available = state[feature_dim:feature_dim + n_branches] > 0.5
    if not observed[0]:
        raise ValueError("raw feedback required")
    # Safety masks only. Confidence/margin decisions come from the learned model.
    valid_return = observed & (bf[:, 4] < 0.5) & (bf[:, 3] <= clip_max + 1e-6)
    valid_return[0] = True
    best = int(np.where(valid_return, p, -np.inf).argmax())
    pending = available & ~observed
    probs = [float(v) for v in p]
    if state[-1] <= 0:
        return RouterAction("stop", order[best], "budget_exhausted", probs)
    if not pending.any():
        return RouterAction("stop", order[best], "candidates_exhausted", probs)
    nxt = int(np.where(pending, p - call_cost, -np.inf).argmax())
    # Ties prefer returning an existing result; raw wins return ties by order.
    if p[nxt] - call_cost <= p[best]:
        return RouterAction("stop", order[best], "learned_stop", probs)
    return RouterAction("call", order[nxt], "learned_next_tool", probs)


class IterativeRouter:
    def __init__(self, weight_path, device="cpu", classifier_weight=None):
        import torch
        from ._router_net import RouterNet

        ck = torch.load(weight_path, map_location="cpu", weights_only=False)
        if (not isinstance(ck, dict) or ck.get("schema") != SCHEMA
                or ck.get("output_activation") != "sigmoid"):
            raise ValueError("Iterative router needs a compatible partial-feedback checkpoint; old routers cannot be reused")
        self.branch_order = _validated_order(ck.get("branch_order"))
        self.n_branches = len(self.branch_order)
        self.feature_dim = 6 + 5 * self.n_branches
        self.state_dim = self.feature_dim + self.n_branches + 1
        if ck.get("feature_dim") != self.state_dim:
            raise ValueError("Checkpoint state dimension does not match its branch order")
        self.budget = int(ck["max_calls"])
        self.call_cost = float(ck["call_cost"])
        self.clip_max = float(ck["clip_max"])
        if not 0 <= self.budget < self.n_branches or not 0 <= self.call_cost < 1 or not 0 <= self.clip_max <= 1:
            raise ValueError("Invalid checkpoint policy settings")
        if classifier_weight and file_hash(classifier_weight) != ck.get("classifier_sha256"):
            raise ValueError("Classifier does not match iterative router training cache")
        self.device = device
        self.diag_mean = np.asarray(ck["diag_mean"], dtype=np.float32)
        self.diag_std = np.asarray(ck["diag_std"], dtype=np.float32)
        if (self.diag_mean.shape != (6,) or self.diag_std.shape != (6,)
                or not np.isfinite(self.diag_mean).all() or not np.isfinite(self.diag_std).all()
                or np.any(self.diag_std <= 0)):
            raise ValueError("Invalid diagnostic normalization")
        self.model = RouterNet(self.state_dim, hidden=int(ck["hidden"]), num_branches=self.n_branches).to(device)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()

    def probabilities(self, states):
        import torch

        arr = np.asarray(states, dtype=np.float32).copy()
        single = arr.ndim == 1
        arr = np.atleast_2d(arr)
        if arr.ndim != 2 or arr.shape[1] != self.state_dim or not np.isfinite(arr).all():
            raise ValueError("Invalid state shape or values for this router")
        arr[:, :6] = (arr[:, :6] - self.diag_mean) / self.diag_std
        with torch.inference_mode():
            probs = torch.sigmoid(self.model(torch.from_numpy(arr).to(self.device))).cpu().numpy()
        return probs[0] if single else probs

    def decide(self, diagnosis, outcomes, candidates):
        # Legacy test doubles may omit checkpoint metadata; real instances own their layout.
        order = getattr(self, "branch_order", tuple(FULL_BRANCH_ORDER))
        names = {o.upstream for o in outcomes}
        if not names <= set(order):
            raise ValueError("Observed outcomes include a branch outside this router")
        available = np.array([up in candidates for up in order])
        observed = np.array([up in names for up in order])
        features = extract_features_for(diagnosis, outcomes, order)
        remaining = max(0, self.budget - sum(bool(up) for up in names))
        state = make_state(features, available, observed, remaining, self.budget, branch_order=order)
        return choose_action(self.probabilities(state), state, self.call_cost, self.clip_max,
                             branch_order=order)
