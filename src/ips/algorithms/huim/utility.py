"""Fingerprint -> utility transaction, and the three external utilities p(i).

Item = one AP column index (0..519), present iff the AP was detected (RSSI != 100).
Internal utility q(i,T) = RSSI rescaled to an integer 1..q_max with the RSSI range of
the train split only; external utility p(i) is fitted on train labels only:

    A  p(i) = 1                                         (control: signal strength only)
    B  p(i) = max(I(A_i;Y), 1e-6) / max_j I(A_j;Y)      (mutual information with the class)
    C  p(i) ~ I(A_i;Y) * log(N / (1 + support_i))       (MI x IDF, normalised to (0,1])

I(A;Y) is the plug-in mutual information (Shannon 1948) over 6 AP states: missing and
five RSSI bins. These are the definitions of ThaiTaka/HUIM-IPS- (src/utility_models.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MISSING = 100
RSSI_EDGES = (-104.0, -90.0, -80.0, -70.0, -60.0, 0.0)
N_STATES = len(RSSI_EDGES)          # 5 detected bins + missing
MODELS = ("A", "B", "C")
EPS = 1e-12


@dataclass(frozen=True)
class TransactionEncoder:
    rssi_min: float
    rssi_max: float
    q_max: int = 100

    @classmethod
    def fit(cls, rssi: np.ndarray, q_max: int = 100) -> TransactionEncoder:
        detected = np.asarray(rssi)[np.asarray(rssi) != MISSING]
        if detected.size == 0:
            raise ValueError("Train split has no detected AP; cannot fit RSSI range.")
        lo, hi = float(detected.min()), float(detected.max())
        if hi <= lo:
            raise ValueError("Train RSSI range is degenerate.")
        return cls(lo, hi, q_max)

    def quantities(self, rssi: np.ndarray) -> np.ndarray:
        """q(i,T) as int64[N,520]; 0 = missing, never 0 for a detection."""
        rssi = np.asarray(rssi, dtype=np.float64)
        detected = rssi != MISSING
        scaled = np.clip((rssi - self.rssi_min) / (self.rssi_max - self.rssi_min), 0.0, 1.0)
        q = np.rint(1.0 + scaled * (self.q_max - 1.0)).astype(np.int64)
        q[~detected] = 0
        q[detected & (q < 1)] = 1
        return q

    def transactions(self, rssi: np.ndarray) -> list[dict[int, int]]:
        """One {ap_index: q} transaction per fingerprint, aligned with the input rows."""
        q = self.quantities(rssi)
        return [{int(j): int(row[j]) for j in np.flatnonzero(row)} for row in q]

    def as_dict(self) -> dict:
        return {"rssi_min": self.rssi_min, "rssi_max": self.rssi_max, "q_max": self.q_max,
                "missing_code": MISSING, "fitted_on": "train split only"}


def states(rssi: np.ndarray) -> np.ndarray:
    """0 = missing, 1..5 = RSSI bin (right-closed edges, -104 in bin 1, 0 dBm in bin 5)."""
    rssi = np.asarray(rssi, dtype=np.float64)
    bins = np.digitize(np.clip(rssi, RSSI_EDGES[0], RSSI_EDGES[-1]),
                       np.asarray(RSSI_EDGES[1:]), right=True) + 1
    return np.where(rssi == MISSING, 0, np.clip(bins, 1, N_STATES - 1)).astype(np.int64)


def mutual_information(state: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """I(A_j;Y) in bits for every column j, from the joint histogram."""
    _, y = np.unique(np.asarray(labels), return_inverse=True)
    n, n_classes = len(y), int(y.max()) + 1
    py = np.bincount(y, minlength=n_classes) / n
    mi = np.zeros(state.shape[1])
    for j in range(state.shape[1]):
        joint = np.bincount(state[:, j] * n_classes + y,
                            minlength=N_STATES * n_classes).reshape(N_STATES, n_classes) / n
        expected = joint.sum(axis=1, keepdims=True) * py[None, :]
        nonzero = joint > 0
        mi[j] = float(np.sum(joint[nonzero] * np.log2(joint[nonzero] / (expected[nonzero] + EPS))))
    return np.maximum(mi, 0.0)


def external_utility(model: str, rssi: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict]:
    """p(i) > 0 for every AP column plus diagnostics; train rows and labels only."""
    if model not in MODELS:
        raise ValueError(f"Unknown HUIM utility model {model!r}; expected one of {MODELS}.")
    rssi = np.asarray(rssi)
    if model == "A":
        return np.ones(rssi.shape[1]), {"model": "A", "formula": "p(i) = 1"}
    mi = mutual_information(states(rssi), labels)
    if mi.max() <= 0:
        raise ValueError("Every AP has zero mutual information with the class; use utility_model A.")
    if model == "B":
        raw = mi / mi.max()
        formula = "p(i) = max(I(A_i;Y), 1e-6) / max_j I(A_j;Y)"
    else:
        support = (rssi != MISSING).sum(axis=0)
        raw = mi * np.maximum(np.log(len(rssi) / (1.0 + support)), 0.0)
        raw = raw / raw.max() if raw.max() > 0 else raw
        formula = "p(i) ~ I(A_i;Y) * log(N/(1+support_i)), normalised to (0,1]"
    p = np.where(raw <= 0, 1e-6, raw)
    order = np.argsort(-mi, kind="stable")[:5]
    return p, {"model": model, "formula": formula, "mi_bits_max": float(mi.max()),
               "top_mi_ap_indices": [int(j) for j in order]}


def contributions(huis) -> dict:
    """contrib(i) = sum over HUIs X containing i of u(X)/|X| (each HUI shares its utility)."""
    score: dict = {}
    for itemset, utility, _support in huis:
        share = utility / len(itemset)
        for item in itemset:
            score[item] = score.get(item, 0.0) + share
    return score


def ranking(huis) -> list:
    """APs of the HUI set, highest contribution first (ties by AP index)."""
    score = contributions(huis)
    return sorted(score, key=lambda item: (-score[item], item))
