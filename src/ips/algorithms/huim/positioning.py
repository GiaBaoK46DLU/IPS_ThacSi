"""Weighted k-NN fingerprint positioning and its staged selection on internal validation.

Port of the PHASE I3 estimator of ThaiTaka/HUIM-IPS- (src/optimize_positioning.py), which took
the UJIIndoorLoc mean error from 9.28 to 5.36 ground metres there. The stages, grids and the
0.05 gain rule are unchanged; the selection set is this repo's internal validation split
instead of grouped folds, because here validation is the split reserved for choosing.

    S0  frozen baseline: positive representation, Manhattan, k = 1
    S1  best representation x distance at k = 1
    S2  + k neighbours and weights w = 1/d^p                       (Shepard 1968)
    S3  + per-query device offset: RSSI + delta dB, keep the delta whose nearest
          train fingerprint is closest (uses the query's own signal only)
    S4  + average with the best model on a second representation
S3 and S4 are kept only if they lower the validation mean error by at least min_gain.
Representations and distances: Torres-Sospedra et al., ESWA 42(23), 2015.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .utility import MISSING

K_MAX = 30
K_GRID = tuple(range(1, 21)) + (25, 30)
POWERS = (0, 1, 2, 3, 4)
REPRESENTATIONS = ("positive", "powed", "exponential")
DISTANCES = ("euclidean", "manhattan", "braycurtis", "cosine")
OFFSET_RANGES = (3, 6, 9)
OFFSET_STEP = 3


@dataclass(frozen=True)
class KNNConfig:
    representation: str = "positive"
    distance: str = "manhattan"
    k: int = 1
    power: float = 0
    offset_range: int = 0            # 0 = no device-offset search
    ensemble_with: str = ""          # second representation, "" = none

    def label(self) -> str:
        text = f"{self.representation}+{self.distance}, k={self.k}, w=1/d^{self.power:g}"
        if self.offset_range:
            text += f", offset +-{self.offset_range} dB"
        if self.ensemble_with:
            text += f", ensemble with {self.ensemble_with}"
        return text

    def as_dict(self) -> dict:
        return asdict(self)


BASELINE = KNNConfig()


def representation(rssi: np.ndarray, kind: str) -> np.ndarray:
    """Missing (100) -> 0; detected RSSI on the documented [-104, 0] dBm scale."""
    rssi = np.asarray(rssi, dtype=np.float64)
    detected = rssi != MISSING
    r = np.clip(rssi, -104.0, 0.0)
    if kind == "positive":
        return np.where(detected, 1.0 + (r + 104.0) / 104.0 * 99.0, 0.0)
    if kind == "powed":
        return np.where(detected, ((r + 105.0) / 105.0) ** np.e, 0.0)
    if kind == "exponential":
        return np.where(detected, np.exp((r + 105.0) / 24.0) / np.exp(105.0 / 24.0), 0.0)
    raise ValueError(f"Unknown representation {kind!r}")


def shift(rssi: np.ndarray, delta: float) -> np.ndarray:
    rssi = np.asarray(rssi, dtype=np.float64)
    return np.where(rssi != MISSING, np.clip(rssi + delta, -104.0, 0.0), MISSING)


@dataclass
class Neighbours:
    index: np.ndarray        # [n_query, K] rows of the reference set
    distance: np.ndarray     # [n_query, K] ascending


def neighbours(reference: np.ndarray, query: np.ndarray, metric: str, k: int = K_MAX) -> Neighbours:
    """k nearest reference rows ordered by (distance, row): integer RSSI makes exact ties common,
    so the explicit secondary key keeps every estimate independent of sklearn's tie handling."""
    k = min(k, len(reference))
    index = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric, n_jobs=-1).fit(reference)
    distance, rows = index.kneighbors(query)
    order = np.lexsort((rows, distance), axis=1)
    return Neighbours(np.take_along_axis(rows, order, axis=1), np.take_along_axis(distance, order, axis=1))


def weights(distance: np.ndarray, k: int, power: float) -> np.ndarray:
    w = 1.0 / np.maximum(distance[:, :k], 1e-9) ** power
    return w / w.sum(axis=1, keepdims=True)


def estimate(found: Neighbours, xy: np.ndarray, k: int, power: float) -> np.ndarray:
    return np.einsum("ij,ijk->ik", weights(found.distance, k, power), xy[found.index[:, :k]])


def vote(found: Neighbours, labels: np.ndarray, n_classes: int, k: int, power: float):
    """Weighted class vote of the same neighbours; confidence = winning weight share."""
    w = weights(found.distance, k, power)
    score = np.zeros((len(w), n_classes))
    np.add.at(score, (np.repeat(np.arange(len(w)), w.shape[1]), labels[found.index[:, :k]].ravel()),
              w.ravel())
    return score.argmax(axis=1), score.max(axis=1)


def offset_select(by_delta: dict[float, Neighbours], limit: int) -> tuple[Neighbours, np.ndarray]:
    """Per query, the delta whose nearest neighbour is closest; tried from 0 outwards."""
    grid = sorted((d for d in by_delta if abs(d) <= limit), key=lambda d: (abs(d), d))
    first = by_delta[grid[0]]
    best = np.full(len(first.index), np.inf)
    index, distance = np.zeros_like(first.index), np.zeros_like(first.distance)
    chosen = np.zeros(len(first.index))
    for d in grid:
        found = by_delta[d]
        better = found.distance[:, 0] < best
        best[better], chosen[better] = found.distance[better, 0], d
        index[better], distance[better] = found.index[better], found.distance[better]
    return Neighbours(index, distance), chosen


def deltas(limit: int) -> list[float]:
    return [float(d) for d in np.arange(-limit, limit + 1e-9, OFFSET_STEP)] if limit else [0.0]


class Positioner:
    """A fixed KNNConfig over a train reference set restricted to columns `cols`. With
    `fallback_cols`, a query that detects none of `cols` is located on those columns instead
    (same configuration), so an AP subset never sees an all-missing query vector."""

    def __init__(self, rssi: np.ndarray, xy: np.ndarray, labels: np.ndarray, n_classes: int,
                 cols: np.ndarray, config: KNNConfig, fallback_cols: np.ndarray | None = None):
        self.rssi, self.xy, self.labels = np.asarray(rssi)[:, cols], xy, labels
        self.cols, self.config, self.n_classes = np.asarray(cols), config, n_classes
        self.fallback = (None if fallback_cols is None else
                         Positioner(rssi, xy, labels, n_classes, fallback_cols, config))
        self._reference = {}

    def reference(self, kind):
        if kind not in self._reference:
            self._reference[kind] = representation(self.rssi, kind)
        return self._reference[kind]

    def member(self, rssi: np.ndarray, kind: str) -> tuple[Neighbours, np.ndarray]:
        query = np.asarray(rssi)[:, self.cols]
        found = {d: neighbours(self.reference(kind), representation(shift(query, d), kind),
                               self.config.distance) for d in deltas(self.config.offset_range)}
        if len(found) == 1:
            return found[0.0], np.zeros(len(query))
        return offset_select(found, self.config.offset_range)

    def _predict(self, rssi: np.ndarray) -> dict:
        c = self.config
        found, delta = self.member(rssi, c.representation)
        xy = estimate(found, self.xy, c.k, c.power)
        classes, confidence = vote(found, self.labels, self.n_classes, c.k, c.power)
        if c.ensemble_with:
            second, _ = self.member(rssi, c.ensemble_with)
            xy = (xy + estimate(second, self.xy, c.k, c.power)) / 2.0
        return {"xy": xy, "class_ids": classes, "confidence": confidence, "delta": delta}

    def predict(self, rssi: np.ndarray) -> dict:
        rssi = np.asarray(rssi)
        unmatched = unmatched_rows(rssi, self.cols)
        routed = unmatched if self.fallback is not None else np.zeros(len(rssi), bool)
        out = {"xy": np.zeros((len(rssi), 2)), "class_ids": np.zeros(len(rssi), np.int64),
               "confidence": np.zeros(len(rssi)), "delta": np.zeros(len(rssi)), "unmatched": unmatched}
        for mask, model in ((~routed, self), (routed, self.fallback)):
            if mask.any():
                part = model._predict(rssi[mask])
                for key in ("xy", "class_ids", "confidence", "delta"):
                    out[key][mask] = part[key]
        return out


def unmatched_rows(rssi: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Queries in which none of `cols` was detected."""
    return (np.asarray(rssi)[:, cols] == MISSING).all(axis=1)


def mean_error(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(pred - truth, axis=1).mean())


def _split_lists(ref_rssi, query_rssi, cols, fallback_cols, kind, metric, delta=0.0):
    """Neighbour lists of the matched queries on `cols` and of the unmatched ones on
    `fallback_cols`, exactly as Positioner.predict splits them."""
    ref_rssi, query_rssi = np.asarray(ref_rssi), np.asarray(query_rssi)
    unmatched = (unmatched_rows(query_rssi, cols) if fallback_cols is not None
                 else np.zeros(len(query_rssi), bool))
    lists = []
    for mask, columns in ((~unmatched, cols), (unmatched, fallback_cols)):
        if mask.any():
            reference = representation(ref_rssi[:, columns], kind)
            query = representation(shift(query_rssi[mask][:, columns], delta), kind)
            lists.append((mask, neighbours(reference, query, metric)))
    return lists


def _estimate(lists, xy, n, k, power) -> np.ndarray:
    out = np.zeros((n, 2))
    for mask, found in lists:
        out[mask] = estimate(found, xy, k, power)
    return out


def grid_search(ref_rssi, ref_xy, query_rssi, query_xy, cols, fallback_cols=None) -> list[dict]:
    """S1/S2 grid on one column set: every representation x distance x k x p, scored on the
    pipeline that predict() runs (fallback included)."""
    rows, n = [], len(query_xy)
    k_max = min(K_MAX, len(ref_xy))
    for kind in REPRESENTATIONS:
        for metric in DISTANCES:
            lists = _split_lists(ref_rssi, query_rssi, cols, fallback_cols, kind, metric)
            for k in K_GRID:
                if k > k_max:
                    continue
                for p in POWERS:
                    if k == 1 and p:
                        continue                       # one neighbour: weighting is moot
                    rows.append({"representation": kind, "distance": metric, "k": k, "power": p,
                                 "mean_error": mean_error(_estimate(lists, ref_xy, n, k, p), query_xy)})
    return rows


def refine(ref_rssi, ref_xy, query_rssi, query_xy, cols, start: KNNConfig, start_error: float,
           s2_grid: list[dict], min_gain: float, fallback_cols=None) -> tuple[list[dict], list[dict]]:
    """S3 (offset search) and S4 (ensemble) on top of the S2 configuration."""
    stages, rows, best_error, best, n = [], [], start_error, start, len(query_xy)
    by_delta = {d: _split_lists(ref_rssi, query_rssi, cols, fallback_cols, start.representation,
                                start.distance, d) for d in deltas(max(OFFSET_RANGES))}
    s3 = None
    for limit in OFFSET_RANGES:
        lists = []
        for part in range(len(by_delta[0.0])):
            mask = by_delta[0.0][part][0]
            lists.append((mask, offset_select({d: v[part][1] for d, v in by_delta.items()}, limit)[0]))
        for k in sorted({max(1, start.k - 3), start.k, start.k + 3}):
            for p in sorted({start.power, 2.0, 4.0}):
                if k > min(K_MAX, len(ref_xy)):
                    continue
                candidate = replace(start, k=k, power=float(p), offset_range=limit)
                error = mean_error(_estimate(lists, ref_xy, n, k, p), query_xy)
                rows.append({"stage": "S3", **candidate.as_dict(), "mean_error": error})
                if s3 is None or error < s3[1]:
                    s3 = (candidate, error)
    if s3 and best_error - s3[1] >= min_gain:
        best, best_error = s3
        stages.append({"stage": "S3", "what": "+ device-offset search", "config": best,
                       "validation_mean_error": best_error})
    others = [r for r in s2_grid if r["representation"] != best.representation
              and r["distance"] == best.distance]
    if others:
        second = min(others, key=lambda r: r["mean_error"])["representation"]
        labels = np.zeros(len(ref_xy), dtype=np.int64)
        candidate = replace(best, ensemble_with=second)
        out = Positioner(ref_rssi, ref_xy, labels, 1, cols, candidate, fallback_cols).predict(query_rssi)
        error = mean_error(out["xy"], query_xy)
        rows.append({"stage": "S4", **candidate.as_dict(), "mean_error": error})
        if best_error - error >= min_gain:
            stages.append({"stage": "S4", "what": "+ ensemble of two representations",
                           "config": candidate, "validation_mean_error": error})
    return stages, rows
