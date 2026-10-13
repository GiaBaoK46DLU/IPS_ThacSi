"""EFIM: EFficient high-utility Itemset Mining (Zida et al., KAIS 51(2):595-625, 2017,
doi:10.1007/s10115-016-0986-0), ported unchanged from ThaiTaka/HUIM-IPS- (src/huim/efim.py).

Pure Python because the reference implementation (SPMF) needs a JVM. Implemented: depth-first
search in increasing-TWU order, local-utility and subtree-utility bounds, high-utility database
projection, transaction merging (with an exact occurrence count, so support stays exact) and Fast
Utility Counting. Correctness is checked against a brute-force oracle in tests/test_efim.py.

A projected transaction is (items: tuple[int], utils: tuple[float], prefix_util, count), items
strictly increasing in the internal order; callers get their original item labels back.
"""
from __future__ import annotations

import time
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Sequence, Tuple

Item = Hashable
PTrans = Tuple[Tuple[int, ...], Tuple[float, ...], float, int]


class ResourceLimitExceeded(RuntimeError):
    """Raised when a run exceeds its time / output budget. Carries the partial
    state so an experiment can be recorded as ABORTED instead of lost."""

    def __init__(self, reason: str, n_huis: int, elapsed: float,
                 nodes: int) -> None:
        super().__init__(f"{reason} (after {n_huis} HUIs, {elapsed:.1f}s, "
                         f"{nodes} search-tree nodes)")
        self.reason = reason
        self.n_huis = n_huis
        self.elapsed = elapsed
        self.nodes = nodes


@dataclass
class EFIMStats:
    n_huis: int = 0
    n_nodes_visited: int = 0
    n_candidates_pruned_by_su: int = 0
    n_items_pruned_by_lu: int = 0
    n_items_after_twu_pruning: int = 0
    n_items_initial: int = 0
    n_transactions_initial: int = 0
    n_transactions_after_pruning: int = 0
    n_merge_operations: int = 0
    n_transactions_saved_by_merging: int = 0
    max_depth_reached: int = 0
    total_db_utility: float = 0.0
    elapsed_seconds: float = 0.0
    status: str = "COMPLETED"
    abort_reason: str | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class EFIM:
    """One configured EFIM run.

    Parameters
    ----------
    min_util
        Absolute minimum utility threshold. An itemset X is reported iff
        u(X) >= min_util.
    time_limit_s, max_huis
        Controlled-abort budgets. Exceeding either raises
        ResourceLimitExceeded; the caller records STATUS=ABORTED_RESOURCE_LIMIT.
    collect_support
        Report |g(X)| alongside u(X). Exact thanks to the merged `count` field.
    """

    min_util: float
    time_limit_s: float | None = None
    max_huis: int | None = None
    collect_support: bool = True
    check_every_nodes: int = 2048

    stats: EFIMStats = field(default_factory=EFIMStats)
    _huis: List[Tuple[Tuple[int, ...], float, int]] = field(default_factory=list)
    _id2item: List[Item] = field(default_factory=list)
    _t0: float = 0.0
    _since_check: int = 0

    # ---------------------------------------------------------------- helpers
    def _budget_check(self) -> None:
        self._since_check += 1
        if self._since_check < self.check_every_nodes:
            return
        self._since_check = 0
        if self.time_limit_s is not None:
            el = time.perf_counter() - self._t0
            if el > self.time_limit_s:
                raise ResourceLimitExceeded(
                    f"time limit {self.time_limit_s}s exceeded",
                    self.stats.n_huis, el, self.stats.n_nodes_visited)

    def _emit(self, itemset: Tuple[int, ...], utility: float,
              support: int) -> None:
        self._huis.append((itemset, utility, support))
        self.stats.n_huis += 1
        if self.max_huis is not None and self.stats.n_huis > self.max_huis:
            raise ResourceLimitExceeded(
                f"output limit {self.max_huis} HUIs exceeded",
                self.stats.n_huis, time.perf_counter() - self._t0,
                self.stats.n_nodes_visited)

    @staticmethod
    def _merge(trans: List[PTrans], stats: EFIMStats) -> List[PTrans]:
        """Transaction merging. Projected transactions with the same item
        sequence are fused: utilities and prefix utilities are summed
        element-wise, and so are the occurrence counts."""
        if len(trans) < 2:
            return trans
        buckets: Dict[Tuple[int, ...], List[float | int]] = {}
        for items, utils, pu, cnt in trans:
            slot = buckets.get(items)
            if slot is None:
                buckets[items] = [list(utils), pu, cnt]
            else:
                acc = slot[0]
                for k, v in enumerate(utils):
                    acc[k] += v
                slot[1] += pu
                slot[2] += cnt
        if len(buckets) < len(trans):
            stats.n_merge_operations += 1
            stats.n_transactions_saved_by_merging += len(trans) - len(buckets)
        return [(items, tuple(acc), pu, cnt)
                for items, (acc, pu, cnt) in buckets.items()]

    @staticmethod
    def _fast_utility_counting(db: List[PTrans], n_items: int,
                               lo: int) -> Tuple[List[float], List[float]]:
        """One backward pass per transaction fills both utility-bin arrays.

        For a projected transaction with prefix utility `pu` and suffix items
        e_1 < ... < e_m carrying utilities u_1..u_m:

            lu bin[e_k] += pu + sum_{j=1..m} u_j      (same for every k)
            su bin[e_k] += pu + sum_{j=k..m} u_j

        which are exactly the definitions of local utility and subtree utility
        restricted to this transaction.
        """
        lu = [0.0] * n_items
        su = [0.0] * n_items
        for items, utils, pu, _cnt in db:
            rest = 0.0
            # backward pass: `rest` is sum_{j>=k} u_j once we are at position k
            for k in range(len(items) - 1, -1, -1):
                rest += utils[k]
                su[items[k]] += pu + rest
            total = pu + rest  # pu + sum of the whole suffix
            for it in items:
                lu[it] += total
        return lu, su

    # ------------------------------------------------------------------- main
    def mine(self, database: Sequence[Dict[Item, float]],
             external_utility: Dict[Item, float]
             ) -> List[Tuple[Tuple[Item, ...], float, int]]:
        """Mine all HUIs. Returns [(itemset_of_original_labels, utility,
        support)]. Raises ResourceLimitExceeded if a budget is hit."""
        self._t0 = time.perf_counter()
        self._huis.clear()
        self.stats = EFIMStats()
        st = self.stats
        st.n_transactions_initial = len(database)

        # ---- utilities must be positive: EFIM's bounds assume it ------------
        for it, pv in external_utility.items():
            if pv <= 0:
                raise ValueError(f"external utility p({it!r}) = {pv} <= 0; "
                                 "EFIM's TWU/local/subtree bounds require "
                                 "strictly positive utilities")
        for ti, t in enumerate(database):
            for it, q in t.items():
                if q <= 0:
                    raise ValueError(f"internal utility q({it!r}, T{ti}) = {q} "
                                     "<= 0; positive utilities required")
                if it not in external_utility:
                    raise KeyError(f"item {it!r} has no external utility")

        # ---- step 1: TU per transaction, TWU per item -----------------------
        items_all = sorted({i for t in database for i in t}, key=repr)
        st.n_items_initial = len(items_all)
        if not items_all:
            st.elapsed_seconds = time.perf_counter() - self._t0
            return []

        tu: List[float] = []
        twu: Dict[Item, float] = {i: 0.0 for i in items_all}
        for t in database:
            v = 0.0
            for i, q in t.items():
                v += q * external_utility[i]
            tu.append(v)
        for t, v in zip(database, tu):
            for i in t:
                twu[i] += v
        st.total_db_utility = float(sum(tu))

        # ---- step 2: TWU pruning == local utility pruning at the root -------
        # lu(empty, z) = sum_{T contains z} [ u(empty,T) + re(empty,T) ]
        #              = sum_{T contains z} TU(T) = TWU(z)
        secondary0 = [i for i in items_all if twu[i] >= self.min_util]
        st.n_items_after_twu_pruning = len(secondary0)
        st.n_items_pruned_by_lu += len(items_all) - len(secondary0)
        if not secondary0:
            st.elapsed_seconds = time.perf_counter() - self._t0
            return []

        # ---- step 3: total order '<' = increasing TWU (ties by label) -------
        secondary0.sort(key=lambda i: (twu[i], repr(i)))
        self._id2item = secondary0
        item2id = {it: k for k, it in enumerate(secondary0)}
        n_ids = len(secondary0)

        # ---- step 4: build the pruned, ordered initial database -------------
        db: List[PTrans] = []
        for t in database:
            pairs = [(item2id[i], q * external_utility[i])
                     for i, q in t.items() if i in item2id]
            if not pairs:
                continue
            pairs.sort()
            db.append((tuple(p[0] for p in pairs),
                       tuple(p[1] for p in pairs), 0.0, 1))
        st.n_transactions_after_pruning = len(db)
        db = self._merge(db, st)
        if not db:
            st.elapsed_seconds = time.perf_counter() - self._t0
            return []

        # ---- step 5: root Primary / Secondary via Fast Utility Counting -----
        lu, su = self._fast_utility_counting(db, n_ids, 0)
        secondary = [z for z in range(n_ids) if lu[z] >= self.min_util]
        primary = [z for z in secondary if su[z] >= self.min_util]
        st.n_candidates_pruned_by_su += len(secondary) - len(primary)

        sec_set = set(secondary)
        if len(sec_set) < n_ids:
            db = self._filter_items(db, sec_set)
            db = self._merge(db, st)

        # ---- step 6: depth-first search -------------------------------------
        try:
            self._search((), db, primary, secondary, 1)
            st.status = "COMPLETED"
        except ResourceLimitExceeded as e:
            st.status = "ABORTED_RESOURCE_LIMIT"
            st.abort_reason = str(e)
            st.elapsed_seconds = time.perf_counter() - self._t0
            raise
        st.elapsed_seconds = time.perf_counter() - self._t0
        return [(tuple(self._id2item[i] for i in xs), u, s)
                for xs, u, s in self._huis]

    @staticmethod
    def _filter_items(db: List[PTrans], keep: set) -> List[PTrans]:
        """Drop items that local-utility pruning excluded from this subtree."""
        out: List[PTrans] = []
        for items, utils, pu, cnt in db:
            sel = [k for k, it in enumerate(items) if it in keep]
            if len(sel) == len(items):
                out.append((items, utils, pu, cnt))
            elif sel:
                out.append((tuple(items[k] for k in sel),
                            tuple(utils[k] for k in sel), pu, cnt))
            # a transaction with no surviving item still contributed its prefix
            # utility to an already-emitted itemset; it cannot extend further.
        return out

    def _search(self, alpha: Tuple[int, ...], db: List[PTrans],
                primary: List[int], secondary: List[int], depth: int) -> None:
        st = self.stats
        if depth > st.max_depth_reached:
            st.max_depth_reached = depth
        sec_sorted = sorted(secondary)
        for i in primary:
            st.n_nodes_visited += 1
            self._budget_check()
            beta = alpha + (i,)

            # --- project on i, accumulating u(beta) and |g(beta)| ------------
            u_beta = 0.0
            sup_beta = 0
            proj: List[PTrans] = []
            for items, utils, pu, cnt in db:
                k = bisect_left(items, i)
                if k == len(items) or items[k] != i:
                    continue
                new_pu = pu + utils[k]
                u_beta += new_pu
                sup_beta += cnt
                if k + 1 < len(items):
                    proj.append((items[k + 1:], utils[k + 1:], new_pu, cnt))
            if sup_beta == 0:
                continue
            if u_beta >= self.min_util:
                self._emit(beta, u_beta, sup_beta if self.collect_support else -1)
            if not proj:
                continue
            proj = self._merge(proj, st)

            # --- extensions of beta are the secondary items greater than i ---
            pos = bisect_left(sec_sorted, i)
            cand = [z for z in sec_sorted[pos:] if z > i]
            if not cand:
                continue

            lu, su = self._fast_utility_counting(
                proj, max(cand) + 1, i)
            new_secondary = [z for z in cand if lu[z] >= self.min_util]
            new_primary = [z for z in new_secondary if su[z] >= self.min_util]
            st.n_items_pruned_by_lu += len(cand) - len(new_secondary)
            st.n_candidates_pruned_by_su += len(new_secondary) - len(new_primary)
            if not new_primary:
                continue
            keep = set(new_secondary)
            proj = self._filter_items(proj, keep)
            if not proj:
                continue
            proj = self._merge(proj, st)
            self._search(beta, proj, new_primary, new_secondary, depth + 1)


def mine_efim(database: Sequence[Dict[Item, float]],
              external_utility: Dict[Item, float],
              min_util: float,
              time_limit_s: float | None = None,
              max_huis: int | None = None,
              ) -> Tuple[List[Tuple[Tuple[Item, ...], float, int]], EFIMStats]:
    """Convenience wrapper. On a controlled abort the partial HUI list and the
    stats (with status ABORTED_RESOURCE_LIMIT) are returned instead of raising,
    so an experiment grid can continue."""
    eng = EFIM(min_util=min_util, time_limit_s=time_limit_s, max_huis=max_huis)
    try:
        huis = eng.mine(database, external_utility)
    except ResourceLimitExceeded:
        huis = [(tuple(eng._id2item[i] for i in xs), u, s)
                for xs, u, s in eng._huis]
    return huis, eng.stats
