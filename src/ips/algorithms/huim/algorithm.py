"""HUIM access-point selection + weighted k-NN positioning (ThaiTaka/HUIM-IPS-).

prepare(train)       fit q(i,T) range and p(i) on train, encode train fingerprints as utility
                     transactions and mine HUIs with EFIM down a minUtil grid (train only)
execute(validation)  turn every mined HUI set into AP candidates (union / top-n), choose the AP
                     set and the k-NN estimator S0..S4 on internal validation, then freeze
predict(inputs)      k-NN over the train fingerprints restricted to the chosen APs; a query that
                     detects none of them falls back to every AP detected in train
"""
from __future__ import annotations

import csv
import json
import threading
import time
from dataclasses import replace

import numpy as np

from ...data import AP_COLUMNS
from ...data_access import FingerprintInputs, FingerprintSplit
from ...experiments.contracts import AlgorithmExecution, ExperimentConfig, IPSPredictions, ModelDiagram
from .adapter import HUIMAlgorithm
from .efim import mine_efim
from .positioning import BASELINE, KNNConfig, Positioner, grid_search, refine, unmatched_rows
from .utility import MISSING, MODELS, TransactionEncoder, external_utility, ranking

DEFAULTS = {
    "utility_model": "B",
    "min_util_pct": [20.0, 15.0, 10.0, 7.5, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0],
    "max_bisections": 3,           # refine between the last completed and first aborted threshold
    "rich_enough_huis": 20000,     # stop refining once a completed HUI set is this large
    "time_limit_s": 600.0,
    "max_huis": 1000000,
    "candidate_sets": ["union", "topn"],
    "topn_budgets": [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300],
    "min_gain": 0.05,
    "fallback": "all_detected",    # or "none": an unmatched query keeps its all-missing AP vector
}


class _PeakRSS:
    """Peak resident memory of this process while a block runs, sampled every 20 ms."""

    def __enter__(self):
        import psutil
        self._process = psutil.Process()
        self.baseline = self.peak = self._process.memory_info().rss
        self._stop = threading.Event()

        def sample():
            while not self._stop.wait(0.02):
                self.peak = max(self.peak, self._process.memory_info().rss)
        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)
        self.peak = max(self.peak, self._process.memory_info().rss)

    @property
    def delta_mb(self) -> float:
        return round((self.peak - self.baseline) / 2**20, 2)


class HUIMKNNAlgorithm(HUIMAlgorithm):
    """High-utility AP groups mined by EFIM, located by validation-selected weighted k-NN."""
    model_id = "huim"

    def __init__(self, config: ExperimentConfig):
        super().__init__(config)
        unknown = set(config.params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown HUIM params: {sorted(unknown)}")
        self.params = {**DEFAULTS, **config.params}
        p = self.params
        if p["utility_model"] not in MODELS:
            raise ValueError(f"utility_model must be one of {MODELS}.")
        grid = p["min_util_pct"]
        if (not isinstance(grid, list) or not grid or any(not isinstance(v, (int, float)) or not 0 < v <= 100
                                                          for v in grid)):
            raise ValueError("min_util_pct must be a nonempty list of percentages in (0, 100].")
        if not set(p["candidate_sets"]) <= {"union", "topn", "all_detected"} or not p["candidate_sets"]:
            raise ValueError("candidate_sets must use union, topn and/or all_detected.")
        if p["fallback"] not in ("all_detected", "none"):
            raise ValueError("fallback must be all_detected or none.")
        self._train = self._encoder = self._final = None

    # ------------------------------------------------------------------ train
    def prepare(self, train: FingerprintSplit) -> None:
        if train.name != "train":
            raise ValueError("HUIM preparation requires the train split.")
        self._train = train
        rssi = train.inputs.rssi
        self._encoder = TransactionEncoder.fit(rssi)
        self._utility, self._utility_info = external_utility(self.params["utility_model"], rssi,
                                                             train.class_ids)
        self._detected = np.flatnonzero((rssi != MISSING).any(axis=0))
        transactions = [t for t in self.encode_transactions(train.inputs) if t]
        p_items = {int(j): float(self._utility[j]) for j in self._detected}
        total = sum(q * p_items[i] for t in transactions for i, q in t.items())
        self._runs = []

        def mine(pct, phase):
            with _PeakRSS() as memory:
                started = time.perf_counter()
                huis, stats = mine_efim(transactions, p_items, total * pct / 100.0,
                                        time_limit_s=self.params["time_limit_s"],
                                        max_huis=self.params["max_huis"])
                seconds = time.perf_counter() - started
            run = {"min_util_pct": float(pct), "min_util": total * pct / 100.0, "phase": phase,
                   "status": stats.status, "n_huis": len(huis), "seconds": round(seconds, 4),
                   "peak_rss_delta_mb": memory.delta_mb, "nodes": stats.n_nodes_visited,
                   "ranking": [int(i) for i in ranking(huis)]}
            self._runs.append(run)
            print(f"HUIM mine {self.params['utility_model']} {pct:g}%: {stats.status}, {len(huis)} HUIs, "
                  f"{len(run['ranking'])} APs, {seconds:.2f}s", flush=True)
            return run

        for pct in sorted(self.params["min_util_pct"], reverse=True):
            if mine(pct, "grid")["status"] != "COMPLETED":
                break
        self._refine(mine)
        self._total_utility = total
        self._n_transactions = len(transactions)

    def _refine(self, mine) -> None:
        """Bisect geometrically between the lowest completed and the highest aborted threshold
        (or halve below the grid when nothing aborted), as ThaiTaka/HUIM-IPS- PHASE H2 does."""
        done = [r for r in self._runs if r["status"] == "COMPLETED"]
        aborted = [r["min_util_pct"] for r in self._runs if r["status"] != "COMPLETED"]
        if not done:
            return
        low, best = min(r["min_util_pct"] for r in done), max(r["n_huis"] for r in done)
        high = max(aborted) if aborted else None
        for _ in range(self.params["max_bisections"]):
            if best >= self.params["rich_enough_huis"]:
                return
            middle = low / 2.0 if high is None else float(np.sqrt(low * high))
            if high is not None and not high < middle < low:
                return
            run = mine(round(middle, 6), "refine")
            if run["status"] == "COMPLETED":
                low, best = middle, max(best, run["n_huis"])
            else:
                high = middle

    def encode_transactions(self, inputs: FingerprintInputs):
        if self._encoder is None:
            raise RuntimeError("Call prepare(train) before encoding transactions.")
        return self._encoder.transactions(inputs.rssi)

    # ------------------------------------------------------------ validation
    def _candidates(self) -> list[dict]:
        """Union of every HUI set, truncated ones included (as the study's PHASE I did); top-n only
        from the richest COMPLETED set, because a truncated set is biased by the DFS order."""
        found = [r for r in self._runs if r["status"] == "COMPLETED" and r["ranking"]]
        candidates, seen = [], set()

        def add(name, columns, run=None):
            key = tuple(sorted(columns))
            if columns and key not in seen:
                seen.add(key)
                candidates.append({"name": name, "columns": [int(c) for c in columns],
                                   "min_util_pct": None if run is None else run["min_util_pct"]})
        if "union" in self.params["candidate_sets"]:
            for run in self._runs:
                suffix = "" if run["status"] == "COMPLETED" else "_truncated"
                add(f"union_{run['min_util_pct']:g}pct{suffix}", run["ranking"], run)
        if "topn" in self.params["candidate_sets"] and found:
            richest = min(found, key=lambda r: r["min_util_pct"])
            for n in self.params["topn_budgets"]:
                if n < len(richest["ranking"]):
                    add(f"top{n}_{richest['min_util_pct']:g}pct", richest["ranking"][:n], richest)
        if "all_detected" in self.params["candidate_sets"]:
            add("all_detected", self._detected)
        if not candidates:
            raise ValueError("No HUI set was mined; lower min_util_pct or raise the budgets.")
        return candidates

    def execute(self, validation: FingerprintSplit | None) -> AlgorithmExecution:
        if self._train is None:
            raise RuntimeError("Call prepare(train) before execute().")
        if validation is None or validation.name != "validation":
            raise ValueError("HUIM selection requires internal validation, never test.")
        if (validation.info != self._train.info or
                set(validation.inputs.row_indices) & set(self._train.inputs.row_indices)):
            raise ValueError("Training/validation provenance differs or rows overlap.")
        train, started = self._train, time.perf_counter()
        fallback = self._detected if self.params["fallback"] == "all_detected" else None
        ref, xy, val, val_xy = train.inputs.rssi, train.xy, validation.inputs.rssi, validation.xy
        rows, best = [], None
        for candidate in self._candidates():
            grid = grid_search(ref, xy, val, val_xy, candidate["columns"], fallback)
            for row in grid:
                rows.append({"candidate": candidate["name"], "n_aps": len(candidate["columns"]), **row})
            top = min(grid, key=lambda r: r["mean_error"])
            print(f"HUIM select {candidate['name']} ({len(candidate['columns'])} APs): "
                  f"validation mean {top['mean_error']:.3f}", flush=True)
            if best is None or top["mean_error"] < best[1]["mean_error"]:
                best = (candidate, top, grid)
        candidate, top, grid = best
        columns = np.asarray(candidate["columns"])
        k1 = min((r for r in grid if r["k"] == 1), key=lambda r: r["mean_error"])
        s0 = next(r for r in grid if (r["representation"], r["distance"], r["k"], r["power"]) ==
                  (BASELINE.representation, BASELINE.distance, BASELINE.k, BASELINE.power))
        config2 = KNNConfig(top["representation"], top["distance"], int(top["k"]), float(top["power"]))
        stages = [{"stage": "S0", "what": "baseline k-NN", "config": BASELINE,
                   "validation_mean_error": s0["mean_error"]},
                  {"stage": "S1", "what": "best representation x distance (k=1)",
                   "config": KNNConfig(k1["representation"], k1["distance"], 1, 0),
                   "validation_mean_error": k1["mean_error"]},
                  {"stage": "S2", "what": "+ k and 1/d^p weighting", "config": config2,
                   "validation_mean_error": top["mean_error"]}]
        later, extra = refine(ref, xy, val, val_xy, columns, config2, top["mean_error"], grid,
                              self.params["min_gain"], fallback)
        stages += later
        rows += [{"candidate": candidate["name"], "n_aps": len(columns), **r} for r in extra]
        self._final = stages[-1]["config"]
        n_classes = len(train.info.classes)
        self._columns = columns
        self._positioner = Positioner(ref, xy, train.class_ids, n_classes, columns, self._final,
                                      fallback)
        self._candidate = candidate
        self._stages = stages
        selection_seconds = time.perf_counter() - started
        artifacts = self._write_artifacts(rows, stages, candidate)
        history = [{"stage": s["stage"], "what": s["what"], "config": s["config"].label(),
                    "validation_mean_error": round(float(s["validation_mean_error"]), 6)} for s in stages]
        resources = {"utility_model": self.params["utility_model"], "utility": self._utility_info,
                     "encoder": self._encoder.as_dict(), "n_train_transactions": self._n_transactions,
                     "total_database_utility": self._total_utility,
                     "mining": [{k: v for k, v in r.items() if k != "ranking"} | {"n_aps": len(r["ranking"])}
                                for r in self._runs],
                     "mining_seconds": round(sum(r["seconds"] for r in self._runs), 4),
                     "selection_seconds": round(selection_seconds, 4),
                     "selected_candidate": candidate["name"], "selected_n_aps": int(len(columns)),
                     "validation_rows_without_selected_ap": int(unmatched_rows(val, columns).sum()),
                     "selected_config": self._final.as_dict(),
                     "fallback": self.params["fallback"]}
        finalized = not self.config.smoke
        return AlgorithmExecution(finalized, "huim_mining_selection" if finalized else "smoke_check",
                                  history, artifacts, resources)

    def _write_artifacts(self, rows, stages, candidate) -> dict:
        folder = self.config.output_dir/"algorithm"
        folder.mkdir(parents=True, exist_ok=True)
        grid_path = folder/"huim_selection_grid.csv"
        fields = ["candidate", "n_aps", "stage", "representation", "distance", "k", "power",
                  "offset_range", "ensemble_with", "mean_error"]
        with grid_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows([{k: r.get(k, "") for k in fields} for r in rows])
        selection_path = folder/"huim_selection.json"
        selection_path.write_text(json.dumps({
            "selected_candidate": candidate["name"], "min_util_pct": candidate["min_util_pct"],
            "selected_aps": [AP_COLUMNS[c] for c in candidate["columns"]],
            "stages": [{"stage": s["stage"], "what": s["what"], "config": s["config"].as_dict(),
                        "label": s["config"].label(),
                        "validation_mean_error": float(s["validation_mean_error"])} for s in stages],
            "mining": [{**{k: v for k, v in r.items() if k != "ranking"},
                        "aps_by_contribution": [AP_COLUMNS[c] for c in r["ranking"]]} for r in self._runs]},
            ensure_ascii=False, indent=2), encoding="utf-8")
        return {"huim_selection_grid": grid_path, "huim_selection": selection_path}

    # ------------------------------------------------------------- inference
    def itemsets_to_position(self, inputs: FingerprintInputs) -> IPSPredictions:
        if self._final is None:
            raise RuntimeError("Call execute(validation) to select and freeze the HUIM state before predict().")
        out = self._positioner.predict(inputs.rssi)
        self.last_unmatched = int(out["unmatched"].sum())
        return IPSPredictions(inputs.row_indices.copy(), inputs.source_id, xy=out["xy"],
                              class_ids=out["class_ids"].astype(np.int64),
                              confidence=np.clip(out["confidence"], 0.0, 1.0))

    def predict(self, inputs: FingerprintInputs) -> IPSPredictions:
        return self.itemsets_to_position(inputs)

    def get_diagram(self) -> ModelDiagram | None:
        if self._final is None:
            return None
        c = self._candidate
        return ModelDiagram(
            'flowchart LR\n    A["520 AP RSSI"] --> B["q(i,T) 1..100 + p(i) Model '
            f'{self.params["utility_model"]} (train)"]\n'
            f'    B --> C["EFIM HUIs, {len(self._runs)} minUtil runs"]\n'
            f'    C --> D["{c["name"]}: {len(c["columns"])} APs (validation)"]\n'
            f'    D --> E["k-NN {self._final.label()}"]\n'
            '    E --> F["class vote + xy (original metres)"]\n', "mermaid")
