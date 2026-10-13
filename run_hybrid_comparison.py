"""Comprehensive Benchmark: RF+KNN vs Pure AutoLoC (LLM) vs Hybrid RF+AutoLoC (RF+LLM).

Evaluates across Data Scarcity Ratios: 20%, 30%, 40%, 50%, and 100%
on both:
1. Held-out Validation Split (N=300, with Ground-Truth SpaceID, Building, Floor, 2D Coordinates)
2. Official Blind Test Split (validationData.csv, N=1,111)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, str(Path(__file__).parent))

from src.ips.algorithms.llm.autoloc import AutoLoCEngine
from src.ips.algorithms.llm.hybrid_autoloc import ComparisonMetrics, HybridRFAutoLoCEngine


def evaluate_predictions(
    method_name: str,
    split_name: str,
    split_ratio: float,
    train_size: int,
    pred_b: np.ndarray,
    pred_f: np.ndarray,
    pred_s: np.ndarray,
    pred_lat: np.ndarray,
    pred_lon: np.ndarray,
    test_df: pd.DataFrame,
    fallback_rate: float,
    total_time_sec: float,
) -> ComparisonMetrics:
    true_b = test_df["BUILDINGID"].to_numpy(dtype=int)
    true_f = test_df["FLOOR"].to_numpy(dtype=int)
    true_s = test_df["SPACEID"].to_numpy(dtype=int)
    true_lat = test_df["LATITUDE"].to_numpy(dtype=float)
    true_lon = test_df["LONGITUDE"].to_numpy(dtype=float)

    b_acc = float(np.mean(pred_b == true_b) * 100.0)
    f_acc = float(np.mean(pred_f == true_f) * 100.0)
    joint_mask = (pred_b == true_b) & (pred_f == true_f)
    comb_acc = float(np.mean(joint_mask) * 100.0)

    if (true_s > 0).any():
        s_acc = float(np.mean(pred_s == true_s) * 100.0)
    else:
        s_acc = 0.0

    errors = np.sqrt((pred_lat - true_lat) ** 2 + (pred_lon - true_lon) ** 2)
    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    p90_err = float(np.percentile(errors, 90))
    cond_err = float(np.mean(errors[joint_mask])) if np.any(joint_mask) else mean_err

    return ComparisonMetrics(
        method_name=method_name,
        split_name=split_name,
        split_ratio=split_ratio,
        train_size=train_size,
        test_size=len(test_df),
        building_acc=b_acc,
        floor_acc=f_acc,
        combined_acc=comb_acc,
        space_acc=s_acc,
        mean_error_meters=mean_err,
        median_error_meters=median_err,
        p90_error_meters=p90_err,
        conditional_mean_error=cond_err,
        fallback_rate=fallback_rate,
        total_time_sec=total_time_sec,
    )


def run_rf_knn_scarcity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    wap_cols: List[str],
    split_name: str,
    ratio: float,
    k: int = 9,
    metric: str = "manhattan",
    penalty_weight: float = 1000.0,
    seed: int = 42,
) -> ComparisonMetrics:
    """Run Hierarchical RF + KNN (K=9, Manhattan, W=1000) for a given training subset."""
    t0 = time.time()
    X_train_raw = train_df[wap_cols].to_numpy(dtype=np.float32)
    X_test_raw = test_df[wap_cols].to_numpy(dtype=np.float32)

    X_train = HybridRFAutoLoCEngine.preprocess_positive_shift(X_train_raw)
    X_test = HybridRFAutoLoCEngine.preprocess_positive_shift(X_test_raw)

    b_train = train_df["BUILDINGID"].to_numpy(dtype=int)
    f_train = train_df["FLOOR"].to_numpy(dtype=int)
    s_train = train_df["SPACEID"].to_numpy(dtype=int)
    y_train_coords = train_df[["LATITUDE", "LONGITUDE"]].to_numpy(dtype=float)

    clf_b = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
    clf_b.fit(X_train, b_train)
    pred_b = clf_b.predict(X_test).astype(int)

    clf_f = RandomForestClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
    clf_f.fit(np.column_stack((X_train, b_train)), f_train)
    pred_f = clf_f.predict(np.column_stack((X_test, pred_b))).astype(int)

    X_train_c = np.column_stack((X_train, b_train * penalty_weight, f_train * penalty_weight))
    X_test_c = np.column_stack((X_test, pred_b * penalty_weight, pred_f * penalty_weight))

    reg = KNeighborsRegressor(n_neighbors=k, weights="distance", metric=metric, n_jobs=-1)
    reg.fit(X_train_c, y_train_coords)
    pred_coords = reg.predict(X_test_c)

    _, indices = reg.kneighbors(X_test_c, n_neighbors=1)
    pred_s = s_train[indices[:, 0]]

    elapsed = time.time() - t0
    return evaluate_predictions(
        method_name="RF + KNN (K=9, L1)",
        split_name=split_name,
        split_ratio=ratio,
        train_size=len(train_df),
        pred_b=pred_b,
        pred_f=pred_f,
        pred_s=pred_s,
        pred_lat=pred_coords[:, 0],
        pred_lon=pred_coords[:, 1],
        test_df=test_df,
        fallback_rate=0.0,
        total_time_sec=elapsed,
    )


def run_hybrid_rf_autoloc(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    wap_cols: List[str],
    split_name: str,
    ratio: float,
    model_name: str = "llama3.2:1b",
    max_workers: int = 4,
    seed: int = 42,
) -> ComparisonMetrics:
    """Run Hybrid RF + AutoLoC (Cascaded RF + Floor-Locked Dual-Space RAG + Ollama LLM)."""
    t0 = time.time()
    engine = HybridRFAutoLoCEngine(
        n_estimators=100,
        n_components=15,
        n_neighbors=5,
        knn_k=7,
        power_gamma=1.5,
        idw_exponent=2.0,
        spatial_sigma=25.0,
        llm_gate_threshold=50.0,
        model_name=model_name,
        seed=seed,
    )
    engine.fit(train_df, wap_cols=wap_cols)
    X_test_raw = test_df[wap_cols].to_numpy(dtype=np.float32)
    preds = engine.predict_batch(X_test_raw, max_workers=max_workers)
    elapsed = time.time() - t0

    pred_b = np.array([p.predicted_building for p in preds], dtype=int)
    pred_f = np.array([p.predicted_floor for p in preds], dtype=int)
    pred_s = np.array([p.predicted_space for p in preds], dtype=int)
    pred_lat = np.array([p.predicted_lat for p in preds], dtype=float)
    pred_lon = np.array([p.predicted_lon for p in preds], dtype=float)
    fb_rate = float(np.mean([p.fallback_used for p in preds]) * 100.0)

    return evaluate_predictions(
        method_name="Hybrid RF + AutoLoC (LLM)",
        split_name=split_name,
        split_ratio=ratio,
        train_size=len(train_df),
        pred_b=pred_b,
        pred_f=pred_f,
        pred_s=pred_s,
        pred_lat=pred_lat,
        pred_lon=pred_lon,
        test_df=test_df,
        fallback_rate=fb_rate,
        total_time_sec=elapsed,
    )


def main():
    full_df = pd.read_csv("data/raw/ujiindoorloc/trainingData.csv")
    blind_test_df = pd.read_csv("data/raw/ujiindoorloc/validationData.csv")
    wap_cols = [c for c in full_df.columns if c.startswith("WAP")]

    rng = np.random.RandomState(42)
    shuffled = rng.permutation(len(full_df))
    heldout_df = full_df.iloc[shuffled[:300]].copy().reset_index(drop=True)
    pool_df = full_df.iloc[shuffled[300:]].copy().reset_index(drop=True)

    ratios = [0.2, 0.3, 0.4, 0.5, 1.0]
    all_metrics: List[ComparisonMetrics] = []

    print("=" * 90)
    print("  RUNNING FULL COMPARISON: RF+KNN vs PURE AUTOLOC vs HYBRID RF+AUTOLOC")
    print("=" * 90)

    for r in ratios:
        n_sub = int(len(pool_df) * r)
        sub_df = pool_df.sample(n=n_sub, random_state=42 + int(r * 100)).reset_index(drop=True)
        pct = int(r * 100)
        print(f"\n--- Memory Scarcity Ratio: {pct}% ({n_sub:,} reference samples) ---")

        # 1. RF + KNN on Heldout-300 and BlindTest-1111
        m_rf_val = run_rf_knn_scarcity(sub_df, heldout_df, wap_cols, "Heldout-300", r)
        m_rf_test = run_rf_knn_scarcity(sub_df, blind_test_df, wap_cols, "BlindTest-1111", r)
        all_metrics.extend([m_rf_val, m_rf_test])
        print(f"  [RF + KNN]          Heldout-300:   Comb={m_rf_val.combined_acc:.2f}%, Space={m_rf_val.space_acc:.2f}%, Mean={m_rf_val.mean_error_meters:.2f}m, Med={m_rf_val.median_error_meters:.2f}m")
        print(f"  [RF + KNN]          BlindTest-1111: Bld={m_rf_test.building_acc:.2f}%, Flr={m_rf_test.floor_acc:.2f}%, Comb={m_rf_test.combined_acc:.2f}%, Mean={m_rf_test.mean_error_meters:.2f}m, Med={m_rf_test.median_error_meters:.2f}m")

        # 2. Hybrid RF + AutoLoC on Heldout-300 and BlindTest-1111
        m_hyb_val = run_hybrid_rf_autoloc(sub_df, heldout_df, wap_cols, "Heldout-300", r)
        m_hyb_test = run_hybrid_rf_autoloc(sub_df, blind_test_df, wap_cols, "BlindTest-1111", r)
        all_metrics.extend([m_hyb_val, m_hyb_test])
        print(f"  [Hybrid RF+AutoLoC] Heldout-300:   Comb={m_hyb_val.combined_acc:.2f}%, Space={m_hyb_val.space_acc:.2f}%, Mean={m_hyb_val.mean_error_meters:.2f}m, Med={m_hyb_val.median_error_meters:.2f}m ({m_hyb_val.total_time_sec:.1f}s)")
        print(f"  [Hybrid RF+AutoLoC] BlindTest-1111: Bld={m_hyb_test.building_acc:.2f}%, Flr={m_hyb_test.floor_acc:.2f}%, Comb={m_hyb_test.combined_acc:.2f}%, Mean={m_hyb_test.mean_error_meters:.2f}m, Med={m_hyb_test.median_error_meters:.2f}m ({m_hyb_test.total_time_sec:.1f}s)")

    out_path = Path("docs/Hoc/hybrid_comparison_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(m) for m in all_metrics], f, indent=2)
    print(f"\nSaved full benchmark results to {out_path}")


if __name__ == "__main__":
    main()
