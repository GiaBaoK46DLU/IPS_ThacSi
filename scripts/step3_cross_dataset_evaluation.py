"""Step 3: Cross-Dataset Evaluation & In-Domain IPS Benchmark.
1. Evaluates pre-trained UJIIndoorLoc model (Hierarchical RF + KNN / RF + AutoLoC).
2. Performs Cross-Dataset Zero-Shot transfer to demonstrate domain gap.
3. Implements & evaluates RF + AutoLoC methodology on the new benchmark datasets (DLULIB, TUJI).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def load_uji_pretrained_models(run_dir: str = "runs/hierarchical_final_test/algorithm"):
    p = Path(run_dir)
    clf_b = joblib.load(p / "model_building.pkl")
    clf_f = joblib.load(p / "model_floor.pkl")
    reg_coord = joblib.load(p / "model_coordinates.pkl")
    return clf_b, clf_f, reg_coord


def evaluate_uji_test_set():
    print("\n" + "=" * 80)
    print("PART 1: BASELINE EVALUATION ON UJIINDOORLOC (ORIGINAL DOMAIN)")
    print("=" * 80)
    clf_b, clf_f, reg_coord = load_uji_pretrained_models()
    
    val_path = Path("data/raw/ujiindoorloc/validationData.csv")
    df_val = pd.read_csv(val_path)
    wap_cols = [f"WAP{i:03d}" for i in range(1, 521)]
    
    X_raw = df_val[wap_cols].to_numpy(dtype=np.float32)
    # Preprocess: +100 -> -110, then shift + 110
    X_clean = np.where(X_raw == 100, -110.0, X_raw)
    X_pos = np.maximum(0, X_clean + 110.0)
    
    pred_b = clf_b.predict(X_pos)
    pred_f = clf_f.predict(np.column_stack((X_pos, pred_b)))
    penalty_weight = 1000.0
    X_coord = np.column_stack((X_pos, pred_b * penalty_weight, pred_f * penalty_weight))
    pred_coords = reg_coord.predict(X_coord)
    
    true_coords = df_val[["LONGITUDE", "LATITUDE"]].to_numpy()
    errors = np.sqrt(np.sum((pred_coords - true_coords)**2, axis=1))
    
    b_acc = float(np.mean(pred_b == df_val["BUILDINGID"]) * 100.0)
    f_acc = float(np.mean(pred_f == df_val["FLOOR"]) * 100.0)
    joint_acc = float(np.mean((pred_b == df_val["BUILDINGID"]) & (pred_f == df_val["FLOOR"])) * 100.0)
    mean_err = float(np.mean(errors))
    median_err = float(np.median(errors))
    p90_err = float(np.percentile(errors, 90))
    
    print(f"UJI Validation Test Set (N={len(df_val):,}):")
    print(f"  - Building Accuracy: {b_acc:.2f}%")
    print(f"  - Floor Accuracy:    {f_acc:.2f}%")
    print(f"  - Combined (B+F) Acc:{joint_acc:.2f}%")
    print(f"  - Mean Positioning Error (MPE):   {mean_err:.2f} meters")
    print(f"  - Median Positioning Error:       {median_err:.2f} meters")
    print(f"  - 90th Percentile Error (P90):    {p90_err:.2f} meters")
    
    return {
        "dataset": "UJIIndoorLoc (validationData)",
        "samples": len(df_val),
        "building_acc": b_acc,
        "floor_acc": f_acc,
        "mean_error_m": mean_err,
        "median_error_m": median_err,
        "p90_error_m": p90_err
    }


def evaluate_cross_dataset_direct(dataset_name: str = "TUJI"):
    print("\n" + "=" * 80)
    print(f"PART 2: DIRECT ZERO-SHOT CROSS-DATASET INFERENCE ON: [{dataset_name}]")
    print("=" * 80)
    clf_b, clf_f, reg_coord = load_uji_pretrained_models()
    
    if dataset_name == "TUJI":
        df_te = pd.read_csv("data/benchmarks/tuji/TUJI_testing.csv")
        waps_avail = [f"WAP{i:03d}" for i in range(1, 311)]
        X_avail = df_te[waps_avail].to_numpy(dtype=np.float32)
        X_pos_310 = np.maximum(0, np.where(X_avail == 100, -110.0, X_avail) + 110.0)
        # Pad remaining 210 WAPs to 520
        X_te_520 = np.pad(X_pos_310, ((0, 0), (0, 210)), mode='constant', constant_values=0)
        
        pred_b = clf_b.predict(X_te_520)
        pred_f = clf_f.predict(np.column_stack((X_te_520, pred_b)))
        penalty_weight = 1000.0
        X_coord = np.column_stack((X_te_520, pred_b * penalty_weight, pred_f * penalty_weight))
        pred_coords = reg_coord.predict(X_coord)
        
        true_coords = df_te[["LONGITUDE", "LATITUDE"]].to_numpy()
        errors = np.sqrt(np.sum((pred_coords - true_coords)**2, axis=1))
        
        mean_err = float(np.mean(errors))
        median_err = float(np.median(errors))
        
        print(f"Result of Direct Zero-Shot Inference on {dataset_name}:")
        print(f"  - Mean Positioning Error (Euclidean): {mean_err:,.2f} meters (~{mean_err/1000:.1f} km)")
        print(f"  - Median Positioning Error:          {median_err:,.2f} meters (~{median_err/1000:.1f} km)")
        print(f"  - Ground Truth Coordinates (Local):  X in [{true_coords[:,0].min():.1f}, {true_coords[:,0].max():.1f}], Y in [{true_coords[:,1].min():.1f}, {true_coords[:,1].max():.1f}]")
        print(f"  - Predicted Coordinates (UTM Spain): Lon in [{pred_coords[:,0].min():.1f}, {pred_coords[:,0].max():.1f}], Lat in [{pred_coords[:,1].min():.1f}, {pred_coords[:,1].max():.1f}]")
        print("  -> Scientific Finding: The model CANNOT maintain 7-8m accuracy across disparate buildings")
        print("     due to AP MAC address disjointness and Coordinate Reference System (CRS) mismatch.")
        
        return {
            "dataset": f"{dataset_name} (Zero-shot direct)",
            "mean_error_m": mean_err,
            "median_error_m": median_err,
            "status": "Incompatible Coordinate Reference System & Disjoint APs"
        }


def benchmark_new_dataset_in_domain(dataset: str = "DLULIB"):
    print("\n" + "=" * 80)
    print(f"PART 3: IN-DOMAIN IPS BENCHMARKING (RF + AUTOLOC VS BASELINES) ON: [{dataset}]")
    print("=" * 80)
    
    if dataset == "DLULIB":
        # Load & merge 3 floors
        train_dfs, test_dfs = [], []
        for flr in [1, 2, 3]:
            tr = pd.read_excel(f"data/benchmarks/dlulib/DLULIB_Floor_{flr}/Training_DLULIB_Floor{flr}_All_20.xlsx")
            te = pd.read_excel(f"data/benchmarks/dlulib/DLULIB_Floor_{flr}/Testing_DLULIB_Floor{flr}.xlsx")
            train_dfs.append(tr)
            test_dfs.append(te)
        df_train = pd.concat(train_dfs, ignore_index=True)
        df_test = pd.concat(test_dfs, ignore_index=True)
        
        ap_cols = [f"AP{i}" for i in range(1, 18)]
        X_tr = np.maximum(0, df_train[ap_cols].to_numpy(dtype=np.float32) + 100.0)
        X_te = np.maximum(0, df_test[ap_cols].to_numpy(dtype=np.float32) + 100.0)
        
        y_flr_tr = df_train["FloorID"].to_numpy(dtype=int)
        y_flr_te = df_test["FloorID"].to_numpy(dtype=int)
        y_coord_tr = df_train[["X", "Y"]].to_numpy(dtype=float)
        y_coord_te = df_test[["X", "Y"]].to_numpy(dtype=float)
        
        has_floor = True
        
    elif dataset == "TUJI":
        df_train = pd.read_csv("data/benchmarks/tuji/TUJI_training.csv")
        df_test = pd.read_csv("data/benchmarks/tuji/TUJI_testing.csv")
        
        wap_cols = [f"WAP{i:03d}" for i in range(1, 311)]
        X_tr_raw = df_train[wap_cols].to_numpy(dtype=np.float32)
        X_te_raw = df_test[wap_cols].to_numpy(dtype=np.float32)
        X_tr = np.maximum(0, np.where(X_tr_raw == 100, -110.0, X_tr_raw) + 110.0)
        X_te = np.maximum(0, np.where(X_te_raw == 100, -110.0, X_te_raw) + 110.0)
        
        y_coord_tr = df_train[["LONGITUDE", "LATITUDE"]].to_numpy(dtype=float)
        y_coord_te = df_test[["LONGITUDE", "LATITUDE"]].to_numpy(dtype=float)
        has_floor = False
        
    results = []
    
    # Baseline 1: KNN (K=5)
    t0 = time.time()
    knn = KNeighborsRegressor(n_neighbors=5, metric="euclidean", weights="distance")
    knn.fit(X_tr, y_coord_tr)
    pred_knn = knn.predict(X_te)
    err_knn = np.sqrt(np.sum((pred_knn - y_coord_te)**2, axis=1))
    t_knn = time.time() - t0
    results.append({
        "Method": "Standard KNN (K=5, Euclidean)",
        "Mean Error (m)": np.mean(err_knn),
        "Median Error (m)": np.median(err_knn),
        "P90 Error (m)": np.percentile(err_knn, 90),
        "Inference Time (s)": t_knn
    })
    
    # Baseline 2: Random Forest Regressor
    t0 = time.time()
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_coord_tr)
    pred_rf = rf.predict(X_te)
    err_rf = np.sqrt(np.sum((pred_rf - y_coord_te)**2, axis=1))
    t_rf = time.time() - t0
    results.append({
        "Method": "Random Forest Regressor (T=100)",
        "Mean Error (m)": np.mean(err_rf),
        "Median Error (m)": np.median(err_rf),
        "P90 Error (m)": np.percentile(err_rf, 90),
        "Inference Time (s)": t_rf
    })
    
    # Method 3: Hierarchical RF + KNN (if multi-floor)
    if has_floor:
        t0 = time.time()
        clf_f = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_f.fit(X_tr, y_flr_tr)
        pred_flr = clf_f.predict(X_te)
        flr_acc = float(np.mean(pred_flr == y_flr_te) * 100.0)
        
        penalty = 500.0
        X_tr_pen = np.column_stack((X_tr, y_flr_tr * penalty))
        X_te_pen = np.column_stack((X_te, pred_flr * penalty))
        
        knn_hier = KNeighborsRegressor(n_neighbors=7, metric="manhattan", weights="distance")
        knn_hier.fit(X_tr_pen, y_coord_tr)
        pred_hier = knn_hier.predict(X_te_pen)
        err_hier = np.sqrt(np.sum((pred_hier - y_coord_te)**2, axis=1))
        t_hier = time.time() - t0
        results.append({
            "Method": f"Hierarchical RF + KNN (L1, Floor Acc={flr_acc:.1f}%)",
            "Mean Error (m)": np.mean(err_hier),
            "Median Error (m)": np.median(err_hier),
            "P90 Error (m)": np.percentile(err_hier, 90),
            "Inference Time (s)": t_hier
        })
    else:
        pred_flr = np.zeros(len(X_te), dtype=int)
        y_flr_tr = np.zeros(len(X_tr), dtype=int)
        
    # Method 4: RF + AutoLoC (Dual-Space Manhattan L1 + PCA Latent RAG)
    t0 = time.time()
    n_comp = min(15, X_tr.shape[1] - 1)
    pca = PCA(n_components=n_comp, random_state=42)
    scaler = StandardScaler()
    X_tr_norm = scaler.fit_transform(X_tr)
    X_te_norm = scaler.transform(X_te)
    Z_tr = pca.fit_transform(X_tr_norm)
    Z_te = pca.transform(X_te_norm)
    
    pred_autoloc = []
    for i in range(len(X_te)):
        locked_f = pred_flr[i]
        mask = (y_flr_tr == locked_f)
        sub_X_tr = X_tr[mask]
        sub_Z_tr = Z_tr[mask]
        sub_y = y_coord_tr[mask]
        
        # Dual-space distance: 60% Manhattan L1 + 40% PCA Latent Euclidean
        d_l1 = np.sum(np.abs(sub_X_tr - X_te[i]), axis=1)
        d_l1_norm = d_l1 / (np.max(d_l1) + 1e-6)
        
        d_pca = np.sqrt(np.sum((sub_Z_tr - Z_te[i])**2, axis=1))
        d_pca_norm = d_pca / (np.max(d_pca) + 1e-6)
        
        comp_d = 0.65 * d_l1_norm + 0.35 * d_pca_norm
        k_idx = np.argsort(comp_d)[:7]
        
        # Non-linear Inverse Distance Weighting
        weights = 1.0 / (comp_d[k_idx] + 1e-4)**2.0
        weights /= np.sum(weights)
        pred_pt = np.sum(sub_y[k_idx] * weights[:, None], axis=0)
        pred_autoloc.append(pred_pt)
        
    pred_autoloc = np.array(pred_autoloc)
    err_autoloc = np.sqrt(np.sum((pred_autoloc - y_coord_te)**2, axis=1))
    t_autoloc = time.time() - t0
    results.append({
        "Method": "Proposed RF + AutoLoC (Dual-Space RAG)",
        "Mean Error (m)": np.mean(err_autoloc),
        "Median Error (m)": np.median(err_autoloc),
        "P90 Error (m)": np.percentile(err_autoloc, 90),
        "Inference Time (s)": t_autoloc
    })
    
    res_df = pd.DataFrame(results)
    print(f"Summary Table for {dataset} Benchmark:")
    print(res_df.to_string(index=False))
    return results


if __name__ == "__main__":
    evaluate_uji_test_set()
    evaluate_cross_dataset_direct("TUJI")
    benchmark_new_dataset_in_domain("DLULIB")
    benchmark_new_dataset_in_domain("TUJI")

