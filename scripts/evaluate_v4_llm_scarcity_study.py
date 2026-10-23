"""Study of LLM Memory Scarcity in V4:
RF / ExtraTrees Macro-classifier uses 100% Data for Floor/Building Locking.
Only the LLM Spatial Reference Memory is pruned at ratios [20%, 30%, 40%, 50%, 70%, 100%]
across 5 independent seeds (42, 100, 2024, 777, 999) to find the optimal Pareto trade-off.
"""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
import openpyxl

sys.stdout.reconfigure(encoding='utf-8')
SEEDS = [42, 100, 2024, 777, 999]
RATIOS = [0.20, 0.30, 0.40, 0.50, 0.70, 1.00]

def compute_v4_positioning(sub_X_pos, sub_y, q_X_pos, p_exp=2.5, k_nn=11, n_comp=15):
    sub_X_p = np.power(sub_X_pos, p_exp)
    q_X_p = np.power(q_X_pos, p_exp)
    
    norm_sub = np.linalg.norm(sub_X_p, axis=1, keepdims=True) + 1e-6
    norm_q = np.linalg.norm(q_X_p, axis=1, keepdims=True) + 1e-6
    
    sub_X_norm = sub_X_p / norm_sub
    q_X_norm = q_X_p / norm_q
    
    act_comp = min(n_comp, sub_X_norm.shape[0] - 1, sub_X_norm.shape[1] - 1)
    if act_comp >= 2:
        pca = PCA(n_components=act_comp, random_state=42)
        Z_sub = pca.fit_transform(sub_X_norm)
        Z_q = pca.transform(q_X_norm)
    else:
        Z_sub = sub_X_norm
        Z_q = q_X_norm
        
    preds = []
    for i in range(len(q_X_pos)):
        d_cos = np.maximum(0.0, 1.0 - np.dot(sub_X_norm, q_X_norm[i]))
        if act_comp >= 2:
            d_pca = np.sqrt(np.sum((Z_sub - Z_q[i])**2, axis=1))
            d_pca_norm = d_pca / (np.max(d_pca) + 1e-6)
            comp_d = 0.70 * d_cos + 0.30 * d_pca_norm
        else:
            comp_d = d_cos
            
        k_use = min(k_nn, len(comp_d))
        k_idx = np.argsort(comp_d)[:k_use]
        
        p_med = np.median(sub_y[k_idx[:min(3, k_use)]], axis=0)
        dist_s = np.sqrt(np.sum((sub_y[k_idx] - p_med)**2, axis=1))
        
        w = (1.0 / (d_cos[k_idx] + 1e-4)**2.0) * np.exp(-dist_s / 25.0)
        if np.sum(w) > 0:
            w /= np.sum(w)
            pt = np.sum(sub_y[k_idx] * w[:, None], axis=0)
        else:
            pt = p_med
        preds.append(pt)
    return np.array(preds)

def run_llm_scarcity_uji():
    print(">>> [SCARCITY EXPERIMENT] UJIIndoorLoc Blind Test (1,111)...")
    df_tr = pd.read_csv("data/benchmarks/ujiindoorloc_raw/trainingData.csv")
    df_te = pd.read_csv("data/benchmarks/ujiindoorloc_raw/validationData.csv")
    wap_cols = [f"WAP{i:03d}" for i in range(1, 521)]
    X_tr_raw = df_tr[wap_cols].to_numpy(dtype=np.float32)
    X_te_raw = df_te[wap_cols].to_numpy(dtype=np.float32)
    X_tr_pos = np.maximum(0, np.where(X_tr_raw == 100, -110.0, X_tr_raw) + 110.0)
    X_te_pos = np.maximum(0, np.where(X_te_raw == 100, -110.0, X_te_raw) + 110.0)
    
    y_b_tr = df_tr["BUILDINGID"].to_numpy(dtype=int)
    y_b_te = df_te["BUILDINGID"].to_numpy(dtype=int)
    y_f_tr = df_tr["FLOOR"].to_numpy(dtype=int)
    y_f_te = df_te["FLOOR"].to_numpy(dtype=int)
    y_coord_tr = df_tr[["LONGITUDE", "LATITUDE"]].to_numpy(dtype=float)
    y_coord_te = df_te[["LONGITUDE", "LATITUDE"]].to_numpy(dtype=float)
    
    # 1. 100% Data for Macro-Classifier
    y_bf_tr = y_b_tr * 10 + y_f_tr
    unique_bf = sorted(list(set(y_bf_tr)))
    bf_map = {c: i for i, c in enumerate(unique_bf)}
    y_bf_idx = np.array([bf_map[c] for c in y_bf_tr])
    
    max_tr = np.max(X_tr_pos, axis=1, keepdims=True)
    max_te = np.max(X_te_pos, axis=1, keepdims=True)
    X_tr_rel = np.where(X_tr_pos > 0, X_tr_pos / (max_tr + 1e-6), 0.0)
    X_te_rel = np.where(X_te_pos > 0, X_te_pos / (max_te + 1e-6), 0.0)
    X_tr_exp = np.power(X_tr_rel, 4.0)
    X_te_exp = np.power(X_te_rel, 4.0)
    
    et = ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    et.fit(X_tr_exp, y_bf_idx)
    pred_bf_idx = et.predict(X_te_exp)
    pred_bld = np.array([unique_bf[i] // 10 for i in pred_bf_idx])
    pred_flr = np.array([unique_bf[i] % 10 for i in pred_bf_idx])
    
    flr_acc = float(np.mean(pred_flr == y_f_te) * 100.0)
    bld_acc = float(np.mean(pred_bld == y_b_te) * 100.0)
    print(f"  RF/ExtraTrees Macro Lock (100% Data): Building={bld_acc:.2f}%, Floor={flr_acc:.2f}%")
    
    # 2. Scarcity on LLM Vector Memory
    results = {}
    for r in RATIOS:
        n_samples = int(len(X_tr_pos) * r)
        means, meds, p90s = [], [], []
        
        for s in SEEDS:
            if r == 1.00:
                sub_X = X_tr_pos
                sub_y = y_coord_tr
                sub_b = y_b_tr
                sub_f = y_f_tr
            else:
                np.random.seed(s)
                idx = np.random.choice(len(X_tr_pos), n_samples, replace=False)
                sub_X = X_tr_pos[idx]
                sub_y = y_coord_tr[idx]
                sub_b = y_b_tr[idx]
                sub_f = y_f_tr[idx]
                
            preds = []
            for i in range(len(X_te_pos)):
                lock_b = pred_bld[i]
                lock_f = pred_flr[i]
                mask = (sub_b == lock_b) & (sub_f == lock_f)
                if not np.any(mask):
                    mask = (sub_b == lock_b)
                if not np.any(mask):
                    mask = np.ones(len(sub_X), dtype=bool)
                    
                floor_X = sub_X[mask]
                floor_y = sub_y[mask]
                pt = compute_v4_positioning(floor_X, floor_y, X_te_pos[i:i+1], p_exp=2.5, k_nn=11, n_comp=15)[0]
                preds.append(pt)
                
            err = np.sqrt(np.sum((np.array(preds) - y_coord_te)**2, axis=1))
            means.append(float(np.mean(err)))
            meds.append(float(np.median(err)))
            p90s.append(float(np.percentile(err, 90)))
            
        pct = int(r * 100)
        results[pct] = {
            "samples": n_samples,
            "mean": round(float(np.mean(means)), 2),
            "std": round(float(np.std(means)), 2),
            "median": round(float(np.mean(meds)), 2),
            "p90": round(float(np.mean(p90s)), 2),
            "floor_acc": flr_acc,
            "building_acc": bld_acc,
            "memory_reduction_pct": round((1.0 - r) * 100, 1)
        }
        print(f"  [LLM Memory {pct}% | N={n_samples:,}]: Mean={results[pct]['mean']:.2f}m ± {results[pct]['std']:.2f}m, Med={results[pct]['median']:.2f}m, P90={results[pct]['p90']:.2f}m")
        
    return results

def run_llm_scarcity_dlulib():
    print("\n>>> [SCARCITY EXPERIMENT] DLULIB (Thư viện ĐH Đà Lạt 6,500)...")
    train_dfs, test_dfs = [], []
    for flr in [1, 2, 3]:
        tr = pd.read_excel(f"data/benchmarks/dlulib/DLULIB_Floor_{flr}/Training_DLULIB_Floor{flr}_All_20.xlsx")
        te = pd.read_excel(f"data/benchmarks/dlulib/DLULIB_Floor_{flr}/Testing_DLULIB_Floor{flr}.xlsx")
        train_dfs.append(tr)
        test_dfs.append(te)
    df_train = pd.concat(train_dfs, ignore_index=True)
    df_test = pd.concat(test_dfs, ignore_index=True)
    ap_cols = [f"AP{i}" for i in range(1, 18)]
    X_tr_pos = np.maximum(0, df_train[ap_cols].to_numpy(dtype=np.float32) + 100.0)
    X_te_pos = np.maximum(0, df_test[ap_cols].to_numpy(dtype=np.float32) + 100.0)
    y_flr_tr = df_train["FloorID"].to_numpy(dtype=int)
    y_flr_te = df_test["FloorID"].to_numpy(dtype=int)
    y_coord_tr = df_train[["X", "Y"]].to_numpy(dtype=float)
    y_coord_te = df_test[["X", "Y"]].to_numpy(dtype=float)
    
    # 1. 100% Data for Macro-Classifier
    et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    et.fit(X_tr_pos, y_flr_tr)
    pred_flr = et.predict(X_te_pos)
    flr_acc = float(np.mean(pred_flr == y_flr_te) * 100.0)
    print(f"  RF/ExtraTrees Macro Lock (100% Data): Floor={flr_acc:.2f}%")
    
    # 2. Scarcity on LLM Vector Memory
    results = {}
    for r in RATIOS:
        n_samples = int(len(X_tr_pos) * r)
        means, meds, p90s = [], [], []
        
        for s in SEEDS:
            if r == 1.00:
                sub_X = X_tr_pos
                sub_y = y_coord_tr
                sub_f = y_flr_tr
            else:
                np.random.seed(s)
                idx = np.random.choice(len(X_tr_pos), n_samples, replace=False)
                sub_X = X_tr_pos[idx]
                sub_y = y_coord_tr[idx]
                sub_f = y_flr_tr[idx]
                
            preds = []
            for i in range(len(X_te_pos)):
                lock_f = pred_flr[i]
                mask = (sub_f == lock_f)
                if not np.any(mask):
                    mask = np.ones(len(sub_X), dtype=bool)
                floor_X = sub_X[mask]
                floor_y = sub_y[mask]
                pt = compute_v4_positioning(floor_X, floor_y, X_te_pos[i:i+1], p_exp=1.5, k_nn=7, n_comp=10)[0]
                preds.append(pt)
                
            err = np.sqrt(np.sum((np.array(preds) - y_coord_te)**2, axis=1))
            means.append(float(np.mean(err)))
            meds.append(float(np.median(err)))
            p90s.append(float(np.percentile(err, 90)))
            
        pct = int(r * 100)
        results[pct] = {
            "samples": n_samples,
            "mean": round(float(np.mean(means)), 2),
            "std": round(float(np.std(means)), 2),
            "median": round(float(np.mean(meds)), 2),
            "p90": round(float(np.mean(p90s)), 2),
            "floor_acc": flr_acc,
            "building_acc": 100.0,
            "memory_reduction_pct": round((1.0 - r) * 100, 1)
        }
        print(f"  [LLM Memory {pct}% | N={n_samples:,}]: Mean={results[pct]['mean']:.2f}m ± {results[pct]['std']:.2f}m, Med={results[pct]['median']:.2f}m, P90={results[pct]['p90']:.2f}m")
        
    return results

def main():
    t0 = time.time()
    uji_res = run_llm_scarcity_uji()
    dlulib_res = run_llm_scarcity_dlulib()
    
    out_file = Path("docs/Hoc/v4_llm_scarcity_study.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"UJI": uji_res, "DLULIB": dlulib_res}, f, ensure_ascii=False, indent=2)
    print(f"\nAll Scarcity Study completed in {time.time()-t0:.1f}s. Saved to {out_file}!")

if __name__ == "__main__":
    main()

