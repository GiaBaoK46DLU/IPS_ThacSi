"""Evaluation of V4 (ExtraTrees Floor Lock + Power-Scaled Cosine PCA Dual-Space Engine)
Across ALL 8 benchmark datasets:
1. 100% Full Training Data
2. 50% Data Scarcity (averaged across 5 independent seeds: 42, 100, 2024, 777, 999)

Outputs comprehensive comparative metrics against V1, V2, and V3.
"""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
import openpyxl

sys.stdout.reconfigure(encoding='utf-8')
SEEDS = [42, 100, 2024, 777, 999]

def load_xlsx_fast(file_path):
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    sheet = wb.active
    rows = list(sheet.iter_rows(values_only=True))
    header = rows[0]
    data = rows[1:]
    wb.close()
    return pd.DataFrame(data, columns=header)

def compute_v4_positioning(sub_X_pos, sub_y, q_X_pos, p_exp=2.0, k_nn=9, n_comp=15, metric_type='cosine'):
    """Compute V4 2D coordinate using Power-Scaled Cosine + PCA Dual Space or L1."""
    if metric_type == 'l1':
        preds = []
        for i in range(len(q_X_pos)):
            d1 = np.sum(np.abs(sub_X_pos - q_X_pos[i]), axis=1)
            k_idx = np.argsort(d1)[:min(k_nn, len(d1))]
            p_med = np.median(sub_y[k_idx[:min(3, len(k_idx))]], axis=0)
            dist_s = np.sqrt(np.sum((sub_y[k_idx] - p_med)**2, axis=1))
            w = (1.0 / (d1[k_idx] + 1e-4)**2.0) * np.exp(-dist_s / 10.0)
            if np.sum(w) > 0:
                w /= np.sum(w)
                pt = np.sum(sub_y[k_idx] * w[:, None], axis=0)
            else:
                pt = p_med
            preds.append(pt)
        return np.array(preds)

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

def eval_dataset_v4(name, X_tr_pos, y_tr_coord, X_te_pos, y_te_coord, 
                    flr_tr=None, flr_te=None, bld_tr=None, bld_te=None,
                    p_exp=2.0, k_nn=9, n_comp=15, is_scarcity=True, metric_type='cosine'):
    print(f"\n>>> [EVALUATION V4] Dataset: {name} (N_tr={len(X_tr_pos)}, N_te={len(X_te_pos)})...")
    
    has_bld = (bld_tr is not None and bld_te is not None and len(np.unique(bld_tr)) > 1)
    has_flr = (flr_tr is not None and flr_te is not None and len(np.unique(flr_tr)) > 1)
    
    # ---------------- 1. 100% FULL DATA ----------------
    # A. Floor/Building Classification
    if has_bld and has_flr:
        y_bf_tr = bld_tr * 10 + flr_tr
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
        
        bld_acc_100 = float(np.mean(pred_bld == bld_te) * 100.0)
        flr_acc_100 = float(np.mean(pred_flr == flr_te) * 100.0)
    elif has_flr:
        et = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        et.fit(X_tr_pos, flr_tr)
        pred_flr = et.predict(X_te_pos)
        bld_acc_100 = 100.0
        flr_acc_100 = float(np.mean(pred_flr == flr_te) * 100.0)
        pred_bld = np.zeros(len(X_te_pos), dtype=int)
    else:
        bld_acc_100 = 100.0
        flr_acc_100 = 100.0
        pred_flr = np.zeros(len(X_te_pos), dtype=int)
        pred_bld = np.zeros(len(X_te_pos), dtype=int)
        
    # B. 2D Positioning 100%
    preds_100 = []
    if has_bld and has_flr:
        for i in range(len(X_te_pos)):
            lock_b = pred_bld[i]
            lock_f = pred_flr[i]
            mask = (bld_tr == lock_b) & (flr_tr == lock_f)
            if not np.any(mask):
                mask = (bld_tr == lock_b)
            if not np.any(mask):
                mask = np.ones(len(X_tr_pos), dtype=bool)
            sub_X = X_tr_pos[mask]
            sub_y = y_tr_coord[mask]
            pt = compute_v4_positioning(sub_X, sub_y, X_te_pos[i:i+1], p_exp=p_exp, k_nn=k_nn, n_comp=n_comp, metric_type=metric_type)[0]
            preds_100.append(pt)
    elif has_flr:
        for i in range(len(X_te_pos)):
            lock_f = pred_flr[i]
            mask = (flr_tr == lock_f)
            if not np.any(mask):
                mask = np.ones(len(X_tr_pos), dtype=bool)
            sub_X = X_tr_pos[mask]
            sub_y = y_tr_coord[mask]
            pt = compute_v4_positioning(sub_X, sub_y, X_te_pos[i:i+1], p_exp=p_exp, k_nn=k_nn, n_comp=n_comp, metric_type=metric_type)[0]
            preds_100.append(pt)
    else:
        preds_100 = compute_v4_positioning(X_tr_pos, y_tr_coord, X_te_pos, p_exp=p_exp, k_nn=k_nn, n_comp=n_comp, metric_type=metric_type)
        
    err_100 = np.sqrt(np.sum((np.array(preds_100) - y_te_coord)**2, axis=1))
    res_100 = {
        "mean": float(np.mean(err_100)),
        "median": float(np.median(err_100)),
        "p90": float(np.percentile(err_100, 90)),
        "floor_acc": flr_acc_100,
        "building_acc": bld_acc_100
    }
    print(f"  [100% Data] Mean={res_100['mean']:.2f}m, Med={res_100['median']:.2f}m, P90={res_100['p90']:.2f}m, Floor={res_100['floor_acc']:.2f}%, Bld={res_100['building_acc']:.2f}%")
    
    # ---------------- 2. 50% SCARCITY DATA ----------------
    if is_scarcity:
        m_list, med_list, p90_list, f_list, b_list = [], [], [], [], []
        n_50 = int(len(X_tr_pos) * 0.5)
        for s in SEEDS:
            np.random.seed(s)
            sub_idx = np.random.choice(len(X_tr_pos), n_50, replace=False)
            s_X = X_tr_pos[sub_idx]
            s_y = y_tr_coord[sub_idx]
            s_flr = flr_tr[sub_idx] if flr_tr is not None else None
            s_bld = bld_tr[sub_idx] if bld_tr is not None else None
            
            if has_bld and has_flr:
                y_bf_sub = s_bld * 10 + s_flr
                unique_bf = sorted(list(set(y_bf_sub)))
                bf_map = {c: i for i, c in enumerate(unique_bf)}
                y_bf_sub_idx = np.array([bf_map[c] for c in y_bf_sub])
                
                max_tr_s = np.max(s_X, axis=1, keepdims=True)
                X_tr_s_rel = np.where(s_X > 0, s_X / (max_tr_s + 1e-6), 0.0)
                X_tr_s_exp = np.power(X_tr_s_rel, 4.0)
                
                et = ExtraTreesClassifier(n_estimators=100, random_state=s, n_jobs=-1)
                et.fit(X_tr_s_exp, y_bf_sub_idx)
                pred_bf_idx_s = et.predict(X_te_exp)
                pred_bld_s = np.array([unique_bf[i] // 10 for i in pred_bf_idx_s])
                pred_flr_s = np.array([unique_bf[i] % 10 for i in pred_bf_idx_s])
                
                b_list.append(float(np.mean(pred_bld_s == bld_te) * 100.0))
                f_list.append(float(np.mean(pred_flr_s == flr_te) * 100.0))
                
                preds_s = []
                for i in range(len(X_te_pos)):
                    lock_b = pred_bld_s[i]
                    lock_f = pred_flr_s[i]
                    mask = (s_bld == lock_b) & (s_flr == lock_f)
                    if not np.any(mask):
                        mask = (s_bld == lock_b)
                    if not np.any(mask):
                        mask = np.ones(len(s_X), dtype=bool)
                    sub_sub_X = s_X[mask]
                    sub_sub_y = s_y[mask]
                    pt = compute_v4_positioning(sub_sub_X, sub_sub_y, X_te_pos[i:i+1], p_exp=p_exp, k_nn=k_nn, n_comp=n_comp, metric_type=metric_type)[0]
                    preds_s.append(pt)
            elif has_flr:
                et = ExtraTreesClassifier(n_estimators=100, random_state=s, n_jobs=-1)
                et.fit(s_X, s_flr)
                pred_flr_s = et.predict(X_te_pos)
                b_list.append(100.0)
                f_list.append(float(np.mean(pred_flr_s == flr_te) * 100.0))
                
                preds_s = []
                for i in range(len(X_te_pos)):
                    lock_f = pred_flr_s[i]
                    mask = (s_flr == lock_f)
                    if not np.any(mask):
                        mask = np.ones(len(s_X), dtype=bool)
                    sub_sub_X = s_X[mask]
                    sub_sub_y = s_y[mask]
                    pt = compute_v4_positioning(sub_sub_X, sub_sub_y, X_te_pos[i:i+1], p_exp=p_exp, k_nn=k_nn, n_comp=n_comp, metric_type=metric_type)[0]
                    preds_s.append(pt)
            else:
                b_list.append(100.0)
                f_list.append(100.0)
                preds_s = compute_v4_positioning(s_X, s_y, X_te_pos, p_exp=p_exp, k_nn=k_nn, n_comp=n_comp, metric_type=metric_type)
                
            err_s = np.sqrt(np.sum((np.array(preds_s) - y_te_coord)**2, axis=1))
            m_list.append(float(np.mean(err_s)))
            med_list.append(float(np.median(err_s)))
            p90_list.append(float(np.percentile(err_s, 90)))
            
        res_50 = {
            "mean": float(np.mean(m_list)),
            "std": float(np.std(m_list)),
            "median": float(np.mean(med_list)),
            "p90": float(np.mean(p90_list)),
            "floor_acc": float(np.mean(f_list)),
            "building_acc": float(np.mean(b_list))
        }
        print(f"  [50% Scarcity] Mean={res_50['mean']:.2f}m ± {res_50['std']:.2f}m, Med={res_50['median']:.2f}m, P90={res_50['p90']:.2f}m, Floor={res_50['floor_acc']:.2f}%, Bld={res_50['building_acc']:.2f}%")
    else:
        res_50 = None
        
    return {"res_100": res_100, "res_50": res_50}

def load_uji_lib_floor(floor_num):
    dir_path = Path(f'data/benchmarks/UJI_LIB_DB_Floor_{floor_num}')
    df_tr = load_xlsx_fast(dir_path / 'Data-Training.xlsx')
    df_te = load_xlsx_fast(dir_path / 'Test-01.xlsx')

    tr_mask = ~df_tr[['x', 'y']].isna().any(axis=1)
    df_tr = df_tr[tr_mask].reset_index(drop=True)
    te_mask = ~df_te[['x', 'y']].isna().any(axis=1)
    df_te = df_te[te_mask].reset_index(drop=True)

    ap_cols = [c for c in df_tr.columns if str(c).startswith('AP') and str(c) != 'AP']
    X_tr_raw = np.nan_to_num(df_tr[ap_cols].to_numpy(dtype=np.float32), nan=100.0)
    X_te_raw = np.nan_to_num(df_te[ap_cols].to_numpy(dtype=np.float32), nan=100.0)
    X_tr = np.maximum(0, np.where(X_tr_raw == 100, -110.0, X_tr_raw) + 110.0)
    X_te = np.maximum(0, np.where(X_te_raw == 100, -110.0, X_te_raw) + 110.0)
    y_tr = df_tr[['x', 'y']].to_numpy(dtype=float)
    y_te = df_te[['x', 'y']].to_numpy(dtype=float)
    return X_tr, y_tr, X_te, y_te

def main():
    print("=" * 90)
    print(" EXPERIMENT V4: EVALUATING EXTRATREES FLOOR LOCK + POWER-SCALED COSINE DUAL-SPACE")
    print(" ACROSS ALL 8 BENCHMARKS (100% DATA & 50% SCARCITY)")
    print("=" * 90)
    t0 = time.time()
    all_res = {}
    
    # 1. UJIIndoorLoc
    df_tr = pd.read_csv("data/benchmarks/ujiindoorloc_raw/trainingData.csv")
    df_te = pd.read_csv("data/benchmarks/ujiindoorloc_raw/validationData.csv")
    wap_cols = [f"WAP{i:03d}" for i in range(1, 521)]
    X_tr_raw = df_tr[wap_cols].to_numpy(dtype=np.float32)
    X_te_raw = df_te[wap_cols].to_numpy(dtype=np.float32)
    X_tr = np.maximum(0, np.where(X_tr_raw == 100, -110.0, X_tr_raw) + 110.0)
    X_te = np.maximum(0, np.where(X_te_raw == 100, -110.0, X_te_raw) + 110.0)
    all_res["UJI"] = eval_dataset_v4("UJIIndoorLoc (Blind Test 1,111)", X_tr, df_tr[["LONGITUDE", "LATITUDE"]].to_numpy(),
                                     X_te, df_te[["LONGITUDE", "LATITUDE"]].to_numpy(),
                                     flr_tr=df_tr["FLOOR"].to_numpy(dtype=int), flr_te=df_te["FLOOR"].to_numpy(dtype=int),
                                     bld_tr=df_tr["BUILDINGID"].to_numpy(dtype=int), bld_te=df_te["BUILDINGID"].to_numpy(dtype=int),
                                     p_exp=2.5, k_nn=11, n_comp=15)
                                     
    # 2. DLULIB
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
    all_res["DLULIB"] = eval_dataset_v4("DLULIB (Thư viện ĐH Đà Lạt 6,500)", X_tr, df_train[["X", "Y"]].to_numpy(),
                                       X_te, df_test[["X", "Y"]].to_numpy(),
                                       flr_tr=df_train["FloorID"].to_numpy(dtype=int), flr_te=df_test["FloorID"].to_numpy(dtype=int),
                                       p_exp=1.5, k_nn=7, n_comp=10)
                                       
    # 3. TUJI
    df_tr = pd.read_csv("data/benchmarks/tuji/TUJI_training.csv")
    df_te = pd.read_csv("data/benchmarks/tuji/TUJI_testing.csv")
    ap_cols = [c for c in df_tr.columns if c.startswith("WAP")]
    X_tr = np.maximum(0, np.where(df_tr[ap_cols].to_numpy(dtype=np.float32) == 100, -110.0, df_tr[ap_cols].to_numpy(dtype=np.float32)) + 110.0)
    X_te = np.maximum(0, np.where(df_te[ap_cols].to_numpy(dtype=np.float32) == 100, -110.0, df_te[ap_cols].to_numpy(dtype=np.float32)) + 110.0)
    all_res["TUJI"] = eval_dataset_v4("TUJI (ĐH Đồng Tế 2,147)", X_tr, df_tr[["LONGITUDE", "LATITUDE"]].to_numpy(),
                                     X_te, df_te[["LONGITUDE", "LATITUDE"]].to_numpy(),
                                     p_exp=1.5, k_nn=7, n_comp=15)
                                     
    # 4. SYL
    df_tr = pd.read_csv("data/benchmarks/syl/SYL_Train.csv")
    df_te = pd.read_csv("data/benchmarks/syl/SYL_Test.csv")
    ap_cols = [c for c in df_tr.columns if c.startswith("MAC")]
    X_tr = np.maximum(0, np.where(df_tr[ap_cols].to_numpy(dtype=np.float32) == 100, -110.0, df_tr[ap_cols].to_numpy(dtype=np.float32)) + 110.0)
    X_te = np.maximum(0, np.where(df_te[ap_cols].to_numpy(dtype=np.float32) == 100, -110.0, df_te[ap_cols].to_numpy(dtype=np.float32)) + 110.0)
    all_res["SYL"] = eval_dataset_v4("SYL (Thư viện Thẩm Dương 1,020)", X_tr, df_tr[["LONGITUDE", "LATITUDE"]].to_numpy(),
                                    X_te, df_te[["LONGITUDE", "LATITUDE"]].to_numpy(),
                                    p_exp=1.5, k_nn=7, n_comp=15)
                                    
    # 5. HCXY
    map_hcxy = pd.read_csv('data/benchmarks/hcxy/mapxy_HCXY.txt', sep=r'\s+', header=None, names=['RP', 'X', 'Y'])
    tr_hcxy = pd.read_csv('data/benchmarks/hcxy/train_HCXY.txt', sep=r'\s+').merge(map_hcxy, on='RP')
    te_hcxy = pd.read_csv('data/benchmarks/hcxy/test_HCXY.txt', sep=r'\s+').merge(map_hcxy, on='RP')
    mac_cols = [c for c in tr_hcxy.columns if c.startswith('MAC')]
    X_tr = np.maximum(0, np.where(tr_hcxy[mac_cols].to_numpy(dtype=np.float32) == 100, -110.0, tr_hcxy[mac_cols].to_numpy(dtype=np.float32)) + 110.0)
    X_te = np.maximum(0, np.where(te_hcxy[mac_cols].to_numpy(dtype=np.float32) == 100, -110.0, te_hcxy[mac_cols].to_numpy(dtype=np.float32)) + 110.0)
    all_res["HCXY"] = eval_dataset_v4("HCXY (Tòa nhà Hành chính 860)", X_tr, tr_hcxy[['X', 'Y']].to_numpy(),
                                     X_te, te_hcxy[['X', 'Y']].to_numpy(),
                                     p_exp=1.5, k_nn=7, n_comp=15)
                                     
    # 6. HUS
    map_hus = pd.read_csv('data/benchmarks/hus/hus_mapXY.txt', sep=r'\s+', header=None, names=['RP', 'X', 'Y'])
    tr_hus = pd.read_csv('data/benchmarks/hus/hus_trainall.txt', sep=r'\s+').merge(map_hus, on='RP')
    te_hus = pd.read_csv('data/benchmarks/hus/testcase1.txt', sep=r'\s+').merge(map_hus, on='RP')
    ap_cols = ['AP1', 'AP2', 'AP3', 'AP4', 'AP5']
    X_tr = tr_hus[ap_cols].to_numpy(dtype=np.float32) + 110.0
    X_te = te_hus[ap_cols].to_numpy(dtype=np.float32) + 110.0
    all_res["HUS"] = eval_dataset_v4("HUS (Thực nghiệm HUS 205)", X_tr, tr_hus[['X', 'Y']].to_numpy(),
                                    X_te, te_hus[['X', 'Y']].to_numpy(),
                                    p_exp=1.0, k_nn=5, n_comp=5, metric_type='l1')
                                    
    # 7. UJI Lib Floor 3
    X_tr, y_tr, X_te, y_te = load_uji_lib_floor(3)
    all_res["UJI_LIB_F3"] = eval_dataset_v4(f"UJI Lib Tầng 3 ({len(X_te):,} mẫu)", X_tr, y_tr, X_te, y_te,
                                           p_exp=1.5, k_nn=7, n_comp=15)
                                           
    # 8. UJI Lib Floor 5
    X_tr, y_tr, X_te, y_te = load_uji_lib_floor(5)
    all_res["UJI_LIB_F5"] = eval_dataset_v4(f"UJI Lib Tầng 5 ({len(X_te):,} mẫu)", X_tr, y_tr, X_te, y_te,
                                           p_exp=1.5, k_nn=7, n_comp=15)
                                           
    # Save results to a dedicated V4 json
    out_file = Path("docs/Hoc/v4_all_benchmarks_scarcity_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_res, f, ensure_ascii=False, indent=2)
    print(f"\nAll evaluations complete in {time.time()-t0:.1f}s. Saved to {out_file}!")

if __name__ == "__main__":
    main()

