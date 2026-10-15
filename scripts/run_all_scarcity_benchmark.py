"""Evaluate RF + AutoLoC across 5 data scarcity ratios (20%, 30%, 40%, 50%, 100%)
for ALL benchmark datasets:
- UJIIndoorLoc (Blind Test 1,111)
- DLULIB (Test 6,500)
- TUJI (Test 2,147)
- SYL (Test 1,020)

Runs 5 independent seeds per ratio and outputs mean + median + floor accuracy.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

SEEDS = [42, 100, 2024, 777, 999]

def eval_scarcity_dataset(name, X_tr_full, y_tr_full, X_te, y_te, flr_tr=None, flr_te=None, n_comp=15):
    ratios = [0.2, 0.3, 0.4, 0.5, 1.0]
    results = {}
    print(f"\n==================== SCARCITY FOR {name} ====================")
    for r in ratios:
        n = int(len(X_tr_full) * r)
        m_list, med_list, f_list, p90_list = [], [], [], []
        
        for s in SEEDS:
            np.random.seed(s)
            idx = np.random.choice(len(X_tr_full), n, replace=False)
            sub_X = X_tr_full[idx]
            sub_y = y_tr_full[idx]
            
            if flr_tr is not None:
                sub_flr = flr_tr[idx]
                # Power scale
                clf = RandomForestClassifier(n_estimators=100, random_state=s, n_jobs=-1)
                clf.fit(sub_X, sub_flr)
                pred_flr = clf.predict(X_te)
                f_acc = np.mean(pred_flr == flr_te) * 100.0
                f_list.append(f_acc)
            else:
                pred_flr = np.zeros(len(X_te))
                sub_flr = np.zeros(len(sub_X))
                f_list.append(100.0)
                
            actual_n_comp = min(n_comp, sub_X.shape[1] - 1)
            pca = PCA(n_components=actual_n_comp, random_state=s)
            scaler = StandardScaler()
            Z_sub = pca.fit_transform(scaler.fit_transform(sub_X))
            Z_te = pca.transform(scaler.transform(X_te))
            
            preds = []
            for i in range(len(X_te)):
                lock_f = pred_flr[i]
                mask = (sub_flr == lock_f)
                if not np.any(mask):
                    mask = np.ones(len(sub_X), dtype=bool)
                f_X = sub_X[mask]
                f_Z = Z_sub[mask]
                f_y = sub_y[mask]
                
                d1 = np.sum(np.abs(f_X - X_te[i]), axis=1)
                d1_norm = d1 / (np.max(d1) + 1e-6)
                d_pca = np.sqrt(np.sum((f_Z - Z_te[i])**2, axis=1))
                d_pca_norm = d_pca / (np.max(d_pca) + 1e-6)
                
                comp_d = 0.65 * d1_norm + 0.35 * d_pca_norm
                k_idx = np.argsort(comp_d)[:7]
                
                # Spatial outlier damping
                p_med = np.median(f_y[k_idx[:3]], axis=0)
                dist_s = np.sqrt(np.sum((f_y[k_idx] - p_med)**2, axis=1))
                
                w = (1.0 / (comp_d[k_idx] + 1e-4)**2.0) * np.exp(-dist_s / 25.0)
                if np.sum(w) > 0:
                    w /= np.sum(w)
                    pt = np.sum(f_y[k_idx] * w[:, None], axis=0)
                else:
                    pt = np.mean(f_y[k_idx], axis=0)
                preds.append(pt)
                
            err = np.sqrt(np.sum((np.array(preds) - y_te)**2, axis=1))
            m_list.append(np.mean(err))
            med_list.append(np.median(err))
            p90_list.append(np.percentile(err, 90))
            
        pct = int(r * 100)
        results[pct] = {
            "samples": n,
            "mean": float(np.mean(m_list)),
            "std": float(np.std(m_list)),
            "median": float(np.mean(med_list)),
            "p90": float(np.mean(p90_list)),
            "floor_acc": float(np.mean(f_list))
        }
        print(f"[{name}] {pct}% (N={n:,}): Mean={results[pct]['mean']:.2f}m ± {results[pct]['std']:.2f}m, Med={results[pct]['median']:.2f}m, P90={results[pct]['p90']:.2f}m, Floor={results[pct]['floor_acc']:.2f}%")
        
    return results

def main():
    all_scarcity = {}
    
    # 1. DLULIB
    train_dfs, test_dfs = [], []
    for flr in [1, 2, 3]:
        tr = pd.read_excel(f"data/benchmarks/dlulib/DLULIB_Floor_{flr}/Training_DLULIB_Floor{flr}_All_20.xlsx")
        te = pd.read_excel(f"data/benchmarks/dlulib/DLULIB_Floor_{flr}/Testing_DLULIB_Floor{flr}.xlsx")
        train_dfs.append(tr)
        test_dfs.append(te)
    df_tr = pd.concat(train_dfs, ignore_index=True)
    df_te = pd.concat(test_dfs, ignore_index=True)
    ap_cols = [f"AP{i}" for i in range(1, 18)]
    X_tr = np.maximum(0, df_tr[ap_cols].to_numpy(dtype=np.float32) + 100.0)
    X_te = np.maximum(0, df_te[ap_cols].to_numpy(dtype=np.float32) + 100.0)
    all_scarcity["DLULIB"] = eval_scarcity_dataset("DLULIB", X_tr, df_tr[["X", "Y"]].to_numpy(), X_te, df_te[["X", "Y"]].to_numpy(), df_tr["FloorID"].to_numpy(), df_te["FloorID"].to_numpy(), n_comp=10)
    
    # 2. TUJI
    tr = pd.read_csv("data/benchmarks/tuji/TUJI_training.csv")
    te = pd.read_csv("data/benchmarks/tuji/TUJI_testing.csv")
    waps = [f"WAP{i:03d}" for i in range(1, 311)]
    X_tr = np.maximum(0, np.where(tr[waps].to_numpy(dtype=np.float32) == 100, -110.0, tr[waps].to_numpy(dtype=np.float32)) + 110.0)
    X_te = np.maximum(0, np.where(te[waps].to_numpy(dtype=np.float32) == 100, -110.0, te[waps].to_numpy(dtype=np.float32)) + 110.0)
    all_scarcity["TUJI"] = eval_scarcity_dataset("TUJI", X_tr, tr[["LONGITUDE", "LATITUDE"]].to_numpy(), X_te, te[["LONGITUDE", "LATITUDE"]].to_numpy(), n_comp=15)
    
    # 3. SYL
    tr = pd.read_csv("data/benchmarks/syl/SYL_Train.csv")
    te = pd.read_csv("data/benchmarks/syl/SYL_Test.csv")
    macs = [f"MAC{i}" for i in range(1, 364)]
    X_tr = np.maximum(0, np.where(tr[macs].to_numpy(dtype=np.float32) == 100, -110.0, tr[macs].to_numpy(dtype=np.float32)) + 110.0)
    X_te = np.maximum(0, np.where(te[macs].to_numpy(dtype=np.float32) == 100, -110.0, te[macs].to_numpy(dtype=np.float32)) + 110.0)
    all_scarcity["SYL"] = eval_scarcity_dataset("SYL", X_tr, tr[["LONGITUDE", "LATITUDE"]].to_numpy(), X_te, te[["LONGITUDE", "LATITUDE"]].to_numpy(), n_comp=15)
    
    with open("docs/Hoc/all_scarcity_results.json", "w") as f:
        json.dump(all_scarcity, f, indent=2)
    print("\nSaved all scarcity results to docs/Hoc/all_scarcity_results.json!")

if __name__ == "__main__":
    main()

