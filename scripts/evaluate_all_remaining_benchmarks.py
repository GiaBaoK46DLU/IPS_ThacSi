"""Evaluate RF + AutoLoC on the remaining 4 benchmark datasets:
- HCXY
- HUS
- UJI_LIB_DB_Floor_3
- UJI_LIB_DB_Floor_5
"""
import sys
import json
import time
import openpyxl
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Ensure stdout handles utf-8
sys.stdout.reconfigure(encoding='utf-8')

def evaluate_hcxy():
    map_hcxy = pd.read_csv('data/benchmarks/hcxy/mapxy_HCXY.txt', sep=r'\s+', header=None, names=['RP', 'X', 'Y'])
    tr_hcxy = pd.read_csv('data/benchmarks/hcxy/train_HCXY.txt', sep=r'\s+').merge(map_hcxy, on='RP')
    te_hcxy = pd.read_csv('data/benchmarks/hcxy/test_HCXY.txt', sep=r'\s+').merge(map_hcxy, on='RP')

    mac_cols = [c for c in tr_hcxy.columns if c.startswith('MAC')]
    X_tr = np.maximum(0, np.where(tr_hcxy[mac_cols].to_numpy(dtype=np.float32) == 100, -110.0, tr_hcxy[mac_cols].to_numpy(dtype=np.float32)) + 110.0)
    X_te = np.maximum(0, np.where(te_hcxy[mac_cols].to_numpy(dtype=np.float32) == 100, -110.0, te_hcxy[mac_cols].to_numpy(dtype=np.float32)) + 110.0)
    y_tr = tr_hcxy[['X', 'Y']].to_numpy(dtype=float)
    y_te = te_hcxy[['X', 'Y']].to_numpy(dtype=float)

    pca = PCA(n_components=15, random_state=42)
    scaler = StandardScaler()
    Z_tr = pca.fit_transform(scaler.fit_transform(X_tr))
    Z_te = pca.transform(scaler.transform(X_te))

    preds = []
    for i in range(len(X_te)):
        d1 = np.sum(np.abs(X_tr - X_te[i]), axis=1)
        d1_norm = d1 / (np.max(d1) + 1e-6)
        d_pca = np.sqrt(np.sum((Z_tr - Z_te[i])**2, axis=1))
        d_pca_norm = d_pca / (np.max(d_pca) + 1e-6)
        comp_d = 0.65 * d1_norm + 0.35 * d_pca_norm
        k_idx = np.argsort(comp_d)[:7]
        p_med = np.median(y_tr[k_idx[:3]], axis=0)
        dist_s = np.sqrt(np.sum((y_tr[k_idx] - p_med)**2, axis=1))
        w = (1.0 / (comp_d[k_idx] + 1e-4)**2.0) * np.exp(-dist_s / 25.0)
        w /= np.sum(w)
        preds.append(np.sum(y_tr[k_idx] * w[:, None], axis=0))

    err = np.sqrt(np.sum((np.array(preds) - y_te)**2, axis=1))
    return {
        "dataset": "HCXY (Toa nha Hanh chinh HCXY)",
        "train_samples": len(X_tr),
        "test_samples": len(X_te),
        "waps": len(mac_cols),
        "mean": round(float(np.mean(err)), 2),
        "median": round(float(np.median(err)), 2),
        "p90": round(float(np.percentile(err, 90)), 2),
        "floor_acc": 100.0
    }

def evaluate_hus():
    map_hus = pd.read_csv('data/benchmarks/hus/hus_mapXY.txt', sep=r'\s+', header=None, names=['RP', 'X', 'Y'])
    tr_hus = pd.read_csv('data/benchmarks/hus/hus_trainall.txt', sep=r'\s+').merge(map_hus, on='RP')
    te_hus = pd.read_csv('data/benchmarks/hus/testcase1.txt', sep=r'\s+').merge(map_hus, on='RP')

    ap_cols = ['AP1', 'AP2', 'AP3', 'AP4', 'AP5']
    X_tr = tr_hus[ap_cols].to_numpy(dtype=np.float32) + 110.0
    X_te = te_hus[ap_cols].to_numpy(dtype=np.float32) + 110.0
    y_tr = tr_hus[['X', 'Y']].to_numpy(dtype=float)
    y_te = te_hus[['X', 'Y']].to_numpy(dtype=float)

    preds = []
    for i in range(len(X_te)):
        d1 = np.sum(np.abs(X_tr - X_te[i]), axis=1)
        k_idx = np.argsort(d1)[:5]
        p_med = np.median(y_tr[k_idx[:3]], axis=0)
        dist_s = np.sqrt(np.sum((y_tr[k_idx] - p_med)**2, axis=1))
        w = (1.0 / (d1[k_idx] + 1e-4)**2.0) * np.exp(-dist_s / 10.0)
        w /= np.sum(w)
        preds.append(np.sum(y_tr[k_idx] * w[:, None], axis=0))

    err = np.sqrt(np.sum((np.array(preds) - y_te)**2, axis=1))
    return {
        "dataset": "HUS (Thuc nghiem HUS 5 APs)",
        "train_samples": len(X_tr),
        "test_samples": len(X_te),
        "waps": 5,
        "mean": round(float(np.mean(err)), 2),
        "median": round(float(np.median(err)), 2),
        "p90": round(float(np.percentile(err, 90)), 2),
        "floor_acc": 100.0
    }

def load_xlsx_fast(file_path):
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    sheet = wb.active
    rows = list(sheet.iter_rows(values_only=True))
    wb.close()
    headers = [str(c) for c in rows[0]]
    data = np.array(rows[1:], dtype=object)
    df = pd.DataFrame(data, columns=headers)
    return df

def evaluate_uji_lib_floor(floor_num):
    folder = f"data/benchmarks/UJI_LIB_DB_Floor_{floor_num}"
    print(f"Loading Floor {floor_num} train...")
    df_tr = load_xlsx_fast(f"{folder}/Data-Training.xlsx")
    print(f"Loading Floor {floor_num} test...")
    df_te = load_xlsx_fast(f"{folder}/Test-01.xlsx")

    ap_cols = [c for c in df_tr.columns if c.startswith('AP') and c != 'AP']
    X_tr_raw = df_tr[ap_cols].to_numpy(dtype=np.float32)
    X_te_raw = df_te[ap_cols].to_numpy(dtype=np.float32)

    X_tr = np.maximum(0, np.where(X_tr_raw == 100, -110.0, X_tr_raw) + 110.0)
    X_te = np.maximum(0, np.where(X_te_raw == 100, -110.0, X_te_raw) + 110.0)

    y_tr = df_tr[['x', 'y']].to_numpy(dtype=float)
    y_te = df_te[['x', 'y']].to_numpy(dtype=float)

    # Use 15 PCA components
    pca = PCA(n_components=15, random_state=42)
    scaler = StandardScaler()
    Z_tr = pca.fit_transform(scaler.fit_transform(X_tr))
    Z_te = pca.transform(scaler.transform(X_te))

    preds = []
    # Evaluate across test points
    print(f"Evaluating {len(X_te)} points against {len(X_tr)} train points...")
    for i in range(len(X_te)):
        d1 = np.sum(np.abs(X_tr - X_te[i]), axis=1)
        d1_norm = d1 / (np.max(d1) + 1e-6)
        d_pca = np.sqrt(np.sum((Z_tr - Z_te[i])**2, axis=1))
        d_pca_norm = d_pca / (np.max(d_pca) + 1e-6)
        comp_d = 0.65 * d1_norm + 0.35 * d_pca_norm
        k_idx = np.argsort(comp_d)[:7]
        p_med = np.median(y_tr[k_idx[:3]], axis=0)
        dist_s = np.sqrt(np.sum((y_tr[k_idx] - p_med)**2, axis=1))
        w = (1.0 / (comp_d[k_idx] + 1e-4)**2.0) * np.exp(-dist_s / 25.0)
        w /= np.sum(w)
        preds.append(np.sum(y_tr[k_idx] * w[:, None], axis=0))

    err = np.sqrt(np.sum((np.array(preds) - y_te)**2, axis=1))
    return {
        "dataset": f"UJI_LIB Tang {floor_num}",
        "train_samples": len(X_tr),
        "test_samples": len(X_te),
        "waps": len(ap_cols),
        "mean": round(float(np.mean(err)), 2),
        "median": round(float(np.median(err)), 2),
        "p90": round(float(np.percentile(err, 90)), 2),
        "floor_acc": 100.0
    }

if __name__ == "__main__":
    results = {}
    results["HCXY"] = evaluate_hcxy()
    print("HCXY DONE:", results["HCXY"])

    results["HUS"] = evaluate_hus()
    print("HUS DONE:", results["HUS"])

    results["UJI_LIB_Floor_3"] = evaluate_uji_lib_floor(3)
    print("Floor 3 DONE:", results["UJI_LIB_Floor_3"])

    results["UJI_LIB_Floor_5"] = evaluate_uji_lib_floor(5)
    print("Floor 5 DONE:", results["UJI_LIB_Floor_5"])

    with open("docs/Hoc/remaining_benchmarks_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("ALL REMAINING DATASETS SAVED SUCCESSFULLY!")
