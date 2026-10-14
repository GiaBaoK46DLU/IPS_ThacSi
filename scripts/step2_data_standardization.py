"""Step 2: Data Standardization Module.
Provides robust functions to:
1. Traverse and merge multi-file datasets (e.g. DLULIB floor files or UJI_LIB test files).
2. Clean and standardize Wi-Fi RSSI signals (+100 -> -110 dBm, positive shift max(0, X + 110), power scaling).
3. Split into feature matrix X (normalized/shifted WAPs) and target labels Y (Building, Floor, Coordinates).
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def load_and_merge_tabular_files(
    file_paths: List[Union[str, Path]],
    sort_by: Optional[str] = None
) -> pd.DataFrame:
    """Read and concatenate multiple CSV/Excel files into a single unified DataFrame.
    
    Handles schema alignment across files, auto-detects separators, and resets indices.
    """
    dfs = []
    for fp in file_paths:
        p = Path(fp)
        if not p.exists():
            continue
        if p.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(p)
        elif p.suffix == ".csv":
            df = pd.read_csv(p)
        elif p.suffix == ".txt":
            try:
                df = pd.read_csv(p, sep=r"\s+")
            except Exception:
                df = pd.read_csv(p)
        else:
            continue
        dfs.append(df)
        
    if not dfs:
        raise ValueError("No valid tabular files found to merge.")
        
    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    if sort_by and sort_by in merged_df.columns:
        merged_df = merged_df.sort_values(by=sort_by).reset_index(drop=True)
    return merged_df


def standardize_rssi(
    X_raw: np.ndarray,
    undetected_value: float = 100.0,
    replacement_dbm: float = -110.0,
    shift_offset: float = 110.0,
    power_gamma: Optional[float] = 1.0,
) -> np.ndarray:
    """Preprocess raw RSSI signals to standardized positive space:
    
    Step 1: Replace undetected values (+100 dBm or missing) with replacement_dbm (-110 dBm).
    Step 2: Positive shift: X_pos = max(0, X + shift_offset).
            Maps -110 dBm -> 0.0, and strong signals (-30 dBm) -> 80.0.
    Step 3: Optional non-linear power scaling: (X_pos)^gamma to amplify strong line-of-sight APs.
    """
    X = X_raw.copy()
    if undetected_value > 0:
        # Standard UJI style (+100 indicates undetectable)
        X = np.where(X >= (undetected_value - 1e-3), replacement_dbm, X)
    else:
        # Dataset where undetectable is already represented as -100 dBm or lower
        X = np.where(X <= (undetected_value + 1e-3), replacement_dbm, X)
    
    # Clip any noise lower than replacement
    X = np.maximum(replacement_dbm, X)
    
    # Shift to positive range [0, 80+]
    X_pos = np.maximum(0.0, X + shift_offset)
    
    # Power scaling
    if power_gamma is not None and power_gamma != 1.0:
        X_pos = np.power(X_pos, power_gamma)
        
    return X_pos.astype(np.float32)


def extract_features_and_labels(
    df: pd.DataFrame,
    ap_prefix: str = "WAP",
    coord_cols: Optional[List[str]] = None,
    floor_col: Optional[str] = None,
    building_col: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], List[str]]:
    """Extract RSSI feature matrix X and dictionary of target labels Y.
    
    Auto-detects AP columns if not explicitly specified.
    """
    col_names = list(df.columns)
    
    # Identify AP columns
    ap_cols = [c for c in col_names if c.upper().startswith(ap_prefix.upper()) or "MAC" in c.upper()]
    if not ap_cols:
        # Fallback: look for columns starting with AP
        ap_cols = [c for c in col_names if c.upper().startswith("AP")]
        
    if not ap_cols:
        raise ValueError(f"No AP/WAP columns found with prefix '{ap_prefix}' or 'AP' or 'MAC'.")
        
    X_raw = df[ap_cols].to_numpy(dtype=np.float32)
    
    # Detect Coordinate Columns
    if coord_cols is None:
        if "LONGITUDE" in col_names and "LATITUDE" in col_names:
            coord_cols = ["LONGITUDE", "LATITUDE"]
        elif "X" in col_names and "Y" in col_names:
            coord_cols = ["X", "Y"]
        elif "x" in col_names and "y" in col_names:
            coord_cols = ["x", "y"]
            
    # Detect Floor & Building
    if floor_col is None:
        for candidate in ["FLOOR", "FloorID", "Floor", "floor"]:
            if candidate in col_names:
                floor_col = candidate
                break
                
    if building_col is None:
        for candidate in ["BUILDINGID", "BuildingID", "building", "BUILDING"]:
            if candidate in col_names:
                building_col = candidate
                break
                
    labels = {}
    if coord_cols and all(c in col_names for c in coord_cols):
        labels["coordinates"] = df[coord_cols].to_numpy(dtype=np.float64)
        labels["coord_cols"] = coord_cols
    if floor_col and floor_col in col_names:
        labels["floor"] = df[floor_col].to_numpy(dtype=np.int64)
    if building_col and building_col in col_names:
        labels["building"] = df[building_col].to_numpy(dtype=np.int64)
        
    return X_raw, labels, ap_cols


if __name__ == "__main__":
    print("Testing Step 2 Data Standardization on DLULIB multi-floor data...")
    dlulib_train_files = sorted(glob.glob("data/benchmarks/dlulib/DLULIB_Floor_*/Training_DLULIB_Floor*_All_20.xlsx"))
    print(f"Found {len(dlulib_train_files)} DLULIB training files across floors:")
    for f in dlulib_train_files:
        print(f"  - {f}")
        
    merged_dlulib = load_and_merge_tabular_files(dlulib_train_files)
    print(f"\nMerged DataFrame Shape: {merged_dlulib.shape}")
    
    X_raw, labels, ap_cols = extract_features_and_labels(merged_dlulib, ap_prefix="AP")
    print(f"Detected AP features: {len(ap_cols)} ({ap_cols[:3]} ... {ap_cols[-3:]})")
    print(f"Extracted Labels: {list(labels.keys())}")
    
    X_standardized = standardize_rssi(X_raw, undetected_value=-100.0, replacement_dbm=-100.0, shift_offset=100.0)
    print(f"Standardized X shape: {X_standardized.shape}, Min: {X_standardized.min()}, Max: {X_standardized.max()}")
    print("Step 2 standardization completed successfully!")
