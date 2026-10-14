"""Comprehensive Data Profiling Script for New_Test_Data.
Scans directory structure, reads samples of each dataset, inspects schemas,
and compares them against the standard UJIIndoorLoc 520-WAP benchmark format.
"""
import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np

def profile_dataset(base_dir: str):
    print("=" * 80)
    print(f"STEP 1: SCANNING DIRECTORY TREE & FILES IN: {base_dir}")
    print("=" * 80)
    
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: {base_dir} does not exist!")
        return

    file_list = []
    for root, dirs, files in os.walk(base_path):
        level = len(Path(root).relative_to(base_path).parts)
        indent = "  " * level
        folder_name = os.path.basename(root) if root != str(base_path) else str(base_path)
        print(f"{indent}[DIR] {folder_name}/")
        subindent = "  " * (level + 1)
        for f in files:
            full_f = os.path.join(root, f)
            size_mb = os.path.getsize(full_f) / (1024 * 1024)
            print(f"{subindent}- {f} ({size_mb:.2f} MB)")
            file_list.append(full_f)

    print("\n" + "=" * 80)
    print("STEP 2: DEEP INSPECTION & PROFILING OF SAMPLE DATASETS")
    print("=" * 80)

    # Standard UJIIndoorLoc schema definition
    uji_std_waps = [f"WAP{i:03d}" for i in range(1, 521)]
    uji_std_targets = ["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID"]

    # We will group files by their logical dataset parent
    datasets = {}
    for f in file_list:
        rel = Path(f).relative_to(base_path)
        dataset_name = rel.parts[0]
        datasets.setdefault(dataset_name, []).append(f)

    summary_records = []

    for ds_name, files in sorted(datasets.items()):
        print(f"\n>>> DATASET / SUBFOLDER: [{ds_name}] (Total files: {len(files)})")
        # Sample the first data file
        sample_file = None
        for f in files:
            if f.endswith(('.csv', '.xlsx', '.txt')):
                sample_file = f
                break

        if not sample_file:
            print("  No tabular file (.csv, .xlsx, .txt) found.")
            continue

        print(f"  Inspecting sample file: {os.path.basename(sample_file)}")
        try:
            if sample_file.endswith('.csv'):
                df = pd.read_csv(sample_file, nrows=10)
                # full length count
                # check sep if only 1 col
                if df.shape[1] == 1:
                    df = pd.read_csv(sample_file, sep=';', nrows=10)
                if df.shape[1] == 1:
                    df = pd.read_csv(sample_file, sep=r'\s+', nrows=10)
            elif sample_file.endswith('.xlsx'):
                df = pd.read_excel(sample_file, nrows=10)
            elif sample_file.endswith('.txt'):
                try:
                    df = pd.read_csv(sample_file, sep=r'\s+', nrows=10, header=None)
                except Exception:
                    df = pd.read_csv(sample_file, sep=',', nrows=10)
            
            rows_preview, cols = df.shape
            col_names = [str(c) for c in df.columns]
            
            # Check for WAP columns
            wap_cols = [c for c in col_names if 'WAP' in c.upper() or 'AP' in c.upper()]
            exact_uji_waps = [c for c in col_names if c in uji_std_waps]
            target_cols = [c for c in col_names if c.upper() in ["LONGITUDE", "LATITUDE", "FLOOR", "BUILDINGID", "SPACEID", "X", "Y"]]

            print(f"  - Shape (sample): {rows_preview} rows x {cols} cols")
            print(f"  - Detected AP/WAP columns: {len(wap_cols)} (Matches exact UJI WAP001-520: {len(exact_uji_waps)})")
            print(f"  - Key Target columns detected: {target_cols}")
            print(f"  - Column sample (first 10): {col_names[:10]}")
            print(f"  - Column sample (last 10): {col_names[-10:]}")
            
            # Value summary
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                min_val = df[numeric_cols].min().min()
                max_val = df[numeric_cols].max().max()
                print(f"  - Value range in numeric columns: [{min_val}, {max_val}]")

            # Check compatibility with UJIIndoorLoc
            is_uji_compatible = (len(exact_uji_waps) == 520) and all(t in col_names for t in uji_std_targets)
            compatibility_status = "100% IDENTICAL TO UJI 520-WAP" if is_uji_compatible else "DIFFERENT SCHEMA / BENCHMARK"

            print(f"  - UJIIndoorLoc 520-WAP Compatibility: {compatibility_status}")

            summary_records.append({
                "Dataset": ds_name,
                "Sample File": os.path.basename(sample_file),
                "Total Files": len(files),
                "Total Columns": cols,
                "AP/WAP Count": len(wap_cols),
                "Exact UJI WAPs (520)": len(exact_uji_waps),
                "Detected Coordinates/Labels": ", ".join(target_cols[:5]),
                "Status": compatibility_status
            })

        except Exception as e:
            print(f"  Error reading {sample_file}: {e}")

    print("\n" + "=" * 80)
    print("STEP 3: SUMMARY COMPARISON TABLE ACROSS ALL SUBSETS")
    print("=" * 80)
    summary_df = pd.DataFrame(summary_records)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    profile_dataset("New_Test_Data")

