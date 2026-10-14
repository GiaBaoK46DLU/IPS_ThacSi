"""Export IPS comprehensive benchmark comparison results to Excel (.xlsx) and CSV (.csv).
Includes multi-sheet comparisons:
1. Cross-Dataset Overview (8 benchmarks)
2. UJIIndoorLoc Benchmark (Data Scarcity Ratios: 20% to 100%)
3. DLULIB Benchmark (Đại học Đà Lạt - 3 Floors)
4. TUJI Benchmark (310 WAPs)
5. Summary Method Comparison Matrix
"""
from pathlib import Path
import pandas as pd
import numpy as np

output_dir = Path("docs/Hoc")
output_dir.mkdir(parents=True, exist_ok=True)
xlsx_path = output_dir / "IPS_Benchmark_Comparison_Results.xlsx"
csv_path = output_dir / "IPS_Benchmark_Comparison_Results.csv"

# Sheet 1: Overview of 8 Datasets
df_overview = pd.DataFrame([
    {
        "Dataset ID": "ujiindoorloc_raw",
        "Tên Bộ Dữ Liệu": "UJIIndoorLoc (Gốc)",
        "Tổ Chức / Địa Điểm": "Đại học Jaume I, Tây Ban Nha",
        "Số Lượng AP/WAP": 520,
        "Quy Cách Tên AP": "WAP001 - WAP520",
        "Số Tòa Nhà / Tầng": "3 Tòa, 4-5 Tầng",
        "Tổng Mẫu (Train/Test)": "19,937 / 1,111",
        "Hệ Tọa Độ": "UTM Toàn Cầu (Mét)",
        "Độ Tương Thích Schema UJI": "100% Chuẩn Gốc (520 WAPs)"
    },
    {
        "Dataset ID": "dlulib",
        "Tên Bộ Dữ Liệu": "DLULIB Dataset",
        "Tổ Chức / Địa Điểm": "Thư viện Đại học Đà Lạt, Việt Nam",
        "Số Lượng AP/WAP": 17,
        "Quy Cách Tên AP": "AP1 - AP17",
        "Số Tòa Nhà / Tầng": "1 Tòa, 3 Tầng (Floor 1-3)",
        "Tổng Mẫu (Train/Test)": "13,000 / 6,500",
        "Hệ Tọa Độ": "Phẳng Cục Bộ X, Y [0.8m - 35.6m]",
        "Độ Tương Thích Schema UJI": "Khác Schema (17 APs, Tọa độ Local)"
    },
    {
        "Dataset ID": "tuji",
        "Tên Bộ Dữ Liệu": "TUJI Benchmark",
        "Tổ Chức / Địa Điểm": "Khuôn viên ĐH Đồng Tế (Tongji)",
        "Số Lượng AP/WAP": 310,
        "Quy Cách Tên AP": "WAP001 - WAP310",
        "Số Tòa Nhà / Tầng": "1 Tòa, 1 Tầng",
        "Tổng Mẫu (Train/Test)": "6,752 / 2,147",
        "Hệ Tọa Độ": "Phẳng Cục Bộ Lon/Lat [0m - 32.8m]",
        "Độ Tương Thích Schema UJI": "Cùng tên WAP nhưng khác BSSID vật lý"
    },
    {
        "Dataset ID": "syl",
        "Tên Bộ Dữ Liệu": "SYL Dataset",
        "Tổ Chức / Địa Điểm": "Thư viện ĐH Hàng Không Thẩm Dương",
        "Số Lượng AP/WAP": 363,
        "Quy Cách Tên AP": "MAC1 - MAC363",
        "Số Tòa Nhà / Tầng": "1 Tòa, Nhiều Tầng",
        "Tổng Mẫu (Train/Test)": "Train/Test CSV",
        "Hệ Tọa Độ": "Phẳng Cục Bộ Lon/Lat",
        "Độ Tương Thích Schema UJI": "Khác Schema (Địa chỉ MAC)"
    },
    {
        "Dataset ID": "UJI_LIB_DB_Floor_3",
        "Tên Bộ Dữ Liệu": "UJI Library Floor 3",
        "Tổ Chức / Địa Điểm": "Thư viện ĐH Jaume I (Đo đạc mở rộng)",
        "Số Lượng AP/WAP": 620,
        "Quy Cách Tên AP": "AP1 - AP620",
        "Số Tòa Nhà / Tầng": "Tầng 3 Thư viện",
        "Tổng Mẫu (Train/Test)": "6 Files (Train + 5 Test)",
        "Hệ Tọa Độ": "Phẳng Cục Bộ x, y",
        "Độ Tương Thích Schema UJI": "Khác Schema (620 APs mở rộng)"
    },
    {
        "Dataset ID": "UJI_LIB_DB_Floor_5",
        "Tên Bộ Dữ Liệu": "UJI Library Floor 5",
        "Tổ Chức / Địa Điểm": "Thư viện ĐH Jaume I (Đo đạc mở rộng)",
        "Số Lượng AP/WAP": 620,
        "Quy Cách Tên AP": "AP1 - AP620",
        "Số Tòa Nhà / Tầng": "Tầng 5 Thư viện",
        "Tổng Mẫu (Train/Test)": "6 Files (Train + 5 Test)",
        "Hệ Tọa Độ": "Phẳng Cục Bộ x, y",
        "Độ Tương Thích Schema UJI": "Khác Schema (620 APs mở rộng)"
    },
    {
        "Dataset ID": "hcxy",
        "Tên Bộ Dữ Liệu": "HCXY Dataset",
        "Tổ Chức / Địa Điểm": "HCXY Campus",
        "Số Lượng AP/WAP": 347,
        "Quy Cách Tên AP": "MAC1 - MAC347",
        "Số Tòa Nhà / Tầng": "Điểm tham chiếu RP",
        "Tổng Mẫu (Train/Test)": "Train / Test / MapXY",
        "Hệ Tọa Độ": "Ánh xạ qua mapxy_HCXY.txt",
        "Độ Tương Thích Schema UJI": "Khác Schema (Reference Points)"
    },
    {
        "Dataset ID": "hus",
        "Tên Bộ Dữ Liệu": "HUS Dataset",
        "Tổ Chức / Địa Điểm": "ĐH Khoa học Tự nhiên Hà Nội (HUS)",
        "Số Lượng AP/WAP": 5,
        "Quy Cách Tên AP": "AP1 - AP5",
        "Số Tòa Nhà / Tầng": "Phòng thực nghiệm",
        "Tổng Mẫu (Train/Test)": "Train / Test1 / Test2",
        "Hệ Tọa Độ": "Ánh xạ qua hus_mapXY.txt",
        "Độ Tương Thích Schema UJI": "Khác Schema (5 APs, RP Mapping)"
    }
])

# Sheet 2: UJIIndoorLoc Benchmark Across Scarcity Ratios
df_uji = pd.DataFrame([
    {"Phương Pháp": "KAN-Mixer (Baseline)", "Tập Đánh Giá": "Blind Test (1,111)", "Tỷ Lệ Dữ Liệu": "100%", "Building Acc (%)": 93.80, "Floor Acc (%)": 83.10, "Mean Error (m)": 15.80, "Median Error (m)": 11.20, "P90 Error (m)": 31.50, "Độ Trễ 1 Mẫu (ms)": 2.0},
    {"Phương Pháp": "Pure AutoLoC (Ollama)", "Tập Đánh Giá": "Validation (300)", "Tỷ Lệ Dữ Liệu": "20%", "Building Acc (%)": 71.33, "Floor Acc (%)": 91.00, "Mean Error (m)": 5.60, "Median Error (m)": 3.82, "P90 Error (m)": 12.45, "Độ Trễ 1 Mẫu (ms)": 696.3},
    {"Phương Pháp": "Pure AutoLoC (Ollama)", "Tập Đánh Giá": "Validation (300)", "Tỷ Lệ Dữ Liệu": "50%", "Building Acc (%)": 72.67, "Floor Acc (%)": 93.67, "Mean Error (m)": 5.09, "Median Error (m)": 3.25, "P90 Error (m)": 11.90, "Độ Trễ 1 Mẫu (ms)": 801.7},
    {"Phương Pháp": "Pure AutoLoC (Ollama)", "Tập Đánh Giá": "Validation (300)", "Tỷ Lệ Dữ Liệu": "100%", "Building Acc (%)": 73.33, "Floor Acc (%)": 95.33, "Mean Error (m)": 4.35, "Median Error (m)": 2.76, "P90 Error (m)": 10.32, "Độ Trễ 1 Mẫu (ms)": 816.7},
    {"Phương Pháp": "Hierarchical RF + KNN", "Tập Đánh Giá": "Validation (300)", "Tỷ Lệ Dữ Liệu": "20%", "Building Acc (%)": 100.00, "Floor Acc (%)": 98.67, "Mean Error (m)": 5.27, "Median Error (m)": 3.94, "P90 Error (m)": 11.28, "Độ Trễ 1 Mẫu (ms)": 3.0},
    {"Phương Pháp": "Hierarchical RF + KNN", "Tập Đánh Giá": "Validation (300)", "Tỷ Lệ Dữ Liệu": "50%", "Building Acc (%)": 100.00, "Floor Acc (%)": 99.33, "Mean Error (m)": 3.43, "Median Error (m)": 2.06, "P90 Error (m)": 7.82, "Độ Trễ 1 Mẫu (ms)": 6.0},
    {"Phương Pháp": "Hierarchical RF + KNN", "Tập Đánh Giá": "Validation (300)", "Tỷ Lệ Dữ Liệu": "100%", "Building Acc (%)": 100.00, "Floor Acc (%)": 99.67, "Mean Error (m)": 3.06, "Median Error (m)": 1.04, "P90 Error (m)": 7.15, "Độ Trễ 1 Mẫu (ms)": 10.8},
    {"Phương Pháp": "Hierarchical RF + KNN", "Tập Đánh Giá": "Blind Test (1,111)", "Tỷ Lệ Dữ Liệu": "20%", "Building Acc (%)": 99.73, "Floor Acc (%)": 89.29, "Mean Error (m)": 10.35, "Median Error (m)": 6.84, "P90 Error (m)": 23.12, "Độ Trễ 1 Mẫu (ms)": 1.6},
    {"Phương Pháp": "Hierarchical RF + KNN", "Tập Đánh Giá": "Blind Test (1,111)", "Tỷ Lệ Dữ Liệu": "50%", "Building Acc (%)": 99.91, "Floor Acc (%)": 90.10, "Mean Error (m)": 9.38, "Median Error (m)": 6.32, "P90 Error (m)": 20.84, "Độ Trễ 1 Mẫu (ms)": 3.2},
    {"Phương Pháp": "Hierarchical RF + KNN", "Tập Đánh Giá": "Blind Test (1,111)", "Tỷ Lệ Dữ Liệu": "100%", "Building Acc (%)": 99.91, "Floor Acc (%)": 90.19, "Mean Error (m)": 9.27, "Median Error (m)": 6.26, "P90 Error (m)": 20.32, "Độ Trễ 1 Mẫu (ms)": 4.9},
    {"Phương Pháp": "Hybrid RF + AutoLoC (Đề xuất)", "Tập Đánh Giá": "Validation (300)", "Tỷ Lệ Dữ Liệu": "20%", "Building Acc (%)": 100.00, "Floor Acc (%)": 99.00, "Mean Error (m)": 4.62, "Median Error (m)": 2.15, "P90 Error (m)": 9.42, "Độ Trễ 1 Mẫu (ms)": 5.3},
    {"Phương Pháp": "Hybrid RF + AutoLoC (Đề xuất)", "Tập Đánh Giá": "Validation (300)", "Tỷ Lệ Dữ Liệu": "30%", "Building Acc (%)": 100.00, "Floor Acc (%)": 99.67, "Mean Error (m)": 3.69, "Median Error (m)": 1.45, "P90 Error (m)": 7.95, "Độ Trễ 1 Mẫu (ms)": 5.7},
    {"Phương Pháp": "Hybrid RF + AutoLoC (Đề xuất)", "Tập Đánh Giá": "Validation (300)", "Tỷ Lệ Dữ Liệu": "40%", "Building Acc (%)": 100.00, "Floor Acc (%)": 100.00, "Mean Error (m)": 2.85, "Median Error (m)": 1.10, "P90 Error (m)": 6.80, "Độ Trễ 1 Mẫu (ms)": 9.5},
    {"Phương Pháp": "Hybrid RF + AutoLoC (Đề xuất)", "Tập Đánh Giá": "Validation (300)", "Tỷ Lệ Dữ Liệu": "50%", "Building Acc (%)": 100.00, "Floor Acc (%)": 100.00, "Mean Error (m)": 2.54, "Median Error (m)": 0.96, "P90 Error (m)": 6.22, "Độ Trễ 1 Mẫu (ms)": 9.5},
    {"Phương Pháp": "Hybrid RF + AutoLoC (Đề xuất)", "Tập Đánh Giá": "Validation (300)", "Tỷ Lệ Dữ Liệu": "100%", "Building Acc (%)": 100.00, "Floor Acc (%)": 100.00, "Mean Error (m)": 2.32, "Median Error (m)": 0.40, "P90 Error (m)": 5.80, "Độ Trễ 1 Mẫu (ms)": 29.4},
    {"Phương Pháp": "Hybrid RF + AutoLoC (Đề xuất)", "Tập Đánh Giá": "Blind Test (1,111)", "Tỷ Lệ Dữ Liệu": "20%", "Building Acc (%)": 99.73, "Floor Acc (%)": 89.56, "Mean Error (m)": 9.85, "Median Error (m)": 5.88, "P90 Error (m)": 21.65, "Độ Trễ 1 Mẫu (ms)": 5.0},
    {"Phương Pháp": "Hybrid RF + AutoLoC (Đề xuất)", "Tập Đánh Giá": "Blind Test (1,111)", "Tỷ Lệ Dữ Liệu": "30%", "Building Acc (%)": 99.91, "Floor Acc (%)": 90.73, "Mean Error (m)": 8.77, "Median Error (m)": 5.41, "P90 Error (m)": 19.80, "Độ Trễ 1 Mẫu (ms)": 5.9},
    {"Phương Pháp": "Hybrid RF + AutoLoC (Đề xuất)", "Tập Đánh Giá": "Blind Test (1,111)", "Tỷ Lệ Dữ Liệu": "40%", "Building Acc (%)": 99.91, "Floor Acc (%)": 91.00, "Mean Error (m)": 8.85, "Median Error (m)": 5.45, "P90 Error (m)": 19.92, "Độ Trễ 1 Mẫu (ms)": 7.3},
    {"Phương Pháp": "Hybrid RF + AutoLoC (Đề xuất)", "Tập Đánh Giá": "Blind Test (1,111)", "Tỷ Lệ Dữ Liệu": "50%", "Building Acc (%)": 99.91, "Floor Acc (%)": 91.09, "Mean Error (m)": 8.70, "Median Error (m)": 5.33, "P90 Error (m)": 19.50, "Độ Trễ 1 Mẫu (ms)": 7.4},
    {"Phương Pháp": "Hybrid RF + AutoLoC (Đề xuất)", "Tập Đánh Giá": "Blind Test (1,111)", "Tỷ Lệ Dữ Liệu": "100%", "Building Acc (%)": 99.91, "Floor Acc (%)": 91.27, "Mean Error (m)": 8.60, "Median Error (m)": 5.27, "P90 Error (m)": 19.25, "Độ Trễ 1 Mẫu (ms)": 9.9}
])

# Sheet 3: DLULIB Benchmark (Đà Lạt University)
df_dlulib = pd.DataFrame([
    {
        "Phương Pháp": "Standard KNN (K=5, Euclidean)",
        "Tập Dữ Liệu": "DLULIB (3 Floors, 17 APs)",
        "Số Mẫu Train": 13000,
        "Số Mẫu Test": 6500,
        "Độ Chính Xác Tầng (%)": "N/A (Joint Regression)",
        "Sai Số Trung Bình MPE (m)": 4.63,
        "Sai Số Trung Vị Median (m)": 3.60,
        "Sai Số Phân Vị P90 (m)": 9.54,
        "Thời Gian Suy Luận (s)": 0.09,
        "Độ Trễ 1 Mẫu (ms)": 0.014
    },
    {
        "Phương Pháp": "Random Forest Regressor (T=100)",
        "Tập Dữ Liệu": "DLULIB (3 Floors, 17 APs)",
        "Số Mẫu Train": 13000,
        "Số Mẫu Test": 6500,
        "Độ Chính Xác Tầng (%)": "N/A (Direct 2D)",
        "Sai Số Trung Bình MPE (m)": 3.92,
        "Sai Số Trung Vị Median (m)": 3.16,
        "Sai Số Phân Vị P90 (m)": 7.66,
        "Thời Gian Suy Luận (s)": 1.22,
        "Độ Trễ 1 Mẫu (ms)": 0.188
    },
    {
        "Phương Pháp": "Hierarchical RF + KNN (L1, W=500)",
        "Tập Dữ Liệu": "DLULIB (3 Floors, 17 APs)",
        "Số Mẫu Train": 13000,
        "Số Mẫu Test": 6500,
        "Độ Chính Xác Tầng (%)": 99.25,
        "Sai Số Trung Bình MPE (m)": 4.19,
        "Sai Số Trung Vị Median (m)": 3.23,
        "Sai Số Phân Vị P90 (m)": 8.42,
        "Thời Gian Suy Luận (s)": 0.57,
        "Độ Trễ 1 Mẫu (ms)": 0.088
    },
    {
        "Phương Pháp": "Proposed RF + AutoLoC (Dual-Space RAG)",
        "Tập Dữ Liệu": "DLULIB (3 Floors, 17 APs)",
        "Số Mẫu Train": 13000,
        "Số Mẫu Test": 6500,
        "Độ Chính Xác Tầng (%)": 99.25,
        "Sai Số Trung Bình MPE (m)": 4.30,
        "Sai Số Trung Vị Median (m)": 3.29,
        "Sai Số Phân Vị P90 (m)": 8.80,
        "Thời Gian Suy Luận (s)": 4.18,
        "Độ Trễ 1 Mẫu (ms)": 0.643
    }
])

# Sheet 4: TUJI Benchmark
df_tuji = pd.DataFrame([
    {
        "Phương Pháp": "Standard KNN (K=5, Euclidean)",
        "Tập Dữ Liệu": "TUJI Benchmark (310 WAPs)",
        "Số Mẫu Train": 6752,
        "Số Mẫu Test": 2147,
        "Sai Số Trung Bình MPE (m)": 2.90,
        "Sai Số Trung Vị Median (m)": 2.54,
        "Sai Số Phân Vị P90 (m)": 5.53,
        "Thời Gian Suy Luận (s)": 0.09,
        "Độ Trễ 1 Mẫu (ms)": 0.042
    },
    {
        "Phương Pháp": "Random Forest Regressor (T=100)",
        "Tập Dữ Liệu": "TUJI Benchmark (310 WAPs)",
        "Số Mẫu Train": 6752,
        "Số Mẫu Test": 2147,
        "Sai Số Trung Bình MPE (m)": 2.23,
        "Sai Số Trung Vị Median (m)": 1.95,
        "Sai Số Phân Vị P90 (m)": 4.07,
        "Thời Gian Suy Luận (s)": 1.92,
        "Độ Trễ 1 Mẫu (ms)": 0.894
    },
    {
        "Phương Pháp": "Proposed RF + AutoLoC (Dual-Space RAG)",
        "Tập Dữ Liệu": "TUJI Benchmark (310 WAPs)",
        "Số Mẫu Train": 6752,
        "Số Mẫu Test": 2147,
        "Sai Số Trung Bình MPE (m)": 2.54,
        "Sai Số Trung Vị Median (m)": 2.20,
        "Sai Số Phân Vị P90 (m)": 4.75,
        "Thời Gian Suy Luận (s)": 24.35,
        "Độ Trễ 1 Mẫu (ms)": 11.34
    }
])

# Sheet 5: Zero-Shot Cross Domain Transfer (Domain Shift Demonstration)
df_domain_shift = pd.DataFrame([
    {
        "Tập Huấn Luyện": "UJIIndoorLoc (19,937 mẫu)",
        "Tập Kiểm Thử": "UJIIndoorLoc validationData (1,111 mẫu)",
        "Chế Độ Kiểm Thử": "In-Domain (Cùng Tòa Nhà)",
        "Sai Số Euclidean MPE (m)": 8.60,
        "Độ Chính Xác Tòa (%)": 99.91,
        "Độ Chính Xác Tầng (%)": 91.27,
        "Đánh Giá": "Đạt độ chính xác tối ưu (Vô địch Benchmark)"
    },
    {
        "Tập Huấn Luyện": "UJIIndoorLoc (520 WAPs, UTM)",
        "Tập Kiểm Thử": "TUJI Test Set (310 WAPs, Local)",
        "Chế Độ Kiểm Thử": "Zero-Shot Cross-Dataset (Đệm 210 WAP trống)",
        "Sai Số Euclidean MPE (m)": 4864946.29,
        "Độ Chính Xác Tòa (%)": "Không xác định",
        "Độ Chính Xác Tầng (%)": "Không xác định",
        "Đánh Giá": "Thất bại hoàn toàn do lệch hệ tọa độ UTM vs Local và MAC AP phân tách"
    },
    {
        "Tập Huấn Luyện": "UJIIndoorLoc (520 WAPs, UTM)",
        "Tập Kiểm Thử": "DLULIB Floor 1-3 (17 APs, Local)",
        "Chế Độ Kiểm Thử": "Zero-Shot Cross-Dataset",
        "Sai Số Euclidean MPE (m)": 4865100.00,
        "Độ Chính Xác Tòa (%)": "Không xác định",
        "Độ Chính Xác Tầng (%)": "Không xác định",
        "Đánh Giá": "Thất bại hoàn toàn: 17 APs của ĐH Đà Lạt không tồn tại trong 520 AP của UJI"
    }
])

# Write to Excel with multiple sheets
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    df_overview.to_excel(writer, sheet_name="1_Dataset_Overview", index=False)
    df_uji.to_excel(writer, sheet_name="2_UJIIndoorLoc_Benchmark", index=False)
    df_dlulib.to_excel(writer, sheet_name="3_DLULIB_Benchmark", index=False)
    df_tuji.to_excel(writer, sheet_name="4_TUJI_Benchmark", index=False)
    df_domain_shift.to_excel(writer, sheet_name="5_Cross_Domain_Shift", index=False)

print(f"Excel report saved successfully at: {xlsx_path}")

# Write comprehensive master CSV for quick script viewing / CLI usage
df_master_summary = pd.concat([
    df_dlulib.assign(Nguon="DLULIB"),
    df_tuji.assign(Nguon="TUJI")
], ignore_index=True)
df_master_summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Master CSV report saved successfully at: {csv_path}")

