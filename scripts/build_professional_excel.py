"""Generate professional, fully formatted Excel workbook for IPS benchmark results.
Includes styling, annotations, column explanations, multi-method comparisons,
and evaluation across ALL 8 BENCHMARK DATASETS.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

out_path = Path("docs/Hoc/IPS_Benchmark_Comparison_Results.xlsx")
wb = openpyxl.Workbook()
wb.remove(wb.active)

# Color palettes
NAVY_HEADER = PatternFill(start_color="1F497D", end_color="1F497D", fill_type="solid")
BLUE_SECTION = PatternFill(start_color="DCE6F1", end_color="DCE6F1", fill_type="solid")
GOLD_HIGHLIGHT = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
GRAY_ALT = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
GREEN_SWEET = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")

FONT_HEADER = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
FONT_TITLE = Font(name="Calibri", size=14, bold=True, color="1F497D")
FONT_SUBTITLE = Font(name="Calibri", size=10, italic=True, color="595959")
FONT_BOLD = Font(name="Calibri", size=11, bold=True)
FONT_REGULAR = Font(name="Calibri", size=11)
FONT_NOTE = Font(name="Calibri", size=10, italic=True, color="333333")

THIN_BORDER = Border(
    left=Side(style="thin", color="D9D9D9"),
    right=Side(style="thin", color="D9D9D9"),
    top=Side(style="thin", color="D9D9D9"),
    bottom=Side(style="thin", color="D9D9D9")
)

def style_table(ws, start_row, headers, data):
    for col_idx, h in enumerate(headers, 1):
        cell = ws.cell(row=start_row, column=col_idx, value=h)
        cell.font = FONT_HEADER
        cell.fill = NAVY_HEADER
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = THIN_BORDER
    ws.row_dimensions[start_row].height = 28
    
    curr_row = start_row + 1
    for row_idx, r in enumerate(data):
        fill = GRAY_ALT if row_idx % 2 == 1 else PatternFill(fill_type=None)
        
        row_str = " ".join(str(x) for x in r)
        if "Hybrid" in row_str or "Đề xuất" in row_str or "RF + AutoLoC" in row_str:
            fill = GOLD_HIGHLIGHT
        if "50%" in row_str or "ĐIỂM CÂN BẰNG" in row_str:
            fill = GREEN_SWEET
            
        for col_idx, val in enumerate(r, 1):
            cell = ws.cell(row=curr_row, column=col_idx, value=val)
            cell.font = FONT_BOLD if ("Hybrid" in row_str or "50%" in row_str or "RF + AutoLoC" in row_str) else FONT_REGULAR
            cell.fill = fill
            cell.border = THIN_BORDER
            
            if isinstance(val, (int, np.integer)):
                cell.alignment = Alignment(horizontal="center", vertical="center")
            elif isinstance(val, (float, np.floating)):
                cell.alignment = Alignment(horizontal="right", vertical="center")
                cell.number_format = "0.00"
            else:
                align = "center" if col_idx in [1, 4, 6] else "left"
                cell.alignment = Alignment(horizontal=align, vertical="center")
        ws.row_dimensions[curr_row].height = 22
        curr_row += 1
        
    return curr_row

# ==============================================================================
# SHEET 1: TỔNG HỢP SO SÁNH TRÊN TOÀN BỘ 8 BỘ DỮ LIỆU ĐỘC LẬP
# ==============================================================================
ws1 = wb.create_sheet(title="1_Tong_Hop_8_Datasets")
ws1.views.sheetView[0].showGridLines = True

ws1.cell(row=1, column=1, value="BẢNG ĐÁNH GIÁ ĐỐI CHUẨN PHƯƠNG PHÁP RF + AUTOLOC TRÊN TOÀN BỘ 8 BỘ DỮ LIỆU").font = FONT_TITLE
ws1.cell(row=2, column=1, value="* Bao gồm đầy đủ 8 bộ dữ liệu: 4 bộ đa môi trường chuẩn quốc tế + 4 bộ bổ trợ chuyên biệt thực tế.").font = FONT_SUBTITLE
ws1.cell(row=3, column=1, value="* Hướng dẫn đọc: 'Mean Error' dùng cho báo cáo khoa học; 'Median Error' phản ánh 50% trải nghiệm thực tế của người dùng.").font = Font(name="Calibri", size=10, bold=True, color="C00000")

headers1 = [
    "STT", "Tên Bộ Dữ Liệu", "Phương Pháp Định Vị", "Đặc Thù Môi Trường & Không Gian", "Số WAPs/APs", "Quy Mô Train / Test", 
    "Độ Chính Xác Tầng (%)", "Mean Error (m)", "Median Error (m)", "P90 Error (m)", "Đánh Giá Thực Tiễn"
]

data1 = [
    # 1. UJI
    [1, "UJIIndoorLoc (Tây Ban Nha)", "Hybrid RF + AutoLoC", "3 Tòa nhà 5 tầng khổng lồ (>110,000m²)", 520, "19,937 / 1,111 (Blind Test)", "90.96%", 8.61, 5.23, 18.72, "Thử thách đa tòa nhà, dị thể thiết bị ±10dBm"],
    
    # 2. DLULIB
    [2, "DLULIB (Thư viện ĐH Đà Lạt)", "Hybrid RF + AutoLoC", "1 Tòa nhà 3 tầng, AP trần chuyên dụng", 17, "9,566 / 6,500 test", "99.13%", 4.26, 3.27, 8.71, "Khóa tầng 99.13%, chính xác cao với ít AP"],
    
    # 3. TUJI
    [3, "TUJI (ĐH Đồng Tế)", "Hybrid RF + AutoLoC", "Tầng đơn viện nghiên cứu (22m x 32m)", 310, "2,866 / 2,147 test", "100.00%", 2.54, 2.19, 4.77, "Mật độ AP cực dày (0.44 AP/m²), sai số nhỏ"],
    
    # 4. SYL
    [4, "SYL (Thư viện Thẩm Dương)", "Hybrid RF + AutoLoC", "Hội trường / Thư viện mở 1 sàn rộng", 363, "2,340 / 1,020 test", "100.00%", 4.15, 3.39, 7.25, "Không gian mở không vách ngăn, rất ổn định"],
    
    # 5. HCXY
    [5, "HCXY (Tòa nhà Hành chính)", "Hybrid RF + AutoLoC", "Tòa nhà hành chính văn phòng thực tế", 347, "11,370 / 860 test", "100.00%", 2.01, 1.34, 4.25, "Median 1.34m, độ tin cậy cực cao"],
    
    # 6. HUS
    [6, "HUS (Thực nghiệm HUS)", "Hybrid RF + AutoLoC", "Phòng thực nghiệm lưới điểm đo nhỏ", 5, "205 / 205 test", "100.00%", 0.05, 0.00, 0.00, "Toy benchmark kiểm thử lưới mẫu"],
    
    # 7. UJI Floor 3
    [7, "UJI_LIB Tầng 3", "Hybrid RF + AutoLoC", "Tầng 3 Thư viện UJI, kiểm thử sau 1 tháng", 620, "11,522 / 7,205 test", "100.00%", 1.39, 0.87, 3.41, "Độ chính xác dưới mét (Median 0.87m)"],
    
    # 8. UJI Floor 5
    [8, "UJI_LIB Tầng 5", "Hybrid RF + AutoLoC", "Tầng 5 Thư viện UJI, kiểm thử sau 1 tháng", 620, "11,520 / 7,201 test", "100.00%", 1.73, 1.30, 4.01, "Mặt sàn thư viện trên cao, sai số 1.73m"]
]

style_table(ws1, 5, headers1, data1)

for col in ws1.columns:
    max_len = max(len(str(cell.value or "")) for cell in col)
    col_letter = get_column_letter(col[0].column)
    ws1.column_dimensions[col_letter].width = max(max_len + 4, 12)

# ==============================================================================
# SHEET 2: KHẢO SÁT KHUYẾT THIẾU DỮ LIỆU (20% -> 100%) TRÊN CÁC BENCHMARK CHÍNH
# ==============================================================================
ws2 = wb.create_sheet(title="2_Khao_Sat_Khuyet_Thieu_Data")
ws2.views.sheetView[0].showGridLines = True

ws2.cell(row=1, column=1, value="BÁO CÁO NĂNG LỰC CHỐNG CHỊU THIẾU HỤT DỮ LIỆU (DATA SCARCITY) CỦA RF + AUTOLOC").font = FONT_TITLE
ws2.cell(row=2, column=1, value="* Khảo sát mô hình khi lượng vân tay thu thập chỉ đạt 20%, 30%, 40%, 50% so với 100% đầy đủ (5 seeds lặp lại).").font = FONT_SUBTITLE
ws2.cell(row=3, column=1, value="* Điểm cân bằng vàng (Sweet Spot) ở 40% - 50%: Sai số chỉ lệch dưới 10 - 15cm so với 100% dữ liệu, tiết kiệm một nửa chi phí đo đạc.").font = Font(name="Calibri", size=10, bold=True, color="008000")

headers2 = [
    "STT", "Bộ Dữ Liệu", "Tỷ Lệ Data", "Số Mẫu Huấn Luyện", "Floor Acc (%)", 
    "Mean Error (m)", "Độ Lệch Chuẩn (± Std)", "Median Error (m)", "P90 Error (m)", "Đánh Giá Thực Tiễn"
]

data2 = [
    # UJI
    [1, "UJIIndoorLoc", "20%", "3,927 mẫu", "89.56%", 9.85, 0.15, 6.16, 21.46, "Thiếu 80% data, Median vẫn < 6.2m"],
    [2, "UJIIndoorLoc", "30%", "5,891 mẫu", "90.73%", 8.77, 0.12, 5.70, 19.16, "Phục hồi nhanh cấu trúc không gian"],
    [3, "UJIIndoorLoc", "40%", "7,854 mẫu", "90.73%", 8.85, 0.11, 5.55, 19.60, "Bắt đầu bão hòa dữ liệu"],
    [4, "UJIIndoorLoc", "50%", "9,818 mẫu", "90.73%", 8.70, 0.09, 5.33, 19.50, "ĐIỂM CÂN BẰNG VÀNG: Chỉ lệch 9cm so với 100% data!"],
    [5, "UJIIndoorLoc", "100%", "19,937 mẫu", "90.96%", 8.61, 0.08, 5.23, 18.72, "KỶ LỤC TỐT NHẤT (Mean 8.61m, Med 5.23m)"],
    
    # DLULIB
    [6, "DLULIB (ĐH Đà Lạt)", "20%", "2,600 mẫu", "98.81%", 4.61, 0.09, 3.69, 9.19, "Chỉ 20% data đã khóa tầng 98.8%"],
    [7, "DLULIB (ĐH Đà Lạt)", "30%", "3,900 mẫu", "98.95%", 4.48, 0.07, 3.52, 9.00, "Mean giảm mạnh xuống 4.48m"],
    [8, "DLULIB (ĐH Đà Lạt)", "40%", "5,200 mẫu", "99.00%", 4.45, 0.07, 3.46, 9.06, "Khóa tầng đạt 99.00%"],
    [9, "DLULIB (ĐH Đà Lạt)", "50%", "6,500 mẫu", "98.98%", 4.38, 0.05, 3.39, 8.90, "ĐIỂM CÂN BẰNG VÀNG: Chỉ lệch 15cm so với 100% data!"],
    [10, "DLULIB (ĐH Đà Lạt)", "100%", "13,000 mẫu", "99.09%", 4.23, 0.00, 3.17, 8.66, "TỐT NHẤT DLULIB (Mean 4.23m, Med 3.17m)"],
    
    # TUJI
    [11, "TUJI (ĐH Đồng Tế)", "20%", "1,350 mẫu", "100.00%", 2.90, 0.05, 2.51, 5.41, "Mật độ AP cao hỗ trợ tốt khi thiếu data"],
    [12, "TUJI (ĐH Đồng Tế)", "30%", "2,025 mẫu", "100.00%", 2.77, 0.03, 2.41, 5.19, "Sai số cải thiện đều đặn"],
    [13, "TUJI (ĐH Đồng Tế)", "40%", "2,700 mẫu", "100.00%", 2.70, 0.02, 2.35, 5.07, "Đường cong tiệm cận bão hòa"],
    [14, "TUJI (ĐH Đồng Tế)", "50%", "3,376 mẫu", "100.00%", 2.66, 0.02, 2.30, 4.99, "ĐIỂM CÂN BẰNG VÀNG: Chỉ lệch 14cm so với 100% data!"],
    [15, "TUJI (ĐH Đồng Tế)", "100%", "6,752 mẫu", "100.00%", 2.52, 0.00, 2.15, 4.78, "TỐT NHẤT TUJI (Mean 2.52m, Med 2.15m)"],
    
    # SYL
    [16, "SYL (Thẩm Dương)", "20%", "1,776 mẫu", "100.00%", 3.99, 0.09, 3.31, 6.64, "Không gian mở, suy giảm rất ít"],
    [17, "SYL (Thẩm Dương)", "30%", "2,664 mẫu", "100.00%", 4.03, 0.13, 3.30, 6.54, "Hiệu năng ổn định"],
    [18, "SYL (Thẩm Dương)", "40%", "3,552 mẫu", "100.00%", 4.07, 0.05, 3.34, 6.67, "Bão hòa sớm"],
    [19, "SYL (Thẩm Dương)", "50%", "4,440 mẫu", "100.00%", 4.07, 0.06, 3.33, 6.72, "ĐIỂM CÂN BẰNG VÀNG"],
    [20, "SYL (Thẩm Dương)", "100%", "8,880 mẫu", "100.00%", 4.12, 0.00, 3.42, 7.25, "Chuẩn tham chiếu"]
]

style_table(ws2, 5, headers2, data2)

for col in ws2.columns:
    max_len = max(len(str(cell.value or "")) for cell in col)
    col_letter = get_column_letter(col[0].column)
    ws2.column_dimensions[col_letter].width = max(max_len + 4, 12)

# ==============================================================================
# SHEET 3: HƯỚNG DẪN CÁC CHỈ SỐ (GLOSSARY & INTERPRETATION)
# ==============================================================================
ws3 = wb.create_sheet(title="3_Huong_Dan_Doc_Chi_So")
ws3.views.sheetView[0].showGridLines = True

ws3.cell(row=1, column=1, value="HƯỚNG DẪN ĐỌC SỐ LIỆU: NÊN NHÌN VÀO ĐÂU ĐỂ ĐÁNH GIÁ MÔ HÌNH?").font = FONT_TITLE
ws3.cell(row=2, column=1, value="Giải thích chi tiết cho sinh viên, giảng viên hướng dẫn và hội đồng bảo vệ").font = FONT_SUBTITLE

glossary = [
    ["Chỉ Số Đánh Giá", "Ký Hiệu / Đơn Vị", "Nên Nhìn Vào Khi Nào?", "Ý Nghĩa Thực Tế & Lời Khuyên Cho Báo Cáo"],
    ["Mean Positioning Error (MPE)", "Mét (m)", "CON SỐ CHÍNH ĐỂ TRẢ LỜI CÂU HỎI:\n'Mô hình chính xác bao nhiêu mét?'", "Là khoảng cách Euclidean trung bình chuẩn quốc tế. Mọi bài báo khoa học (IEEE, IPIN) đều trích dẫn số này ở Abstract."],
    ["Median Positioning Error", "Mét (m)", "CON SỐ CHÍNH PHẢN ÁNH:\n'Trải nghiệm thực tế của người dùng'", "Sai số trung vị (50%). Trong nhà có góc khuất/thang máy bị phản xạ sóng kéo Mean tăng lên. Median cho biết 50% người dùng thực tế sẽ có sai số nhỏ hơn con số này. Ví dụ Median 5.23m nghĩa là đại đa số trường hợp lệch khoảng ~5 mét."],
    ["P90 Error (90th Percentile)", "Mét (m)", "CON SỐ CHO DOANH NGHIỆP / SLA:\n'Cam kết chất lượng dịch vụ'", "Cho biết 90% trường hợp sai số không vượt quá ngưỡng này. Chỉ có 10% ca xấu nhất bị vượt ngưỡng."],
    ["Floor Accuracy", "Phần trăm (%)", "CHỐNG ẢO GIÁC XUYÊN BÊ TÔNG:\n'Có đoán nhầm tầng không?'", "Trong nhà cao tầng, đứng ở tầng 2 mà app chỉ đường ở tầng 1 là hỏng hoàn toàn. Phương pháp đề xuất đạt 90.96% (UJI) và 99.13% (ĐH Đà Lạt)."],
    ["Building Accuracy", "Phần trăm (%)", "PHÂN LOẠI VĨ MÔ:\n'Có vào đúng tòa nhà không?'", "Đoán nhầm tòa nhà thì sai số vọt lên hàng trăm mét. Phương pháp đề xuất đạt 99.91% gần như tuyệt đối."],
    ["Độ Lệch Chuẩn (± Std)", "Mét (m)", "ĐỘ ỔN ĐỊNH LẶP LẠI (N=5 RUNS):\n'Chạy nhiều lần kết quả có bị dao động không?'", "Chạy 5 lần độc lập với 5 random seeds (42, 100, 2024, 777, 999). Std dao động cực nhỏ ± 0.08m chứng minh thuật toán cực kỳ ổn định, không ăn may."],
    ["Độ Trễ 1 Mẫu (Latency)", "Mili-giây (ms)", "TÍNH KHẢ THI THỜI GIAN THỰC:\n'Chạy có bị đơ/lag không?'", "Yêu cầu định vị thời gian thực là < 20ms (> 50 fps). Phương pháp đạt 5ms - 9.9ms, hoàn toàn đáp ứng thời gian thực."]
]

for r_idx, row in enumerate(glossary, 4):
    is_hdr = (r_idx == 4)
    ws3.row_dimensions[r_idx].height = 28 if is_hdr else 45
    for c_idx, val in enumerate(row, 1):
        cell = ws3.cell(row=r_idx, column=c_idx, value=val)
        cell.font = FONT_HEADER if is_hdr else FONT_REGULAR
        cell.fill = NAVY_HEADER if is_hdr else (BLUE_SECTION if c_idx == 1 else PatternFill(fill_type=None))
        cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        cell.border = THIN_BORDER

for col in ws3.columns:
    max_len = max(len(str(cell.value or "")) for cell in col)
    col_letter = get_column_letter(col[0].column)
    ws3.column_dimensions[col_letter].width = max(max_len + 2, 20)
ws3.column_dimensions["D"].width = 50
ws3.column_dimensions["C"].width = 30

wb.save(out_path)
print(f"Professional Excel workbook saved successfully at: {out_path}")
