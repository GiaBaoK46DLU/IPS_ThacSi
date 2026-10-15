# Các bộ dữ liệu IPS được cung cấp

Nếu đã chạy audit trên máy này, đọc báo cáo cục bộ
`analysis/2026-10-14/BAO_CAO_DANH_GIA.md` trước khi đưa dữ liệu vào pipeline.

- `raw/`: 34 file gốc, giữ nguyên byte; DLU và UJI Library theo tầng, HCXY/SYL gom dưới SODIndoorLoc.
- `analysis/2026-10-14/`: báo cáo, bảng kiểm kê, đối chiếu nguồn và biểu đồ.
- `tools/`: script đọc/audit và tổng hợp; không train mô hình hay sửa raw.

`raw/` và `analysis/` được Git ignore; clone mới cần đặt dữ liệu được cung cấp
vào đúng đường dẫn. Mã audit và README được chia sẻ, không đưa 267 MiB dữ liệu
thô hoặc cache/báo cáo sinh ra vào lịch sử Git.

Những điểm cần chú ý: HUS test gần như trùng hoàn toàn train; TUJI có header
metadata sai nghĩa; DLU −100 cần xác nhận; Library có phần thống kê cuối workbook
và chứa 25 tháng. Không tự đưa mọi bộ vào cùng schema hoặc gộp AP cùng tên.

## Bốn bộ đã có adapter thực nghiệm KAN

| Dataset | Đường dẫn dưới `raw/` | Đầu vào |
| --- | --- | --- |
| DLULIB | `dlulib/floor_1/`, `floor_2/`, `floor_3/` | `Training_DLULIB_Floor{f}_All_20.xlsx`, `Testing_DLULIB_Floor{f}.xlsx`; 17 AP, gộp ba tầng |
| HCXY | `sodindoorloc/hcxy/` | `train_HCXY.txt`, `test_HCXY.txt`, `mapxy_HCXY.txt`; 347 AP |
| TUJI | `tuji/` | `TUJI_training.csv`, `TUJI_testing.csv`; 310 AP |
| SYL | `sodindoorloc/syl/` | `SYL_Train.csv`, `SYL_Test.csv`; 363 AP |

Reader dùng chung: `src/ips/data_access/supplied.py`. Các bộ được giữ riêng
namespace AP và checkpoint; HCXY/SYL là hai phần của SODIndoorLoc. Pipeline
padding AP trống đến 520 để giữ backbone UJI, không tạo tín hiệu Wi-Fi giả.
CLI/Factory UJI giữ schema cũ; chạy các bộ mới qua
[hướng dẫn KAN](../src/ips/algorithms/kan/README.md).

DLULIB dùng các giả định đã được người dùng cho phép: −100 là không phát hiện,
X/Y theo mét, AP1…AP17 cùng ánh xạ vật lý giữa tầng; chưa thay thế xác nhận từ
chủ dữ liệu. TUJI remap ý nghĩa các cột metadata; không dùng device ID làm feature.

Đường dẫn cũ→mới và SHA-256 ở bản audit cục bộ
`analysis/2026-10-14/relocation_manifest.json`. Hai CSV UJI ở đây là bản sao bộ
operational `data/raw/ujiindoorloc`, không hợp nhất để tăng số mẫu.
