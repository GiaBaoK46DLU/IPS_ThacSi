# Đo thời gian và bộ nhớ trên RTX 4060 Ti

Đây là **đo thử 3 epoch trên toàn train/validation, seed42**, không phải kết quả so sánh chất lượng cuối cùng của chặng 7. Chưa đánh giá tập test. Các tệp trong thư mục là bản sao nguyên byte từ `runs/gpu_calibration_v1`; không bao gồm checkpoint.

Môi trường thực tế: RTX 4060 Ti 8GB, driver591.86, PyTorch2.9.0+cu128, Python3.12.1; CPU i5-13400F. Batch64, FP32, 4 luồng CPU, cấu hình trong `resolved_config.json`, cùng split train16.025/validation3.912. Cả ba lượt đã hoàn thành. `run_metadata.json` ghi hash mã nguồn và split tại lúc đo.

| Mô hình | Trung bình epoch 2–3 (giây) | Ước tính 100 epoch × 3 seed (phút) | VRAM peak allocated (MiB) | VRAM peak reserved (MiB) |
|---|---:|---:|---:|---:|
| MLP | 2,1654 | 10,83 | 67,99 | 70 |
| MLP-Mixer | 4,4223 | 22,11 | 76,92 | 100 |
| KAN-Mixer | 13,4653 | 67,33 | 414,54 | 466 |

Tổng ngoại suy: **100,27 phút** nếu cả 9 lượt đều chạy đủ100 epoch. Dự trù khoảng **1 giờ40 phút–2 giờ**; đây là ước tính từ hai epoch ổn định hơn, không phải khoảng tin cậy thống kê. Early stopping có thể làm lượt chính thức ngắn hơn. Không chọn siêu tham số hay kết luận về độ chính xác từ lần đo này.

Thời gian mỗi epoch gồm train, validation, diagnostics và ghi checkpoint; đã đồng bộ CUDA trước/sau khoảng đo. Chưa tính khởi tạo tiến trình, tải dữ liệu, đánh giá test hay tạo báo cáo cuối. VRAM là allocated/reserved của allocator PyTorch trong vòng huấn luyện; không phải tổng GPU của Windows và các ứng dụng khác. RAM trong `resources.json` là RSS toàn tiến trình, lấy mẫu10ms, có thể bỏ lỡ đỉnh rất ngắn.

## Kiểm tra bằng chứng và chạy lại

Đọc [timing_estimate.json](timing_estimate.json), rồi đối chiếu `history.csv`, `summary.json`, `resources.json` và `run_metadata.json` của mỗi mô hình. Các chỉ số validation trong log là số đo thật ở ba epoch đầu, không được dùng như bảng kết quả nghiên cứu cuối cùng.

```powershell
.\scripts\setup_cuda.ps1
.\scripts\profile_gpu.ps1 -OutputRoot runs/gpu_calibration_repeat
```

Script `scripts/summarize_gpu_profile.py` tính lại các ước tính từ log, kiểm tra ba lượt CUDA hoàn thành đủ ba epoch và dùng đúng số mẫu. Dùng thư mục mới khi chạy lại. Môi trường khóa tại [requirements-lock-cuda.txt](../../requirements-lock-cuda.txt); [hướng dẫn PyTorch2.9.0](https://pytorch.org/get-started/previous-versions/#v290) là nguồn lệnh cài CUDA12.8.
