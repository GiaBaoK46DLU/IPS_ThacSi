# Bảng mẫu trước khi có số đo

`—` nghĩa là **chưa chạy/chưa có số đo**, không phải0. Không lấy số từ ví dụ giảng dạy, smoke, pilot hoặc bài báo khác để điền vào đây. Report chính thức tự sinh từ9 lượt đã kiểm tra.

| Mô hình | Split | Seed | Building acc | Building macro-F1 | Floor acc | Floor macro-F1 | Joint acc | Joint macro-F1 |
|---|---|---|---|---|---|---|---|---|
| MLP | test | mean±SD42/43/44 | — | — | — | — | — | — |
| MLP-Mixer | test | mean±SD42/43/44 | — | — | — | — | — | — |
| KAN-Mixer | test | mean±SD42/43/44 | — | — | — | — | — | — |

| Mô hình | MAE x/y (m) | RMSE x/y (m) | Mean2D (m) | Median2D (m) | P90 (m) | RMSE2D (m) | Đúng tòa/tầng và lỗi≤8m |
|---|---|---|---|---|---|---|---|
| MLP | — | — | — | — | — | — | — |
| MLP-Mixer | — | — | — | — | — | — | — |
| KAN-Mixer | — | — | — | — | — | — | — |

| Mô hình | Epoch/steps thực | Train+val (s) | RAM train peak (MiB) | VRAM allocated/reserved (MiB) | B1 end-to-end p50/p90 (ms) | B64 throughput (mẫu/s) | RAM/VRAM suy luận |
|---|---|---|---|---|---|---|---|
| MLP | — | — | — | — | — | — | — |
| MLP-Mixer | — | — | — | — | — | — | — |
| KAN-Mixer | — | — | — | — | — | — | — |

Khi có kết quả: ghi hash split/checkpoint, config, device, precision, seed, đơn vị, scope timing/memory và support của các nhóm. SD mẫu dùng ddof=1; một seed thì SD không xác định.
