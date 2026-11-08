# Kiểm chứng việc gom thư mục KAN

Thay đổi trên nhánh `feature/kan-only-algorithm`, từ commit `d1db4c1`.
Triển khai KAN, cấu hình, vendor và tests riêng nằm trong `src/ips/algorithms/kan/`.
`KANMixerAlgorithm` vẫn kế thừa trực tiếp `Algorithm`. Không thay kiến trúc mạng,
key state_dict, loss, metric, split hoặc quy tắc chọn checkpoint.

Sau yêu cầu dọn dự án, các alias backend ở vị trí cũ đã bỏ. Import Python dùng
`ips.algorithms.kan.*`. `Bundle` chỉ cung cấp NumPy/split; không còn `dataset()`
phụ thuộc KAN. `build_model` của KAN không tải baseline cá nhân.
CLI vẫn hỗ trợ train/predict/evaluate/check/benchmark. Hai tên config cũ được
ánh xạ khi file cũ không còn. Config không nhận diện vẫn báo lỗi, không tự dùng mặc định.

## Kiểm tra đã chạy

Môi trường: Python 3.12.1, PyTorch 2.9.0+cpu trong `.venv`.

| Kiểm tra | Kết quả |
|---|---|
| `python -m unittest discover -s tests -v` trên máy hiện tại | 27 tests đạt; 7 tests lịch sử được giữ trong archive cá nhân |
| Cùng lệnh trên bản xuất chỉ gồm file dùng chung | 27 tests đạt |
| `python -m ips algorithms` trên bản xuất | Chỉ đăng ký `kan_mixer`; nhánh chưa triển khai được thông báo rõ |
| `python -m ips check --device cpu` | Tensor/forward/data integrity đạt; 365199 tham số |
| `python -m ips check --config configs/kan_mixer.json --device cpu` | Tên config cũ hoạt động |
| `python -u -m ips run --config src/ips/algorithms/kan/configs/experiment_smoke.json` trên bản xuất | Một epoch CPU hoàn tất: train 256, validation 128; không đánh giá test |
| `python -u -m ips train --device cpu --smoke --run-dir runs/shared_layout_cleanup/direct` | Lệnh train trực tiếp hoàn tất một epoch CPU |
| `python -m ips predict --checkpoint runs/shared_layout_cleanup/direct/best.pt --csv examples/wifi_scans.csv --output runs/shared_layout_cleanup/direct/predictions.csv --device cpu` | Xuất 5 dự đoán |
| `python -m ips evaluate --checkpoint runs/shared_layout_cleanup/direct/best.pt --split validation --output-dir runs/shared_layout_cleanup/direct/validation --device cpu` | Đánh giá 3912 mẫu validation nội bộ; không chạy test |
| Checkpoint KAN v1 thực tế `runs/stage7_cuda_v1/kan_mixer_seed42/best.pt` | Nạp epoch 54; nhãn và tọa độ giống hệt mã mạng cũ trên 26 mẫu validation |
| Wheel xây từ bản xuất dùng chung | Có configs, vendor provenance/license; không chứa alias/backend ở vị trí cũ hoặc `ips/analysis.py` cá nhân |
| Manifest 191 file cá nhân/lịch sử | Đã chuyển vào personal_learning/archive/workspace/, SHA-256 giữ nguyên |
| MLP/MLP-Mixer cá nhân qua legacy_models.py | Forward hai mẫu đạt; chỉ import vendor được cập nhật sang đường dẫn mới |

Smoke Factory trên workspace sau dọn mất khoảng 8,67 giây theo `RunResult.runtime.total_seconds`, gồm audit dữ liệu,
chuẩn bị, thực thi, dự đoán, đánh giá và xuất artifact; không gồm khởi động Python.
Đây chỉ là kiểm tra pipeline, không phải kết quả định vị của mô hình huấn luyện đầy đủ.

Bằng chứng cục bộ, ignored:

- `runs/kan_folder_smoke/`: terminal.log, checkpoint, predictions, metrics và run_result.json.
- `runs/shared_layout_cleanup/direct/`: lệnh train/predict/evaluate trực tiếp.
- `runs/shared_layout_cleanup/shared_export/`: bản chỉ có file dùng chung, tests/check/smoke đều đạt.
- `runs/shared_layout_cleanup/real_kan_v1.json`: kiểm tra checkpoint thực tế.
- `runs/shared_layout_cleanup/wheel/`: wheel sau dọn.

Nội dung cá nhân không còn trong src/ hoặc tests/ dùng chung. Manifest đường dẫn lưu trữ
nằm trong personal_learning/archived_files_manifest.json. Các script lịch sử cần commit
gốc để tái lập đúng giao thức; không thuộc CLI mới.

`main` giữ nguyên tại `6bd94c4d74d4a2239e882a9890738583eacb89d4`.
Không sửa CSV/split/preprocessing, `metrics.py`, báo cáo cũ hoặc 191 file cá nhân.
