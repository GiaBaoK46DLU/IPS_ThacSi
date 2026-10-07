# Định vị Wi-Fi với MLP, MLP-Mixer và RSSI-KAN-Mixer

Dự án học và thực nghiệm PyTorch trên UJIIndoorLoc. Mỗi mô hình nhận RSSI/mask của 520 AP và dự đoán 13 tổ hợp tòa–tầng cùng tọa độ 2D. Bắt đầu với [bài thực hành chặng 6](docs/lessons/chang_06/README.md), gồm thiết kế, giải thích mã theo từng khối, lệnh chạy, kiểm tra đầu ra và bài tập.

## Cấu trúc

```text
configs/                 cấu hình mô hình và huấn luyện dùng chung
src/ips/                 đọc dữ liệu, mô hình, train, predict, evaluate, CLI
tests/                   kiểm tra trục mixing, gradient, inverse, checkpoint
scripts/                 chạy ba mô hình trên cùng seed và ngân sách
examples/                CSV chỉ có RSSI để thử dự đoán
data/raw/ujiindoorloc/    hai CSV gốc, giữ nguyên byte
data/processed/          indices, AP order, class mapping, preprocessing, manifest
docs/lessons/            tài liệu và thực hành các chặng
references/              DOCX đầu vào và bài báo PDF
runs/                    checkpoint, history, diagnostics và dự đoán từng lượt
.venv/                   môi trường riêng của dự án
```

Repo đã được sắp xếp lại trong chặng 6. Tài liệu cũ nằm trong `docs/lessons/`; PDF ở `references/papers/`, DOCX ở `references/course_materials/`. Nguồn dữ liệu và thành viên split giữ nguyên; xem [báo cáo di chuyển](docs/lessons/chang_05/artifacts/relocation_report.json).

## Cài đặt và kiểm tra

Python đã kiểm chứng: 3.12.1. Các phiên bản chính: NumPy 2.2.6, scikit-learn 1.6.1, PyTorch 2.9.0+cpu. `.venv` đã được tạo ở máy hiện tại. Trên máy mới, chạy PowerShell tại repo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONUTF8 = '1'
.\.venv\Scripts\python.exe -m ips prepare --verify-only
.\.venv\Scripts\python.exe -m ips check
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Lệnh CPU dựa trên [hướng dẫn PyTorch cho 2.9.0](https://pytorch.org/get-started/previous-versions/#v290). Chỉ cần torch; không cần torchvision/torchaudio cho RSSI. Để tái tạo cả các phiên bản phụ thuộc đã cài trên CPU/Windows/Python 3.12, dùng `pip install -r requirements-lock-cpu.txt` thay cho hai lệnh pip trên. Bản snapshot này không dành cho CUDA.

Nếu chưa có raw, chạy `python docs/lessons/chang_05/download_data.py` bằng môi trường dự án, rồi `python -m ips prepare`. Nguồn: [UCI UJIIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc). Dataset được cấp phép CC BY 4.0; cần ghi công khi sử dụng.

## Chạy theo thứ tự

Thử baseline trước:

```powershell
.\.venv\Scripts\python.exe -m ips train --model mlp --seed 42 --device cpu --smoke --run-dir runs/my_first_mlp
.\.venv\Scripts\python.exe -m ips predict --checkpoint runs/my_first_mlp/best.pt --csv examples/wifi_scans.csv --output runs/my_first_mlp/predictions.csv --device cpu
```

Thay `--model` bằng `mlp_mixer` hoặc `kan_mixer`. `--smoke` dùng 256 train/128 validation, chọn cố định theo lớp, chạy một epoch và đánh dấu checkpoint là kiểm tra pipeline. Không dùng để kết luận mô hình nào tốt hơn. Các lượt đã kiểm chứng ở `runs/check_stage6/`; baseline hai epoch trên toàn train ở `runs/baseline_stage6/`.

Huấn luyện theo giao thức tối thiểu đã chốt:

```powershell
.\.venv\Scripts\python.exe -m ips train --model mlp --seed 42 --device cpu --run-dir runs/mlp_seed42_full
.\.venv\Scripts\python.exe -m ips evaluate --checkpoint runs/mlp_seed42_full/best.pt --split validation --output-dir runs/mlp_seed42_full/validation --device cpu
```

Mặc định tối đa 100 epoch, early stopping patience15, chọn checkpoint theo lỗi 2D validation nội bộ. `--epochs` thay ngân sách cho một lượt thử. [configs/base.json](configs/base.json) giữ cùng tiền xử lý, optimizer, loss, batch và quy tắc chọn checkpoint. Chạy ba mô hình × ba seed bằng:

```powershell
.\scripts\run_experiment.ps1 -OutputRoot runs/experiment_v1 -Device cpu
```

Mỗi run/output phải dùng đường dẫn mới để tránh trộn kết quả. Sau khi đã chốt mô hình và giao thức, `evaluate --split test` giữ nguyên 1.111 mẫu test gốc. Vòng huấn luyện không đánh giá test.

## Nguồn và phạm vi kiến trúc

- [MLP-Mixer, Tolstikhin và cộng sự](https://arxiv.org/abs/2105.01601): cơ chế token-mixing và channel-mixing.
- [KAN-Mixers, Canuto và cộng sự, v1](https://arxiv.org/abs/2503.08939v1): dùng KAN ở cả hai nhánh, nghiên cứu phân loại ảnh.
- [efficient-kan khóa tại commit 7b6ce1c…](https://github.com/Blealtan/efficient-kan/tree/7b6ce1c87f18c8bc90c208f6b494042344216b11): giữ nguyên source và giấy phép MIT trong `src/ips/models/vendor/`; có hash đối chiếu.

Token hóa 26×20 AP, hai head IPS và các siêu tham số là đề xuất từ chặng 4, không phải tái lập nguyên xi thí nghiệm ảnh. Số tham số thực tế: MLP142.607, MLP-Mixer40.835, KAN-Mixer365.199; giao thức tối thiểu chưa khớp ngân sách tham số. Không suy ra kiến trúc tốt hơn chỉ từ một lượt ngắn.

Split hiện có train16.025 / validation3.912 / test1.111. Nhóm train–validation không giao nhau nhưng người, điểm và một số input RSSI vẫn chung; đọc [giới hạn ở chặng 5](docs/lessons/chang_05/README.md). Tọa độ giữ quy ước mét của dataset, chưa xác minh đầy đủ CRS/EPSG.
