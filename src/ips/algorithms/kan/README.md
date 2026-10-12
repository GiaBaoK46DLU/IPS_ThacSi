# Thư mục review KAN-Mixer

Toàn bộ triển khai KAN nằm tại đây. KANMixerAlgorithm kế thừa trực tiếp Algorithm ở ../base.py;
không có lớp trung gian NeuralAlgorithm. Mạng vẫn dùng efficient-kan B-spline.

```text
kan/
  algorithm.py                 vòng đời IPS: prepare → execute → predict
  paths.py                     đường dẫn config và ánh xạ tên config cũ
  dataset.py                   NumPy/RSSI → tensor PyTorch
  models/
    networks.py                token/channel mixing, chọn mạng cũ hoặc reference
    reference.py               điểm train và KAN-Mixer head tham chiếu
    refinement.py              wrapper lai fingerprint, catalogue train
    ensemble.py                wrapper kết hợp các mạng reference
    vendor/                    efficient_kan.py, provenance.json, license
  train.py                     loss, optimizer, checkpoint theo validation
  predict.py                   inference và evaluate checkpoint
  runtime.py                   seed, device, nạp checkpoint v1/v2
  accuracy.py                  chọn decoder thuần/lai bằng validation
  accuracy_ensemble.py         chọn ensemble bằng validation (hướng mở rộng)
  final_refit.py                refit development theo lựa chọn đã khóa
  freeze_refit.py               đóng gói refit, không chọn lại bằng validation
  resources.py                 RAM/VRAM và đồng bộ CUDA
  diagnostics.py               chẩn đoán input từng KANLinear
  benchmark.py                 thời gian/bộ nhớ suy luận
  configs/
    model.json                 kiến trúc/loss/training KAN
    experiment_smoke.json      config Factory: một epoch CPU
    accuracy_reference.json    head điểm tham chiếu, flatten
    accuracy_spatial.json      nhãn mềm và augmentation RSSI
    accuracy_mean_ablation.json reference head dùng mean pooling
  tests/test_pipeline.py       trục mixing, gradient, vendor hash, checkpoint
  tests/test_reference.py      catalogue, decoder, lai/ensemble, refit fixture
  README.md
```

Review theo thứ tự: algorithm.py → dataset.py → models/networks.py → train.py → predict.py.
Khi phân tích chi phí hoặc lỗi spline, xem resources.py, diagnostics.py và benchmark.py.
Nguồn vendor giữ nguyên byte/hash; không thay thuật toán hoặc key state_dict chỉ vì đổi folder.

Các phần dùng chung vẫn ở ngoài folder này: data_access/, experiments/, metrics.py, data.py,
algorithms/base.py, CLI và dữ liệu CSV/split. LLM/HUIM không cần dùng tensor hay training của KAN.
Các alias ở đường dẫn KAN cũ đã bỏ; import Python dùng ips.algorithms.kan.*.
DatasetRepository chỉ trả NumPy/split. Chuyển tensor được thực hiện trong folder KAN.
tests/test_pipeline.py ở root chỉ là điểm vào discovery, tests thực sự nằm trong folder này.

Từ root repo:

```powershell
.\.venv\Scripts\python.exe -u -m ips run --config src/ips/algorithms/kan/configs/experiment_smoke.json
.\.venv\Scripts\python.exe -m ips check --device cpu
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
# Chỉ tests mạng KAN
.\.venv\Scripts\python.exe -m unittest ips.algorithms.kan.tests.test_pipeline -v
```

Đổi output_dir nếu đã chạy config smoke. Kết quả vẫn ở runs/ và bị ignore, không nằm cùng source.
Lệnh train/predict/evaluate/benchmark cũ vẫn hoạt động; CLI ánh xạ hai đường dẫn config KAN cũ
sang configs/ bên trong folder này khi file cũ không tồn tại. Checkpoint KAN v1 vẫn nạp bằng state_dict.
Mã dùng chung không tải MLP/MLP-Mixer cá nhân; phần học đó gọi trực tiếp personal_learning/legacy_models.py.

## Biến thể điểm tham chiếu

Factory vẫn dùng ID `kan_mixer`; `positioning.variant=reference` trong model config
chọn mạng điểm tham chiếu. Thiếu trường này giữ mạng hồi quy cũ để đọc checkpoint lịch sử.
Backbone vẫn là efficient-kan B-spline; head và decoder RSSI là thiết kế IPS của dự án.

Chạy train và hiệu chỉnh suy luận (output-dir phải mới):

```powershell
.\.venv-cuda\Scripts\python.exe -u -m ips train --config src/ips/algorithms/kan/configs/accuracy_spatial.json --seed 44 --device cuda --run-dir runs/kan_spatial_new
.\.venv-cuda\Scripts\python.exe -m ips.algorithms.kan.accuracy --run-dir runs/kan_spatial_new --output-dir runs/kan_calibration_new --device cuda
```

Hoặc dùng Runner chung: `python -m ips run --config examples/kan_accuracy_validation.json`.
Ví dụ này chỉ đánh giá validation. Checkpoint và decoder được chọn bằng validation;
chốt lựa chọn trước khi đánh giá test. `pure_selected.pt` giữ mạng thuần;
`selected.pt` có thể là bản lai với catalogue fingerprint train.

Ensemble/refit có mã và tests kỹ thuật, chưa tạo số liệu trong báo cáo ba seed hiện tại.
Refit toàn development là giao thức riêng: validation cũ trở thành train, nên
CLI/Factory từ chối báo validation đó như held-out. Không dùng nó để chọn decoder lại.
FastKAN/S-KAN chưa được triển khai trong nhánh này.

Chi tiết từng file và giới hạn: [Giải thích thay đổi](GIAI_THICH_THAY_DOI_2026-10-09.md).

[Kiến trúc nhóm](../../../../docs/architecture/shared_factory.md) · [README dự án](../../../../README.md).
