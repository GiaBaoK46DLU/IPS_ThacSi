# Thư mục review KAN-Mixer

Toàn bộ triển khai KAN nằm tại đây. KANMixerAlgorithm kế thừa trực tiếp Algorithm ở ../base.py;
không có lớp trung gian NeuralAlgorithm. Mạng vẫn dùng efficient-kan B-spline.

```text
kan/
  algorithm.py                 vòng đời IPS: prepare → execute → predict
  paths.py                     đường dẫn config và ánh xạ tên config cũ
  dataset.py                   NumPy/RSSI → tensor PyTorch
  models/
    networks.py                embedding, token/channel mixing, hai head IPS
    vendor/                    efficient_kan.py, provenance.json, license
  train.py                     loss, optimizer, checkpoint theo validation
  predict.py                   inference và evaluate checkpoint
  runtime.py                   seed, device, nạp checkpoint v1
  resources.py                 RAM/VRAM và đồng bộ CUDA
  diagnostics.py               chẩn đoán input từng KANLinear
  benchmark.py                 thời gian/bộ nhớ suy luận
  configs/
    model.json                 kiến trúc/loss/training KAN
    experiment_smoke.json      config Factory: một epoch CPU
  tests/test_pipeline.py       trục mixing, gradient, vendor hash, checkpoint
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

[Kiến trúc nhóm](../../../../docs/architecture/shared_factory.md) · [README dự án](../../../../README.md).
