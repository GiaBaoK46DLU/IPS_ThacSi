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
  supplied_experiment.py        chuyển cấu hình UJI sang DLULIB/HCXY/TUJI/SYL
  supplied_figures.py           xuất bảng và hình từ kết quả thật trên các bộ mới
  verify_supplied.py            đối chiếu checkpoint, catalogue, row và metric sau chạy
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

## Thực nghiệm trên DLULIB và HCXY

Lối chạy nghiên cứu riêng nằm trong `supplied_experiment.py`, tái sử dụng nguyên mạng,
loss và vòng train hiện tại. Reader NumPy dùng chung ở `ips/data_access/supplied.py`.
Các lệnh Factory/UJI cũ vẫn giữ schema cũ; không đổi tên CSV của bộ khác thành UJI.

Giữ 26×40 bằng cách padding AP trống đến520. Chỉ fit transform tọa độ và catalogue
trên train; train/validation tách theo vị trí. DLULIB dùng All_20, không gộp Mean.
HCXY có một tòa/tầng nên chỉ số classification mang tính mô tả. Các giả định của
DLULIB về −100, đơn vị mét và AP order giữa tầng được ghi trong manifest.

```powershell
# Chọn thư mục mới cho mỗi lượt; cần raw trong dataset/raw/.
./.venv-cuda/Scripts/python.exe -m ips.algorithms.kan.supplied_experiment prepare --output-dir runs/kan_supplied_new
./.venv-cuda/Scripts/python.exe -u -m ips.algorithms.kan.supplied_experiment train --output-dir runs/kan_supplied_new --device cuda
./.venv-cuda/Scripts/python.exe -u -m ips.algorithms.kan.supplied_experiment evaluate --output-dir runs/kan_supplied_new --device cuda
./.venv-cuda/Scripts/python.exe -m ips.algorithms.kan.supplied_figures --output-dir runs/kan_supplied_new
./.venv-cuda/Scripts/python.exe -m ips.algorithms.kan.verify_supplied --output-dir runs/kan_supplied_new
```

Ba seed42/43/44 dùng cấu hình training UJI đã xác minh. Decoder và hệ số lai giữ
đúng cấu hình từng seed UJI, không calibration mới. Hiện source_settings đọc
các run/calibration UJI gốc local; prepare chép cấu hình/provenance vào output
để review. Thiếu source run phải khôi phục snapshot cấu hình đã kiểm chứng,
không suy đoán decoder. Kết quả mới nằm dưới output/{dlulib,hcxy}/; train metrics
chỉ chẩn đoán vì catalogue lai có chứa chính truy vấn train.

Các phần dùng chung vẫn ở ngoài folder này: data_access/, experiments/, metrics.py, data.py,
algorithms/base.py, CLI và dữ liệu CSV/split. LLM/HUIM không cần dùng tensor hay training của KAN.
Các alias ở đường dẫn KAN cũ đã bỏ; import Python dùng ips.algorithms.kan.*.
DatasetRepository chỉ trả NumPy/split. Chuyển tensor được thực hiện trong folder KAN.

Adapter supplied hiện hỗ trợ thêm `tuji` và `syl` (phần SYL của SODIndoorLoc).
TUJI dùng đúng mapping native: FLOOR→z, BUILDINGID→floor, SPACEID→building,
RELATIVEPOSITION→device. Metadata thiết bị chỉ theo dòng nguồn để phân tích,
không đi vào forward. SYL và HCXY giữ namespace AP/site riêng; không nối MAC
cùng tên giữa hai site. `RP` trong TUJI/SYL là ID vị trí dẫn xuất theo từng role.

Ví dụ chạy tiếp, dùng output mới để giữ split và checkpoint đã khóa:

```powershell
.\.venv-cuda\Scripts\python.exe -m ips.algorithms.kan.supplied_experiment prepare --datasets tuji syl --output-dir runs/kan_tuji_sod_new
.\.venv-cuda\Scripts\python.exe -u -m ips.algorithms.kan.supplied_experiment train --datasets tuji syl --output-dir runs/kan_tuji_sod_new --device cuda
.\.venv-cuda\Scripts\python.exe -m ips.algorithms.kan.supplied_experiment evaluate --datasets tuji syl --output-dir runs/kan_tuji_sod_new --device cuda
.\.venv-cuda\Scripts\python.exe -m ips.algorithms.kan.supplied_figures --datasets tuji syl --output-dir runs/kan_tuji_sod_new
.\.venv-cuda\Scripts\python.exe -m ips.algorithms.kan.verify_supplied --datasets tuji syl --output-dir runs/kan_tuji_sod_new
```

TUJI/SYL chỉ một building/floor trong mỗi phần; confusion matrices một ô là
mô tả, không chứng minh khả năng phân loại nhiều tòa/tầng. Các bảng thiết bị
và số lần quét không phải đơn vị thống kê độc lập khi có fingerprint lặp.
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

[Hợp đồng Algorithm](../base.py) · [README dự án](../../../../README.md).
