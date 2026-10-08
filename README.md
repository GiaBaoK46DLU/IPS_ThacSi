# IPS dùng chung: KAN-Mixer, LLM và HUIM

Dự án Python/PyTorch trên UJIIndoorLoc cho nhóm ba người. Thuật toán đã chạy được trong dự án chung:
**KAN-Mixer dùng efficient-kan B-spline**. LLM/HUIM có adapter trừu tượng để nhóm phát triển tiếp,
chưa được đăng ký như thuật toán dự đoán IPS hoàn chỉnh. FastKAN/S-KAN chưa được triển khai.

MLP và MLP-Mixer là phần học/so sánh cá nhân, không hiển thị trong Factory hoặc CLI của dự án chung.

## Cấu trúc

```text
src/ips/
  algorithms/
    base.py                    Algorithm: hợp đồng IPS chung
    kan_mixer.py               KANMixerAlgorithm kế thừa trực tiếp Algorithm
    llm/adapter.py             hợp đồng LLM trừu tượng
    huim/adapter.py            hợp đồng HUIM trừu tượng
  data_access/                CSV, NumPy, nguồn/row index, split và audit
  experiments/                config/result, factory, runner, evaluator, exporter
  models/                     KAN-Mixer và efficient-kan vendor
  data.py / dataset.py        schema/RSSI và chuyển tensor cho KAN
  train.py / predict.py       backend KAN và lệnh trực tiếp
  metrics.py                  metric IPS dùng chung
  runtime.py / resources.py   seed, checkpoint v1, thiết bị và đo tài nguyên
  diagnostics.py              chẩn đoán KAN
  benchmark.py / cli.py       benchmark và điểm vào lệnh
configs/kan_mixer.json         cấu hình KAN: kiến trúc, loss, training
configs/experiments/           cấu hình Factory KAN
data/raw/ujiindoorloc/         CSV gốc
data/processed/                split, preprocessing, AP order, mapping, manifest
docs/architecture/            hợp đồng và sơ đồ nhóm
docs/lessons/chang_05/         chuẩn bị/audit dữ liệu còn cần cho dự án chung
scripts/setup_cuda.ps1         thiết lập môi trường GPU
tests/                        tests của phần dùng chung
personal_learning/            baseline và bản lưu cá nhân; ignored
runs/                         checkpoint/log/output mới; ignored
```

## Cài đặt và chạy

Python repo đã dùng: 3.12. Các dependency được khóa trong requirements.txt và pyproject.toml.
Từ root repo, trên máy mới:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Nếu chưa có CSV: tải từ nguồn UCI qua script dữ liệu
.\.venv\Scripts\python.exe docs/lessons/chang_05/download_data.py
.\.venv\Scripts\python.exe -m ips prepare --verify-only
.\.venv\Scripts\python.exe -m ips algorithms
.\.venv\Scripts\python.exe -m ips check --device cpu
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Factory:

```powershell
.\.venv\Scripts\python.exe -u -m ips run --config configs/experiments/kan_mixer_smoke.json
# Terminal khác để theo dõi epoch
Get-Content runs/shared_kan_mixer_smoke/terminal.log -Tail 20 -Wait
```

Chạy lại phải đổi output_dir sang đường dẫn chưa tồn tại. Smoke một epoch trên subset256/128 chỉ kiểm tra pipeline.
Ví dụ JSON:

```json
{
  "algorithm_id": "kan_mixer",
  "params": {"model_config": "configs/kan_mixer.json"},
  "output_dir": "runs/my_kan_smoke",
  "split_dir": "data/processed/uji_group_split_v1",
  "task": "position",
  "seed": 42,
  "device": "cpu",
  "smoke": true,
  "evaluation_splits": ["validation"]
}
```

Lệnh trực tiếp vẫn có cho KAN:

```powershell
.\.venv\Scripts\python.exe -u -m ips train --device cpu --smoke --run-dir runs/my_kan_direct
.\.venv\Scripts\python.exe -m ips predict --checkpoint runs/my_kan_direct/best.pt --csv examples/wifi_scans.csv --output runs/my_kan_direct/predictions.csv --device cpu
.\.venv\Scripts\python.exe -m ips evaluate --checkpoint runs/my_kan_direct/best.pt --split validation --output-dir runs/my_kan_direct/validation --device cpu
```

Training đầy đủ dùng smoke=false, params.epochs nếu cần và device=cuda với môi trường CUDA phù hợp.
Giữ chọn checkpoint bằng validation nội bộ; test chỉ đánh giá sau khi chốt, checkpoint smoke không được dùng test.
Factory có thể nạp checkpoint KAN v1 bằng params.checkpoint. Trọng số/state_dict KAN hiện có được giữ tương thích.

## Phần học cá nhân và lịch sử

Theo phạm vi đã thống nhất, baseline config, tài liệu học cũ, reports so sánh và script suite ba mô hình đã
bỏ tracking, **vẫn giữ trên máy**. .gitignore có danh sách cụ thể; Git không tự bỏ tracking file cũ chỉ bằng ignore.
Các nội dung đó không xuất hiện trong clone mới sau khi commit thay đổi này.

Bản mã MLP/MLP-Mixer riêng được lưu trong personal_learning/legacy_models.py; bản README trước thay đổi và
mã wrapper cũ cũng được lưu ở personal_learning/. build_model có cầu nối cục bộ cho baseline khi file này tồn tại,
nhưng chúng không được đăng ký trong Factory và không được quảng bá là chức năng dùng chung.
Clone mới không cần thư mục cá nhân để chạy KAN.

Một số file cũ vẫn ở vị trí cũ trên máy nhưng đã ignored, gồm docs/lessons/chang_02,04,06,07,08,
reports/, references/course_materials/, scripts chặng7/8, configs/base.json, src/ips/analysis.py
và tests/test_evaluation.py. Bản học cũ còn trong lịch sử Git tại commit 6bd94c4;
tái lập nguyên suite chặng7 dùng mã gốc 2ad74a0 cùng môi trường/dữ liệu/checkpoint tương ứng.
Không đổi số liệu báo cáo cũ để phù hợp cấu trúc mới.

## Hợp đồng và nguồn

Xem [kiến trúc nhóm](docs/architecture/shared_factory.md), [sơ đồ](docs/architecture/shared_factory.png)
và [kiểm chứng KAN-only](docs/architecture/kan_only_verification.md).
Dataset vẫn giữ520 AP,13 cặp tòa/tầng, split16025/3912/1111 và preprocessing chỉ fit internal train.
Tọa độ theo quy ước mét của UJIIndoorLoc; chưa xác minh đầy đủ CRS. Dataset:
[UCI UJIIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc), CC BY4.0.
KAN vendor có commit/hash/license trong src/ips/models/vendor/provenance.json và LICENSE.efficient-kan.
Token hóa26×20 AP và hai head IPS là thiết kế ứng dụng RSSI của repo; không tuyên bố đã tái lập benchmark ảnh.
