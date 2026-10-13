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
    kan/                       toàn bộ phần KAN để review
      algorithm.py             KANMixerAlgorithm kế thừa trực tiếp Algorithm
      dataset.py               RSSI/NumPy → tensor PyTorch
      models/                  mạng KAN và efficient-kan vendor
      train.py / predict.py    huấn luyện, suy luận, đánh giá checkpoint
      runtime.py / resources.py / diagnostics.py / benchmark.py
      configs/                 model.json, experiment_smoke.json
      tests/                   tests mạng, tensor, checkpoint KAN
      README.md                hướng dẫn review KAN
    llm/adapter.py             hợp đồng LLM trừu tượng
    huim/adapter.py            hợp đồng HUIM trừu tượng
  data_access/                 CSV, NumPy, nguồn/row index, split và audit
  experiments/                 config/result, factory, runner, evaluator, exporter
  data.py / metrics.py         schema/RSSI và metric IPS dùng chung
  cli.py / __main__.py         điểm vào lệnh
data/raw/ujiindoorloc/          CSV gốc
data/processed/                split, preprocessing, AP order, mapping, manifest
scripts/data/                  tải/audit/tạo/kiểm split dùng chung
scripts/setup_cuda.ps1         thiết lập môi trường GPU
tests/                         tests chung và discovery tests KAN
personal_learning/             baseline cá nhân; ignored
  archive/workspace/           tài liệu, config, reports, scripts cũ; giữ nguyên nội dung
docs/ / references/            tài liệu và bài báo cá nhân; ignored, không có trong clone mới
runs/                          checkpoint/log/output; ignored
```

Phần của thành viên KAN nằm trong [thư mục KAN](src/ips/algorithms/kan/README.md).
Review theo thứ tự algorithm.py → dataset.py → models/networks.py → train.py → predict.py.
Các alias backend tại đường dẫn cũ đã bỏ. Import Python dùng `ips.algorithms.kan.*`;
lệnh CLI và checkpoint KAN v1 vẫn được hỗ trợ. `data_access` chỉ trả NumPy/split,
không phụ thuộc mạng KAN hoặc tạo tensor.

## Cài đặt và chạy

Python repo đã dùng: 3.12. Các dependency được khóa trong requirements.txt và pyproject.toml.
Từ root repo, trên máy mới:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Nếu chưa có CSV: tải từ nguồn UCI qua script dữ liệu
.\.venv\Scripts\python.exe scripts/data/download_data.py
.\.venv\Scripts\python.exe -m ips prepare --verify-only
.\.venv\Scripts\python.exe -m ips algorithms
.\.venv\Scripts\python.exe -m ips check --device cpu
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Factory:

```powershell
.\.venv\Scripts\python.exe -u -m ips run --config src/ips/algorithms/kan/configs/experiment_smoke.json
# Terminal khác để theo dõi epoch
Get-Content runs/kan_folder_smoke/terminal.log -Tail 20 -Wait
```

Chạy lại phải đổi output_dir sang đường dẫn chưa tồn tại. Smoke một epoch trên subset256/128 chỉ kiểm tra pipeline.
Ví dụ JSON:

```json
{
  "algorithm_id": "kan_mixer",
  "params": {"model_config": "src/ips/algorithms/kan/configs/model.json"},
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

Bản mã MLP/MLP-Mixer riêng nằm trong personal_learning/legacy_models.py và được gọi trực tiếp từ phần cá nhân.
build_model của KAN chỉ tạo KAN, không tải baseline cá nhân. Clone mới không cần personal_learning/ để chạy KAN.

Tài liệu chặng 02/04/06/07/08, reports, course_materials, script chặng7/8, config baseline,
analysis.py và tests đánh giá cũ đã gom vào personal_learning/archive/workspace/ theo cây đường dẫn cũ.
191 file được kiểm tra SHA-256, nội dung giữ nguyên. Đây là bản lưu lịch sử; các script đó
không thuộc CLI hiện tại. Bản học cũ còn trong lịch sử Git tại commit 6bd94c4;
tái lập nguyên suite chặng7 dùng mã gốc 2ad74a0 cùng môi trường/dữ liệu/checkpoint tương ứng.
Không đổi số liệu báo cáo cũ để phù hợp cấu trúc mới.

## Hợp đồng và nguồn

Hợp đồng nhóm nằm trong [Algorithm](src/ips/algorithms/base.py),
[kiểu dữ liệu thực nghiệm](src/ips/experiments/contracts.py) và các adapter trong `src/ips/algorithms/`.
Xem [hướng dẫn KAN](src/ips/algorithms/kan/README.md) và [công cụ dữ liệu](scripts/data/README.md).
Toàn bộ `docs/` và `references/` được bỏ tracking, giữ nguyên trên máy; clone mới không cần chúng
để chạy pipeline. `prepare` chỉ tạo split nếu chưa có manifest, tránh đổi provenance/hash lịch sử.
Dataset vẫn giữ520 AP,13 cặp tòa/tầng, split16025/3912/1111 và preprocessing chỉ fit internal train.
Tọa độ theo quy ước mét của UJIIndoorLoc; chưa xác minh đầy đủ CRS. Dataset:
[UCI UJIIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc), CC BY4.0.
KAN vendor có commit/hash/license trong src/ips/algorithms/kan/models/vendor/provenance.json và LICENSE.efficient-kan.
Token hóa26×20 AP và hai head IPS là thiết kế ứng dụng RSSI của repo; không tuyên bố đã tái lập benchmark ảnh.
