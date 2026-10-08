# Kiểm chứng refactor Factory

Đã thực hiện ngày 08/10/2026 trên nhánh `feature/shared-ips-factory`, từ
`main@2ad74a0fd2f925966a1ca18f121a8a8701526870`. Workspace ban đầu sạch, không tìm thấy AGENTS.md.
Các kiểm tra dưới đây được thực hiện trước khi commit/push/merge; main lúc kiểm chứng giữ nguyên commit gốc.

Môi trường thực tế: Windows, Python3.12.1, `.venv` với PyTorch2.9.0+cpu, NumPy2.2.6,
scikit-learn1.6.1, bốn thread neural. Không chạy suite GPU dài.

## Tests và CLI

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kết quả sau các chỉnh sửa mã cuối: **27 tests PASS**, gồm 15 test cũ và 12 test mới (14,688 giây).
Test mới bảo đảm default registry chỉ có ba neural backends; từ chối adapter/nn.Module/ID thiếu;
hợp đồng IPS/source/order; metric đầy đủ bằng legacy và metric riêng không giả lập output thiếu;
train-only coordinate fit; AP/nhãn/tensor/subset/indices gốc; non-neural execution không optimizer;
test sau finalization; export và không ghi đè output; CLI cũ và fixture checkpoint v1.

Các lệnh cũ đã chạy thật, ngoài tests mock dispatch:

```powershell
.\.venv\Scripts\python.exe -u -m ips train --model mlp --device cpu --smoke --run-dir runs/factory_refactor/legacy_cli_train
.\.venv\Scripts\python.exe -m ips predict --checkpoint runs/stage7_cuda_v1/kan_mixer_seed42/best.pt --csv examples/wifi_scans.csv --output runs/factory_refactor/legacy_predictions.csv --device cpu
.\.venv\Scripts\python.exe -m ips evaluate --checkpoint runs/stage7_cuda_v1/mlp_seed42/best.pt --split validation --output-dir runs/factory_refactor/legacy_validation --device cpu
```

Train: một epoch hoàn tất. Predict: xuất 5 mẫu. Evaluate: đủ 3.912 validation nội bộ,
mean2D khoảng13,551 m, joint accuracy≈0,9831. Đây là kiểm tra tương thích checkpoint lịch sử,
không phải kết quả thực nghiệm mới để nhập vào reports. Lệnh `check --device cpu` được chạy thật trong test;
forward/parameter counts của cả ba mạng đều PASS. `ips algorithms` liệt kê đúng ba ID và báo rõ các nhánh chưa triển khai.

## Smoke CPU qua Factory

Đã chạy cả ba config trong configs/experiments:

```powershell
.\.venv\Scripts\python.exe -u -m ips run --config configs/experiments/kan_mixer_smoke.json
.\.venv\Scripts\python.exe -u -m ips run --config configs/experiments/mlp_smoke.json
.\.venv\Scripts\python.exe -u -m ips run --config configs/experiments/mlp_mixer_smoke.json
```

Mỗi lượt một epoch, cùng subset train256/validation128, seed42; không đánh giá test.
Các số dưới đây được đọc từ run_result.json/resources.json thực tế của ba lượt đầu, không phải ngoại suy full dataset:

| Mô hình | Thời gian Runner (giây) | Vòng train+validation (giây) | Đỉnh RSS lấy mẫu trong pha train (MiB) |
|---|---:|---:|---:|
| MLP | 6,939 | 0,077 | 343,15 |
| MLP-Mixer | 6,931 | 0,183 | 368,62 |
| KAN-Mixer, efficient-kan | 11,557 | 2,944 | 767,04 |

Runner bao gồm audit dữ liệu/chuẩn bị/thực thi/dự đoán/metric/export, không gồm khởi động Python/import CLI.
RSS là toàn tiến trình, không phải bộ nhớ riêng trọng số; lấy mẫu10ms có thể bỏ lỡ đỉnh ngắn. CPU không có số đo VRAM.
Ngân sách một epoch trên subset không dùng để so sánh chất lượng hoặc dự báo 100epoch trên full dataset.

Sau khi bổ sung provenance/hash mã và ghi rõ phạm vi thời gian backend, đã chạy lại KAN một lượt và MLP một lượt
ở `runs/factory_refactor/final_kan_smoke`, `final_mlp_smoke`. MLP cuối hoàn tất≈6,972 giây;
tất cả đường dẫn artifact tồn tại, provenance ghi hash 33 tệp Python. Các artifact có CSV predictions,
metrics, matrices JSON/CSV, PNG confusion/CDF, log epoch, history/resources và sơ đồ model .mmd.
Các output trong runs/ bị Git ignore, không thay reports/.

## Checkpoint v1 thật và fixture

Máy này **có checkpoint thật**, dù runs/ bị ignore. Đã nạp qua Algorithm adapter và trực tiếp qua loader legacy
trên cùng26 validation samples, CPU, cả ba checkpoint seed42:

| Checkpoint | Epoch được chọn | Kết quả đối chiếu |
|---|---:|---|
| runs/stage7_cuda_v1/mlp_seed42/best.pt | 21 | class và tọa độ khớp hoàn toàn |
| runs/stage7_cuda_v1/mlp_mixer_seed42/best.pt | 37 | class và tọa độ khớp hoàn toàn |
| runs/stage7_cuda_v1/kan_mixer_seed42/best.pt | 54 | class và tọa độ khớp hoàn toàn |

Bằng chứng máy hiện tại: `runs/factory_refactor/real_v1_verification.json`.
Tests có fixture v1 nên checkout mới không cần các checkpoint ignored vẫn chạy được tests tương thích.
Không đổi format hoặc state_dict mạng để làm phép kiểm tra này.

## Audit lịch sử và Git

Đã gọi `audit_suite(Path('runs/stage7_cuda_v1'))` chỉ đọc. Kết quả **từ chối đúng dự kiến**:
`Training implementation changed since protocol lock: src/ips/dataset.py`.
Đây là kiểm tra provenance lịch sử đang hoạt động, không phải lỗi loader v1; không nới rule và không tạo lại reports cũ.
Để tái audit suite dùng commit gốc và môi trường/dữ liệu/checkpoint tương ứng, như tài liệu kiến trúc giải thích.

`git diff --name-only -- reports data scripts src/ips/metrics.py src/ips/runtime.py src/ips/models configs/base.json`
không có output; các phần đó giữ nguyên. `git -c core.whitespace=cr-at-eol diff --check` PASS.
Sơ đồ Mermaid đã được validate/render bằng mmdc11.15.0, xuất SVG/PNG và kiểm tra trực quan.

Phạm vi chưa hoàn thành có chủ ý: LLM RSSI→vị trí, HUIM itemset→vị trí, FastKAN và S-KAN.
Các quyết định phương pháp và phân công nằm ở [hướng dẫn kiến trúc](shared_factory.md).
