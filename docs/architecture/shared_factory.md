# Kiến trúc IPS dùng chung cho nhóm ba người

Các thuật toán chạy được: `kan_mixer` (**efficient-kan B-spline**), `mlp`, `mlp_mixer`.
LLM và HUIM mới có adapter trừu tượng, chưa có bộ chuyển RSSI thành vị trí và chưa đăng ký.
FastKAN/S-KAN trong hình tham khảo chưa có trong repo.

![Sơ đồ phụ thuộc](shared_factory.png)

[Nguồn Mermaid](shared_factory.mmd) · [SVG phóng to](shared_factory.svg).

## Luồng gọi

Trước refactor: CLI → `dataset.load_bundle()` đọc/audit CSV và split → `FingerprintDataset` tạo tensor
→ `build_model()` → `train()` hoặc `load_checkpoint(v1)` → metric/export riêng.

CLI cũ vẫn chạy pipeline neural. CLI mới `ips run` đi qua:

1. `ExperimentConfig` → `AlgorithmFactory.create()` chọn lớp theo registry.
2. `DatasetRepository.load()` audit hash/schema/split; `Bundle.split()` trả NumPy.
3. `Algorithm.prepare(train)` fit biểu diễn riêng, **chỉ train**.
4. `Algorithm.execute(validation)` train, khai phá hoặc chốt pipeline suy luận; validation chỉ chọn trạng thái.
5. `Algorithm.predict(FingerprintInputs)` trả `IPSPredictions` đúng nguồn/thứ tự, không được nhận nhãn vị trí.
6. `IPSEvaluator` tính metric; `ReportExporter` xuất artifact; Runner trả `RunResult`.

`Algorithm` không kế thừa `torch.nn.Module`, không yêu cầu optimizer/epoch/loss. Nhánh neural chứa những
chi tiết đó và tái sử dụng `train.py`, `infer()`, `build_model()` và checkpoint v1. Factory từ chối abstract
adapter, nn.Module, ID trùng và ID chưa có. Chuẩn bị biểu diễn thuộc Algorithm; Repository chỉ quản lý dữ liệu/split.

## Trách nhiệm và cộng tác

```text
configs/base.json                   cấu hình neural cũ, giữ nguyên
configs/experiments/*_smoke.json     cấu hình Runner, CPU một epoch
src/ips/
  data.py                          parser/schema/RSSI transform cũ
  data_access/
    contracts.py                   FingerprintInputs / FingerprintSplit / DatasetInfo
    repository.py                  CSV, hash, split, audit; không import torch
  dataset.py                       adapter NumPy → tensor; API cũ vẫn tồn tại
  algorithms/
    base.py                        Algorithm ABC
    neural.py                      KANMixerAlgorithm / MLPAlgorithm / MLPMixerAlgorithm
    llm/adapter.py                 LLMAlgorithm trừu tượng
    huim/adapter.py                HUIMAlgorithm trừu tượng
  experiments/
    contracts.py                   ExperimentConfig / IPSPredictions / AlgorithmExecution / RunResult
    factory.py                     registry cấp IPS, khác build_model cấp neural
    runner.py                      vòng đời, thời gian và terminal log
    evaluator.py                   metric theo đầu ra thực có
    exporter.py                    CSV/JSON/confusion/CDF/diagram/history/resources
  models/                          mạng neural/vendor cũ
  train.py / predict.py / cli.py    CLI cũ + Factory mới
tests/test_factory.py              contract/split/extension/v1/CLI
docs/architecture/                 hướng dẫn và sơ đồ
reports/                           kết quả lịch sử, không ghi lại từ Runner
runs/<output_dir>/                 artifact riêng; bị Git ignore
```

Thành viên KAN phụ trách `algorithms/neural.py` và `models/`; LLM phụ trách `algorithms/llm/`;
HUIM phụ trách `algorithms/huim/`. Đổi hợp đồng chung/split cần cả nhóm review. Mỗi người dùng nhánh và
output riêng. Chạy các lượt ở tiến trình riêng: tee stdout và RNG neural là trạng thái toàn tiến trình,
Runner không hỗ trợ chạy đồng thời trong nhiều thread của một tiến trình.

## Dữ liệu và rò rỉ

- RSSI nguyên `[N,520]`, đúng WAP001…WAP520, miền `−104..0` hoặc `100` = không quan sát AP.
  Hợp đồng chung giữ raw missing; neural dùng `(z,mask)` cũ, các nhánh khác tự chọn biểu diễn.
- Row index bắt đầu từ 0 trong **CSV nguồn**, không đánh lại theo split. `source_id` là đường dẫn resolve;
  cùng row index ở hai CSV không phải cùng mẫu. Dòng CSV = row index + 2. Array dùng chung được sao chép/chỉ đọc.
- `FingerprintSplit` có nhãn joint class, tọa độ gốc float64 và metadata để học/đánh giá;
  `FingerprintInputs` cho predict chỉ có RSSI, AP order và danh tính mẫu, không có xy/class/USERID/PHONEID.
- 13 cặp tòa–tầng theo mapping đã audit, không dùng tích Descartes 3×5. Class ID và building/floor phải nhất quán.
  Dự đoán xy phải inverse về tọa độ gốc theo quy ước mét; chưa xác minh đầy đủ CRS, không tự tính Haversine/3D.
- Giữ nguyên split train16.025/validation3.912/test1.111, manifest/hash/nhóm. Đây chưa phải split tách riêng người đo;
  các user/điểm/RSSI vẫn có thể chung như chặng 5 đã ghi.
- Center/scale lưu sẵn phải khớp phép fit **toàn internal train** và hash train indices. Smoke subset256/128 vẫn
  dùng transform này, đúng pipeline cũ. Các binning/scaler/vocabulary/index/item utilities học mới chỉ fit trong prepare(train).
- Không dùng validation/test làm retrieval corpus, ví dụ few-shot hay dữ liệu học itemset. Không refit transform
  trong execute(validation): đây là trách nhiệm backend cần review/test, interface không tự phát hiện mọi rò rỉ.
- Runner chỉ mở mẫu test **sau execute** và `AlgorithmExecution.finalized=True`. Cờ này là cam kết đã chốt backend;
  không phải cơ chế tự phát hiện tuning trên test. Smoke không được đánh giá test.
  Audit hash vẫn đọc byte CSV test trước đó để xác minh nguồn; không parse hoặc truyền mẫu/nhãn test vào thuật toán.

## Chạy và cấu hình

Không cần dependency mới. Từ root repo:

```powershell
.\.venv\Scripts\python.exe -m ips algorithms
.\.venv\Scripts\python.exe -u -m ips run --config configs/experiments/kan_mixer_smoke.json
.\.venv\Scripts\python.exe -u -m ips run --config configs/experiments/mlp_smoke.json
.\.venv\Scripts\python.exe -u -m ips run --config configs/experiments/mlp_mixer_smoke.json
# Terminal thứ hai
Get-Content runs/factory_kan_mixer_smoke/terminal.log -Tail 20 -Wait
```

Output directory phải **chưa tồn tại**, kể cả sau lượt lỗi. Chạy lại bằng bản copy JSON có output_dir mới.
Lỗi có failure.json/terminal.log. Đường dẫn cấu hình tương đối tính từ root repo.

Ví dụ full neural (chỉ hướng dẫn, không tự chạy suite GPU):

```json
{
  "algorithm_id": "kan_mixer",
  "params": {"model_config": "configs/base.json", "epochs": 100},
  "split_dir": "data/processed/uji_group_split_v1",
  "output_dir": "runs/team_kan_seed42_v1",
  "task": "position",
  "seed": 42,
  "device": "cuda",
  "smoke": false,
  "evaluation_splits": ["validation"],
  "export_plots": true
}
```

`params` thuộc backend, Runner không diễn giải epoch. `task=auto` đánh giá mọi đầu ra thực có;
position đòi xy và joint building/floor, coordinates đòi xy, classification đòi ít nhất một output phân loại.
Thiếu đầu ra bắt buộc gây lỗi, không điền số giả.

Đánh giá checkpoint v1: thay params bằng `{"checkpoint":"runs/<run>/best.pt"}`; không kết hợp checkpoint
với epochs. Loader kiểm tra model ID/AP order/class mapping/preprocessing/split manifest với dữ liệu hiện dùng.
Sau khi chốt giao thức, có thể chọn evaluation_splits=["validation","test"] với checkpoint run_type=training.
Checkpoint smoke không được đánh giá test.

API đăng ký **backend thật**, không đăng ký adapter trừu tượng:

```python
from pathlib import Path
from ips.experiments.contracts import ExperimentConfig
from ips.experiments.factory import default_factory
from ips.experiments.runner import ExperimentRunner
from my_team_backend import MyLLMPositionAlgorithm  # do nhóm triển khai, concrete Algorithm

factory = default_factory()
factory.register("team_llm_v1", MyLLMPositionAlgorithm)
config = ExperimentConfig("team_llm_v1", Path("runs/team_llm_v1_seed42"),
                          params={"model_path": "..."}, task="position")
result = ExperimentRunner(factory=factory).run(config)
```

## Đầu ra, metric và tài nguyên

IPSPredictions giữ source/row order; có xy, class_ids, buildings, floors và confidence tùy khả năng.
Nếu có xy+class IDs, evaluator gọi **nguyên hàm metric cũ**. Các đầu ra riêng dùng cùng công thức/label domain;
không có xy thì không có mean2D/target8m, không có phân loại thì không có confusion/conditional joint-correct error.
RunResult giữ array dự đoán trong bộ nhớ, metrics/matrices theo split, thời gian, config và artifact path;
history/resources/diagram tùy chọn.

```text
<output_dir>/
  terminal.log                    stdout/stderr, log epoch trực tiếp
  data_provenance.json             manifest, preprocessing, AP order, classes
  experiment_config.json
  run_result.json                  metrics, runtime, danh mục artifact; không nhân đôi array
  history.json / resources.json    chỉ khi backend có
  model_diagram.mmd hoặc .txt      chỉ khi backend có
  algorithm/                      checkpoint v1/history/diagnostics neural
  validation/ hoặc test/
    predictions.csv               dự đoán thật + truth để phân tích, thêm sau predict
    metrics.json / matrices.json
    *_confusion_matrix.csv/.png    nếu có phân loại; hàng=true, cột=predicted
    error_cdf.png                 nếu có xy
```

Runtime đo thời gian thực từng pha, export tách riêng; không coi thời gian predict toàn split là latency một scan.
Resources neural giữ RAM toàn tiến trình lấy mẫu và VRAM allocated/reserved như cũ; có thể bỏ lỡ đỉnh RAM ngắn.
Trường backend pipeline_seconds trong Factory không tính audit dữ liệu và tạo tensor đã làm trước đó;
resources.json ở root ghi factory_scope_note, dùng runtime của RunResult để xem toàn lượt.
Backend chưa đo memory có resources=None, không ghi 0 như đã đo. LLM/HUIM cần ghi phạm vi tài nguyên, model/provider,
token budget/thời gian mạng nếu có; lịch sử train không bị bắt buộc.

So sánh dùng cùng manifest/row indices/domain/metric/quy tắc chọn trạng thái. Ghi rõ tiền xử lý khác nhau;
epoch neural không tự tương đương token LLM hay vòng khai phá HUIM. Chốt ngân sách thời gian/tài nguyên và
siêu tham số trước test; dùng nhiều seed và báo cáo mean/SD khi phương pháp có ngẫu nhiên.

## Tương thích lịch sử

CLI train/predict/evaluate/check vẫn hoạt động; dataset.load_bundle()/Bundle.dataset() là adapter tương thích.
Giữ metrics.py, model/vendor, dữ liệu/split, base config, reports và script khóa ba mô hình chặng 7.

Refactor làm hash dataset.py/train.py thay đổi. Audit suite chặng 7 cố ý kiểm tra mã đã sinh kết quả:
**không tái audit/tạo lại báo cáo suite cũ bằng checkout đã refactor**, không nới kiểm tra hash.
Tái lập suite lịch sử bằng commit gốc `2ad74a0fd2f925966a1ca18f121a8a8701526870` trong checkout/worktree riêng,
với môi trường/dữ liệu/checkpoint tương ứng. Nạp v1 được không đồng nghĩa đã chạy lại nguyên suite.
Factory ghi run mới, không tự nhập kết quả vào bảng lịch sử đã khóa.

## Nhóm cần quyết định tiếp

**LLM:** model cục bộ/API, cách serialization RSSI/missing/AP identity, fine-tune hay inference,
train retrieval/few-shot, decoding tọa độ/cặp nhãn, phản hồi không hợp lệ và tính tái lập/provider/token/memory.

**HUIM:** item và binning RSSI fit trên train, định nghĩa utility/ngưỡng, thuật toán khai phá,
khớp pattern với mẫu mới, quy tắc itemset → tòa/tầng/xy, không khớp/nhiều khớp và chi phí mine/predict.
Itemset chưa phải dự đoán IPS.

**KAN:** chỉ efficient-kan B-spline đã có. FastKAN/S-KAN cần mã/nguồn/config/tests mới trước khi đăng ký;
không đổi ý nghĩa ID kan_mixer hiện tại.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m ips check --device cpu
```

Tests mới: registry/concrete subclass/ID thiếu, nguồn/thứ tự và output contract, metric so với legacy,
AP/label/tensor/subset gốc, train-only coordinate fit, vòng đời non-neural, chặn test trước finalization,
artifact/refusal overwrite, CLI cũ và fixture v1. Smoke không chứng minh đạt 7–8 m.
