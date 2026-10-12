# Giải thích các file thay đổi và file mới trong thư mục KAN

Ngày kiểm tra: **09/10/2026**. Nhánh: `feature/kan-accuracy-20261009`. Mốc đối chiếu: **Git HEAD `d473d6c`** với mã đang có trên máy.

Tài liệu này giải thích mã nguồn thực tế của đợt cải tiến độ chính xác. Tại mốc phân tích ngày 09/10, trong `src/ips/algorithms/kan/` có **6 file đã sửa và 11 file nghiên cứu mới** so với HEAD nói trên. File Markdown bạn đang đọc được tạo sau khi kiểm tra, không tính vào 11 file nghiên cứu mới. Ngày 12/10/2026, nhánh được đổi tên thành `feature/kan-cai-thien`, tài liệu được bỏ ignore và các thay đổi được chia thành commit theo nhiệm vụ. README trong KAN cũng được cập nhật để chỉ tới lối chạy mới.

## 1. Vì sao có đợt thay đổi này?

KAN-Mixer cũ nhận dấu vân tay Wi-Fi rồi dự đoán trực tiếp tòa/tầng và hai tọa độ. Kết quả test trung bình ba seed là khoảng **15,726 m**. Đợt nghiên cứu mới giữ các khối KAN-Mixer, nhưng thay cách biểu diễn đầu ra: mạng học mức phù hợp của một mẫu với các **điểm tham chiếu đã đo trong train**, sau đó kết hợp tọa độ của những điểm ấy để định vị.

Ví dụ, nếu mạng cho rằng một mẫu giống các điểm A và B trên cùng tầng, ta lấy tọa độ giữa A và B theo trọng số mạng dự đoán. Cách này đưa cấu trúc vị trí đã biết vào đầu ra, thay vì yêu cầu một head tuyến tính tự dự đoán hai số bất kỳ.

Các file mới xuất hiện để tách những trách nhiệm khác nhau: định nghĩa mạng, tìm cấu hình giải mã, kết hợp fingerprint, lưu/nạp checkpoint và thử các hướng mở rộng. Không phải mọi file mới đều đã được dùng để tạo kết quả.

### Kết quả nào thuộc thay đổi nào?

| Cấu hình | Mean2D test seed 44 | Mean2D test ba seed 42/43/44 ± sample SD | Trạng thái |
|---|---:|---:|---|
| KAN-Mixer cũ, hồi quy trực tiếp | 14,081635 m | 15,726113 ± 1,509809 m | Đối chứng lịch sử |
| KAN-Mixer điểm tham chiếu thuần | 7,719440 m | 8,058153 ± 0,306071 m | Đã huấn luyện và đánh giá |
| KAN-Mixer lai fingerprint | 7,605120 m | 7,838428 ± 0,211871 m | Đã hiệu chỉnh bằng validation và đánh giá |
| Ensemble nhiều mạng | — | — | Có mã và test kỹ thuật; chưa có thực nghiệm đầy đủ trong báo cáo |
| Refit trên toàn bộ development | — | — | Có mã và fixture test; chưa chạy giao thức này để tạo số liệu báo cáo |

**Mean2D** là trung bình khoảng cách Euclid 2D trên toàn bộ 1.111 mẫu test. Dấu ± ở bảng là độ lệch chuẩn mẫu giữa ba seed, không phải độ lệch chuẩn của 1.111 lỗi hoặc khoảng tin cậy. Bản lai đạt dưới 8 m theo trung bình ba seed; bản thuần chưa đạt tiêu chí này. Seed 42 lai vẫn là 8,018838 m.

Đây là cải thiện đã đo của **một gói thiết kế**. Vì head, số tham số, loss, augmentation và lịch learning rate cùng thay đổi, chưa thể kết luận riêng flatten hay riêng spline KAN gây ra toàn bộ mức giảm.

## 2. Bản đồ file để review

Các đường dẫn từ đây, trừ khi ghi khác, tính tương đối từ thư mục `kan/`.

```text
kan/
  algorithm.py                      ĐÃ SỬA: adapter IPS cấp thuật toán
  dataset.py                        ĐÃ SỬA: bổ sung nguồn gốc split của tensor
  runtime.py                        ĐÃ SỬA: config và checkpoint v1/v2
  train.py                          ĐÃ SỬA: reference loss, catalogue, cosine LR
  predict.py                        ĐÃ SỬA: chặn validation của final refit
  accuracy.py                       MỚI: hiệu chỉnh decoder bằng validation
  accuracy_ensemble.py              MỚI: chọn cách kết hợp nhiều mạng
  final_refit.py                    MỚI: train cuối trên toàn bộ development
  freeze_refit.py                   MỚI: đóng gói refit bằng lựa chọn đã khóa
  models/
    networks.py                    ĐÃ SỬA: chọn mạng cũ hoặc reference
    reference.py                   MỚI: điểm tham chiếu và mạng reference
    refinement.py                  MỚI: kết hợp với fingerprint train
    ensemble.py                    MỚI: kết hợp các mạng đã train
    vendor/                        GIỮ NGUYÊN: efficient-kan B-spline
  configs/
    model.json                     GIỮ NGUYÊN: cấu hình lịch sử
    accuracy_reference.json        MỚI: reference với nhãn điểm cứng
    accuracy_spatial.json          MỚI: nhãn mềm và augmentation mạnh hơn
    accuracy_mean_ablation.json    MỚI: reference head nhưng mean pooling
  tests/
    test_pipeline.py               GIỮ NGUYÊN: kiểm tra pipeline cũ
    test_reference.py              MỚI: kiểm tra các tính chất nhánh mới
  GIAI_THICH_THAY_DOI_2026-10-09.md  Tài liệu giải thích thay đổi, có thể đưa vào Git
```

Ba nhóm cần phân biệt:

1. **Lõi đã tạo ra kết quả mới:** `reference.py`, các bổ sung trong `train.py`, cấu hình spatial, `accuracy.py`, `refinement.py` và đường nạp checkpoint trong `runtime.py`.
2. **Tích hợp và bảo vệ giao thức:** `algorithm.py`, `dataset.py`, `networks.py`, `predict.py`, tests.
3. **Hướng mở rộng có mã:** `ensemble.py`, `accuracy_ensemble.py`, `final_refit.py`, `freeze_refit.py`. Sự có mặt của chúng không có nghĩa kết quả ensemble/refit đã hoàn tất.

## 3. Sáu file đã sửa

### 3.1. `algorithm.py`: đưa biến thể mới vào vòng đời IPS chung

**Vai trò trước đây:** chứa `KANMixerAlgorithm`, lớp kế thừa trực tiếp `Algorithm` dùng chung. Nó nhận split NumPy, chuẩn bị tensor, train hoặc nạp checkpoint, trả `IPSPredictions` và thông tin sơ đồ.

**Hai thay đổi cụ thể:**

- Sau khi nạp checkpoint, nếu metadata có `full_development_refit` và cấu hình yêu cầu đánh giá `validation`, hàm `execute()` báo lỗi. Refit đã học cả validation cũ, nên không được báo dữ liệu đó như tập giữ lại độc lập.
- `get_diagram()` bổ sung mô tả số điểm tham chiếu, số thành viên ensemble và dấu hiệu bản lai fingerprint nếu checkpoint có các thành phần tương ứng.

**Vì sao cần:** Factory/Runner phải biết checkpoint mình đang dùng thuộc dạng nào, đồng thời bảo vệ người sử dụng khỏi việc so validation đã học với validation chưa học.

**Phần giữ nguyên:** hợp đồng `prepare → execute → predict`, kiểu đầu ra IPS, kiểm tra mapping/AP/manifest/preprocessing và cách gọi từ Factory. Không tạo một `algorithm_id` mới cho mỗi biến thể mạng.

**Lưu ý khi đọc sơ đồ:** đây vẫn là sơ đồ tổng quan có chú thích, chưa vẽ riêng mọi bước top-k hay tìm láng giềng. Dòng chữ trong sơ đồ mô tả khả năng checkpoint, không chứng minh khả năng ấy đã được thực nghiệm.

### 3.2. `dataset.py`: tensor phải biết mình đến từ split nào

**Vai trò:** chuyển dữ liệu/split dùng chung thành `FingerprintDataset` cho PyTorch: features RSSI, class ID, tọa độ chuẩn hóa, tọa độ gốc và row index.

**Thay đổi:** thêm thuộc tính `split_name`:

- Constructor trực tiếp đặt `"unspecified"` vì một mảng chỉ số không tự chứng minh nó thuộc train.
- `from_split()` lấy tên thực từ `FingerprintSplit.name`, chẳng hạn `"train"` hoặc `"validation"`.

**Vì sao cần:** `fit_reference_points()` kiểm tra tên split trước khi dựng điểm tham chiếu. Nếu vô tình truyền validation, hàm phải từ chối thay vì âm thầm đưa vị trí validation vào catalogue.

**Không đổi:** công thức chuẩn hóa, thứ tự 520 AP, class mapping, row index, tọa độ gốc và `__getitem__()`. Đây là nhãn nguồn gốc dữ liệu, không phải một cách chia tập mới.

**Giới hạn:** thuộc tính này là kiểm tra hợp đồng trong mã; người viết Python vẫn có thể gán lại thuộc tính. Độ tin cậy còn đến từ Repository, manifest/hash, chỉ số split và các kiểm tra provenance; không chỉ từ một chuỗi `"train"`.

### 3.3. `models/networks.py`: giữ mạng cũ và bổ sung lựa chọn reference

**Vai trò:** định nghĩa `MixerBlock`, KAN token/channel mixing, `RSSIMixer` cũ và factory nội bộ `build_model()`.

**Thay đổi trong `build_model()`:** đọc `config["positioning"]["variant"]`:

| Giá trị | Mạng được tạo |
|---|---|
| Không có `positioning.variant` | `RSSIMixer` cũ, mặc định `regression_v1` |
| `regression_v1` | `RSSIMixer` cũ |
| `reference` | `ReferenceKANMixer` từ `models/reference.py` |
| Giá trị khác | Báo lỗi rõ ràng |

**Vì sao cần:** mở rộng mô hình mà không phá cấu hình/checkpoint cũ. Import `ReferenceKANMixer` ngay trong nhánh lựa chọn còn tránh import vòng ở cấp module: `reference.py` tái sử dụng `MixerBlock` từ chính file này.

**Không đổi:** thuật toán của `MixerBlock`, các nhánh B-spline KAN, head cũ và việc loại MLP/MLP-Mixer cá nhân khỏi dự án dùng chung. KAN backend vẫn là efficient-kan, không phải FastKAN hay S-KAN.

`build_model()` tạo một **`nn.Module`** bên trong nhánh KAN; `AlgorithmFactory` ở ngoài tạo một **`Algorithm`** cấp IPS. Hai factory phục vụ hai cấp khác nhau.

### 3.4. `train.py`: học vị trí tham chiếu thay vì chỉ hồi quy hai số

File này có nhiều bổ sung nhất trong sáu file đã sửa.

#### a. Thêm nhánh reference trong `combined_loss()`

Nếu đầu ra mạng có `reference_logits`, hàm dùng loss mới:

1. Tính khoảng cách bình phương giữa tọa độ mẫu train và các điểm tham chiếu.
2. Che điểm ở lớp tòa/tầng khác bằng khoảng cách vô hạn.
3. Nếu `reference_sigma_normalized > 0`, tạo nhãn mềm Gaussian từ khoảng cách; điểm gần trên cùng tầng có trọng số mục tiêu cao hơn.
4. Nếu sigma bằng 0 hoặc không cấu hình, chọn điểm gần nhất làm nhãn cứng, có thể dùng `label_smoothing` nếu bật.
5. Tính loss tọa độ Euclid từ `dense_conditional_xy` của lớp thật trong train.
6. Gộp loss lớp, loss điểm và loss tọa độ theo các hệ số cấu hình.

Trong cấu hình spatial:

```text
L = 0,5 × L_class + 1 × L_reference + 3 × L_coordinate
```

Loss tọa độ nằm trong hệ chuẩn hóa; mean2D để báo cáo được tính sau khi đổi về tọa độ gốc. Dùng nhãn thật để tạo mục tiêu **khi train** là hợp lệ; `forward()` khi dự đoán không nhận nhãn thật của mẫu.

Với mạng cũ không có `reference_logits`, nhánh CrossEntropy + SmoothL1 vẫn giữ nguyên.

**Chi tiết log dễ hiểu nhầm:** giá trị trả về cho cột `train_class_loss` của reference là **classification + reference_loss**, chưa nhân hệ số của loss tổng. Vì thế không đọc cột này như chỉ riêng CrossEntropy tòa/tầng, và không so loss cũ/mới như cùng một đại lượng.

#### b. Fit catalogue trước khi tạo mạng

`config` được `deepcopy`, rồi gọi `fit_reference_points(config, train_data)`. Danh sách điểm được thêm vào cấu hình hiệu lực và checkpoint. Copy sâu tránh việc một lượt train sửa cấu hình gốc của người gọi.

Với cấu hình mạng cũ, `fit_reference_points()` trả về ngay, nên không đổi hành vi cũ.

#### c. Thêm cosine learning-rate scheduler

Nếu `training.scheduler == "cosine"`, dùng `CosineAnnealingLR`; nếu thiếu trường này, learning rate vẫn cố định như trước. Scheduler chạy sau mỗi epoch; learning rate ghi trong history là giá trị đã dùng cho epoch đó.

Mức thấp nhất lấy từ `minimum_learning_rate` nếu có, nếu không bằng 5% learning rate ban đầu. Với lr 0,001 thì mặc định là 0,00005.

#### d. Bổ sung hash nguồn reference

Khi train biến thể reference, `models/reference.py` được đưa vào `training_implementation_sha256`, giúp đối chiếu mã thực sự tham gia huấn luyện.

**Phần giữ nguyên:** chọn `best.pt` bằng validation mean2D, early stopping, không đánh giá test trong vòng train, log từng epoch, RAM/VRAM, checkpoint có metadata và từ chối ghi đè run-dir.

**Chưa được bổ sung:** lưu đầy đủ RNG, trạng thái generator và scheduler để resume bit-exact. Có `optimizer_state_dict` không đồng nghĩa có thể tiếp tục chính xác một lượt bị ngắt. Vì vậy seed 44 đã được train lại từ đầu ở thư mục mới.

### 3.5. `runtime.py`: một đường nạp cho mạng cũ, reference, lai và ensemble

**Thay đổi kiểm tra cấu hình:**

- Từ chối `reference_sigma_normalized` âm.
- Chỉ chấp nhận scheduler `cosine` hoặc không cấu hình.

**Thay đổi nạp checkpoint:**

1. Chấp nhận `format_version` 1 hoặc 2, thay vì chỉ 1.
2. Nếu có `ensemble_members`, dựng từng mạng từ config/state của thành viên, rồi bọc bằng `EnsembleReferenceKANMixer`.
3. Nếu không có ensemble, dựng mạng đơn như trước.
4. Nếu có `fingerprint_refinement`, bọc mạng vừa dựng bằng `RefinedKANMixer`.
5. Đưa mạng lên device và đặt `eval()`.

**Vì sao cần:** CLI và Factory đều gọi `load_checkpoint()`. Một đường nạp chung giúp `predict/evaluate` dùng cùng cách suy luận với kiểu checkpoint đã chọn, thay vì mỗi lệnh tự diễn giải một kiểu.

**Tương thích:** checkpoint hồi quy v1 cũ vẫn được dựng bằng `RSSIMixer`. Reference thuần cũng có thể ở v1: số phiên bản mô tả định dạng payload, không trực tiếp nói mạng là kiến trúc cũ hay mới. Bản lai/ensemble/refit sử dụng v2 trong các công cụ mới.

**Không đổi:** `torch.load(..., weights_only=True)`, nạp `state_dict` với `strict=True`, cách đặt seed/device và metadata phần mềm. Checkpoint reference chứa điểm tham chiếu; checkpoint lai chứa catalogue fingerprint, nên khi suy luận không cần fit lại từ CSV train.

### 3.6. `predict.py`: ngăn đánh giá nhầm validation sau refit

**Thay đổi duy nhất:** trong `evaluate()`, nếu chọn split `validation` nhưng checkpoint được đánh dấu `full_development_refit`, báo lỗi rằng validation đã được đưa vào train.

**Vì sao cần:** một mô hình refit có thể chạy dự đoán trên validation về mặt kỹ thuật, nhưng kết quả đó không còn đo khả năng trên dữ liệu giữ lại. Kiểm tra này bảo vệ ý nghĩa báo cáo.

**Không đổi:** định nghĩa metric, đổi tọa độ về đơn vị gốc, kiểm hash/mapping/AP và xuất predictions/metrics. Mạng reference/lai chạy được nhờ `runtime.load_checkpoint()`, không cần sửa công thức đánh giá.

## 4. Mười một file mới

### 4.1. `models/reference.py`: lõi của mô hình cải tiến

File có hai trách nhiệm liên quan chặt chẽ.

#### `fit_reference_points()` — chuẩn bị điểm để mạng học

- Chỉ chạy khi `positioning.variant == "reference"`.
- Với train thông thường, yêu cầu dataset được xác định là split train.
- Tạo điểm duy nhất theo **class ID và tọa độ gốc**. Cùng tọa độ 2D nhưng khác tầng vẫn là điểm khác.
- Dùng tọa độ gốc float64 để xác định điểm, tránh gộp nhầm do làm tròn float32.
- Lưu tọa độ đã chuẩn hóa và class ID vào `positioning.references`, cùng `fit_scope`.
- Có nhánh riêng `final_refit=True` cho development sau khi chọn cấu hình; đó là giao thức khác, không phải ngoại lệ để fit validation trong thí nghiệm thường.

Trên train hiện tại có **872 điểm**. Đây là số sinh từ dữ liệu của split, không phải hằng số 872 viết cứng trong lớp.

#### `ReferenceKANMixer` — mạng PyTorch và decoder vị trí

Luồng tensor với cấu hình đã chạy:

```text
features [B, 520, 2]
  → xử lý signal/mask, augmentation chỉ khi train
  → reshape [B, 26, 40]
  → Linear embedding [B, 26, 64]
  → 2 MixerBlock dùng B-spline KAN
  → LayerNorm [B, 26, 64]
  → flatten [B, 1664]
  → Linear 1664→256, GELU, Dropout, Linear 256→872
  → reference_logits [B, 872]
  → tổng hợp 13 lớp tòa/tầng và tọa độ theo lớp dự đoán
```

Head cuối có lớp Linear/GELU. Tên KAN-Mixer ở đây nói về các khối trộn token/channel bằng KAN; không có nghĩa mọi phép biến đổi trong mạng đều là KAN.

**Các điểm quan trọng:**

- Có thể chọn `pooling="mean"` để thử đối chứng; mặc định nhánh reference là flatten.
- Xác suất lớp được tạo bằng cách cộng xác suất các điểm thuộc cùng lớp, thực hiện ổn định bằng `logsumexp`.
- Tọa độ được lấy từ top-k điểm **trong lớp dự đoán**, chuẩn hóa trọng số bằng softmax với temperature.
- `top_k=1000` là giới hạn yêu cầu; số điểm thực lấy còn bị giới hạn bởi catalogue. Không có 1.000 điểm trên mỗi tầng trong split hiện tại.
- `dense_conditional_xy` dùng toàn bộ điểm trong mỗi lớp để tính loss train; `conditional_xy` dùng top-k để suy luận. Hai mục đích này được tách rõ.
- `training_temperature` cho phép giữ temperature dùng học khi temperature suy luận đã được hiệu chỉnh, phục vụ công cụ refit.
- Các buffer `catalogue_indices/catalogue_valid` được tạo lại từ điểm tham chiếu, không lưu vào `state_dict`; chuẩn bị chỉ số trên CPU một lần giúp tránh indexing boolean và đồng bộ CUDA lặp nhiều lần.

**Augmentation:** thêm nhiễu Gaussian từng AP, một độ lệch chung cho fingerprint và bỏ ngẫu nhiên AP đã thấy. Các giá trị dB là độ lệch chuẩn, không phải giới hạn cứng. Mức nhiễu được đổi theo thang signal bằng chia 104, signal được clamp và nhân mask. `eval()` tắt các phép ngẫu nhiên này.

**Vì sao có file riêng:** để giữ nguyên `RSSIMixer` lịch sử, tái sử dụng `MixerBlock`, và có nơi độc lập để review những giả định về điểm tham chiếu. Đây là biến thể IPS của dự án, không phải lời khẳng định tái lập nguyên bản kiến trúc ảnh của bài báo.

### 4.2. `models/refinement.py`: bản lai KAN và fingerprint

**Vấn đề cần giải quyết:** tọa độ neural đã cải thiện đáng kể, nhưng các fingerprint train gần mẫu mới vẫn có thể cung cấp thông tin cục bộ bổ sung.

File gồm:

- `db_features()`: chuyển signal/mask về biểu diễn tương đương `(RSSI + 105)/105`, AP thiếu bằng 0. Khoảng cách Euclid tương đương điền −105 dBm cho AP không thấy, chỉ khác hệ số thang đo chung.
- `refine_predictions()`: tính khoảng cách fingerprint tới catalogue; bỏ phiếu tòa/tầng có trọng số nếu bật; tìm láng giềng trong lớp dự đoán; trộn tọa độ neighbor và neural.
- `RefinedKANMixer`: wrapper `nn.Module`, giữ backbone và catalogue trong buffer, kiểm tra phạm vi fit/shape/tham số.

Quy tắc tọa độ:

```text
xy cuối = alpha × xy neural + (1 − alpha) × xy láng giềng
```

Validation của cả ba seed chọn `alpha=0.75`, `neighbors=10`, `distance_power=2`, `neighbor_class_weight=0`. Tức là **75% neural, 25% fingerprint**, nhãn tòa/tầng giữ theo neural. File hỗ trợ trộn nhãn, nhưng cấu hình đã chọn không dùng khả năng ấy.

**Vì sao tách khỏi reference:** đây là phương pháp lai, có chi phí tìm kiếm và dữ liệu catalogue bổ sung. Cần báo riêng với mô hình neural thuần, không gọi mọi mức sai số của nó là kết quả KAN thuần.

**Ràng buộc:** catalogue thông thường lấy train; alpha phải dương và không quá 1; không nhận nhãn thật của query. File cũng chấp nhận catalogue development của giao thức final refit có đánh dấu riêng.

### 4.3. `accuracy.py`: chọn decoder trên validation, không train lại backbone

**Mục đích:** sau khi train và chọn `best.pt`, thử cách chuyển reference logits thành vị trí phù hợp hơn.

`calibrate()`:

1. Yêu cầu run đã có `summary.json`, checkpoint là reference và `run_type="training"`; từ chối smoke và final refit.
2. Kiểm manifest; lấy tensor train và validation từ split đã lưu.
3. Chạy mạng lấy reference scores validation một lần.
4. Thử **24** cấu hình thuần: top-k `1,3,5,10,20,1000` × temperature `0.5,1,2,4`.
5. Nếu bật hybrid, giữ decoder thuần tốt nhất rồi thử **72** cấu hình lai: 4 số neighbor × 2 power × 3 class blend × 3 alpha.
6. Chọn theo validation mean2D và ghi lịch sử lựa chọn.

Tổng là **96 ứng viên** khi có hybrid, 24 khi `--pure-only`. Đây là tìm kiếm hai bước, không phải thử mọi tổ hợp top-k/temperature với mọi tổ hợp hybrid.

**File xuất:**

| Artifact | Ý nghĩa |
|---|---|
| `pure_selected.pt` | Checkpoint thuần với top-k/temperature được chọn |
| `selected.pt` | Tốt nhất giữa thuần và các ứng viên lai; không đảm bảo luôn là lai |
| `calibration.json` | Lưới thử, kết quả validation, lựa chọn, thời gian và hash |

**Vì sao có file riêng:** calibration không phải một epoch huấn luyện mới. Tách riêng giúp giữ checkpoint backbone và làm rõ bước chọn tham số suy luận.

**Điểm cần hiểu chính xác:** hàm này không tính metric hay dự đoán test. `load_bundle()` vẫn nạp bundle dữ liệu chung; sự tách biệt nằm ở việc chỉ dùng train/validation cho phép tính và lựa chọn. Dòng log `official test unopened` chỉ nên hiểu là **entry point này không đánh giá test**; nó không chứng minh chưa ai xem test trong toàn chiến dịch. Test đã được xem ở các lượt trước.

### 4.4. `models/ensemble.py`: kết hợp nhiều mạng đã học

`EnsembleReferenceKANMixer` chứa từ hai mạng reference trở lên trong `ModuleList`.

- Xác suất lớp cuối là trung bình xác suất lớp của các thành viên.
- Tọa độ theo từng lớp có thể lấy trung bình đều, hoặc lấy trọng số từ xác suất mà mỗi thành viên gán cho lớp ấy.
- Chọn lớp cuối rồi lấy tọa độ của lớp dự đoán; không dùng nhãn thật của query.
- Không có optimizer, training loop hay bước fit trong file này.

**Vì sao xuất hiện:** các seed có thể mắc lỗi khác nhau; ensemble là một hướng kiểm tra liệu kết hợp dự đoán có giảm dao động hay không. Đổi lại, phải giữ nhiều mạng và chạy nhiều forward, nên thường tăng chi phí bộ nhớ/tính toán.

**Trạng thái:** đã có implementation và kiểm tra bằng fixture; **không phải nguồn của con số 7,838 m**. Trung bình metric ba seed trong báo cáo cũng **khác** một ensemble: báo cáo tính metric từng mạng rồi tổng hợp; ensemble trộn dự đoán trước khi tính metric.

### 4.5. `accuracy_ensemble.py`: chọn và đóng gói ensemble bằng validation

`assemble()` nhận các checkpoint thuần đã chốt độc lập:

- Từ chối checkpoint đã là hybrid/ensemble hoặc final refit.
- Kiểm cùng preprocessing, mapping, AP order, manifest và run type.
- So hai kiểu tọa độ ensemble: trung bình đều và theo xác suất lớp.
- Sau đó thử 9 cách lai ensemble: 3 class blend × 3 alpha, giữ `k=10/power=2` theo lựa chọn validation trước đó.
- Xuất `pure_selected.pt`, `selected.pt`, `selection.json`, lưu danh sách config/state của các thành viên trong checkpoint v2.

**Vì sao khác `ensemble.py`:** file model chỉ biết kết hợp dự đoán; file này biết đọc dữ liệu, đánh giá validation, chọn chính sách và xuất artifact. Tách hai phần tránh để inference tự chọn tham số bằng dữ liệu đánh giá.

**Trạng thái:** hướng mở rộng chưa chạy đủ thực nghiệm trong báo cáo seed 44. Lớp model kiểm số lớp; công cụ assembly kiểm mapping đầy đủ, nên các kiểm tra có vai trò bổ sung nhau.

### 4.6. `final_refit.py`: train cuối với nhiều dữ liệu hơn sau khi khóa lựa chọn

**Refit** ở đây là huấn luyện một mạng mới từ đầu bằng toàn bộ **19.937 mẫu development**, sau khi cấu hình và số epoch đã được lựa chọn bằng validation nội bộ trước đó. Nó không phải tiếp tục optimizer của checkpoint cũ.

`refit()` yêu cầu selection JSON có:

- `selection_split="internal_validation_only"`;
- `test_evaluated=false`;
- `final_refit_epochs` là số nguyên trong khoảng 1–100;
- hash checkpoint nguồn thuộc danh sách được cho phép.

Sau đó dùng cấu hình đã chọn, khởi tạo trọng số theo seed mới, dựng catalogue development và train số epoch cố định. Không đánh giá validation/test trong vòng train, không dùng early stopping trên validation đã nhập vào train.

**Chi tiết thiết kế:**

- Giữ preprocessing và split files lịch sử, không fit lại transform hay ghi đè split.
- Giữ quỹ đạo cosine theo `training.max_epochs`, tránh nén lịch 100 epoch vào số epoch refit đã chọn.
- Metadata ghi `full_development_refit`, hash selection lock, phạm vi catalogue và train count.
- `validation_mean_2d=None`. File cuối vẫn mang tên `best.pt` để tương thích đường nạp, nhưng thực chất là checkpoint cuối của số epoch cố định, không phải best chọn trên validation mới.

**Vì sao có file riêng:** refit thay phạm vi dữ liệu được học và cách chốt checkpoint. Cần một giao thức riêng để không trộn với thí nghiệm train 16.025 mẫu.

**Trạng thái và giới hạn:** chỉ có fixture CPU nhỏ kiểm cơ chế, chưa chạy refit development thật trong báo cáo. Chuỗi `test_evaluated=false` và hash là điều kiện kiểm trong JSON, không tự chứng minh lịch sử chưa xem test. Không được tạo lại một lock giả “trước test” sau khi đã xem kết quả hiện tại.

### 4.7. `freeze_refit.py`: giữ quyết định cũ, đóng gói mô hình refit

`freeze()` nhận các checkpoint final refit và một selection lock trước đó:

- Kiểm lock hợp lệ và từng checkpoint cùng hash lock.
- Kiểm mapping, preprocessing, AP order và manifest giữa các thành viên.
- Dựng catalogue fingerprint từ toàn development, đánh dấu `development_after_validation_selection`.
- Áp dụng `frozen_hybrid_settings` đã khóa, xuất `hybrid_seed*.pt`.
- Nếu từ hai thành viên trở lên, xuất thêm `pure_ensemble.pt` và `hybrid_ensemble.pt`, dùng `frozen_ensemble_confidence_weighted`.
- Ghi `frozen_artifacts.json` gồm đường dẫn/hash và `test_evaluated=false`.

**Vì sao cần cả freeze và calibration:** sau refit, validation cũ đã được học, nên không được dùng `accuracy.py` để chọn lại decoder. `freeze_refit.py` chỉ áp dụng lựa chọn cũ, không tính dự đoán/metric validation hoặc test.

**Trạng thái:** chưa tạo artifact refit thật cho kết quả báo cáo. Khi dùng phải có đủ các trường frozen settings trong lock; đây không phải file để cải thiện test tự động.

### 4.8–4.10. Ba file cấu hình `configs/accuracy_*.json`

JSON tách các lựa chọn thử nghiệm khỏi lớp mạng. Mỗi file mô tả mixer/kan/loss/training/positioning; không chứa số liệu thành tích và không tự chạy huấn luyện.

| Trường | `accuracy_reference.json` | `accuracy_spatial.json` | `accuracy_mean_ablation.json` |
|---|---|---|---|
| Variant | reference | reference | reference |
| Pooling | flatten | flatten | mean |
| Head width | 256 | 256 | 256 |
| Nhãn điểm | Cứng, sigma mặc định 0 | Mềm, sigma chuẩn hóa 0,05 | Cứng, sigma mặc định 0 |
| RSSI noise std | 1,5 dB | 2 dB | 1,5 dB |
| Fingerprint shift std | 1,5 dB | 3 dB | 1,5 dB |
| AP dropout | 2% | 5% | 2% |
| Max epoch | 100 | 100 | 50 |
| Patience | 25 | 20 | 12 |
| Scheduler | cosine | cosine | cosine |
| Mask scale | 0 | 0 | 0 |
| Batch | 128 | 128 | 128 |
| Top-k/temperature ban đầu | 5 / 1 | 5 / 1 | 5 / 1 |

Các file giữ cùng split v1, 26 token × 64 channel, 2 block, B-spline và các hệ số loss 0,5/1/3.

**Vì sao từng file xuất hiện:**

- `accuracy_reference.json`: thử head điểm tham chiếu với flatten và augmentation ban đầu; làm bước thiết kế trung gian.
- `accuracy_spatial.json`: thử nhãn điểm mềm theo khoảng cách, augmentation mạnh hơn và patience 20; là cấu hình được chọn để chạy đủ seed 42/43/44 trong báo cáo.
- `accuracy_mean_ablation.json`: giữ reference head nhưng lấy trung bình token, để xem hướng giữ thông tin token có hữu ích không.

**Cần tránh hai hiểu nhầm:**

1. Mean ablation là **reference head dùng mean**, không phải nguyên trạng `RSSIMixer` cũ hồi quy trực tiếp. Nó có ngân sách epoch/patience khác các file còn lại, nên phép so sánh đã chạy chưa phải ablation cô lập hoàn toàn công bằng.
2. Top-k/temperature trong JSON là khởi điểm khi train; checkpoint sau calibration có thể khác. Hãy đọc `calibration.json` hoặc `checkpoint.config`, không chỉ nhìn file JSON nguồn để mô tả decoder cuối.

`smooth_l1_beta` vẫn có trong schema cấu hình dùng chung của backend, nhưng nhánh reference không dùng SmoothL1 để tính coordinate loss. Đây không phải một loss SmoothL1 bổ sung bị bỏ quên.

### 4.11. `tests/test_reference.py`: kiểm tra tính đúng của cơ chế mới

File bổ sung **8 test methods**, dùng mô hình nhỏ và dữ liệu giả có chủ đích:

| Test | Tính chất được kiểm tra |
|---|---|
| Catalogue train | Điểm lấy đúng từ train, giữ class/xy; từ chối validation và mạng reference chưa fit |
| Decoder IPS | Shape logits/xy đúng, finite, xác suất lớp đúng cách tổng hợp, tọa độ trong giới hạn từng trục của lớp dự đoán; không nhận query labels |
| Gradient loss | Với nhãn cứng và mềm, các tham số backbone/head đều nhận gradient hữu hạn |
| Reference checkpoint | Save/load tái lập chính xác output, không cần mở CSV train |
| RSSI cho hybrid | AP thiếu khác tín hiệu rất yếu trong biểu diễn khoảng cách |
| Hybrid | Dùng lớp dự đoán, giữ đóng góp neural, từ chối alpha bằng 0 hoặc catalogue test |
| Ensemble checkpoint | Xác suất lớp là trung bình thành viên; wrapper ensemble/lai round-trip tái lập được |
| Final refit fixture | Train số epoch đã khóa, metadata phù hợp và chặn đánh giá validation đã nhập train |

**Vì sao cần:** những lỗi như đưa catalogue validation vào mạng hoặc dùng tầng thật khi dự đoán có thể làm kết quả đẹp lên nhưng sai giao thức. Tests tập trung vào hợp đồng đó và tương thích checkpoint, không đặt assertion rằng mô hình chắc chắn đạt 7–8 m.

**Giới hạn test:** kiểm min/max từng trục là kiểm giới hạn hộp của tọa độ, không phải một thuật toán kiểm hình học bao lồi tổng quát. Tính chất convex combination còn được hiểu từ công thức decoder. Fixture refit một epoch không chứng minh chất lượng refit trên dữ liệu thật.

## 5. Kiến trúc kế thừa có đổi không?

Không. Cấp thuật toán dùng chung vẫn là:

```python
class KANMixerAlgorithm(Algorithm):
    ...
```

Các lớp `RSSIMixer`, `ReferenceKANMixer`, `RefinedKANMixer`, `EnsembleReferenceKANMixer` là các **mạng/wrapper PyTorch bên trong nhánh KAN**, kế thừa `nn.Module`. Chúng không phải bốn thuật toán mới được đăng ký với Factory.

`ReferenceKANMixer` không kế thừa `RSSIMixer`; nó tái sử dụng `MixerBlock`. `RefinedKANMixer` chứa backbone; `EnsembleReferenceKANMixer` chứa danh sách các mạng. Đây là cách ghép thành phần, không phải thêm lớp `NeuralAlgorithm` vào giữa `Algorithm` và KAN.

Factory vẫn chọn ID `kan_mixer`. Thay model config hoặc checkpoint sẽ quyết định biến thể bên trong. LLM/HUIM tiếp tục có thể phát triển theo hợp đồng `Algorithm` riêng, không phải dùng optimizer, tensor hay reference catalogue của KAN.

## 6. Những file bên ngoài `kan` thay đổi để nối vào pipeline

Phần này giải thích liên hệ, không mở rộng danh sách 17 file của thư mục KAN.

| File ở root repo | Thay đổi thực tế | Vì sao cần |
|---|---|---|
| `src/ips/cli.py` | Lệnh `check` fit reference catalogue từ train khi variant là reference, trước `build_model()` | Mạng reference cần danh sách điểm để biết kích thước head |
| `tests/test_pipeline.py` | Import thêm `ReferenceTests` | `unittest discover -s tests` tìm được tests đặt trong folder KAN |
| `examples/kan_accuracy_validation.json` | Cấu hình Factory ID `kan_mixer`, model config spatial, seed42, CUDA, chỉ validation | Có ví dụ gọi nhánh mới qua Runner dùng chung |
| `README.md` | Bổ sung mô tả nghiên cứu, file/lệnh và giới hạn refit | Giúp nhóm hiểu lối chạy mới, giữ hướng dẫn cũ |

`check` vẫn là kiểm kỹ thuật và shape, không phải thí nghiệm đủ mẫu để kết luận chất lượng. Seed 42 trong file example không tự đổi thành 44; muốn chạy seed 44 qua Factory cần cấu hình seed tương ứng và output-dir mới.

## 7. Các file giữ nguyên có còn cần không?

| File/nhóm | Vì sao vẫn cần |
|---|---|
| `paths.py` | Đường dẫn cấu hình và ánh xạ đường dẫn lịch sử |
| `resources.py` | Đo RAM tiến trình, VRAM và đồng bộ CUDA cho thời gian |
| `diagnostics.py` | Chẩn đoán đầu vào từng KANLinear; cấu hình hiện tại đặt số batch lấy mẫu bằng 0, không có nghĩa file hết vai trò |
| `benchmark.py` | Đo suy luận, bộ nhớ; đã dùng trong 18 benchmark của báo cáo |
| `models/vendor/*` | Mã efficient-kan, license và provenance; backbone B-spline vẫn dùng chúng |
| `models/__init__.py` và các `__init__.py` | Điểm import/package; các lớp mới được import tại nơi cần nên không nhất thiết sửa tất cả file init |
| `configs/model.json` | Giữ mô hình cũ và khả năng tái lập/checkpoint lịch sử |
| `configs/experiment_smoke.json` | Lối kiểm tra Factory ngắn trên CPU |
| `tests/test_pipeline.py` trong KAN | Kiểm pipeline, axis mixing, gradient, vendor và tương thích cũ |
| `README.md` trong KAN | Hướng dẫn bố cục, loader v1/v2 và lối chạy mới; được cập nhật ngày 12/10 khi chuẩn bị commit |

`__pycache__`/`.pyc` nếu thấy trên máy là cache Python, không phải triển khai thuật toán mới và không dùng làm tài liệu review nguồn. Kết quả run, checkpoint, ảnh/bảng nằm trong `runs/`, không nằm lẫn với code KAN.

## 8. Từ file đến một lượt thực nghiệm thật

### Luồng thông thường đã dùng cho seed 44

1. Repository đọc CSV và split đã chốt. `FingerprintDataset.from_split()` biến thành tensor, giữ row order và tên split.
2. `train.py` fit điểm từ train; `build_model()` chọn `ReferenceKANMixer`; optimizer học trọng số và chọn checkpoint bằng validation.
3. `accuracy.py` lấy checkpoint đã hoàn tất, chọn decoder thuần/lai bằng validation và ghi `calibration.json`.
4. `runtime.py` nạp checkpoint thuần hoặc bọc refinement. CLI/Factory dự đoán RSSI rồi evaluator dùng nhãn thật để tính metric sau dự đoán.
5. Script báo cáo tổng hợp metric từng seed/split, xuất bảng/hình. Không có bước ensemble hoặc final refit trong kết quả ba seed này.

Lệnh minh họa dưới đây **không được chạy khi tạo tài liệu này**. Dùng tên thư mục mới nếu cần chạy lại:

```powershell
# Từ D:\IPS_ThacSi
& .venv-cuda/Scripts/python.exe -u -m ips train `
  --config src/ips/algorithms/kan/configs/accuracy_spatial.json `
  --device cuda --seed 44 --run-dir runs/seed44_review_new

& .venv-cuda/Scripts/python.exe -m ips.algorithms.kan.accuracy `
  --run-dir runs/seed44_review_new `
  --output-dir runs/seed44_review_calibration_new --device cuda

# Đây chỉ là kiểm tra validation. Chốt lựa chọn trước khi đánh giá test.
& .venv-cuda/Scripts/python.exe -m ips evaluate `
  --checkpoint runs/seed44_review_calibration_new/pure_selected.pt `
  --split validation --output-dir runs/seed44_review_validation_new --device cuda
```

`--pure-only` trên lệnh calibration bỏ bước tìm hybrid. Để xem kết quả vừa hoàn tất, không cần train lại; đọc các artifact có sẵn.

### Artifact kiểm chứng đang có

Các đường dẫn sau tính từ root repo:

- `runs/kan_seed44_complete_20261009_v1/README.md`: kết quả, protocol và file map.
- `training_config.json`, `protocol_lock.json` trong run mới: cấu hình giữ cố định trước lượt seed 44.
- `training/resolved_config.json`: cấu hình hiệu lực đã có catalogue fit từ train.
- `training/history.csv`, `summary.json`, `resources.json`: 57 epoch, best37, thời gian/RAM/VRAM.
- `calibration/calibration.json`, `pure_selected.pt`, `selected.pt`: lựa chọn decoder và checkpoint.
- `report/per_seed_results.json`, `aggregate_results.json`: số liệu 27 đánh giá, nguồn predictions và ma trận.
- `report/pre_evaluation_checkpoint_lock.json`, `verification.json`: hash khóa và kiểm đủ mẫu/thứ tự/metric.
- `references/Bao_cao_cai_tien_KAN_Mixer_UJIIndoorLoc_2026-10-09.docx`: báo cáo đầy đủ.

### Trạng thái kiểm chứng cần đọc đúng

- Bộ **35 tests đã pass trong audit trước** và được chạy lại thành công ngày 12/10 khi chuẩn bị commit. Lệnh `check` mạng cũ/reference và smoke Factory một epoch CPU cũng được kiểm tra. Tác vụ viết Markdown ngày 09/10 không sửa implementation.
- Seed 44 mới đã train lại hoàn tất; epoch1–44 khớp lượt gián đoạn trước. Đã đánh giá đủ mẫu và đo benchmark.
- Nguồn train seed42 khác hiện tại ở bốn file do sửa sau lượt đó; replay validation khớp rất sát. Không khẳng định toàn bộ training source của ba seed byte-identical.
- Split/metric lịch sử giữ nguyên, chỉ train fit các catalogue trong giao thức thông thường. Validation có nhiều vị trí đã được đo trong train, nên chưa chứng minh unseen-location/user generalization.
- Test đã được quan sát trong nghiên cứu trước. Các file calibration không tính test không đồng nghĩa toàn chiến dịch chưa từng xem test.

## 9. Review theo thứ tự nào để dễ hiểu?

1. Đọc bảng kết quả và `configs/accuracy_spatial.json` để biết biến thể thực sự đã chạy.
2. Đọc `dataset.py` rồi `fit_reference_points()` để hiểu các điểm được lấy từ đâu.
3. Đọc `MixerBlock` trong `networks.py`, sau đó `ReferenceKANMixer.forward()` để theo dõi tensor và tọa độ.
4. Đọc `combined_loss()` trong `train.py` để phân biệt mục tiêu train với suy luận.
5. Đọc `accuracy.py`, `refinement.py`, `runtime.load_checkpoint()` để hiểu hiệu chỉnh, bản lai và nạp lại.
6. Đọc `algorithm.py`, `predict.py`, tests để thấy cách nối vào IPS chung và bảo vệ giao thức.
7. Sau cùng mới đọc ensemble/refit/freeze; chúng là hướng mở rộng, không cần để giải thích kết quả seed44 hiện tại.

## 10. Những điều chưa nên kết luận

- Có `ensemble.py` không có nghĩa đã đo ensemble đạt bao nhiêu mét.
- Có `final_refit.py` không có nghĩa số liệu hiện tại được train từ 19.937 mẫu. Ba seed báo cáo vẫn train 16.025 mẫu.
- Có validation dưới 6 m không đồng nghĩa test dưới 6 m.
- Bản lai đạt trung bình 7,838 m không đồng nghĩa KAN thuần trung bình dưới 8 m, mọi seed dưới 8 m hay mọi mẫu dưới 8 m.
- Chưa có đối chứng cùng head/ngân sách để quy cải thiện riêng cho KAN; fingerprint thuần cũng còn cần đối chứng.
- Không có FastKAN/S-KAN hay triển khai IPS LLM/HUIM hoàn tất trong các thay đổi này.

Tài liệu chỉ mô tả mã và artifact có thật tại mốc kiểm tra. Những nhận xét về cơ chế cải thiện là lý giải thiết kế, còn đóng góp riêng của từng yếu tố cần thí nghiệm đối chứng.
