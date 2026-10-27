# KAN Version 4 — nghiên cứu UJIIndoorLoc

Ngày bắt đầu: 26/10/2026. Nhánh: `kan-version-4`.

Bản bàn giao để review: [báo cáo Word V4](../../../../docs/Hoc/KAN_V4/IPS_KAN_V4.docx)
và [kết quả tổng hợp cùng receipt kiểm toán](../../../../docs/Hoc/KAN_V4/README.md).
Checkpoint, dữ liệu và CSV trong `runs/` tiếp tục được giữ cục bộ theo quy ước repository.

Người dùng cho phép nghiên cứu và thực nghiệm nhằm đạt **mean2D Test < 7 m**.
Phạm vi là **một KAN-Mixer chung cho ba tòa, 13 lớp tòa/tầng**, 10 seed độc lập:
42, 43, 44, 100, 2024, 777, 999, 45, 46, 47. Không ensemble hoặc truy xuất fingerprint khi suy luận.

## Kết quả cuối ngày26/10/2026 — dừng để đánh giá lại

**Chưa đạt mục tiêu Test <7 m.** V4 `balanced_points`, đủ10 refit độc lập, đạt
**7,277383 ± 0,211021 m**, so với V3 **7,424524 ± 0,293599 m**.
Giảm **0,147141 m (1,9818%)**, vẫn cao hơn ngưỡng7 m **0,277383 m**.
Mỗi model chấm đủ1.111 dòng Test gốc, giữ nguyên khoảng cách Euclidean sau đảo chuẩn hóa.
Dấu± là sample SD giữa10 mean của các model; cùng1.111 query được dùng lặp lại,
không phải11.110 đối tượng Test độc lập.

| Chỉ số tổng hợp10 model | V3 | V4 |
|---|---:|---:|
| Mean2D, m | 7,424524 ± 0,293599 | 7,277383 ± 0,211021 |
| Trung bình median2D mỗi model, m | 4,522050 | 4,360606 |
| Trung bình P90 mỗi model, m | 15,562094 | 15,742108 |
| Trung bình RMSE mỗi model, m | 13,521332 | 12,682924 |
| Building accuracy, % | 99,765977 | 99,837984 |
| Floor accuracy, % | 93,420342 | 93,456346 |
| Joint building/floor accuracy, % | 93,384338 | 93,447345 |

Mean và median giảm, nhưng P90 tăng0,180014 m. Không mô tả cải thiện đồng đều ở mọi chỉ số.

| Seed đã đăng ký | V3 Test mean2D | V4 Test mean2D | V4−V3, m |
|---|---:|---:|---:|
|42|7,449044|7,499968|+0,050924|
|43|8,064226|7,365463|−0,698764|
|44|7,442032|7,450090|+0,008058|
|100|7,115416|7,080805|−0,034611|
|2024|7,539543|7,240229|−0,299313|
|777|7,343180|7,011597|−0,331582|
|999|7,162227|7,356053|+0,193827|
|45|7,361981|7,565631|+0,203650|
|46|7,678969|7,263517|−0,415452|
|47|7,088622|6,940479|−0,148143|

6/10 cặp cải thiện Test, so với9/10 cặp cải thiện Validation.
Seed47 đạt6,940479 m chỉ là kết quả mô tả sau Test, không được chọn thay cho trung bình10 seed.
Seed42 đã cố định làm model minh họa/triển khai trước Test đạt7,499968 m;
không đổi seed triển khai sang47 vì đã nhìn thấy Test.

### Năm hướng đã thử và quyết định

| Hướng | Bằng chứng Validation seed42 | Kết luận trong vòng này |
|---|---|---|
| AP-token | Dừng sau11 epoch do không học hiệu quả; head GELU bị sụp trên chẩn đoán train | Chưa có accuracy hoàn tất/calibration hợp lệ; không coi148,50 m là kết quả cùng ngân sách |
| Nhóm AP co-observation | 5,801703 m | Kém V3 seed42; loại |
| Joint spatial-risk loss | 5,718381 m | Kém V3 seed42; loại |
| Cân bằng vị trí tham chiếu | 5,542652 m seed42; đủ10 seed5,585354 ± 0,038662 m | Chọn chỉ bằng Validation; Test7,277383 ± 0,211021 m |
| Absolute + relative context | 5,809118 m | Kém V3 seed42; loại |

V3 Validation seed42 là5,582541 m. Thiết kế thành công thay phân phối lấy mẫu;
backbone KAN-Mixer, loss và same-point augmentation của V3 giữ nguyên.
Không tuyên bố đã phát minh một kiến trúc KAN mới.

### Kiểm toán, tái lập và phạm vi kết luận

- Nguồn/kế hoạch/cấu hình/15 artifact baseline được kiểm hash xuyên suốt;10 checkpoint refit
  được khóa trong `runs/kan_v4_20261026_r2/test_checkpoint_lock.json` trước Test.
- Kiểm toán độc lập đạt trên10 model /11.110 dòng dự đoán, với10 trạng thái trọng số khác nhau:
  `runs/kan_v4_20261026_r2/audits/final_20261026T151612036225Z.json`.
  Kiểm row order/truth, catalogue933RP từ development, preprocessing,41 epoch/seed,
  sampling, tọa độ gốc, mọi metric và nguồn/checkpoint. Không train/inference lại trong kiểm toán.
- 86 kiểm thử regression/invariant đã đạt; negative control của auditor từ chối CSV bị sửa tọa độ.
  Mã Python đã khóa không đổi trong suốt lượt; các script phân tích ngoài `src/ips/` không sửa model.
- Đây là so sánh toàn bộ pipeline: V3 refit31 epoch, V4 refit41 epoch theo cùng quy tắc chọn
  từ Validation. Chưa cô lập tác dụng trực tiếp của sampler khỏi tác dụng qua ngân sách refit.
- Test có lịch sử được xem trong các vòng trước. Không coi đây là một bộ blind chưa từng tiếp cận.
  Không đổi lựa chọn mô hình/k/T/epoch hoặc tạo vòng thử Test tiếp theo sau kết quả này.

Các lệnh đọc/kiểm lại artifact đã hoàn tất:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/audit_kan_v4_results.py --output-dir runs/kan_v4_20261026_r2 --phase final
.\.venv-cuda\Scripts\python.exe -B -X utf8 scripts/analyze_kan_v4_validation.py --output-dir runs/kan_v4_20261026_r2
```

### Cơ chế STOP và hướng tiếp theo chưa chạy

Dừng vòng hữu hạn đã đăng ký theo tiêu chí tài nguyên/tiến triển của người dùng:
4/5 hướng không cải thiện Validation, hướng còn lại chỉ giảm Test khoảng2%, chưa đạt7 m.
13 trial hoàn tất và10 refit đã ghi tổng17.897,10 s, khoảng4,97 giờ elapsed training;
chưa gồm lượt r1 bị dừng hoặc lượt47 dở. Thời gian gồm chờ/chia sẻ GPU, không phải GPU compute hours.
**Không kích hoạt ngưỡng5 hướng liên tiếp không cải thiện**, vì `balanced_points` đã cải thiện.
Không lấy việc dừng nghiên cứu làm bằng chứng goal đã hoàn thành.

Giới hạn có thể nằm ở khả năng chống lệch thiết bị/thời điểm mà cân bằng số fingerprint/RP
chưa trực tiếp giải quyết; đây là giả thuyết, chưa có ablation nhân quả chứng minh.
Hướng tiếp theo có căn cứ nhất là kiểm chứng lại AP-token với chuẩn hóa trước readout để xử lý
head GELU sụp đã quan sát, sau đó tách ablation mask-aware readout. Cần chọn bằng Validation
tách thiết bị và một giao thức đánh giá mới trước khi mở thêm nghiên cứu; **chưa chạy các hướng này**.

Báo cáo dừng: `runs/kan_v4_20261026_r2/stop_report.json`.
Đối chiếu yêu cầu: `runs/kan_v4_20261026_r2/completion_audit.json` ghi rõ mục tiêu chưa đạt.
Các mốc dưới đây là nhật ký lịch sử, không phải trạng thái worker hiện tại.

## Mốc xác nhận đủ 10 seed — 26/10/2026, 12:39 UTC

Thiết kế `balanced_points` hoàn tất đủ 10 lượt Validation: **5,585354 ± 0,038662 m**,
so với V3 cùng 10 seed **5,694870 m**; giảm **0,109516 m**, 9/10 cặp cải thiện.
Seed46 hơi xấu hơn và vẫn được giữ. Đây là cải thiện Validation, chưa chứng minh Test <7 m.

Seed47 chạy62 epoch, chọn checkpoint epoch42; calibration k10/T0,5,
Validation **5,557741 m**. Kiểm toán độc lập toàn bộ13 model hoàn tất /50.856 dòng
Validation đạt: `runs/kan_v4_20261026_r2/audits/validation_20261026T123940004168Z.json`.
Kiểm toán kiểm lại tọa độ gốc, row order, metric, catalogue, checkpoint và hash nguồn/baseline.

Khóa lựa chọn tại `runs/kan_v4_20261026_r2/selection_lock.json`, SHA256
`c1e5df5b37dd642c6741cb66f30e6b355c4a532df94985e182e5632d74bd5457`.
Best epoch theo thứ tự seed đã đăng ký: **37,16,40,56,27,44,69,35,47,42**;
median là41, nên cả10 refit dùng **41 epoch**, giữ cosine horizon100 như kế hoạch.
Worker20064 đã bắt đầu refit seed42; Test chỉ được đánh giá sau khi khóa đủ10 checkpoint refit.
Không thay k/T, số epoch hoặc chọn seed dựa trên Test.

Thiết kế được chọn thay **phân phối lấy mẫu huấn luyện**, giữ backbone/loss V3;
không gọi đây là kiến trúc KAN mới hoặc đóng góp mới đã được chứng minh.
Mean Validation của nhóm tọa độ chưa có trong train là **6,999582 m**;
không suy diễn mức giảm full Validation thành bằng chứng tăng khả năng khái quát hóa ở mọi nhóm.
Nhật ký dưới đây giữ các mốc cũ theo thời điểm; trạng thái worker phải kiểm từ tiến trình đang chạy.

### Phân tích cặp Validation đủ10 seed trong khi refit

Script `scripts/analyze_kan_v4_validation.py` chỉ đọc development và CSV Validation đã lưu,
kiểm row/truth/Euclidean/điểm số trước khi phân tích; không train, inference hoặc mở dữ liệu Test.
Receipt: `runs/kan_v4_20261026_r2/paired_validation_ten_20261026T124406Z.json`.
Các bảng là trung bình sai số của10 mạng độc lập, không trung bình tọa độ để tạo ensemble.

| Nhóm Validation | Dòng/model | V3 mean2D | V4 mean2D | V4−V3 |
|---|---:|---:|---:|---:|
| Toàn bộ, chỉ số chính | 3.912 | 5,694871 | 5,585354 | −0,109516 |
| Tọa độ đã có trong train | 3.053 | 5,302998 | 5,187443 | −0,115555 |
| Tọa độ chưa có trong train | 859 | 7,087637 | 6,999582 | −0,088055 |
| Không nghe được AP | 15 | 71,519236 | 65,850188 | −5,669047 |
| Nghe1–9 AP | 365 | 8,222448 | 8,271569 | +0,049121 |
| Nghe10–19 AP | 2.258 | 5,103906 | 4,987237 | −0,116670 |
| Nghe ít nhất20 AP | 1.274 | 5,243117 | 5,166287 | −0,076830 |

Đối chứng chỉ các dòng có AP cho chênh lệch −0,088117 m. Đây là chẩn đoán,
không loại15 dòng khỏi chỉ số chính hoặc thay cohort Test. Khoảng20% mức giảm tổng
đến từ15 dòng không AP; không nên coi toàn bộ giảm trung bình là cải thiện đồng đều.

Bootstrap mô tả2.000 lần, seed20261026, trên359 nhóm đo: khoảng percentile95%
của V4−V3 khi cố định10 seed là **[−0,200350; −0,013554] m**.
Khi lấy mẫu lại cả10 cặp seed và nhóm đo: **[−0,231094; +0,016836] m**, vẫn chứa0.
Đây là phân tích sau lựa chọn, chưa hiệu chỉnh selection bias, không phải gate mới hoặc bằng chứng Test.
V3 dùng CSV CPU replay có sẵn; khác mean so với báo cáo selection GPU dưới0,000001 m
ở từng seed, được kiểm và lưu trong receipt. Nguồn dữ liệu,20 CSV và script đều có SHA256.

## Baseline được bảo toàn

V3 `same_point_relative`, refit development 31 epoch: **7,424524 ± 0,293599 m**,
đủ 1.111 dòng test/model. Giữ nguyên `runs/kan_joint_under6_20261020_v1/`.
Mã, cấu hình, checkpoint và kết quả cũ không được gán lại cho V4.
Hai tệp Markdown untracked có sẵn trong `docs/architecture/` thuộc trạng thái ban đầu và được giữ nguyên.

V4 bắt đầu từ commit `8b77292` trên main, đã chứa bản merge V3 và thay đổi RF/AutoLoC của thành viên khác.
Diff mã KAN/data/metrics so với V3 `66f276a` rỗng trước khi triển khai V4.

## Giả thuyết và thí nghiệm tách biệt

Mọi thiết kế xuất phát từ cấu hình V3 đã chọn, giữ split, loss/augmentation và lịch train trừ thành phần được ghi rõ.

| Thứ tự | Thiết kế | Thành phần thay đổi | Giả thuyết / rủi ro |
|---|---|---|---|
| 1 | `ap_tokens` | Mỗi AP là một token, 520 token × 16 channel; head 128 để khống chế dung lượng | Channel mixing chia sẻ cùng ngữ nghĩa RSSI/mask giữa token. Đây là gói kiến trúc; thay dung lượng là yếu tố nhiễu cần báo cáo. Có CPU smoke lịch sử, chưa có kết quả accuracy hợp lệ của thiết kế này. |
| 2 | `coobservation_patches` | Ghép AP theo cây phân cụm cosine của mask train, giữ 26×20 AP và số tham số | Đưa các AP có vùng phủ đồng xuất hiện vào cùng patch. Không biết tọa độ AP, không khẳng định cụm tương đương vị trí vật lý. |
| 3 | `joint_spatial_risk` | Thêm kỳ vọng khoảng cách tới mọi RP vào loss, hệ số 1 trong hệ tọa độ chuẩn hóa | Phạt khối xác suất ở tọa độ xa thuộc mọi lớp; loss XY hiện tại chỉ chọn nhánh lớp thật. Rủi ro làm phân phối quá tập trung. |
| 4 | `balanced_points` | Lấy mẫu đều RP thay vì đều fingerprint, cùng số lượt cập nhật | Train có 2–80 fingerprint/RP. Giảm ưu tiên vị trí đo nhiều; có thể làm lệch phân phối thực tế nên metric vẫn tính trên mọi dòng. |
| 5 | `absolute_relative_context` | Thêm tín hiệu tuyệt đối vào RSSI tương đối + mask, giữ same-point mixing V3 | Khôi phục ngữ cảnh cường độ bị phép trừ trung bình loại bỏ. Có thể học lại lệch thiết bị; đối chứng cũ không chứa đồng thời same-point protocol được chọn. |

Đây là giả thuyết, chưa phải đóng góp đã được chứng minh. Không gọi chúng là tái lập nguyên thông số bài báo.

## Cơ sở tham khảo

- [MLP-Mixer](https://arxiv.org/abs/2105.01601): phân tách token/channel mixing. Ngữ nghĩa token AP là suy luận ứng dụng của V4.
- [AP identity và RSSI dưới dạng token](https://arxiv.org/abs/2506.00656): cơ sở xem từng AP là một đơn vị biểu diễn. V4 vẫn dùng KAN-Mixer, không thêm Transformer.
- [Learning with a Wasserstein Loss](https://papers.nips.cc/paper_files/paper/2015/file/a9eb812238f753132652ae09963a05e9-Paper.pdf): học phân phối có ground metric. V4 dùng trường hợp target là một điểm: `sum_j p_j * ||a_j-y||`; không triển khai Sinkhorn hoặc tự thêm phạt chiều cao.

## Giao thức đã xác định trước kết quả

- Giữ nguyên train/validation/test 16.025/3.912/1.111 và toàn bộ hash nguồn.
- Chỉ fit AP grouping trên train nội bộ; giữ permutation này khi refit. Catalogue RP mới được mở rộng sau khóa lựa chọn như V3.
- Sàng lọc tối đa năm thiết kế trên seed 42. Checkpoint và k/T chọn bằng toàn Validation theo pipeline cũ.
- Sau **năm thiết kế liên tiếp không cải thiện full Validation ít nhất 0,01 m**, dừng tính toán và báo cáo đánh giá lại.
- Tối đa hai thiết kế cải thiện full Validation được xác nhận trên 42/43/44, xếp theo score cân bằng tọa độ đã có/chưa có trong train.
- Chỉ chọn ứng viên cải thiện cả mean full Validation và score cân bằng so với V3 cùng ba seed. Sau đó chạy đủ 10 seed, yêu cầu mean Validation cải thiện V3 trước khi refit/test.
- Refit epoch là median làm tròn của best epoch từ đủ 10 lượt Validation, giữ cosine horizon 100. Khởi tạo lại từng seed và train development; không chấm Validation cũ sau refit.
- Khóa đủ 10 checkpoint refit trước Test; tính mean và sample SD của 10 mean2D. Không chọn seed thuận lợi.
- Sau Test không chỉnh lại kiến trúc/k/T để thử lại trên cùng Test trong nghiên cứu hữu hạn này. Nếu không đạt mục tiêu, báo cáo trung thực và dừng đánh giá lại.
- Test đã được xem ở các vòng cũ; không mô tả kết quả V4 như đánh giá trên bộ hoàn toàn chưa từng được tiếp cận.

## Tái lập và trạng thái

Controller: `python -m ips.algorithms.kan.research_v4` (`prepare`, `run`).
Artifact đang chạy: `runs/kan_v4_20261026_r2/`; bản r1 được giữ tại `runs/kan_v4_20261026/`.
Mỗi revision gồm protocol/cấu hình/hash/snapshot, log từng trial,
CSV Validation, khóa lựa chọn, refit và kết quả Test nếu qua các gate.
Không tự sửa kế hoạch/nguồn đã khóa khi worker đang chạy.

Kiểm thử trước sàng lọc: 85 tests đạt (23,083 s), log `runs/kan_v4_preflight_20261026.log`.

Thiết kế `ap_tokens` đã dừng vì không học hiệu quả sau 11 epoch hoàn tất:
Validation chưa calibration tốt nhất 148,498928 m, accuracy ghép tòa/tầng luôn 13,4458%;
khoảng 30 s/epoch. Đây là kết quả **bị kiểm duyệt do chi phí/tính khả thi**, không phải kết quả
đủ ngân sách hoặc sau calibration. Không gọi nó là một trial train hoàn tất.
Giữ nguyên protocol/snapshot/log/checkpoint dở ở `runs/kan_v4_20261026/`, kèm `resource_abort.json`.

Trước khi chạy bốn thiết kế còn lại, lập protocol revision mới tại `runs/kan_v4_20261026_r2/`.
Mã/kế hoạch r1 không bị sửa; r2 lưu hash receipt r1, đánh dấu AP-token là thiết kế không cải thiện,
và vẫn chỉ có tổng cộng năm giả thuyết ban đầu. Không dùng kết quả Test để sửa kế hoạch.
Controller r2 thêm khả năng mang receipt dừng do tài nguyên, không tạo điểm accuracy giả cho lượt dở.
Chưa có kết quả Test V4. Chưa đạt mục tiêu dưới 7 m.

### Quan sát trong sàng lọc

- Revision r2: sáu kiểm thử V4 đạt, gồm gate từ chối finalize khi thiếu seed hoặc sai winner.
- Kiểm tra tương thích trên 13 fingerprint train: logits và XY từ checkpoint V3 giống tuyệt đối
  giữa mã hiện tại và snapshot V3 (maximum absolute difference bằng 0).
- `coobservation_patches`, seed42: chạy 46 epoch, checkpoint epoch26;
  sau calibration k1000/T0,5, full Validation **5,801703 m**, score cân bằng **6,320531**.
  Baseline tương ứng là 5,582541 m / 5,961530. Thiết kế này chưa cải thiện.
  Số tham số giữ 1.014.568. Co-observation cosine trong patch tăng 0,093398→0,258177
  trên train; cơ chế nhóm hoạt động nhưng chưa chứng minh lợi ích về accuracy.
- Chẩn đoán lượt AP-token bị dừng, trên 128 dòng train: representation vẫn biến thiên,
  nhưng mọi đầu vào của GELU ở head checkpoint epoch2 nằm trong [-46,70; -9,16],
  đầu ra GELU bằng 0 trong float32; logits RP không thay đổi giữa query và luôn chọn lớp11.
  Đây là bằng chứng trực tiếp về sụp head trong cấu hình đã thử, không phải bằng chứng
  bác bỏ mọi cách token hóa một AP/token. Chuẩn hóa/định tâm trước head là giả thuyết
  cần nghiên cứu riêng nếu mở đợt tiếp theo; chưa triển khai để cứu một trial sau kết quả.
- Bằng chứng cơ chế và tương thích lưu tại `runs/kan_v4_20261026_r2/` dưới các tên
  `ap_grouping_diagnostic.json`, `ap_token_failed_learning_diagnostic.json`,
  `baseline_forward_compatibility.json`. Không dùng dữ liệu held-out cho các chẩn đoán này.
- `joint_spatial_risk`, seed42: 41 epoch, best epoch21, k10/T0,5;
  full Validation **5,718381 m**, balanced score **6,294264**, joint accuracy **99,4376%**.
  Có cải thiện lỗi trong các nhóm nghe từ 10 AP trở lên, nhưng nhóm 1–9 AP và nhóm
  không nghe AP xấu hơn; full Validation không cải thiện. Phân tích nhóm chỉ là
  chẩn đoán, không thay metric và không bỏ mẫu khó. Xem `validation_error_analysis.json`.
- `balanced_points`, seed42: 57 epoch, best epoch37, k10/T0,5;
  full Validation **5,542652 m**, cải thiện **0,039889 m** so với V3 seed42.
  Seen-coordinate **5,198040 m**, unseen-coordinate **6,767450 m**, balanced score **5,982745**.
  Unseen-coordinate và balanced score hơi xấu hơn V3 seed42; mức giảm full Validation
  chưa chứng minh cải thiện khái quát hoặc Test. Đủ điều kiện xác nhận ba seed theo
  kế hoạch, chưa được chọn làm winner. Chưa có Test V4 hoặc kết quả 10 seed V4.
- `absolute_relative_context`, seed42: 43 epoch, best epoch23, k10/T0,5;
  full Validation **5,809118 m**, balanced score **6,383463**; không cải thiện.
- Đã đóng sàng lọc năm hướng: bốn trial hoàn tất và một trial bị dừng do không học hiệu quả.
  Chỉ `balanced_points` qua gate full Validation seed42; bắt đầu xác nhận 43/44.
  Không có năm hướng liên tiếp không cải thiện, vì `balanced_points` giảm full Validation;
  bộ đếm thất bại được đặt lại đúng theo protocol. Điều này không đồng nghĩa đã có winner.
- Bootstrap thăm dò ghép cặp trên 359 nhóm đo Validation (2.000 mẫu bootstrap,
  seed20261026): chênh lệch `balanced_points − V3` là -0,039889 m, khoảng percentile95%
  [-0,268654; 0,206540] m. Khoảng này bao gồm0, nên một seed chưa chứng minh cải thiện ổn định.
  Đây là chẩn đoán bổ sung, không tạo gate lựa chọn mới sau kết quả; xem
  `paired_validation_uncertainty.json`.
- Xác nhận `balanced_points`, seed43: 36 epoch, best epoch16, k20/T0,5;
  full Validation **5,604308 m**, balanced score **6,101534**.
  So với V3 cùng seed (5,660620 m / 6,323923), cả hai chỉ số giảm.
  Seed44 đang chạy; chưa có winner hoặc kết quả Test V4.

### Cách hiểu lấy mẫu đều theo điểm

Với N điểm tham chiếu và n_r fingerprint ở điểm r, sampler có kỳ vọng tối ưu
`(1/N) * sum_r [(1/n_r) * sum_i_in_r L_i]`, thay vì lấy trung bình đều trên mọi fingerprint.
Mỗi epoch vẫn có 16.025 lượt lấy mẫu và cùng batch/optimizer/loss/augmentation với V3;
sampling có hoàn lại, nên số fingerprint phân biệt trong một epoch không bằng 16.025.
Đây là thay đổi thuật toán lấy mẫu huấn luyện, không phải thêm layer hoặc đổi learning rate.
Metric Validation/Test vẫn lấy trung bình trên mọi dòng theo giao thức gốc.

Nếu ứng viên được chọn, epoch refit được suy ra lại từ đủ10 best epoch Validation theo
quy tắc cũ. Một mức giảm Test khi đó là hiệu quả của toàn pipeline, bao gồm sampling và
ngân sách refit được quy tắc lựa chọn tạo ra; không tự quy toàn bộ mức giảm cho sampling riêng lẻ.

## Trạng thái mới nhất — sau xác nhận ba seed

`balanced_points` được khóa làm kiến trúc/phương pháp thắng của đợt sàng lọc này:

| Chỉ số Validation, trung bình seed42/43/44 | V3 | V4 balanced_points |
|---|---:|---:|
| Full mean2D | 5,657807 m | **5,574536 m** |
| Score cân bằng seen/unseen | 6,119681 | **6,067398** |
| Mean2D ở tọa độ chưa có trong train | 6,943224 m | 6,946191 m |

SD mẫu của full Validation V4 qua ba seed: **0,030882 m**.
Cả full mean và score cân bằng qua gate đã khóa (giảm tối thiểu0,01 m).
Lỗi tại tọa độ chưa thấy gần như giữ nguyên; chưa có bằng chứng cải thiện thành phần này.
Không gọi kết quả Validation này là kết quả Test hoặc bằng chứng đã đạt dưới7 m.

Đã hoàn tất seed42/43/44; đang train seed100, sau đó2024/777/999/45/46/47.
Controller tự kiểm mean Validation đủ10 seed trước khi cho phép refit, khóa checkpoint và Test.
Goal vẫn hoạt động; **chưa có Test V4**. Mục tiêu Test<7 m chưa đạt/chưa được chứng minh.

Nguồn trạng thái: `runs/kan_v4_20261026_r2/architecture_lock.json`,
`confirmation_three.json`, `confirmation_readable_summary.json`, `live_observation.json`.
Worker hiện được theo dõi bằng exec session1568; cần poll handle hoặc kiểm process thật trước khi tiếp tục.
Không khởi động lại chỉ vì tệp ghi chú hoặc một lần quan sát hết thời gian.

### Kiểm toán bổ sung trong lúc xác nhận đủ10 seed

- Bộ kiểm thử hiện tại: **86 tests đạt trong17,437 s**, log
  `runs/kan_v4_20261026_r2/audits/regression_suite.log`.
- Công cụ độc lập `scripts/audit_kan_v4_results.py` kiểm CSV/checkpoint đã lưu, không forward model.
  Phase Validation không mở nhãn Test; phase final yêu cầu có summary đủ10 seed và khóa checkpoint trước Test.
- Đã kiểm **6 model/23.472 dòng Validation**: row index, dòng CSV gốc, tọa độ/nhãn thật,
  số AP, sai số Euclidean, mean/median/P90/RMSE, accuracy, macro-F1, ba ma trận nhầm lẫn,
  catalogue train-only, hash nguồn và checkpoint đều khớp.
- Negative control: cố ý dịch một pred_x thêm1 m trong bản sao tạm của CSV; auditor từ chối đúng.
  Bản sao đã được dọn trong thư mục tạm riêng, không thay artifact gốc.
- Receipt: `audits/validation_20261026T042109535351Z.json` và `audits/auditor_negative_control.json`.
  Đây là kiểm toán phần mềm/dữ liệu, không phải thí nghiệm accuracy mới hoặc bằng chứng Test<7 m.

### Tiến độ xác nhận đủ10 seed — sau seed100

- Hoàn tất42/43/44/100. Seed100: 76 epoch, best epoch56, k20/T0,5;
  Validation **5,567222 m**, balanced score **6,117588** (V3 cùng seed:5,740441 m).
- Trung bình bốn seed đầu theo thứ tự đã đăng ký: **5,572708 ± 0,025479 m**;
  V3 cùng bốn seed: **5,678465 m**. Đây là kết quả trung gian, chưa thay báo cáo đủ10 seed.
- Worker exec session1568 đang chạy2024, tiếp theo777/999/45/46/47.
- 86 tests hiện tại đã đạt; source/config/protected-baseline guards còn nguyên.
  Chưa có refit hoặc Test V4; goal Test<7 m tiếp tục hoạt động và chưa được chứng minh.

### Tiến độ sau seed2024

- Hoàn tất42/43/44/100/2024. Seed2024:47 epoch, best epoch27, k20/T0,5;
  Validation **5,612167 m**, balanced score **6,121332** (V3 cùng seed:5,652697 m).
- Trung bình năm seed đầu: **5,580600 ± 0,028254 m**;
  V3 cùng năm seed: **5,673312 m**. Cả năm cặp đều giảm full Validation;
  đây vẫn là kết quả trung gian, không thay kết quả đủ10 seed hoặc Test.
- Worker exec session1568 đang chạy777; còn999/45/46/47 sau đó.
  Goal đang hoạt động, mã/cấu hình giữ nguyên; chưa refit hoặc Test V4.

### Tiến độ sau seed777

- Hoàn tất42/43/44/100/2024/777. Seed777:64 epoch, best epoch44, k20/T0,5;
  Validation **5,604046 m**, balanced score **6,164602** (V3 cùng seed:5,802873 m).
- Trung bình sáu seed đầu: **5,584507 ± 0,027023 m**;
  V3 cùng sáu seed: **5,694905 m**. Cả sáu cặp giảm full Validation.
  Chưa dùng số liệu trung gian để thay kết quả đủ10 seed hoặc kết luận Test.
- Worker exec session1568 đang chạy999, sau đó45/46/47. Cấu hình vẫn giữ nguyên.
  Goal tiếp tục hoạt động; chưa refit hoặc Test V4, chưa chứng minh Test<7 m.

### Tiến độ sau seed999

- Hoàn tất42/43/44/100/2024/777/999. Seed999:89 epoch, best epoch69, k20/T0,5;
  Validation **5,542068 m**, balanced score **6,072723** (V3 cùng seed:5,737755 m).
- Trung bình bảy seed đầu: **5,578445 ± 0,029425 m**;
  V3 cùng bảy seed: **5,701027 m**. Cả bảy cặp giảm full Validation.
  Đây là số liệu trung gian của giao thức10 seed, không phải Test hoặc kết quả chọn seed thuận lợi.
- Worker exec session1568 đang chạy45; còn46/47. Cấu hình vẫn giữ nguyên.
  Goal đang hoạt động; chưa có refit hoặc Test V4, chưa chứng minh Test<7 m.

### Tiến độ sau seed45

- Hoàn tất42/43/44/100/2024/777/999/45. Seed45:55 epoch, best epoch35, k10/T1;
  Validation **5,577112 m**, balanced score **6,057134** (V3 cùng seed:5,619543 m).
- Trung bình tám seed đầu: **5,578278 ± 0,027246 m**;
  V3 cùng tám seed: **5,690841 m**. Cả tám cặp giảm full Validation.
  Đây vẫn là số liệu trung gian, không thay báo cáo đủ10 seed hoặc Test.
- Worker exec session1568 đang chạy46, sau đó47. Cấu hình vẫn giữ nguyên.
  Goal tiếp tục hoạt động; chưa refit/Test V4, chưa chứng minh Test<7 m.

### Cấu hình tái lập được đưa vào phần public

`configs/v4_balanced_points.json` là bản sao byte-identical của thiết kế đã đăng ký
`runs/kan_v4_20261026_r2/configs/balanced_points.json` (SHA256
`71de153ac8609129deacd5baa336ddaf95ac9001ec917b55f5435856a3936a65`).
Không chứa weights hoặc catalogue; lệnh train tự fit catalogue bằng train nội bộ.
Ví dụ tái lập một seed trong thư mục mới:

```powershell
.\.venv-cuda\Scripts\python.exe -B -m ips train --config src/ips/algorithms/kan/configs/v4_balanced_points.json --seed 42 --device cuda --run-dir runs/v4_reproduction_seed42_new
```

Lệnh trên chỉ train/chọn checkpoint bằng Validation, không thay quy trình Test/refit đã khóa.
Pipeline đang chạy vẫn dùng các JSON riêng trong thư mục thực nghiệm; không đổi đầu vào giữa lượt.

### Sau seed46 và phục hồi gián đoạn seed47

- Seed46 hoàn tất67 epoch, best epoch47, k10/T1: Validation **5,669576 m**,
  balanced score **6,108527**. V3 cùng seed là5,665361 m: V4 cao hơn khoảng0,004215 m.
  Không loại lượt hơi xấu hơn để làm đẹp trung bình.
- Chín seed hoàn tất42/43/44/100/2024/777/999/45/46:
  **5,588422 ± 0,039695 m**, so với V3 cùng chín seed **5,688010 m**.
- Handle1568 không còn và kiểm tra hệ điều hành không có research worker/Python còn chạy.
  Seed47 dừng sau2 epoch, không có failure.json hoặc summary.json. Nguyên nhân chưa xác định;
  không quy kết OOM, phần cứng hoặc lỗi mô hình chỉ từ hiện tượng này.
- Đã kiểm toán12 model hoàn tất /46.944 dòng Validation, receipt
  `audits/validation_20261026T120859216883Z.json`: tất cả metric, dữ liệu, catalogue và hash khớp.
- Giữ nguyên8 tệp lượt47 dở trong `interrupted_trials/seed47_20261026T120923Z`;
  hash sau di chuyển khớp trước di chuyển. Receipt `recovery_seed47_20261026T120923Z.json`.
- Chạy lại riêng47 từ đầu cùng mã/config/seed. Prefix2 epoch khớp **0 sai khác số học**
  ở mọi trường ngoài thời gian; resolved config byte-identical. Không dùng điểm số lượt dở
  để chọn checkpoint hoặc thay kết quả47 hoàn tất sau này. Receipt `recovery_seed47_prefix_audit.json`.
- Worker mới exec session **20064**, PID được kiểm tra71988/75684, đang train47.
  Chín seed đã hoàn tất giữ nguyên; chưa có refit/Test V4. Goal Test<7 m vẫn hoạt động và chưa đạt.

### Refit đầu tiên hoàn tất

- Seed42 refit đủ41 epoch trên19.937 dòng development, không dùng tập đánh giá Validation cũ;
  thời gian686,04 s. Checkpoint SHA256
  `f8b9c167a56f96fcb1ec0c3cfc51a6a140e1b5177f82d256ea8d94f35946176d`.
- Kiểm toán metadata/history/checkpoint đạt: đúng seed/41 epoch, train19.937/validation0,
  đúng sampling đã chọn và khóa lựa chọn, mọi tensor trọng số float hữu hạn;
  toàn bộ hash nguồn và baseline bảo toàn. Receipt `audits/refit_seed42_provenance.json`.
- Worker20064 đã chuyển sang seed43. Còn9 refit; chưa Test V4.
  Các kiểm tra trên không thay thế kiểm toán độc lập toàn bộ10 model và1.111 dòng Test/model khi hoàn tất.
