# Nhánh HUIM — EFIM chọn AP + k-NN vân tay

`HUIMKNNAlgorithm` (algorithm.py) kế thừa `HUIMAlgorithm` (adapter.py) và được đăng ký trong
`default_factory` với ID `huim`. Mã port từ nghiên cứu HUIM-IPS của Lê Thành Thái
(github.com/ThaiTaka/HUIM-IPS-), giữ nguyên định nghĩa, lưới ngưỡng và quy tắc chọn; chỉ thay
cách chia: ở đây validation nội bộ của repo là tập dùng để chọn.

## Repo trả lời các câu hỏi của hợp đồng thế nào

| Câu hỏi trong adapter cũ | Lựa chọn | Fit trên |
|---|---|---|
| Item, missing | item = cột AP (0..519), có mặt khi RSSI ≠ 100 | — |
| Binning RSSI | q(i,T) = round(1 + (RSSI − RSSI_min)/(RSSI_max − RSSI_min) × 99), 0 = missing | train |
| Utility | u(i,T) = q(i,T)·p(i); p(i) theo Model A/B/C (bên dưới) | train |
| Thuật toán, ngưỡng | EFIM (Zida et al. 2017); minUtil = θ × tổng utility, θ giảm 20% → 1%, dừng ở lần đầu chạm ngân sách, chia đôi tối đa 3 lần | train |
| Itemset → AP | union mọi AP trong HUI; top-n theo contrib(i) = Σ u(X)/\|X\| | train |
| Itemset → vị trí | k-NN trên vân tay train, chỉ các AP được chọn; tòa/tầng = phiếu có trọng số, xy = trung bình có trọng số | — |
| Chọn AP set, k, khoảng cách | các bước S0→S4 trên validation nội bộ | validation |
| Không khớp | vân tay không thấy AP nào được chọn → định vị bằng mọi AP thấy ở train, cùng cấu hình; bước chọn chấm điểm đúng pipeline này | train |
| Nhiều khớp | trọng số 1/dᵖ; khoảng cách bằng nhau xếp theo dòng train | — |
| Thời gian, bộ nhớ | resources.json: giây và peak RSS của từng lần khai phá; runtime của Runner cho predict | — |

p(i): **A** p = 1 (đối chứng); **B** p = max(I(A;Y), 10⁻⁶)/max I (I = mutual information
với lớp tòa/tầng, 6 trạng thái AP); **C** p ∝ I(A;Y)·log(N/(1+support)).

S0 positive + Manhattan, k = 1; S1 biểu diễn (positive, powed, exponential) × khoảng cách
(Euclid, Manhattan, Bray–Curtis, cosine) ở k = 1; S2 thêm k = 1…30 và w = 1/dᵖ, p = 0…4;
S3 dò lệch thiết bị ±3/±6/±9 dB; S4 ghép hai biểu diễn. S3/S4 chỉ được nhận nếu giảm sai số
validation ≥ min_gain (0,05 m). Ứng viên AP: union của mọi lần khai phá (lần bị dừng vì
ngân sách mang hậu tố `_truncated`) và top-n từ tập HUI hoàn tất giàu nhất.

## Chạy

```powershell
.\.venv\Scripts\python.exe -u -m ips run --config src/ips/algorithms/huim/configs/experiment_smoke.json
.\.venv\Scripts\python.exe -u -m ips run --config src/ips/algorithms/huim/configs/experiment_validation.json
```

Params (đều tùy chọn): `utility_model` A/B/C; `min_util_pct` lưới θ (%); `max_bisections`,
`rich_enough_huis`; `time_limit_s`, `max_huis` (ngân sách mỗi lần EFIM); `candidate_sets` gồm
`union`, `topn`, `all_detected` (thêm mọi AP thấy ở train làm ứng viên đối chiếu); `topn_budgets`;
`min_gain`; `fallback` = `all_detected` (mặc định) hoặc `none` (vân tay không khớp giữ vectơ toàn
missing trên các AP đã chọn, như PHASE I của nghiên cứu gốc). resources.json ghi số dòng validation
không thấy AP nào được chọn. Thêm `"test"` vào evaluation_splits khi muốn chấm test: execute đã chốt mọi lựa chọn
trên validation nên Runner cho phép (smoke thì không).

Kết quả trong output_dir: `algorithm/huim_selection.json` (AP được chọn, các bước S0–S4, mọi lần
khai phá kèm danh sách AP xếp theo contrib), `algorithm/huim_selection_grid.csv` (mọi cấu hình đã
chấm trên validation), `history.json`, `resources.json`, `model_diagram.mmd` cùng các file chung của Runner.

## Lưu ý

- Tọa độ và sai số theo đơn vị gốc của UJIIndoorLoc như phần còn lại của repo. Nghiên cứu gốc xác
  định đây là EPSG:3857 (Web Mercator); mét thực địa = giá trị × 0,766126 tại vĩ độ campus UJI.
- Kết quả k-NN phụ thuộc thứ tự cộng số thực khi tính khoảng cách. Giới hạn số luồng
  (OMP_NUM_THREADS, LOKY_MAX_CPU_COUNT) có thể đổi lựa chọn khi hai cấu hình gần bằng nhau.
- Trên các tập con smoke, nhiều lần quét cùng một điểm làm số HUI bùng nổ ngay dưới khoảng 10%,
  nên smoke dùng ngân sách nhỏ (`max_huis` 100000) và union bị cắt.
- Nghiên cứu gốc cho thấy tập AP do HUIM chọn không tốt hơn top-n theo mutual information hay theo
  số lần phát hiện ở cùng số AP; `candidate_sets` có `all_detected` để kiểm lại điều đó trên split này.

Tests: `tests/test_efim.py` (EFIM so với vét cạn, giá trị tính tay), `tests/test_algorithm.py`
(fit chỉ trên train, chọn chỉ trên validation, hợp đồng dự đoán, không khớp, Factory/Runner).
Hai file được nạp vào bộ test chung qua `tests/test_huim.py`.
