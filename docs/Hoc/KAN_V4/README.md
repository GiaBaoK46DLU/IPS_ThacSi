# KAN Mixer V4 trên UJIIndoorLoc

V4 `balanced_points` đạt **7,277383 ± 0,211021 m** trên 10 seed cố định, mỗi model
đánh giá đủ 1.111 dòng Test gốc. V3 `same_point_relative` đạt
**7,424524 ± 0,293599 m** theo cùng chế độ refit development. Mức giảm là
0,147141 m (1,9818%). **Chưa đạt mục tiêu trung bình dưới 7 m.**

## Nội dung để review

- [Báo cáo Word đầy đủ](IPS_KAN_V4.docx): 34 trang, thuật toán, cấu hình, công thức,
  so sánh V3/V4 theo 10 seed, tòa/tầng và số AP. V2 chỉ là mẫu trình bày thuật toán.
- [Kết quả Test của 10 model](official_test_summary.json): bản sao nguyên byte của
  tổng hợp cuối tại `runs/kan_v4_20261026_r2/official_test_summary.json`.
- [Xác nhận Validation](validation_summary.json): đủ 10 seed, best epoch,
  quy tắc khóa 41 epoch refit và so sánh với V3.
- [Receipt kiểm toán cuối](final_audit.json): đối chiếu 10 bộ trọng số khác nhau,
  11.110 dòng dự đoán, metric tọa độ gốc, phạm vi fit và hash nguồn/checkpoint.
- [Nhật ký và giải thích thay đổi](../../../src/ips/algorithms/kan/UPDATE_VERSION_4.md).
- [Cấu hình chạy](../../../src/ips/algorithms/kan/configs/v4_balanced_points.json).

## Những gì thay đổi

Thay đổi được chọn là `training.sampling = reference_points`: mỗi fingerprint ở
điểm có `n` lần quét nhận trọng số `1/n`, lấy có hoàn lại với cùng số lượt lấy
mẫu mỗi epoch. Backbone KAN-Mixer B-spline, loss và same-point mixing của V3 giữ
nguyên. Refit tăng từ 31 lên 41 epoch theo median best epoch Validation; không
quy toàn bộ mức giảm Test cho tác dụng riêng của sampler.

Các tùy chọn AP grouping và spatial-risk loss phục vụ đối chứng nghiên cứu;
chúng không được bật trong cấu hình `balanced_points`. Thiết kế AP-token bị
dừng vì không học hiệu quả và không được coi là một lượt accuracy hoàn tất.

## Phạm vi bằng chứng

Các checkpoint, CSV dự đoán, snapshot và dữ liệu raw vẫn giữ trong thư mục chạy
thử cục bộ theo quy ước repository; các receipt được công bố ở đây không mang
theo các artifact đó. Muốn chạy lại auditor phải có bộ artifact tương ứng tại
`runs/kan_v4_20261026_r2`. Receipt là bằng chứng của lần kiểm đã thực hiện, không
thay thế một lần tái lập độc lập trên máy khác.

Mười model dùng cùng 1.111 query Test; 11.110 dự đoán không phải 11.110 đối tượng
độc lập. Test đã có lịch sử được xem trong dự án. Một seed đạt 6,940479 m không
thay cho kết quả trung bình 10 seed. Không chọn lại seed/k/nhiệt độ sau Test.

Kiểm thử mã không cần chạy lại nghiên cứu:

```powershell
.\.venv-cuda\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -v
```

Kiểm lại kết quả đã lưu, không training hoặc inference mới:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 scripts/audit_kan_v4_results.py --output-dir runs/kan_v4_20261026_r2 --phase final
```

Nghiên cứu hữu hạn đã dừng để đánh giá lại; việc đưa nhánh này lên PR không
khởi động lại training và không thay đổi baseline V3.
