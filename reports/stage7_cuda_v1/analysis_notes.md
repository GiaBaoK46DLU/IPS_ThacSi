# Diễn giải kết quả thật của chặng 7

Đọc cùng [bảng kết quả](README.md), [JSON tổng hợp](aggregate_metrics.json) và [CSV theo nhóm](group_metrics.csv). Các số dưới đây lấy từ suite CUDA đã hoàn tất; nhận xét nhân quả/đề xuất được phân biệt với quan sát. Không có thực nghiệm bổ sung sau khi xem test.

## 1. Mục tiêu và so sánh ba mô hình

**Quan sát:** mean2D test là MLP15,80±0,43 m, MLP-Mixer15,52±1,75 m và KAN-Mixer15,73±1,51 m. Cả9 checkpoint đều chưa đạt mean≤8 m. Chỉ khoảng25–28% mẫu nằm trong bán kính8 m; khi yêu cầu đồng thời đúng tòa/tầng, tỷ lệ khoảng24–27%.

MLP-Mixer có mean2D và joint accuracy tốt nhất trong bảng trung bình:15,52 m và92,56%. Tuy nhiên, chênh lệch mean2D ghép theo seed so với MLP chỉ−0,28±1,35 m; KAN-Mixer là−0,07±1,42 m. Chênh lệch nhỏ hơn biến thiên quan sát giữa các seed; không có kiểm định/CI để kết luận ưu thế thống kê.

KAN-Mixer có median2D thấp hơn MLP (11,69 so với13,09 m), nhưng RMSE2D cao hơn (23,32 so với19,37 m). **Diễn giải:** một mức lỗi thường gặp tốt hơn có thể đi cùng những lỗi cực lớn hơn; không chọn chỉ median để tuyên bố KAN tốt nhất. KAN có MAE tọa độ validation7,87 m, nhưng đó không phải mean2D test và không đáp ứng mục tiêu đã khóa.

## 2. Validation và test khác nhau

**Quan sát:** mean2D validation lần lượt12,88/12,64/12,25 m, còn test15,80/15,52/15,73 m. Building accuracy test vẫn khoảng99,2–99,6%, nhưng floor accuracy chỉ90,8–92,7%, thấp hơn khoảng98,2–98,4% ở validation.

**Suy luận có giới hạn:** mức giảm phù hợp với khả năng khác phân bố giữa các đợt đo/thiết bị và split validation còn chung người/điểm với train. Các bảng này chưa xác định nguyên nhân riêng; không đủ để kết luận điện thoại, thời gian hay kiến trúc là nguyên nhân duy nhất. Không có dấu hiệu cần bỏ các mẫu khó để sửa báo cáo.

## 3. Building/floor khó và số mẫu

Mean±SD qua ba seed; nhãn bên trái là building/floor thật:

| Nhóm test | Số mẫu | MLP mean2D (m) | MLP-Mixer mean2D (m) | KAN-Mixer mean2D (m) |
|---|---:|---:|---:|---:|
| 1/0 | 30 | 16,79±2,66 | 19,73±1,70 | 24,41±8,11 |
| 1/1 | 143 | 18,16±2,10 | 21,40±4,28 | 22,98±5,77 |
| 2/4 | 39 | 18,20±0,37 | 22,02±3,76 | 23,22±3,40 |

**Quan sát:** KAN-Mixer joint accuracy ở1/0 chỉ70,00%, ở1/1 khoảng72,96%. Nhóm2/4 vẫn có joint accuracy88,89% nhưng lỗi tọa độ23,22 m. Như vậy lỗi IPS không chỉ là nhầm nhãn tầng; hai đầu ra phải được cải thiện/đánh giá riêng. Nhóm30/39 mẫu có support nhỏ hơn nhóm143 mẫu, nên cần thận trọng với độ ổn định của kết luận nhóm.

Không phải mọi nhóm đều kém hơn với Mixer: hình theo building/floor cho thấy KAN cải thiện rõ so với MLP ở0/0 nhưng kém hơn ở1/0 và1/1. Mean toàn tập cân theo số mẫu, nên thứ hạng toàn tập không phản ánh đồng nhất mọi tầng.

## 4. Người đo, điện thoại và AP count

**Người đo:** test USERID0 không được ghi nhận, nên không có kết luận theo từng người trên test. Trên validation, USERID8 có141 mẫu và lỗi MLP/MLP-Mixer/KAN-Mixer lần lượt23,98/26,65/25,18 m, cao hơn mean validation toàn tập. Đây là mô tả nhóm; chưa tách được ảnh hưởng người, địa điểm, điện thoại hay tín hiệu.

**Điện thoại:** PHONEID14 có28 mẫu test; lỗi MLP19,12±0,18 m và KAN28,12±2,66 m. Trong khi PHONEID13 có369 mẫu và mean MLP/KAN đều khoảng15 m. Không kết luận PHONEID14 tự nó gây lỗi: hai nhóm có thể được đo ở các tòa/tầng khác nhau và support chênh lệch lớn. PHONEID0 vẫn là nhóm thiết bị hợp lệ với120 mẫu.

**AP count:**

| Số AP quan sát | Số mẫu test | MLP mean2D (m) | MLP-Mixer mean2D (m) | KAN-Mixer mean2D (m) |
|---|---:|---:|---:|---:|
| 1–9 | 154 | 21,67 | 25,53 | 23,15 |
| 10–19 | 632 | 14,52 | 13,54 | 13,96 |
| 20–29 | 261 | 14,78 | 13,96 | 14,47 |
| ≥30 | 64 | 18,50 | 17,33 | 20,37 |

Không có mẫu0 AP trên test; metric nhóm này để trống. **Quan sát:** nhóm ít AP có lỗi cao hơn nhóm10–29 AP. Nhưng nhóm≥30 AP cũng cao hơn nhóm10–29, nên không có quan hệ đơn điệu “càng nhiều AP càng chính xác” trong dữ liệu này. Các số ở bảng AP là mean qua seed; SD đầy đủ có trong CSV.

## 5. Các lỗi lớn và KAN diagnostics

**Ví dụ lỗi thật:** KAN seed44 tại row_index21 (dòng CSV23) chỉ quan sát2 AP, nhầm floor1 thành0 và lệch223,22 m. MLP-Mixer seed42 tại row_index410 (dòng412),4 AP, nhầm building0/floor1 sang building2/floor3 và lệch204,90 m. Hai ví dụ không chứng minh mọi lỗi lớn đều do ít AP; xem đủ các dòng trong [worst_test_errors.csv](worst_test_errors.csv).

Ngay cả khi đúng tòa/tầng, mean2D KAN của ba seed vẫn12,66–14,69 m; chỉ lọc sai tòa/tầng cũng không đạt8 m. Những mẫu sai building có thể lệch rất xa: KAN seed43 có12 mẫu sai building với mean139,26 m. Không cộng count sai building và sai floor vì các nhóm có thể chồng nhau.

[kan_diagnostics_summary.csv](kan_diagnostics_summary.csv) ghi từng KANLinear ở best/last epoch. Ví dụ best và last có các đầu vào ngoài nominal grid; phần ngoài full knot support có ý nghĩa khác. Thống kê chỉ lấy2 batch train/epoch, không phải toàn bộ test. Nhánh base SiLU vẫn tồn tại ngoài spline support. **Chưa có bằng chứng để gán toàn bộ lỗi test cho grid**; cần ablation chọn bằng validation nếu muốn kiểm tra giả thuyết này.

## 6. Lợi ích và chi phí

**Quan sát:** MLP-Mixer có40.835 tham số, ít hơn MLP142.607, nhưng batch1 end-to-end p50 khoảng1,19ms so với0,46ms của MLP. Số tham số ít hơn không đảm bảo latency thấp hơn: trục token, số thao tác, kích thước tensor và overhead thực thi đều ảnh hưởng.

KAN-Mixer có365.199 tham số; batch1 p50 khoảng4,95ms, khoảng10,7 lần MLP trong phép đo này. Mean thời gian train+validation một seed là403,17s so với82,61s của MLP, nhưng các lượt KAN cũng chạy nhiều epoch hơn; không gọi tỷ lệ đó là chi phí mỗi optimizer step.

RAM peak train toàn tiến trình khoảng1.319/1.348/1.465 MiB; VRAM reserved peak70/100/466 MiB cho MLP/MLP-Mixer/KAN. Các số này có thư viện/dữ liệu/allocator trong scope đã ghi, không phải chỉ storage trọng số. Batch1 inference RAM khoảng808/858/932 MiB; GPU reserved34/34/36 MiB. Tổng9 vòng train33,32 phút; không gộp startup/evaluation/benchmark vào số này.

**Kết luận trong giao thức này:** baseline MLP có độ biến thiên mean2D thấp hơn và suy luận nhanh hơn; MLP-Mixer có mean2D/joint accuracy trung bình tốt hơn một chút với ít tham số hơn; KAN-Mixer chưa cho thấy lợi ích mean2D đủ rõ để bù chi phí. Không suy rộng thành kết luận mọi KAN-Mixer đều kém hoặc tốt hơn MLP.

## 7. Thí nghiệm tiếp theo — đề xuất, chưa chạy

Ưu tiên một giao thức mới chỉ chọn bằng train/validation: thêm baseline Wi-Fi truyền thống như kNN; kiểm tra token/AP ordering, λxy, độ rộng/dropout và hai biến thể lai KAN/MLP. Thêm kiểm tra gần bằng tham số hoặc cùng ngân sách thời gian, cùng số thử siêu tham số cho mỗi mô hình. Nếu muốn điều kiện hóa head tọa độ theo building/floor, lúc suy luận phải dùng nhãn dự đoán/xác suất dự đoán, không dùng nhãn thật test.

Vì test này đã được xem, báo cáo phiên bản cải tiến trên cùng test phải được gọi là thăm dò; cần tập kiểm chứng độc lập hoặc thiết kế nested/group CV phù hợp để xác nhận mục tiêu8 m mạnh hơn. Không có thêm lượt tuning hay kết quả cải tiến nào được tuyên bố trong chặng7 hiện tại.
