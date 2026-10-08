# Ghi chú thuyết trình chặng 8

18 slide chính + 2 slide phụ lục. Thời lượng gợi ý: 15–20 phút.

## Slide 1 — Mở đầu

Đề tài dự đoán vị trí từ tín hiệu Wi-Fi bằng ba kiến trúc mạng. Bài trình bày tập trung vào cách đặt bài toán, quy trình so sánh và kết quả thật. Giao thức tối thiểu đã hoàn tất, nhưng mean sai số 2D trên test chưa đạt mục tiêu 7–8 m. Tôi sẽ đánh giá cả chất lượng, tốc độ và bộ nhớ.

## Slide 2 — Bài toán IPS

IPS là hệ thống định vị trong nhà. RSSI là cường độ tín hiệu mà thiết bị thu được từ mỗi access point. Fingerprint là toàn bộ tín hiệu trong một lần quan sát. Hai giá trị trên slide chỉ minh họa, không phải một dự đoán thật. Theo tài liệu UJIIndoorLoc, 100 là mã không phát hiện AP. Mạng học liên hệ giữa tín hiệu và nhãn vị trí ở train, rồi dự đoán vị trí khi chỉ có tín hiệu.

Nguồn:
https://archive.ics.uci.edu/dataset/310/ujiindoorloc

## Slide 3 — Câu hỏi và giả thuyết

Giả thuyết định hướng của câu một là KAN-Mixer có mean2D thấp hơn MLP-Mixer khi giữ giao thức so sánh. Chúng ta không mặc định giả thuyết đúng. Câu hai xem cả chất lượng và chi phí. Câu ba cần ablation mask, cách nhóm AP và split giữ người/thiết bị ngoài train. Những tiêu chí nghiên cứu mới viết ở chặng 8 là đề xuất cho tương lai, không được gọi là đăng ký trước của chặng 7.

Nguồn:
docs/lessons/chang_08/research_questions.md

## Slide 4 — Quy trình nghiên cứu

Ta kiểm tra dữ liệu, lưu split và chỉ fit tiền xử lý trên train. Mỗi mô hình chạy ba seed. Trong vòng huấn luyện, validation chọn best checkpoint và quyết định early stopping. Khi hoàn tất chín lượt, audit và khóa checkpoint trước đánh giá test. Benchmark diễn ra sau training. Không lấy kết quả test để đổi checkpoint hoặc siêu tham số của giao thức này. Sơ đồ Mermaid chi tiết trong tài liệu còn tách riêng test và benchmark.

Nguồn:
docs/lessons/chang_08/research_workflow.mmd
reports/stage7_cuda_v1/protocol_lock.json

## Slide 5 — UJIIndoorLoc

Dữ liệu có 520 cột Wi-Fi và các cột nhãn, người, điện thoại, thời gian. Số mẫu trên slide lấy từ tài liệu chính thức và CSV đã kiểm tra ở chặng 5. Dataset có ba tòa và 13 tổ hợp building/floor hợp lệ. Tên validationData.csv dễ gây nhầm: trong giao thức này nó là test cuối. Nhãn tọa độ, tòa, tầng và metadata người/điện thoại/thời gian không làm đầu vào mô hình.

Nguồn:
https://archive.ics.uci.edu/dataset/310/ujiindoorloc
docs/lessons/chang_05/README.md

## Slide 6 — Tiền xử lý RSSI

Giá trị thực đã kiểm tra nằm trong khoảng −104 đến 0, và 100 biểu thị không quan sát. Với AP được quan sát, z=(r+104)/104 và m=1. Với mã100, z=0 và m=0. Mask giúp phân biệt AP không quan sát với tín hiệu −104 có quan sát vì cả hai cùng z=0. Các hàng trên là ví dụ biến đổi đúng công thức. Tọa độ dùng một center hai chiều và shared scale fit riêng từ train nội bộ, rồi inverse trước tính sai số.

Nguồn:
https://archive.ics.uci.edu/dataset/310/ujiindoorloc
src/ips/data.py
data/processed/uji_group_split_v1/preprocessing.json

## Slide 7 — Cách chia dữ liệu

StratifiedGroupKFold dùng năm fold, shuffle và seed2026, chọn fold0. Proxy nhóm gồm USERID, PHONEID, building, floor, tọa độ, space và relative position. Nhóm proxy không giao nhau, nhưng còn người, vị trí và một số fingerprint chung giữa train/validation. Do đó split không đảm bảo generalization sang người mới. Test USERID0 nghĩa là chưa ghi nhận người, không phải toàn bộ mẫu do một người thực đo. Không dùng test để fit preprocessing.

Nguồn:
docs/lessons/chang_05/README.md
reports/stage7_cuda_v1/fairness_audit.json

## Slide 8 — Ba kiến trúc

MLP baseline nhận 1.040 số sau làm phẳng, hai lớp 128 và 64. Hai Mixer nhận 26 nhóm 20 AP, mỗi nhóm 40 số, embedding 64 channel và 2 khối. KAN-Mixer thay cả token-mixing và channel-mixing bằng các KANLinear của efficient-kan. Embedding và head vẫn tuyến tính. KAN-Mixers của Canuto v1 nghiên cứu phân loại ảnh. Đây là biến thể RSSI cho IPS do đề tài thiết kế. So sánh hiện tại giữ giao thức nhưng chưa khớp số tham số.

Nguồn:
https://arxiv.org/abs/2105.01601
https://arxiv.org/abs/2503.08939v1
https://github.com/Blealtan/efficient-kan/tree/7b6ce1c87f18c8bc90c208f6b494042344216b11
src/ips/models/networks.py

## Slide 9 — Hai nhánh trộn

B là số mẫu trong batch, 26 là số token và 64 là số channel. Token-mixing áp dụng cùng phép trộn 26 token cho từng channel. Channel-mixing áp dụng cùng phép trộn 64 channel cho từng token. MLP học các trọng số tuyến tính quanh hàm GELU. KANLinear học hàm một biến trên cạnh: phi gồm trọng số base nhân SiLU và tổ hợp B-spline với hệ số học được. Cấu hình dùng grid_size=5, spline_order=3, grid_range=[−3,3] và không cập nhật grid. Không chèn GELU giữa hai KANLinear.

Nguồn:
src/ips/models/networks.py
reports/stage7_cuda_v1/frozen_config.json
https://arxiv.org/abs/2404.19756
https://arxiv.org/abs/2503.08939v1

## Slide 10 — Giao thức so sánh

Chín lượt chạy tuần tự trên cùng RTX 4060 Ti. Cấu hình dùng AdamW với lr=0,001, weight_decay=0,0001, dropout=0,1, gradient_clip=1. Loss là CE cộng SmoothL1 với lambda_xy=1, beta=1 trên tọa độ chuẩn hóa. Chọn best nếu mean2D validation cải thiện hơn min_delta=0,01 m, patience=15, tối đa 100 epoch. Cùng ngân sách tối đa không có nghĩa cùng epoch thực tế, FLOPs hoặc thời gian. Môi trường PyTorch 2.9.0+cu128, FP32, deterministic, bốn luồng CPU. Không tuning thêm sau khi xem test.

Nguồn:
reports/stage7_cuda_v1/frozen_config.json
reports/stage7_cuda_v1/protocol_lock.json

## Slide 11 — Đọc các chỉ số

Accuracy đo tỷ lệ đúng. Macro-F1 tính F1 mỗi lớp rồi lấy trung bình đều, giúp thấy lớp ít mẫu. Joint accuracy yêu cầu cùng đúng tòa và tầng. Sau inverse tọa độ, ta tính khoảng cách Euclid 2D của từng mẫu. Mean nhạy với lỗi lớn, median mô tả điểm giữa, P90 là ngưỡng chứa 90% lỗi. RMSE2D phạt mạnh lỗi lớn. Ví dụ lệch 3 m và 4 m cho khoảng cách 5 m, trong khi MAE trên hai trục của riêng mẫu này là 3,5 m. Mục tiêu 8 m dùng mean2D, giữ mọi mẫu sai tòa/tầng.

Nguồn:
src/ips/metrics.py
reports/stage7_cuda_v1/README.md

## Slide 12 — Kết quả định vị chính

Mean2D test: MLP 15,80 ± 0,43 m, MLP-Mixer 15,52 ± 1,75 m và KAN-Mixer 15,73 ± 1,51 m. Biểu đồ chỉ biểu diễn trung bình, bảng có SD mẫu giữa ba seed. Không có seed nào đạt mean≤8 m. MLP-Mixer thấp hơn MLP 0,28 m khi ghép theo seed, SD chênh lệch 1,35 m. KAN thấp hơn MLP 0,07 m, SD chênh lệch 1,42 m. Chưa đủ bằng chứng về ưu thế thống kê. Không đổi mean2D test thành median hoặc MAE validation để tuyên bố đạt mục tiêu.

Nguồn:
reports/stage7_cuda_v1/aggregate_metrics.json
reports/stage7_cuda_v1/per_seed_metrics.csv

## Slide 13 — Kết quả phân loại

MLP-Mixer có joint accuracy trung bình 92,56%, MLP 90,97% và KAN 90,70%. Building accuracy đều trên 99%, nhưng floor accuracy khoảng 91–93%. Vì vậy cần đánh giá hai đầu ra riêng: tòa đúng chưa đảm bảo tầng đúng hay tọa độ chính xác. Joint macro-F1 theo thứ tự MLP, MLP-Mixer, KAN-Mixer là 89,47%, 91,16% và 88,70%. Ba seed chưa cho phép tuyên bố một kiến trúc tốt nhất nói chung.

Nguồn:
reports/stage7_cuda_v1/aggregate_metrics.json
reports/stage7_cuda_v1/figures/confusion_mlp_mixer.png

## Slide 14 — Phân tích lỗi lớn

Median2D của KAN thấp nhất, nhưng RMSE2D 23,32 m cao hơn MLP 19,37 m. Vì vậy cần đọc nhiều chỉ số. Ví dụ thật: KAN seed 44, row_index=21, dòng CSV 23, có hai AP quan sát, nhầm tầng 1 sang 0 và sai 223,22 m. Một ví dụ không chứng minh ít AP gây mọi lỗi lớn. Phân tích theo tầng còn cho thấy nhóm 1/0 và 1/1 khó với KAN. Nhóm test 1/0 chỉ có 30 mẫu nên phải ghi số mẫu khi so sánh.

Nguồn:
reports/stage7_cuda_v1/worst_test_errors.csv
reports/stage7_cuda_v1/group_metrics.csv
reports/stage7_cuda_v1/aggregate_metrics.json

## Slide 15 — Thời gian huấn luyện

Tổng vòng train + validation là 33,32 phút. Bảng là trung bình và SD qua ba seed của từng mô hình. KAN chạy nhiều epoch hơn MLP, nên tỷ lệ tổng thời gian không đại diện chi phí một optimizer step. Early stopping làm ngân sách thực tế khác nhau dù đều tối đa 100 epoch. Thời gian gồm validation, ghi log, lấy diagnostics và lưu checkpoint. Chưa gồm import/startup/tải dữ liệu trước vòng lặp hoặc evaluation/benchmark cuối. Không dùng ngoại suy pilot ba epoch làm kết quả chính.

Nguồn:
reports/stage7_cuda_v1/aggregate_metrics.json
reports/stage7_cuda_v1/per_seed_metrics.csv

## Slide 16 — Latency và bộ nhớ

Biểu đồ là trung bình p50 end-to-end B1 qua ba seed: MLP 0,46 ms, Mixer 1,19 ms, KAN 4,95 ms. End-to-end từ RSSI trong RAM qua tiền xử lý, GPU, mô hình và inverse đến mảng CPU, chưa gồm quét Wi-Fi hoặc I/O file. Mỗi benchmark có 10 warmup, 100 lần đo, đồng bộ CUDA và tiến trình riêng. Bảng là peak RAM toàn tiến trình và peak VRAM reserved khi train, lấy mean qua ba seed. VRAM allocated riêng là 67,99/76,92/414,54 MiB. Những số này khác storage trọng số. Phiên desktop chưa cô lập hoàn toàn, không ngoại suy nguyên số sang máy khác.

Nguồn:
reports/stage7_cuda_v1/aggregate_metrics.json
src/ips/benchmark.py
src/ips/resources.py

## Slide 17 — Giới hạn

Ba seed cùng split chỉ mô tả biến thiên huấn luyện. SD mẫu không phải CI và không chứng minh độ ổn định trên dữ liệu khác. Proxy nhóm không giao nhau vẫn chưa giữ người/vị trí ngoài train. Số tham số, FLOPs và thời gian của ba kiến trúc khác nhau. Tọa độ theo quy ước mét dataset, CRS/EPSG chưa hoàn toàn xác minh, nên chưa suy ra lỗi 3D. Vì test đã xem, cải tiến chọn sau phân tích này trên test cũ mang tính thăm dò.

Nguồn:
reports/stage7_cuda_v1/analysis_notes.md

## Slide 18 — Kết luận

Đóng góp hiện có là mô hình hóa IPS, cài đặt ba mạng và thực nghiệm tái lập, cùng đánh giá chất lượng và chi phí. Chưa có thuật toán KAN mới hay kết quả đạt mục tiêu 8 m. MLP-Mixer có mean2D trung bình thấp nhất nhưng chênh lệch nhỏ và biến thiên đáng kể. KAN-Mixer chưa có lợi ích mean2D rõ ràng trong cấu hình này. Bước tiếp theo chưa chạy: ablation mask, AP ordering, hai biến thể lai, kNN baseline, so sánh khớp ngân sách và group CV. Chọn bằng validation, lưu giao thức mới và kiểm chứng độc lập.

Nguồn:
docs/lessons/chang_08/completion_checklist.md
docs/lessons/chang_08/research_questions.md
reports/stage7_cuda_v1/aggregate_metrics.json

## Slide 19 — Phụ lục tensor

520 AP, mỗi AP có 2 số, tạo 1.040 features. Reshape thành 26 token với 40 features mỗi token, tương đương 20 AP. Linear embedding biến 40 thành 64, dùng chung mọi token. Token branch đổi B×26×64 thành B×64×26, trộn 26–32–26 rồi đổi trục về. Channel branch trộn 64–128–64. Mỗi branch có pre-LN, residual và dropout. Sau 2 khối, LayerNorm và trung bình trên trục token cho B×64. Hai head tuyến tính cho 13 logits và 2 tọa độ chuẩn hóa.

Nguồn:
src/ips/models/networks.py
reports/stage7_cuda_v1/frozen_config.json

## Slide 20 — Nguồn

Các tài liệu nền giải thích kiến trúc. Những số thực nghiệm IPS trên slide lấy từ suite cục bộ đã hoàn tất. Backend efficient-kan giữ commit 7b6ce1c87f18c8bc90c208f6b494042344216b11, có provenance và giấy phép MIT. Tài liệu KAN-Mixers v1 nghiên cứu ảnh, còn token hóa và head IPS là thiết kế của đề tài. JSON, cấu hình, audit và dự đoán từng mẫu trong repo cho phép truy nguồn kết quả.

Nguồn:
https://archive.ics.uci.edu/dataset/310/ujiindoorloc
https://arxiv.org/abs/2105.01601
https://arxiv.org/abs/2404.19756
https://arxiv.org/abs/2503.08939v1
https://github.com/Blealtan/efficient-kan/tree/7b6ce1c87f18c8bc90c208f6b494042344216b11
reports/stage7_cuda_v1/aggregate_metrics.json
