# Câu hỏi nghiên cứu và giả thuyết có thể kiểm chứng

## Hai loại câu hỏi

**Đối chiếu thực nghiệm đã hoàn tất:** dùng chặng 7 để mô tả chính xác cấu hình đã chạy. Các giả thuyết viết ở chặng 8 là cách hệ thống hóa câu hỏi sau thực nghiệm, không được ghi là đã đăng ký trước. Không có kiểm định ý nghĩa thống kê trong suite ba seed.

**Thực nghiệm tương lai:** khóa giả thuyết, split, chỉ số chính, số lần tìm siêu tham số và quy tắc kết luận trước khi chạy. Các ngưỡng dưới đây là đề xuất nghiên cứu mới, không phải kết quả công bố.

Ký hiệu: E là mean2D trên toàn bộ tập đánh giá; ΔE = E_mô_hình − E_đối_chứng, âm nghĩa là sai số thấp hơn. Mean2D có đơn vị mét. Các seed dùng cùng split nên mô tả biến thiên huấn luyện, không phải ba mẫu dữ liệu độc lập.

## RQ1 — Thay MLP bằng KAN trong Mixer có giảm sai số IPS?

- **Giả thuyết định hướng:** KAN-Mixer có E thấp hơn MLP-Mixer trong cùng giao thức. H0: ΔE≥0; H1: ΔE<0.
- **Thay đổi:** chỉ loại nhánh trộn; giữ token/AP ordering, split, preprocessing, head, loss, batch và ngân sách tìm siêu tham số.
- **Đo:** E là chính; joint accuracy, P90, RMSE2D và chi phí là phụ. Báo cáo từng seed và chênh lệch ghép theo seed.
- **Bằng chứng hiện có:** KAN 15,73±1,51 m so với MLP-Mixer 15,52±1,75 m. Trung bình hiện tại không ủng hộ hướng giảm lỗi. Điều này không bác bỏ mọi cấu hình KAN-Mixer.
- **Cần thêm:** tìm siêu tham số bằng nested group CV với ngân sách bằng nhau; so sánh gần bằng tham số và một so sánh cùng ngân sách thời gian. Báo cáo mức cải thiện có ý nghĩa thực tế trước khi chạy mới, ví dụ giảm ít nhất 1 m; không chọn ngưỡng sau khi nhìn kết quả mới.

## RQ2 — Lợi ích chất lượng có bù được chi phí?

- **Giả thuyết định hướng:** một biến thể Mixer cải thiện E mà không tăng quá mức latency/bộ nhớ đối với yêu cầu triển khai đã xác định. Chưa có yêu cầu latency cụ thể của sản phẩm nên chưa gọi mô hình nào là “đạt thời gian thực”.
- **Phép kiểm chứng:** báo cáo cặp (E, latency B1 p50/p90), RAM toàn tiến trình và VRAM allocated/reserved. Một mô hình bị trội nếu mô hình khác có lỗi và chi phí đều không cao hơn, ít nhất một đại lượng thấp hơn. Không gộp mọi chỉ số thành điểm tùy ý.
- **Bằng chứng hiện có:** B1 end-to-end p50 MLP/MLP-Mixer/KAN-Mixer là 0,46/1,19/4,95 ms trên RTX4060Ti. KAN chưa có lợi ích mean2D rõ ràng để bù chi phí trong bảng này.
- **Cần thêm:** khóa ngân sách ứng dụng trước thử mới, kiểm tra CPU/thiết bị triển khai, p90 và tải đồng thời. Thời gian quét Wi-Fi cần phép đo riêng.

## RQ3 — Split theo người hoặc thiết bị có làm đánh giá khó hơn?

- **Giả thuyết tương lai:** split giữ người/thiết bị ngoài train cho E lớn hơn split proxy nhóm hiện tại. H0: ΔE≤0; H1: ΔE>0. Đây là kỳ vọng chưa được đo.
- **Thiết kế:** so sánh outer group CV theo USERID hoặc PHONEID trong `trainingData.csv`; kiểm tra đủ 13 lớp, độ phủ vị trí và nhãn ở từng fold. Inner split dùng để tuning/early stopping. Không diễn giải nhóm test USERID0 như một người thực.
- **Đo:** E, joint macro-F1 và phân bố tòa/tầng, AP count, số mẫu từng fold. Giữ cùng ngân sách và seed cho mọi kiến trúc.
- **Giới hạn:** holdout người có thể đồng thời thay đổi vị trí/thiết bị. Nếu độ phủ khác nhau, kết quả mô tả tác động tổng hợp của giao thức, chưa tách được nguyên nhân người đo.

## RQ4 — Mask và cách chia AP thành token ảnh hưởng thế nào?

- **H4a tương lai:** có mask làm E thấp hơn bỏ mask. H0: ΔE≥0; H1: ΔE<0.
- **H4b tương lai:** cách nhóm AP chọn bằng validation có E thấp hơn thứ tự 26×20 AP cố định. H0: ΔE≥0; H1: ΔE<0.
- **Đối chứng:** giữ đủ 520 AP và cùng dữ liệu. Với ablation mask, thay mask bằng 0 nhưng giữ kích thước đầu vào trước, để giảm thay đổi tham số; sau đó có thể so sánh thiết kế đầu vào gọn hơn riêng.
- **Thiết kế:** thử một hoán vị cố định dùng nhất quán train/val/test, hoặc nhóm AP bằng thống kê chỉ fit trên train. Khóa số cách nhóm và ngân sách chọn; không dùng vị trí thật test để tạo token.
- **Đo:** E và joint accuracy, lỗi theo AP count, số tham số và latency. Không khẳng định 20 AP liền nhau tương ứng lân cận vật lý: repo không có tọa độ AP để chứng minh điều đó.
- **Hiện trạng:** mọi model trong chặng 7 có mask; chưa có ablation nên chưa thể gán mức cải thiện cho mask hay nhóm AP.

## RQ5 — Biến thể lai có giữ chất lượng với chi phí thấp hơn KAN toàn phần?

- **Biến thể đề xuất:** KAN ở token-mixing, MLP ở channel-mixing; và chiều ngược lại. Hai mô hình này chưa được cài/chạy trong suite hiện tại.
- **Giả thuyết tương lai:** biến thể lai có B1 p50 thấp hơn KAN toàn phần và sai số không tăng quá δ=1 m. δ là ngưỡng chấp nhận thực tế đề xuất ở chặng 8, phải chốt lại trước thử mới.
- **Phép kiểm chứng:** chất lượng không kém hơn đòi hỏi cận trên khoảng bất định thích hợp của ΔE dưới δ; chỉ nhìn hai trung bình gần nhau không đủ. Đồng thời latency phải giảm với cùng cách đo.
- **Thiết kế:** cùng token/head/preprocessing; ngân sách tìm siêu tham số bằng nhau, nhiều seed trên nhiều outer group folds. Ước lượng bất định tôn trọng nhóm và cấu trúc fold; không bootstrap từng dòng như thể mọi fingerprint độc lập.

## Mục tiêu 7–8 m là một tiêu chí riêng

Tiêu chí làm việc: E≤8 m trên toàn bộ tập kiểm chứng, giữ cả các mẫu nhầm tòa/tầng. Trong chặng 7: 0/9 checkpoint đạt. Mục tiêu chưa đạt không làm pipeline mất giá trị; nó giúp xác định khoảng cách cần cải thiện.

Test chính thức đã được xem trong chặng 7. Các thử nghiệm cải tiến trên cùng test phải mang nhãn **thăm dò**. Với nghiên cứu tiếp theo, có thể dùng nested/group CV làm đánh giá ngoài quá trình tuning trên phần training, nhưng lựa chọn câu hỏi đã chịu ảnh hưởng từ test cũ; dữ liệu kiểm chứng độc lập vẫn là bằng chứng mạnh hơn cho xác nhận cuối. Không tự hứa một cấu hình mới sẽ đạt 8 m.

## Phiếu đăng ký thí nghiệm mới

Trước mỗi thí nghiệm: ghi ID phiên bản, RQ/H, biến can thiệp, đối chứng, split/group và seed; chỉ số chính, δ nếu có, số cấu hình được thử, ngân sách, quy tắc checkpoint và loại đánh giá xác nhận/thăm dò. Sau chạy: ghi kết quả mọi lượt, biến thiên, vi phạm giao thức nếu có, kết luận và giới hạn. Giữ lại kết quả bất lợi.

**Bài tập:** chọn RQ4a và điền phiếu cho một phép ablation duy nhất. Giải thích vì sao thay mask và tăng độ rộng mạng đồng thời sẽ khó xác định tác dụng của mask.

Nguồn số liệu: [kết quả chặng 7](../../../reports/stage7_cuda_v1/README.md), [diễn giải và giới hạn](../../../reports/stage7_cuda_v1/analysis_notes.md). Các giả thuyết tương lai là đề xuất của đề tài.
