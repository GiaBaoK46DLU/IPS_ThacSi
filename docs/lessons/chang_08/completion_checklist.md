# Checklist tổng kết đề tài

Đánh dấu theo trạng thái thực tế của repo ngày 08/10/2026. “Hoàn thành đánh giá” khác với “đạt mục tiêu chất lượng”.

## Dữ liệu và thiết kế

- [x] Kiểm tra nguồn, tên cột, số hàng và mã100 không phát hiện AP — [chặng 5](../chang_05/README.md).
- [x] Xác định input520 AP × 2, 13 cặp tòa/tầng, tọa độ2D — [chặng 4](../chang_04/README.md).
- [x] Lưu split cố định16.025 / 3.912 / 1.111, proxy nhóm và kiểm tra giao nhau.
- [x] Fit biến đổi tọa độ chỉ trên train, giữ AP order nhất quán.
- [x] Phân biệt cơ chế từ bài báo với token/head IPS tự thiết kế — [chặng 6](../chang_06/README.md).
- [ ] Xác minh đầy đủ CRS/EPSG trước khi nối dữ liệu địa lý khác hoặc đưa ra sai số 3D.

## Cài đặt và thực nghiệm tối thiểu

- [x] Cài MLP, MLP-Mixer, KAN-Mixer cùng hai đầu ra.
- [x] Khóa backend efficient-kan, phiên bản môi trường và cấu hình.
- [x] Chạy3 mô hình × 3 seed trên RTX4060Ti, tối đa 100 epoch, early stopping.
- [x] Giữ terminal log, history, best checkpoint và trạng thái hoàn tất 9 lượt.
- [x] Khóa checkpoint trước đánh giá toàn bộ 1.111 mẫu test — [audit](../../../reports/stage7_cuda_v1/pretest_audit.json).
- [x] Hoàn thành kiểm tra pipeline và evaluation — [bằng chứng chặng 7](../../../reports/stage7_cuda_v1/verification_report.json).
- [ ] Đóng gói checkpoint `runs/stage7_cuda_v1/*/best.pt` và checksum vào kho lưu trữ lâu dài. Các file này hiện nằm ngoài Git.
- [ ] Tái chạy môi trường từ lockfile trên máy khác để kiểm chứng khả năng tái lập độc lập.

## Đánh giá và phân tích

- [x] Báo cáo building/floor/joint accuracy, macro-F1 và confusion matrix.
- [x] Báo cáo MAE/RMSE tọa độ, mean/median/P90/RMSE2D trên toàn bộ test.
- [x] Báo cáo mean±SD mẫu qua 3 seed, không gọi SD là CI.
- [x] Phân tích tòa/tầng, điện thoại, AP count và người đo trên validation.
- [x] Đo train+val time, RSS RAM và VRAM allocated/reserved.
- [x] Đo B1/B64 inference riêng sau training, warmup và đồng bộ CUDA.
- [x] Ghi phạm vi phép đo và giữ cả mẫu lỗi lớn — [phân tích](../../../reports/stage7_cuda_v1/analysis_notes.md).
- [ ] Đạt mean2D ≤ 8 m trên toàn bộ tập kiểm chứng. **Chưa đạt trong suite hiện tại.**
- [ ] Khẳng định ưu thế thống kê hoặc tính tổng quát của một kiến trúc. **Bằng chứng hiện tại chưa đủ.**

## Thực nghiệm mở rộng — chưa chạy

- [ ] Đăng ký giao thức mới trước tuning, xác định vai trò test đã xem.
- [ ] Thêm kNN fingerprinting và ngân sách chọn siêu tham số tương đương.
- [ ] So sánh gần bằng tham số và cùng ngân sách thời gian.
- [ ] Ablation mask, AP ordering/tokenization và trọng số loss.
- [ ] Cài/chạy hai biến thể lai KAN/MLP.
- [ ] Outer/inner group CV theo người hoặc thiết bị, kiểm tra độ phủ nhãn/vị trí.
- [ ] Tập dữ liệu độc lập để xác nhận cải tiến sau khi xem test cũ.
- [ ] Đo trên thiết bị triển khai, thêm thời gian quét Wi-Fi nếu đánh giá trải nghiệm thực.

## Tổng kết và bảo vệ

- [x] Sơ đồ quy trình, nguồn chỉnh sửa Mermaid, PNG/SVG — [sơ đồ](research_workflow.mmd).
- [x] RQ, giả thuyết và phép kiểm chứng — [bảng nghiên cứu](research_questions.md).
- [x] Checklist có bằng chứng, không đánh dấu mục tiêu chưa đạt.
- [x] Slide 20 trang có số liệu thật, nguồn và ghi chú — [PowerPoint](presentation/IPS_KAN_Mixer_UJIIndoorLoc.pptx).
- [x] BảnPDF và lời dẫn theo từng slide.
- [x] Dàn ý báo cáo, bài tập và câu hỏi vấn đáp — [tài liệu chặng 8](README.md).
- [ ] Điền thông tin người trình bày, giảng viên và cơ sở đào tạo nếu cần. Bộ slide chưa tự suy đoán các thông tin này.
- [ ] Tự thuyết trình thử 15–20 phút, trả lời câu hỏi không dựa hoàn toàn vào ghi chú.
- [ ] Hoàn thiện báo cáo/luận văn theo mẫu chính thức của cơ sở đào tạo.
