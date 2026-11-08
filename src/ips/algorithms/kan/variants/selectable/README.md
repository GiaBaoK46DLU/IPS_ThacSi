# S-KAN-Mixer: thí nghiệm IPS đối chiếu V4

Triển khai độc lập cơ chế của [Yang et al., Activation Space Selectable KAN](https://arxiv.org/abs/2408.08338v1).
Đây là **S-KAN thích nghi cho IPS**, với pool khai báo rõ bên dưới, không khẳng định khớp mã/tập 18 hàm chưa được tác giả cung cấp.
Khung Mixer và head IPS giữ nguyên V4; tám KANLinear được thay bằng lớp chọn hàm theo từng node đầu vào.

## Cây thư mục

- `bases.py`: định nghĩa toán học các ứng viên.
- `layers.py`: logits chọn hàm theo node, mask/prune, mô hình gọn chỉ giữ hàm được chọn.
- `search.py`: pha chọn hàm dùng internal Train; checkpoint phục hồi sau mỗi vòng prune.
- `experiment.py`: đóng băng protocol, mười seed, Validation, refit, khóa checkpoint và Test.
- `audit.py`: tính lại metric từ CSV, kiểm tra đủ hàng và tọa độ thật.
- `report.py`: báo cáo tiếng Việt sinh từ kết quả đã audit vào `docs/Bao/KAN_SKAN/` (gitignore).
- `provenance.json`: nguồn phương pháp và các lựa chọn thích nghi.
- `../../configs/selectable/uji_v4_matched.json`: cấu hình đầu vào, chỉ thêm backend/pool/stage vào V4.
- `../../../../../../scripts/run_skan_uji.py`: entrypoint (từ repo: `scripts/run_skan_uji.py`).

## Pool khai báo: 18 ứng viên, không phải 18 họ thuật toán khác nhau

1. B-spline: grid 5, bậc 3, range [-3,3], vendor efficient-kan hiện có.
2. Gaussian RBF: 8 và 16 tâm đều [-3,3], độ rộng bằng khoảng cách tâm.
3. Chebyshev: bậc 3 và 5 trên tanh(x).
4. Gram rời rạc: bậc 3 và 5 trên tanh(x), hệ số recurrence khởi tạo theo công thức Gram rồi học theo node.
5. Jacobi: bậc 3 và 5, alpha=beta=1, trên tanh(x).
6. Bernstein: bậc 3 và 5 trên sigmoid(x).
7. Mexican hat, đạo hàm Gaussian, Morlet cos(5u): tịnh tiến và log-scale học theo node.
8. Cửa sổ thực lấy cảm hứng Meyer: sin(pi|u|) nhân cửa sổ trơn compact; **không phải** toàn bộ wavelet Meyer thu được bằng biến đổi Fourier ngược.
9. Shannon bandpass thực: 2*sinc(2u) - sinc(u); không phụ thuộc thứ tự batch hay nhóm input.
10. ReLU basis: grid5 + overlap3, biên dưới/độ rộng học được, tích ReLU cục bộ bình phương.
11. RSWAF: 8 tâm đều [-3,3], 1-tanh²((x-center)/spacing).

Tất cả ứng viên có nhánh nền SiLU và hệ số học theo cạnh; không chèn LayerNorm mới.
Scale/width học trong miền log, clamp [-4,4] để tránh chia cho zero. Mỗi node chia sẻ lựa chọn trên các cạnh đầu ra.
Logits ban đầu bằng zero → xác suất 1/18; masked softmax bảo đảm xác suất không âm và tổng bằng 1.
Mỗi vòng bỏ ứng viên có logit nhỏ nhất trên từng node (tie lấy index đầu), rồi chuẩn hóa lại.
Sau vòng 17, chỉ còn một ứng viên/node; mô hình chính tạo các nhóm vật lý chỉ chứa tham số được chọn.
Khởi tạo nhóm dùng fan-in toàn lớp, tránh tăng biên độ chỉ vì một nhóm có ít node.

## Quy trình đã định trước

- Seed: 42,43,44,100,2024,777,999,45,46,47. Mọi seed train và refit mới.
- 16025 Train,3912 Validation; 19937 development; Test chính thức đủ1111 mẫu, đo Euclid2D mét trên tọa độ gốc.
- Chọn cấu trúc trên 1% internal Train =160 mẫu phân tầng13 lớp, catalogue fit toàn internal Train.
- 17 vòng; mỗi vòng 3 epoch full training +3 epoch chỉ logits lựa chọn, rồi prune một hàm/node.
- Pha chọn hàm: batch16, lr0.001, AdamW; wd0 cho logits, wd0.0001 cho phần còn lại; reset optimizer mỗi pha; không cosine. Batch chọn hàm giảm từ dự kiến32 để có khoảng trống VRAM khi GPU8GB đang được các ứng dụng khác sử dụng; batch train chính vẫn128.
- Pha chính giữ cấu hình V4: batch128, lr0.001, wd0.0001, cosine100, patience20, min_delta0.01, clip1.
- Sau lựa chọn **khởi tạo lại tất cả trọng số**, chỉ truyền cấu trúc; Validation chọn checkpoint/decoder như V4.
- Refit epoch = round(median(10 best epoch)); scheduler vẫn horizon100; giữ preprocessing train-only, học lại catalogue trên development.
- Refit từ trọng số mới theo seed, giữ cấu trúc đã chọn; không chọn lại hàm trên Validation hay Test.
- Khóa đủ10 refit trước Test. Không ensemble/KNN; không chỉnh cấu hình sau khi xem Test.

Đối chiếu V4 giữ nguyên cấu hình ngoài, loss, sampling, augmentation và split. S-KAN thêm compute chọn hàm và có thể thay số tham số; không tuyên bố tổng compute/parameter bằng nhau.
V4 Test đã được biết trước, cần nêu lịch sử tiếp xúc benchmark trong báo cáo.

## Chạy và review

```powershell
.\.venv-cuda\Scripts\python.exe scripts/run_skan_uji.py prepare --output-dir runs/kan_skan_uji_20261028
.\.venv-cuda\Scripts\python.exe -u scripts/run_skan_uji.py run --output-dir runs/kan_skan_uji_20261028
.\.venv-cuda\Scripts\python.exe scripts/run_skan_uji.py audit --output-dir runs/kan_skan_uji_20261028
```

Chỉ một worker được giữ OS file lock. Gọi lại `run` tiếp tục các seed đã xong; search tiếp tục sau vòng prune hoàn chỉnh.
Train/refit bị ngắt chưa có summary được chuyển vào `interrupted/` rồi chạy lại seed từ đầu (không giả vờ khôi phục optimizer/RNG chưa lưu).
Protocol hash khóa source/config/data và artifact V4, có source_snapshot riêng. Nếu sửa source, phải lập protocol/run mới, không sửa guard của thí nghiệm cũ.
Kết quả nằm trong `official_test_summary.json`, `completion_audit.json`; cấu hình từng seed trong `activation_search/seed*/selected_config.json`,
decoder cuối và cấu hình refit trong `final_refit/seed*/resolved_config.json`.

## Nguồn định nghĩa cơ sở

- [FastKAN](https://github.com/ZiyaoLi/fast-kan): Gaussian RBF.
- [Wav-KAN](https://github.com/zavareh1/Wav-KAN): họ wavelet.
- [ReLU-KAN](https://github.com/quiqi/relu_kan): local ReLU basis.
- [torch-conv-kan](https://github.com/IvanDrokin/torch-conv-kan): tham khảo recurrence/các họ basis.

Các công thức ngoài B-spline được viết độc lập; vendor B-spline kế thừa attribution/license của dự án.
Các biến thể chuẩn hóa/khởi tạo khác mã từng tác giả được khai báo rõ ở đây, không gọi từng ứng viên là tái lập nguyên bản FastKAN/WavKAN.
