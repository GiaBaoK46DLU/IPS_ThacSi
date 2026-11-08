# Kiểm chứng cấu trúc KAN dùng chung

Thực hiện ngày 08/10/2026 trên nhánh feature/kan-only-algorithm, từ main@6bd94c4.
Workspace ban đầu sạch. Người dùng xác nhận giữ phần học cá nhân trên máy và bỏ khỏi Git dùng chung.

## Cấu trúc và phạm vi

- KANMixerAlgorithm trong algorithms/kan_mixer.py kế thừa **trực tiếp Algorithm**; không còn NeuralAlgorithm.
- Registry và CLI chỉ hiển thị kan_mixer; train mặc định KAN, từ chối --model mlp/mlp_mixer.
- Config dùng chung là configs/kan_mixer.json, không yêu cầu block mlp. Token/channel vẫn dùng efficient-kan B-spline.
- Giữ MLP/MLP-Mixer riêng trong personal_learning/legacy_models.py; cầu nối build_model chỉ dùng file cá nhân
  khi chủ máy gọi baseline trực tiếp. Factory không đăng ký chúng. Clone mới không cần file này để chạy KAN.
- 191 file khớp danh sách .gitignore đã bỏ tracking bằng git rm --cached, xác minh từng file còn trên máy
  với SHA256 không đổi: baseline configs, báo cáo so sánh, tài liệu học cũ, Office materials, scripts suite cũ,
  analysis và tests lịch sử. Manifest kiểm tra nằm ở personal_learning/ignored_files_manifest.json (ignored).
- README, tài liệu kiến trúc và Mermaid/SVG/PNG đã cập nhật; mã wrapper/mạng/README/sơ đồ trước thay đổi có bản lưu cá nhân.
- CSV, split/manifest/preprocessing, metrics.py và efficient-kan vendor giữ nguyên.

## Tests

Sau sửa code chính, unittest discover tại workspace: 30 tests PASS. Khi bổ sung thêm một test về
không phụ thuộc baseline cá nhân, bản xuất dùng chung được kiểm tra đầy đủ như sau.

Xuất file đang tracked hoặc mới chưa ignored sang runs/kan_only_refactor/shared_export, không có
personal_learning/, configs/base.json, src/ips/analysis.py hay test_evaluation.py cũ. Dùng PYTHONPATH
trỏ vào src của bản xuất và cwd của bản xuất để tránh import nhầm mã từ editable install tại workspace gốc.

```powershell
# Các lệnh thực sự chạy trong bản xuất
python -m unittest discover -s tests -v
python -m ips algorithms
python -m ips check --device cpu
python -u -m ips run --config configs/experiments/kan_mixer_smoke.json
```

Kết quả: **24/24 shared tests PASS**, khoảng14,715 giây. Registry chỉ có kan_mixer;
check cho365.199 tham số, input[2,520,2], logits[2,13], xy[2,2].
Tests giữ metric/alignment/source/row order, train-only fit, Factory, finalization trước test,
KAN gradient/token-channel axes, checkpoint v1 fixture và tương thích các lệnh KAN.

Trên máy còn test_evaluation.py lịch sử đã ignored, unittest discover vẫn tìm file vật lý đó;
Git ignore không phải bộ lọc test discovery. Clone mới chỉ chạy bộ24 tests dùng chung.

## Smoke CPU và checkpoint thật

Smoke Factory trong bản xuất: seed42, train256/validation128, một epoch, CPU, không dùng test.
Log epoch hiển thị trực tiếp; hoàn tất với mean2D validation≈133,921 m, joint accuracy0,09375.
Đây là smoke pipeline, không phải kết quả đủ huấn luyện hoặc kết luận chất lượng.
Có checkpoint/history/resources/log, predictions/metrics/matrices, confusion/CDF và model diagram.
Artifacts nằm tại runs/kan_only_refactor/shared_export/runs/shared_kan_mixer_smoke/ (ignored).

Checkpoint thật đã kiểm tra: runs/stage7_cuda_v1/kan_mixer_seed42/best.pt, format v1, best epoch54.
Nạp qua KANMixerAlgorithm mới và mạng cũ được giữ riêng, cùng26 validation samples, CPU:
**class IDs và tọa độ khớp hoàn toàn**. Bằng chứng: runs/kan_only_refactor/real_kan_v1.json (ignored).
Không chạy suite GPU nhiều giờ và không thay số liệu reports cũ.

Mermaid đã validate bằng mmdc11.15.0, xuất SVG/PNG và kiểm tra trực quan.
LLM/HUIM vẫn là hợp đồng trừu tượng; FastKAN/S-KAN chưa được triển khai.
