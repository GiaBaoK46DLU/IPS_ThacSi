# Thư mục review KAN-Mixer

Toàn bộ triển khai KAN nằm tại đây. KANMixerAlgorithm kế thừa trực tiếp Algorithm ở ../base.py;
không có lớp trung gian NeuralAlgorithm. Mạng vẫn dùng efficient-kan B-spline.

**Báo cáo Update Version 3 (22/10/2026):** xem
[UPDATE_VERSION_3.md](UPDATE_VERSION_3.md) để đọc đầy đủ thay đổi, thuật toán,
cấu hình và kết quả test. Nhánh hiện tại là `update-version-3`. Thực nghiệm đã
dừng theo yêu cầu; chưa đạt mean2D dưới 7 m. File này được đổi tên từ
`NGHIEN_CUU_DUOI_SAU_MET.md`; nhật ký cũ được giữ trong phụ lục của báo cáo.

**Phạm vi cải tiến UJI (cập nhật 22/10/2026): một KAN-Mixer chung ba tòa.**
Chạy bằng `improvement.py`; controller từ chối bộ nhớ/truy xuất fingerprint,
hiệu chỉnh kNN, ensemble và trung bình nhiều lượt suy luận. Có thể đổi cấu trúc
KAN-Mixer, pooling/head bên trong mạng, loss, augmentation train và lịch tối ưu.
`SamePointMixing` chỉ tăng cường mẫu train cùng vị trí, không kết hợp bộ dự đoán.
EMA (nếu bật) cập nhật trọng số khi train; suy luận vẫn dùng đúng một bộ trọng số.
Các file lai/ensemble đã có từ thực nghiệm cũ giữ để đọc checkpoint và tái lập
lịch sử, không được dùng cho mục tiêu UJI dưới 7 m của vòng nghiên cứu này.

```text
kan/
  algorithm.py                 vòng đời IPS: prepare → execute → predict
  paths.py                     đường dẫn config và ánh xạ tên config cũ
  dataset.py                   NumPy/RSSI → tensor PyTorch
  models/
    networks.py                token/channel mixing, chọn mạng cũ hoặc reference
    reference.py               điểm train và KAN-Mixer head tham chiếu
    refinement.py              wrapper lai fingerprint, catalogue train
    ensemble.py                wrapper kết hợp các mạng reference
    vendor/                    efficient_kan.py, provenance.json, license
  train.py                     loss, optimizer, checkpoint theo validation
  augmentation.py              tăng cường train giữa các mẫu cùng vị trí
  improvement.py               chạy/kiểm soát nghiên cứu chỉ một KAN-Mixer
  improvement_analysis.py      kiểm dự đoán validation, tọa độ đã/chưa thấy
  confirmation.py              xác nhận đa seed, khóa lựa chọn trước refit/test
  predict.py                   inference và evaluate checkpoint
  runtime.py                   seed, device, nạp checkpoint v1/v2
  accuracy.py                  CLI chọn decoder KAN-Mixer bằng validation
  accuracy_ensemble.py         chọn ensemble bằng validation (hướng mở rộng)
  final_refit.py                refit development theo lựa chọn đã khóa
  freeze_refit.py               đóng gói refit, không chọn lại bằng validation
  supplied_experiment.py        chuyển cấu hình UJI sang DLULIB/HCXY/TUJI/SYL
  supplied_figures.py           xuất bảng và hình từ kết quả thật trên các bộ mới
  verify_supplied.py            đối chiếu checkpoint, catalogue, row và metric sau chạy
  resources.py                 RAM/VRAM và đồng bộ CUDA
  diagnostics.py               chẩn đoán input từng KANLinear
  benchmark.py                 thời gian/bộ nhớ suy luận
  configs/
    model.json                 kiến trúc/loss/training KAN
    experiment_smoke.json      config Factory: một epoch CPU
    accuracy_reference.json    head điểm tham chiếu, flatten
    accuracy_spatial.json      nhãn mềm và augmentation RSSI
    accuracy_mean_ablation.json reference head dùng mean pooling
  tests/test_pipeline.py       trục mixing, gradient, vendor hash, checkpoint
  tests/test_reference.py      catalogue, decoder, lai/ensemble, refit fixture
  README.md
```

Review theo thứ tự: algorithm.py → dataset.py → models/networks.py → train.py → predict.py.
Khi phân tích chi phí hoặc lỗi spline, xem resources.py, diagnostics.py và benchmark.py.
Nguồn vendor giữ nguyên byte/hash; không thay thuật toán hoặc key state_dict chỉ vì đổi folder.

## Thực nghiệm trên DLULIB và HCXY

Lối chạy nghiên cứu riêng nằm trong `supplied_experiment.py`, tái sử dụng nguyên mạng,
loss và vòng train hiện tại. Reader NumPy dùng chung ở `ips/data_access/supplied.py`.
Các lệnh Factory/UJI cũ vẫn giữ schema cũ; không đổi tên CSV của bộ khác thành UJI.

Giữ 26×40 bằng cách padding AP trống đến520. Chỉ fit transform tọa độ và catalogue
trên train; train/validation tách theo vị trí. DLULIB dùng All_20, không gộp Mean.
HCXY có một tòa/tầng nên chỉ số classification mang tính mô tả. Các giả định của
DLULIB về −100, đơn vị mét và AP order giữa tầng được ghi trong manifest.

```powershell
# Chọn thư mục mới cho mỗi lượt; cần raw trong dataset/raw/.
./.venv-cuda/Scripts/python.exe -m ips.algorithms.kan.supplied_experiment prepare --output-dir runs/kan_supplied_new
./.venv-cuda/Scripts/python.exe -u -m ips.algorithms.kan.supplied_experiment train --output-dir runs/kan_supplied_new --device cuda
./.venv-cuda/Scripts/python.exe -u -m ips.algorithms.kan.supplied_experiment evaluate --output-dir runs/kan_supplied_new --device cuda
./.venv-cuda/Scripts/python.exe -m ips.algorithms.kan.supplied_figures --output-dir runs/kan_supplied_new
./.venv-cuda/Scripts/python.exe -m ips.algorithms.kan.verify_supplied --output-dir runs/kan_supplied_new
```

Ba seed42/43/44 dùng cấu hình training UJI đã xác minh. Decoder và hệ số lai giữ
đúng cấu hình từng seed UJI, không calibration mới. Hiện source_settings đọc
các run/calibration UJI gốc local; prepare chép cấu hình/provenance vào output
để review. Thiếu source run phải khôi phục snapshot cấu hình đã kiểm chứng,
không suy đoán decoder. Kết quả mới nằm dưới output/{dlulib,hcxy}/; train metrics
chỉ chẩn đoán vì catalogue lai có chứa chính truy vấn train.

Các phần dùng chung vẫn ở ngoài folder này: data_access/, experiments/, metrics.py, data.py,
algorithms/base.py, CLI và dữ liệu CSV/split. LLM/HUIM không cần dùng tensor hay training của KAN.
Các alias ở đường dẫn KAN cũ đã bỏ; import Python dùng ips.algorithms.kan.*.
DatasetRepository chỉ trả NumPy/split. Chuyển tensor được thực hiện trong folder KAN.

Adapter supplied hiện hỗ trợ thêm `tuji` và `syl` (phần SYL của SODIndoorLoc).
TUJI dùng đúng mapping native: FLOOR→z, BUILDINGID→floor, SPACEID→building,
RELATIVEPOSITION→device. Metadata thiết bị chỉ theo dòng nguồn để phân tích,
không đi vào forward. SYL và HCXY giữ namespace AP/site riêng; không nối MAC
cùng tên giữa hai site. `RP` trong TUJI/SYL là ID vị trí dẫn xuất theo từng role.

Ví dụ chạy tiếp, dùng output mới để giữ split và checkpoint đã khóa:

```powershell
.\.venv-cuda\Scripts\python.exe -m ips.algorithms.kan.supplied_experiment prepare --datasets tuji syl --output-dir runs/kan_tuji_sod_new
.\.venv-cuda\Scripts\python.exe -u -m ips.algorithms.kan.supplied_experiment train --datasets tuji syl --output-dir runs/kan_tuji_sod_new --device cuda
.\.venv-cuda\Scripts\python.exe -m ips.algorithms.kan.supplied_experiment evaluate --datasets tuji syl --output-dir runs/kan_tuji_sod_new --device cuda
.\.venv-cuda\Scripts\python.exe -m ips.algorithms.kan.supplied_figures --datasets tuji syl --output-dir runs/kan_tuji_sod_new
.\.venv-cuda\Scripts\python.exe -m ips.algorithms.kan.verify_supplied --datasets tuji syl --output-dir runs/kan_tuji_sod_new
```

TUJI/SYL chỉ một building/floor trong mỗi phần; confusion matrices một ô là
mô tả, không chứng minh khả năng phân loại nhiều tòa/tầng. Các bảng thiết bị
và số lần quét không phải đơn vị thống kê độc lập khi có fingerprint lặp.
tests/test_pipeline.py ở root chỉ là điểm vào discovery, tests thực sự nằm trong folder này.

Từ root repo:

```powershell
.\.venv\Scripts\python.exe -u -m ips run --config src/ips/algorithms/kan/configs/experiment_smoke.json
.\.venv\Scripts\python.exe -m ips check --device cpu
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
# Chỉ tests mạng KAN
.\.venv\Scripts\python.exe -m unittest ips.algorithms.kan.tests.test_pipeline -v
```

Đổi output_dir nếu đã chạy config smoke. Kết quả vẫn ở runs/ và bị ignore, không nằm cùng source.
Lệnh train/predict/evaluate/benchmark cũ vẫn hoạt động; CLI ánh xạ hai đường dẫn config KAN cũ
sang configs/ bên trong folder này khi file cũ không tồn tại. Checkpoint KAN v1 vẫn nạp bằng state_dict.
Mã dùng chung không tải MLP/MLP-Mixer cá nhân; phần học đó gọi trực tiếp personal_learning/legacy_models.py.

## Biến thể điểm tham chiếu

Factory vẫn dùng ID `kan_mixer`; `positioning.variant=reference` trong model config
chọn mạng điểm tham chiếu. Thiếu trường này giữ mạng hồi quy cũ để đọc checkpoint lịch sử.
Backbone vẫn là efficient-kan B-spline; head và decoder RSSI là thiết kế IPS của dự án.

Chạy train và hiệu chỉnh suy luận (output-dir phải mới):

```powershell
.\.venv-cuda\Scripts\python.exe -u -m ips train --config src/ips/algorithms/kan/configs/accuracy_spatial.json --seed 44 --device cuda --run-dir runs/kan_spatial_new
.\.venv-cuda\Scripts\python.exe -m ips.algorithms.kan.accuracy --run-dir runs/kan_spatial_new --output-dir runs/kan_calibration_new --device cuda
```

Hoặc dùng Runner chung: `python -m ips run --config examples/kan_accuracy_validation.json`.
Ví dụ này chỉ đánh giá validation. Checkpoint và decoder được chọn bằng validation;
chốt lựa chọn trước khi đánh giá test. `pure_selected.pt` giữ mạng thuần;
CLI hiện tại cũng xuất `selected.pt` là mạng thuần. Artifact cũ có thể chứa bản
lai; không dùng những artifact đó cho nghiên cứu chỉ KAN-Mixer hiện tại.

Lối chạy mới lưu config, hash, snapshot mã và log epoch cho từng trial:

```powershell
python -m ips.algorithms.kan.improvement prepare --output-dir runs/kan_only_new
python -m ips.algorithms.kan.improvement extend --output-dir runs/kan_only_new
python -u -m ips.algorithms.kan.improvement trial --output-dir runs/kan_only_new --name same_point --seed 42
python -m ips.algorithms.kan.improvement summarize --output-dir runs/kan_only_new
python -m ips.algorithms.kan.improvement_analysis --output-dir runs/kan_only_new
```

Controller chỉ train và calibration trên validation. Không đổi split, metric
hay dùng test chọn cấu hình. `kan_only_scope.json` ghi phạm vi, còn
`screening.json` tách riêng design bị loại. Mốc dưới 6 m trên validation chưa
đồng nghĩa đạt dưới 6 m trên official test.

Tùy chọn `positioning.head_transform="kan"` thay head cuối bằng hai
KANLinear B-spline; mặc định `linear_gelu` giữ checkpoint cũ. Đây là thay đổi
nội bộ của một KAN-Mixer, cần so sánh thêm tham số, thời gian và sai số.

Refit sau khi chốt hỗ trợ `training.same_point_mixing` và `ema_decay` giống
lượt chọn mô hình; không tính lại validation/test. `training_top_k` và
`training_temperature` giữ decoder lúc train, còn `top_k`/`temperature` giữ
decoder suy luận đã chọn. Checkpoint calibration cũ chưa có training_config
phải được đối chiếu JSON train gốc qua hash trong selection lock.

`same_point_partner_policy="different_phone_first"` chỉ dùng PHONEID chọn cặp
train cùng tọa độ. `same_point_consistency_weight>0` ghép hai view/cùng vị trí,
huấn luyện một mạng với supervised loss và Jensen–Shannon consistency trên
phân phối điểm tham chiếu. Inference không cần cặp/PHONEID. Một bước train xử
lý gấp đôi số view nên phải so sánh thời gian/bộ nhớ và số view bên cạnh epoch.

Sau khi đã đăng ký đầy đủ cấu hình dự kiến, khóa kế hoạch xác nhận. Lệnh
`confirm` chỉ chạy khi toàn bộ lượt sàng lọc seed42 đã hoàn tất:

```powershell
python -m ips.algorithms.kan.confirmation prepare --output-dir runs/kan_only_new
python -u -m ips.algorithms.kan.confirmation confirm --output-dir runs/kan_only_new
```

Kế hoạch v3 giữ 10 seed cố định và hash nguồn. Pool xác nhận gồm hai ứng viên
tốt nhất theo toàn validation seed42, ứng viên tốt nhất ở các tọa độ chưa có
trong train và đối chứng budget100. Chạy pool trên42/43/44, chọn theo trung bình
đều sai số của hai nhóm tọa độ đã có/chưa có trong train; sau đó ứng viên thắng
mới chạy đủ10 seed. Đây là tiêu chí chọn kiến trúc trên validation, không phải
định nghĩa mới của mean2D báo cáo. Checkpoint và decoder vẫn chọn bằng mean2D
trên toàn3912 dòng validation. Không chọn seed theo test và không tạo ensemble.
Controller chỉ xuất selection_lock sau đủ bằng chứng; nó không chạy
official test. Không sửa mã/config đã khóa trong khi xác nhận; nếu cần sửa lỗi,
bảo toàn kế hoạch cũ và lập phiên bản mới trước khi tiếp tục, ghi rõ lý do.

Driver cuối ở `scripts/kan_under6_final.py`, ngoài phần source đã khóa:

```powershell
python scripts/kan_under6_final.py prepare --output-dir runs/kan_only_new
# Chỉ chạy sau khi confirmation đã tạo selection_lock.json
python -u scripts/kan_under6_final.py refit --output-dir runs/kan_only_new
python -u scripts/kan_under6_final.py evaluate --output-dir runs/kan_only_new
python -u scripts/kan_under6_final.py benchmark --output-dir runs/kan_only_new
```

Plan chốt trước chế độ chính là refit toàn development; train-only báo riêng.
Đánh giá test yêu cầu đủ mọi refit và manifest checkpoint trước khi đọc nhãn.
Driver kiểm lại sai số từ CSV trên đủ1111 dòng, giữ metric gốc, không chọn seed
theo test. Benchmark dùng RSSI train cố định, batch1/64 và process riêng.

Ensemble/refit có mã và tests kỹ thuật, chưa tạo số liệu trong báo cáo ba seed hiện tại.
Refit toàn development là giao thức riêng: validation cũ trở thành train, nên
CLI/Factory từ chối báo validation đó như held-out. Không dùng nó để chọn decoder lại.
FastKAN/S-KAN chưa được triển khai trong nhánh này.

### Vòng cải tiến không gian v2

Vòng `kan_joint_under6_20261020_v1` đã đóng: mean test qua10 model refit là
**7,424524 ±0,293599 m**, chưa đạt6 m. Mean validation5,694870 m không phải
kết quả test. Mã nguồn/config, checkpoint,26 CSV đánh giá và8 benchmark của
vòng đó được giữ nguyên; có snapshot mã nguồn và audit tính lại toàn bộ CSV.

`spatial_research.py` chuẩn bị vòng mới có đối chứng cấu hình cũ và các ablation:

- `relative_rank_mask`: thêm thứ hạng trung bình khi RSSI bằng nhau trong
  chính fingerprint; giữ thứ tự AP và không fit trên validation/test.
- `nearby_point_mixing`: trộn hai vị trí train khác nhau cùng tòa/tầng trong
  phạm vi5/10 đơn vị tọa độ gốc; nội suy cả RSSI và nhãn XY với cùng alpha.
  Đây là dữ liệu tổng hợp chưa được xác nhận vật lý. Không có truy xuất khi
  dự đoán và không dùng chung với same-point mixing/consistency.
- `coordinate_residual_scale`: một head KAN B-spline hai lớp trong cùng mạng
  học phần hiệu chỉnh XY cho13 lớp. Tanh chặn từng trục ở ±scale chuẩn hóa;
  scale0,05 tương ứng khoảng4,956 m **mỗi trục**, không phải bán kính2D5 m.
  Khởi tạo hiệu chỉnh bằng0 và giữ head này khi calibration decoder.

Các tùy chọn mặc định tắt; checkpoint v1/v2 cũ vẫn đọc được. Head mới không
phải một mô hình định vị độc lập. Cả train và refit dùng chung hàm chọn
augmentation `training_representation()` để giữ nhất quán.

```powershell
python -m ips.algorithms.kan.spatial_research prepare --output-dir runs/kan_spatial_v2_new
python scripts/kan_under6_final.py prepare --output-dir runs/kan_spatial_v2_new
python -u -m ips.algorithms.kan.spatial_research screen --output-dir runs/kan_spatial_v2_new
python -u -m ips.algorithms.kan.confirmation confirm --output-dir runs/kan_spatial_v2_new
# Sau khi có selection_lock.json, chạy refit/evaluate/benchmark như phần trên.
```

Vòng này là nghiên cứu thăm dò tiếp sau khi đã biết kết quả test của vòng
trước. Cấu hình/decoder mới chọn bằng validation và chốt trước lượt test mới;
không mô tả official test là một bộ hoàn toàn chưa từng được xem.

Chi tiết từng file và giới hạn: [Giải thích thay đổi](GIAI_THICH_THAY_DOI_2026-10-09.md).

[Hợp đồng Algorithm](../base.py) · [README dự án](../../../../README.md).

## UJI huấn luyện riêng từng tòa

`uji_buildings.py` chạy ba KAN-Mixer tham chiếu riêng cho building 0, 1, 2 ở
mỗi seed 42/43/44. Giữ nguyên các dòng của split `uji_group_split_v1`, 520 AP
và cấu hình training `accuracy_spatial`; tâm/scale tọa độ và catalogue chỉ fit
từ train của từng tòa. Đây là điều kiện **đã biết tòa** để chọn mạng, không phải
hệ thống tự nhận diện tòa từ RSSI. Đơn vị vật lý của sigma 0,05 chuẩn hóa cũng
thay đổi theo scale mỗi tòa nên chưa phải ablation chỉ thay số lượng mạng.

```powershell
python -m ips.algorithms.kan.uji_buildings prepare --output-dir runs/uji_buildings_new
python -u -m ips.algorithms.kan.uji_buildings train --output-dir runs/uji_buildings_new --device cuda
python -u -m ips.algorithms.kan.uji_buildings evaluate --output-dir runs/uji_buildings_new --device cuda
python -m ips.algorithms.kan.uji_buildings_report --output-dir runs/uji_buildings_new
```

`train --smoke --device cpu` kiểm tra riêng trên ít mẫu, không tạo checkpoint
dùng đánh giá test. `evaluate` đợi đủ chín checkpoint pure đã khóa; decoder
cuối kế thừa cấu hình UJI theo seed. Không hiệu chỉnh lại trên test. Các file
training/calibration UJI cũ trong `runs/` là điều kiện cần và được ghi hash vào
protocol. Mỗi output-dir phải mới; các checkpoint/kết quả cũ được giữ nguyên.

Báo cáo gồm mạng riêng, mạng chung gốc và đối chứng mạng chung biết tòa (chỉ
giới hạn lớp trong decoder, không đổi trọng số). Mean 2D từng tòa được tính trên
cùng các dòng test. Với mỗi seed, tính trung bình đều ba mean, sau đó mới tổng
hợp mean/SD qua seed; xuất thêm trung bình theo số mẫu và predictions CSV để
kiểm tra lại. Test UJI đã được xem trong các thực nghiệm trước nên đây là lượt
thăm dò tiếp theo, không phải đánh giá trên một test mới chưa từng được xem.
