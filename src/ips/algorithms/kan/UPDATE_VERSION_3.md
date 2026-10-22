# Update Version 3 — Báo cáo thay đổi KAN-Mixer trên UJIIndoorLoc

Ngày cập nhật: **22/10/2026** · Nhánh: **`update-version-3`** · Nền mã: `main` tại `adef7d7` (nhánh nghiên cứu cũ: `feature/kan-duoi-sau-met`).

Tài liệu này thay tên `NGHIEN_CUU_DUOI_SAU_MET.md`. Phần tổng hợp đọc trực tiếp mã hiện có, JSON cấu hình, checkpoint và CSV dự đoán của hai vòng nghiên cứu. Nhật ký 123 mục trước khi đổi tên được giữ nguyên ở Phụ lục D. “Version 3” là phiên bản bàn giao và báo cáo. Backend của các model được ghi nhận vẫn là efficient-kan B-spline.

**Thực nghiệm đã dừng theo yêu cầu.** Lần cập nhật này chỉ viết báo cáo và đổi tên file/nhánh; không huấn luyện, refit, chạy suy luận test mới, commit, push hoặc merge.

## Mục lục

- [1. Kết quả cần đọc trước](#muc-1)
- [2. Những gì được giữ cố định và những gì thực sự thay đổi](#muc-2)
- [3. Thuật toán đang dùng — từ input đến tọa độ](#muc-3)
- [4. Thay đổi về tăng cường dữ liệu, tối ưu và các lựa chọn đã thử](#muc-4)
- [5. Toàn bộ cấu hình chính và cách đọc các giá trị](#muc-5)
- [6. Các cấu hình đã sàng lọc và mọi lượt train đã hoàn tất](#muc-6)
- [7. Toàn bộ kết quả test và model có sai số thấp nhất](#muc-7)
- [8. Vì sao có mức giảm — bằng chứng nào đủ mạnh, điều gì chưa chứng minh](#muc-8)
- [9. Thời gian, bộ nhớ và điều kiện chạy](#muc-9)
- [10. Mọi file thêm/sửa trong phần bàn giao](#muc-10)
- [11. Tính toàn vẹn, gián đoạn và quyết định dừng](#muc-11)
- [12. Các tệp để kiểm lại và tài liệu tham khảo](#muc-12)
- [Phụ lục A: 35 JSON cấu hình hai vòng](#phu-luc-a)
- [Phụ lục B: sáu JSON thiết kế/CPU](#phu-luc-b)
- [Phụ lục C: decoder của từng seed](#phu-luc-c)
- [Phụ lục D: toàn bộ nhật ký cũ](#phu-luc-d)

<a id="muc-1"></a>

## 1. Kết quả cần đọc trước

| Phạm vi | Kết quả mean2D test | Cách hiểu |
|---|---|---|
| Một model thấp nhất trong 33 trường hợp v1/v2 đã đối chiếu | 7,088622 m | v1, same_point_relative, seed 47, refit 31 epoch |
| Nhóm cấu hình có mean nhiều seed thấp nhất | 7,424524 ± 0,293599 m | v1: đủ 10 model refit độc lập |
| Một model thấp nhất của vòng v2 mới nhất | 7,409123 m | nearby_residual, seed 2024, train-only |
| Trung bình lần đánh giá sớm mới nhất | 7,724358 ± 0,401037 m | v2: bảy model train-only; chưa refit |
| Seed 999 mới chạy xong | 7,695771 m | v2: train 70 epoch, checkpoint chọn ở epoch 50 |

Mỗi model đều dùng **đủ 1.111 mẫu test**, một mạng chung cho cả **ba tòa và 13 tổ hợp tòa/tầng**. Không có mô hình riêng từng tòa, không chọn tầng thật để định tuyến, không loại mẫu khó và không ensemble. Dấu ± là **sample SD giữa các model**, không phải khoảng tin cậy hoặc SD của khoảng cách từng mẫu.

**Chưa có bằng chứng đạt dưới 7 m, càng chưa thể báo cáo đạt 6 m.** Giá trị 7,088622 m là mức thấp nhất của một seed sau khi xem kết quả, chỉ dùng mô tả. Nó không thay kết quả nhóm 10 seed và không phải checkpoint được chọn trước test để triển khai. Seed 42 mới là seed minh họa/triển khai đã cố định trước test trong kế hoạch cũ.

Phạm vi “thấp nhất” ở đây là 26 trường hợp đã test của vòng v1 và bảy trường hợp của vòng v2 trên cùng giao thức UJI huấn luyện chung. Không trộn kết quả KAN lai kNN, thí nghiệm tách tòa hoặc các bộ dữ liệu khác vào phép lấy minimum này.

<a id="muc-2"></a>

## 2. Những gì được giữ cố định và những gì thực sự thay đổi

### 2.1. Dữ liệu và phép đo

- Tập development UJI có 19.937 dòng; train nội bộ **16.025**, validation **3.912**. Test là toàn bộ **1.111** dòng của `validationData.csv` gốc UCI. “Validation” trong tên CSV UCI không phải validation nội bộ của dự án.
- Giữ thứ tự `WAP001`…`WAP520`, nhãn tòa/tầng, row index và `uji_group_split_v1`. Manifest/hash và nhóm đo được kiểm trước khi chạy. Nhóm đo không giao nhau không đồng nghĩa người đo, điện thoại hoặc tọa độ đều mới.
- Trong validation có 3.053 dòng tại tọa độ đã có trong train và 859 dòng tại tọa độ chưa có. Score cân bằng hai nhóm chỉ chọn kiến trúc; chỉ số báo cáo test vẫn là mean2D trên mọi dòng.
- Chỉ fit biến đổi học từ dữ liệu và catalogue trên train. Sau khi khóa lựa chọn, refit v1 mới được dùng cả development; validation cũ khi đó không còn là tập đánh giá độc lập.
- Hai cột tọa độ là **LONGITUDE rồi LATITUDE**. Mọi sai số được tính bằng khoảng cách Euclidean trực tiếp sau đảo chuẩn hóa, không haversine, không nhân cos vĩ độ, không co thang để làm nhỏ số. Metadata hiện có `crs_epsg=null`; báo cáo không tự gán một mã EPSG đã được tác giả dữ liệu xác nhận. Ký hiệu m theo thang benchmark đang dùng; phép tính và hệ số gốc được ghi cụ thể dưới đây để kiểm lại.

### 2.2. Ba thế hệ cấu hình trong nhánh này

1. **Đối chứng `budget100`:** tín hiệu tuyệt đối, mask_scale=0, head điểm tham chiếu; patience=100 để đối chiếu ngân sách. Checkpoint được chọn ở ba seed đã kiểm tương đương baseline lịch sử 8,058153 m.
2. **Vòng v1 `same_point_relative`:** thêm RSSI tương đối, mask hữu dụng và trộn các bản ghi cùng vị trí; tiếp đó refit development 31 epoch theo lựa chọn đã khóa. Hai block và công thức KAN cơ sở vẫn giữ.
3. **Vòng v2 `nearby_residual`:** giữ RSSI tương đối nhưng thay same-point mixing bằng nearby-point mixing bán kính10, đồng thời thêm head KAN hiệu chỉnh tọa độ trong cùng mạng. Đây là model đang có kết quả test sớm bảy seed; **chưa có refit v2**.

Thay đổi v2 về rank tín hiệu, dropout mạnh, bán kính5 và residual đơn lẻ là các đối chứng có chạy validation. Chúng không được gán thành những thành phần đồng thời có mặt trong `nearby_residual`.

<a id="muc-3"></a>

## 3. Thuật toán đang dùng — từ input đến tọa độ

### 3.1. Biểu diễn RSSI và mask

Một mẫu là 520 cường độ tín hiệu. Giá trị100 biểu thị không nghe được AP, không phải tín hiệu rất mạnh. Với AP thứ i:

$$m_i=\mathbf{1}[RSSI_i\ne100],\qquad z_i=m_i\frac{RSSI_i+104}{104}.$$

`m_i` là mask 0/1, `z_i` là tín hiệu chuẩn hóa. Khi RSSI=-104 thì z=0 nhưng m=1; AP vắng có z=0,m=0. Mẫu không nghe được AP nào vẫn được giữ.

Biểu diễn tương đối lấy trung bình các AP nghe được trong **chính fingerprint**, không fit trung bình từ validation/test:

$$n=\max(1,\sum_i m_i),\quad \mu_z=\frac{\sum_i m_i z_i}{n},\quad r_i=\frac{m_i(z_i-\mu_z)}{0.2}.$$

Số 0,2 tương ứng20,8 dB trên thang104 dB. Nếu mọi AP cùng lệch một mức và tập AP không đổi, phép trừ trung bình loại độ lệch chung. Đây là tính chất của biểu diễn, không chứng minh mọi sai khác điện thoại đã được xử lý.

Các mode đã cài: `absolute_mask` nhận(z,mask×scale); `absolute_relative_mask` nhận(z,r,mask×scale); `relative_mask` nhận(r,mask×scale); `relative_rank_mask` nhận(r,rank,mask×scale). Mode rank chỉ xếp hạng AP nghe được, dùng trung bình hạng khi tie, đưa hạng0…n−1 về[-1,1]; n≤1 hoặc AP vắng nhận 0. Danh tính AP không bị xáo trộn.

### 3.2. Khối KAN B-spline

KAN học các hàm một biến trên kết nối. Công thức thực thi của `KANLinear` trong vendor có thể viết:

$$y_j=\sum_i\left[w^b_{ji}\,SiLU(x_i)+s_{ji}\sum_q c_{jiq}B_{iq}(x_i)\right].$$

`w^b` là trọng số nhánh nền; `c` là hệ số các hàm cơ sở B-spline; `s` là spline_scaler riêng khi bật; `B` được xác định bởi knot grid. Mô hình học các trọng số/hệ số này. Với grid_size 5 và spline_order 3, có 8 hệ số spline trên một kết nối. Trong các lượt chính, không cập nhật knot grid và không cộng regularization_loss riêng của vendor.

Ý tưởng KAN lấy từ [Liu và cộng sự](https://arxiv.org/abs/2404.19756). Công thức thực thi trên đối chiếu [mã efficient-kan tại commit đã ghim](https://raw.githubusercontent.com/Blealtan/efficient-kan/7b6ce1c87f18c8bc90c208f6b494042344216b11/src/efficient_kan/kan.py) và file vendor tại repo. Không có FastKAN hoặc S-KAN trong các model được báo cáo.

### 3.3. Token mixing, channel mixing và head

Mạng nhận tensor[B,520,2]. Hai đặc trưng/AP được reshape thành[B,26,40], embedding Linear40→64. Mỗi block thực hiện:

$$U=X+\operatorname{transpose}^{-1}\!\left(K_t(\operatorname{transpose}(LN(X)))\right),\qquad Y=U+K_c(LN(U)).$$

`LN` là LayerNorm theo channel. `K_t` gồm KAN26→32→26, dùng chung cho các channel; `K_c` gồm KAN64→128→64, dùng chung cho các token. Mỗi nhánh có Dropout và cộng residual. Hai block dùng tổng cộng8 KANLinear; không chèn GELU giữa hai KAN trong nhánh mixing. GELU có trong head Linear của cấu hình tham chiếu.

Sau final LayerNorm, flatten26×64 thành 1664 số. Head `Linear1664→256 → GELU → Dropout → Linear256→N_RP` xuất logit cho các điểm tham chiếu. `N_RP=872` ở train nội bộ; v1 refit mở thành 933. Đây là số tọa độ phân biệt theo lớp, không phải số fingerprint.

Nguồn kiến trúc Mixer tham khảo là [KAN-Mixers v1 của Canuto và cộng sự](https://arxiv.org/abs/2503.08939v1), nghiên cứu phân loại ảnh. Token RSSI, head điểm tham chiếu, loss định vị, augmentation và quy trình refit ở đây là thiết kế IPS của dự án, không phải thông số tái lập nguyên bản bài ảnh.

| Bước | Tensor/hoạt động của cấu hình chính |
|---|---|
| Đọc RSSI và chuẩn hóa | [B,520] → [B,520,2] gồm z và mask |
| Huấn luyện | Trộn mẫu nếu bật; noise/shift/AP dropout chỉ khi train |
| Biểu diễn tương đối | [B,520,2]; rank nếu bật mới thành[B,520,3] |
| Token và embedding | [B,26,40] → [B,26,64] |
| Hai MixerBlock và final LayerNorm | [B,26,64] |
| Flatten | [B,1664] |
| Head điểm tham chiếu | [B,872] train-only hoặc[B,933] ở v1 refit |
| Head hiệu chỉnh chỉ ở v2 đã chọn | KAN1664→32→26 → reshape[B,13,2] → tanh×0,05 |
| Decoder và chọn lớp | [B,13] logits; tọa độ điều kiện[B,13,2] → XY[B,2] |
| Đảo chuẩn hóa, ánh xạ class | XY hai cột gốc; building và floor mỗi loại[B] |

### 3.4. Decoder tọa độ

Gọi s_j là logit của điểm tham chiếu j, a_j là tọa độ chuẩn hóa, C_c là các điểm thuộc lớp tòa/tầng c:

$$\ell_c=\log\sum_{j\in C_c}e^{s_j},\qquad \hat c=\arg\max_c\ell_c.$$

Trong mỗi lớp, chọn các điểm có logit cao nhất rồi lấy trung bình có trọng số:

$$w_j=\frac{e^{s_j/T}}{\sum_{q\in TopK(C_c)}e^{s_q/T}},\qquad \hat a_c=\sum_{j\in TopK(C_c)}w_j a_j.$$

`T` là temperature. T nhỏ làm trọng số tập trung hơn. Chương trình pad catalogue mọi lớp đến M=max số điểm/lớp rồi gọi topk với min(k,M). Ô pad có logit−∞ nên trọng số 0; số điểm hợp lệ đóng góp thực tế không vượt số điểm của lớp đó. Vì vậy k=1000 không phải lấy 1.000 fingerprint gần nhất và cũng không chỉ đơn giản chặn theo tổng 872/933 điểm.

Suy luận lấy lớp do mạng dự đoán, không dùng nhãn thật của query. Trong loss tọa độ, nhãn lớp thật chỉ chọn nhánh cần tính loss, giống cách dùng nhãn để huấn luyện phân loại.

### 3.5. Head KAN hiệu chỉnh của v2

Với vector đặc trưng h có 1664 số, hai KANLinear tạo13 cặp hiệu chỉnh:

$$\delta_c=0.05\tanh\left([K_2(K_1(h))]_c\right),\qquad \hat y_c=\hat a_c+\delta_c.$$

Hai lớp có chiều 1664→32→26. Chỉ nhánh cuối được zero-init ở base_weight và spline_weight; giữ spline_scaler khác0 để vẫn có gradient học spline. Ban đầu hiệu chỉnh bằng 0, giúp khởi đầu giống decoder cũ. Nhánh này nằm **trong cùng mạng**, không phải bộ định vị thứ hai hay kNN.

0,05 giới hạn từng trục tọa độ chuẩn hóa, tương đương khoảng 4,955751 đơn vị mặt phẳng gốc trên mỗi trục. Nó **không** là bán kính2D và không bảo đảm sai số cuối dưới 5 m. Head được cộng vào cả tọa độ dense dùng train và tọa độ top-k dùng suy luận/calibration. Cấu hình tốt nhất lịch sử7,088622 m **không bật head này**.

Với cùng catalogue872 điểm ở train nội bộ, v1 có **1.014.568** tham số; thêm head residual đưa v2 lên **1.555.368**, tăng đúng **540.800** tham số. Riêng v1 refit mở catalogue thêm61 điểm làm head Linear cuối tăng61×(256+1)=**15.677** tham số, thành **1.030.245**; đó là thay đổi kích thước đầu ra, không phải thêm lớp KAN mixing.

### 3.6. Loss và đảo chuẩn hóa

Các điểm cùng lớp với nhãn train nhận phân bố mục tiêu Gaussian:

$$q_j=\frac{\exp(-\|a_j-y\|^2/(2\sigma^2))}{\sum_{r\in C_c}\exp(-\|a_r-y\|^2/(2\sigma^2))},\quad j\in C_c;\qquad q_j=0\ \text{nếu khác lớp}.$$

$$L=0.5L_{class}+1.0L_{RP}+3.0L_{xy},\qquad L_{RP}=-\sum_jq_j\log p_j.$$

`L_class` là cross entropy lớp tòa/tầng; `L_RP` là cross entropy trên toàn bộ logit RP với nhãn mềm; `L_xy` là trung bình sqrt(||tọa độ dự đoán−tọa độ đích||²+10⁻⁸) trong hệ chuẩn hóa. Mặc định dùng tọa độ **dense** của lớp thật khi train; biến thể `coordinate_source=topk` là đối chứng khác. Khi sigma=0, mã dùng nhãn điểm gần nhất và CE nhãn cứng. Trường `smooth_l1_beta=1` còn trong schema chung nhưng **không được loss nhánh reference sử dụng**.

Sigma=0,05 là **độ rộng Gaussian trên tọa độ chuẩn hóa**, không phải ngưỡng khoảng cách hoặc bán kính chặt. Giá trị0,05 này khác vai trò với coordinate_residual_scale0,05 dù cùng con số.

$$\hat{XY}_{goc}=s\hat y+\mu,\qquad \mu=(-7463.915495586015,4864870.4533671485),\quad s=99.11502300196764.$$

Tâm và shared_scale được fit từ 16.025 dòng train nội bộ, dùng chung hai trục, giữ nguyên khi refit. Tọa độ được đảo về float64 trước tính metric. Không cộng độ cao giả định vào mean2D nếu sai tầng; floor accuracy được báo riêng.

<a id="muc-4"></a>

## 4. Thay đổi về tăng cường dữ liệu, tối ưu và các lựa chọn đã thử

### 4.1. Same-point mixing, phiên bản v1

Với xác suất0,5, chọn một **bản ghi khác** có cùng lớp và đúng cùng tọa độ gốc. Alpha lấy đều[0,2;0,8]. AP nghe ở cả hai nguồn được nội suy; AP chỉ có ở một nguồn được chọn theo xác suất alpha để mask vẫn nhị phân. Tọa độ đích không đổi. Nếu không có đối tác thì giữ mẫu gốc. Biến thể `different_phone_first` ưu tiên đối tác khác điện thoại nếu có; PHONEID chỉ phục vụ sampler train, không đưa vào mạng.

### 4.2. Nearby-point mixing, phiên bản v2

Thay đối tác cùng tọa độ bằng một vị trí **khác tọa độ**, cùng lớp và không quá bán kính5 hoặc10 của mặt phẳng gốc. Chọn đều một vị trí phù hợp rồi chọn đều một bản ghi của vị trí đó, tránh tự ưu tiên điểm có nhiều lượt quét. Tín hiệu/mask trộn như trên; tọa độ đích cũng phải trộn: y_mix=alpha·y1+(1−alpha)·y2. Không dùng đối tác từ validation/test. V1 refit được dùng development sau khóa; v2 chưa chạy refit.

Đây là giả định tăng cường về tín hiệu tại vị trí trung gian, không phải fingerprint đã đo vật lý. Không có bước tìm láng giềng từ RSSI query trong suy luận. Mã từ chối bật nearby mixing cùng same-point mixing/consistency để tránh cách diễn giải mơ hồ.

### 4.3. Nhiễu và AP dropout

Trong forward ở chế độ train, cộng Gaussian RSSI noise theo từng AP và Gaussian shift chung theo fingerprint, sau đó AP dropout, clamp về[0,1] và nhân mask. Cấu hình chính giữ noise 2 dB, shift 3 dB và dropout 0,05. Các thành phần này đã có ở đối chứng; không được kể toàn bộ chúng như kỹ thuật mới làm giảm sai số ở v1/v2.

### 4.4. Những lựa chọn có mã nhưng không cùng xuất hiện trong model cuối

- **Learned pooling:** Linear64→32, tanh, Linear32→1, softmax theo token; đối chứng với flatten/mean. Không được chọn vào hai cấu hình chính.
- **Classifier phụ:** một Linear dự đoán13 lớp từ representation, có thể cộng vào logsumexp lớp và thêm CE phụ. Calibration phải giữ logits này nếu model có bật; các cấu hình chính không bật.
- **Head KAN cho toàn bộ logit RP:** thay head Linear bằng hai KANLinear. Khác với head residual nhỏ của v2; không được gọi hai lựa chọn là một. Chưa được chọn ở v1.
- **EMA:** cập nhật shadow weights sau optimizer step bằng decay hiệu dụng min(decay,(1+steps)/(10+steps)); copy buffers. Khi bật, checkpoint validation chọn weights EMA; lưu live weights riêng. Không phải ensemble khi suy luận. Các cấu hình chính không bật.
- **Consistency:** hai view train của cùng vị trí đi qua cùng một mạng, thêm Jensen–Shannon divergence của phân phối RP ở temperature đã đặt. Hai view train không phải TTA. Biến thể này có chi phí khác một view; không được chọn.
- **Thay độ rộng nhãn mềm, regularization/dropout, tăng số block, token nhỏ hơn, loss top-k:** đều có thử nghiệm riêng, được liệt kê ở mục6 và phụ lục cấu hình. Cấu hình không cải thiện vẫn giữ kết quả.
- **Fingerprint memory/truy xuất, inference nhiều view:** những hướng được phác thảo trước khi người dùng giới hạn “chỉ KAN-Mixer” đã bị loại khỏi pipeline nghiên cứu. Ba JSON metric_memory/metric_centered/metric_centered_ema còn là dấu vết thiết kế; không có điểm số hợp lệ và mã hiện tại từ chối chúng.
- **Đổi knot grid từ activation train:** có thăm dò CPU riêng nhưng chưa đưa vào model của các lượt test. Công thức efficient-kan và vendor bytes không được sửa.

<a id="muc-5"></a>

## 5. Toàn bộ cấu hình chính và cách đọc các giá trị

Bảng sau đọc từ cấu hình input và checkpoint thật, không suy đoán từ tên trial. Phụ lục A chép nguyên 35 JSON input (27 v1,8 v2), bao gồm ba thiết kế bị loại. Phụ lục B chép đủ sáu JSON thiết kế/CPU, không gán kết quả test cho chúng.

| Khóa | V1 thấp nhất, seed 47 refit | V2 thấp nhất, seed 2024 train-only | Ý nghĩa |
|---|---|---|---|
| `model` | "kan_mixer" | "kan_mixer" | Mạng nội bộ KAN-Mixer; Algorithm cấp IPS vẫn là KANMixerAlgorithm. |
| `split_dir` | "data/processed/uji_group_split_v1" | "data/processed/uji_group_split_v1" | Bundle split/hash cố định của UJI. |
| `architecture_reference` | "https://arxiv.org/abs/2503.08939v1" | "https://arxiv.org/abs/2503.08939v1" | Nguồn Mixer ảnh tham khảo, không phải kết quả IPS. |
| `mixer.tokens` | 26 | 26 | Số token. |
| `mixer.features_per_token` | 40 | 40 | Số đặc trưng/token; tokens×features phải giữ đủ AP×kênh. |
| `mixer.channels` | 64 | 64 | Độ rộng embedding và channel. |
| `mixer.blocks` | 2 | 2 | Số MixerBlock. |
| `mixer.token_hidden` | 32 | 32 | Chiều ẩn trộn token. |
| `mixer.channel_hidden` | 128 | 128 | Chiều ẩn trộn channel. |
| `mixer.dropout` | 0.1 | 0.1 | Xác suất Dropout trong mạng khi train. |
| `kan.basis` | "B_spline" | "B_spline" | Hàm cơ sở B-spline. |
| `kan.grid_size` | 5 | 5 | Số khoảng lưới cơ sở. |
| `kan.spline_order` | 3 | 3 | Bậc spline. |
| `kan.base_activation` | "SiLU" | "SiLU" | SiLU của nhánh nền. |
| `kan.enable_standalone_scale_spline` | true | true | Scaler spline riêng cho từng kết nối. |
| `kan.grid_range` | [-3,3] | [-3,3] | Miền knot grid ban đầu, không phải khoảng tọa độ. |
| `kan.scale_noise` | 0.1 | 0.1 | Nhiễu khởi tạo hệ số spline, không phải RSSI noise. |
| `kan.scale_base` | 1.0 | 1.0 | Tham số scale khởi tạo nhánh nền. |
| `kan.scale_spline` | 1.0 | 1.0 | Tham số scale khởi tạo nhánh spline. |
| `kan.grid_eps` | 0.02 | 0.02 | Chỉ tham gia thuật toán đổi lưới nếu có chạy; các lượt này không chạy. |
| `kan.update_grid` | false | false | Có cập nhật lưới khi train hay không. |
| `kan.extra_regularization_loss_coefficient` | 0 | 0 | Hệ số loss regularization riêng KAN. |
| `kan.required_diagnostic` | "input_distribution_and_out_of_knot_support_fraction_per_KANLinear" | "input_distribution_and_out_of_knot_support_fraction_per_KANLinear" | Mô tả nhu cầu chẩn đoán; không tự kích hoạt diagnostic. |
| `kan.note` | "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters" | "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters" | Ghi chú provenance, không đổi phép tính. |
| `loss.smooth_l1_beta` | 1.0 | 1.0 | Trường schema chung, không dùng ở loss reference hiện tại. |
| `loss.lambda_xy` | 3.0 | 3.0 | Trọng số loss tọa độ chuẩn hóa. |
| `loss.lambda_class` | 0.5 | 0.5 | Trọng số CE tòa/tầng. |
| `loss.lambda_reference` | 1.0 | 1.0 | Trọng số CE điểm tham chiếu. |
| `loss.label_smoothing` | 0.0 | 0.0 | Chỉ cho CE nhãn cứng khi sigma=0. |
| `loss.reference_sigma_normalized` | 0.05 | 0.05 | Độ rộng Gaussian nhãn mềm, không phải bán kính. |
| `training.learning_rate` | 0.001 | 0.001 | Learning rate ban đầu của AdamW. |
| `training.weight_decay` | 0.0001 | 0.0001 | Weight decay của AdamW. |
| `training.batch_size` | 128 | 128 | Số mẫu dataset/batch; consistency có thể tạo hai view. |
| `training.max_epochs` | 100 | 100 | Trần epoch, cũng là horizon cosine trong cấu hình chính. |
| `training.patience` | 20 | 20 | Số epoch chờ validation cải thiện đủ min_delta. |
| `training.min_delta` | 0.01 | 0.01 | Mức cải thiện validation mean2D cần để thay checkpoint. |
| `training.gradient_clip_l2` | 1.0 | 1.0 | Chặn norm L2 gradient. |
| `training.threads` | 2 | 2 | Số thread PyTorch CPU. |
| `training.kan_diagnostic_batches` | 0 | 0 | Số batch diagnostic;0 nghĩa không chạy. |
| `training.scheduler` | "cosine" | "cosine" | Cosine theo epoch nếu đặt cosine. |
| `training.same_point_mixing` | 0.5 | 0.0 | Xác suất trộn bản ghi cùng vị trí. |
| `positioning.variant` | "reference" | "reference" | Head tham chiếu, khác regression_v1 cũ. |
| `positioning.pooling` | "flatten" | "flatten" | Flatten giữ đủ 26×64 phần tử. |
| `positioning.head_width` | 256 | 256 | Chiều ẩn head logit RP. |
| `positioning.mask_scale` | 1.0 | 1.0 | Hệ số mask tại embedding. |
| `positioning.top_k` | 10 | 20 | Số điểm yêu cầu trong decoder; thực tế giới hạn theo catalogue lớp. |
| `positioning.temperature` | 1.0 | 1.0 | Temperature suy luận được chọn bằng validation. |
| `positioning.rssi_noise_db` | 2.0 | 2.0 | SD noise riêng AP lúc train. |
| `positioning.device_shift_db` | 3.0 | 3.0 | SD shift chung fingerprint lúc train. |
| `positioning.ap_dropout` | 0.05 | 0.05 | Xác suất bỏ AP khi train. |
| `positioning.input_mode` | "relative_mask" | "relative_mask" | Biểu diễn tín hiệu trong mạng. |
| `positioning.fit_scope` | "development_after_validation_selection" | "internal_train_only" | Nguồn fit catalogue; train-only hoặc development sau khóa. |
| `positioning.training_temperature` | 1.0 | Không đặt; dùng mặc định | Temperature dùng train/refit, tách khỏi temperature suy luận. |
| `positioning.training_top_k` | 5 | Không đặt; dùng mặc định | k dùng train/refit, tách khỏi k suy luận. |
| `training.nearby_point_mixing` | Không đặt; dùng mặc định | 0.5 | Xác suất trộn hai vị trí khác nhau cùng lớp. |
| `training.nearby_point_radius_m` | Không đặt; dùng mặc định | 10.0 | Bán kính chọn vị trí train trên mặt phẳng gốc. |
| `positioning.coordinate_residual_scale` | Không đặt; dùng mặc định | 0.05 | Giới hạn hiệu chỉnh từng trục chuẩn hóa;0 là tắt. |
| `positioning.coordinate_residual_width` | Không đặt; dùng mặc định | 32 | Chiều ẩn head KAN hiệu chỉnh. |

| Đại lượng thực thi | V1 seed 47 | V2 seed 2024 |
|---|---|---|
| Optimizer | AdamW | AdamW |
| LR tối thiểu cosine | 0,00005 | 0,00005 |
| Head mặc định | linear_gelu | linear_gelu |
| Auxiliary/EMA/consistency | Tắt | Tắt |
| coordinate_source | dense (mặc định) | dense (mặc định) |
| Số reference points | 933 | 872 |
| Tham số trainable | 1030245 | 1555368 |
| KANLinear trong model | 8 ở mixing | 8 ở mixing +2 ở residual |
| Epoch của model được test | 31 | 36 |
| Số epoch đã chạy trong lượt | 31 refit cố định | 56, chọn best36 |
| Dữ liệu train của model test | 19.937 development | 16.025 train nội bộ |
| Nguồn lựa chọn epoch/decoder | Validation giai đoạn trước refit | Validation nội bộ |

Checkpoint chứa cả mảng933 hoặc872 tọa độ/nhãn catalogue. Đó là dữ liệu fit được lưu cùng model, không phải933 siêu tham số cần tự đặt. Số lượng, fit_scope, checkpoint path và SHA256 được giữ trong báo cáo; không in lặp hàng nghìn tọa độ vào bảng cấu hình.

### 5.1. Cấu hình checkpoint V1 có sai số thấp nhất

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 10,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "relative_mask",
    "fit_scope": "development_after_validation_selection",
    "training_temperature": 1.0,
    "training_top_k": 5,
    "catalogue_summary_NOT_A_RUNTIME_KEY": {
      "count": 933,
      "source": "checkpoint references; coordinate/class arrays are training data, not hyperparameters"
    }
  }
}
```

`catalogue_summary_NOT_A_RUNTIME_KEY` chỉ là ghi chú rút gọn của báo cáo, không được đưa nguyên vào cấu hình chạy. Toàn bộ `references` gốc nằm trong checkpoint đã kiểm hash sau:

[runs/kan_joint_under6_20261020_v1/final_refit/same_point_relative/seed47/training/best.pt](../../../../runs/kan_joint_under6_20261020_v1/final_refit/same_point_relative/seed47/training/best.pt)

SHA256: `f283c5760d05f12a73bd2e7844255600ab94b8995fb31f90c33b4b897e263e4d`.

### 5.2. Cấu hình checkpoint V2 thấp nhất trong lần test mới

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.0,
    "nearby_point_mixing": 0.5,
    "nearby_point_radius_m": 10.0
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 20,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "relative_mask",
    "coordinate_residual_scale": 0.05,
    "coordinate_residual_width": 32,
    "fit_scope": "internal_train_only",
    "catalogue_summary_NOT_A_RUNTIME_KEY": {
      "count": 872,
      "source": "checkpoint references; coordinate/class arrays are training data, not hyperparameters"
    }
  }
}
```

[runs/kan_joint_under6_20261021_v2/trials/nearby_residual/seed2024/calibration/pure_selected.pt](../../../../runs/kan_joint_under6_20261021_v2/trials/nearby_residual/seed2024/calibration/pure_selected.pt)

SHA256: `8606849c1e800515b8dfcd7b510c1805ab4a35d1bccfc0224fa852c7f7fff9fd`.


### 5.3. Các mặc định thực thi cần biết khi tái lập

Những giá trị sau không phải khóa mới của JSON; chúng được xác định từ mã và mặc định AdamW trong đúng môi trường PyTorch 2.9.0+cu128 đã dùng:

| Thành phần | Giá trị thực thi |
|---|---|
| AdamW betas; epsilon | (0,9; 0,999); 10⁻⁸ |
| AdamW amsgrad / maximize / capturable / differentiable | false / false / false / false |
| AdamW foreach / fused | Không ép chọn (`None`); để thư viện quyết định nhánh hỗ trợ |
| Weight decay | 0,0001, tách khỏi gradient loss theo AdamW |
| Train DataLoader | shuffle=true, num_workers=0, drop_last=false; generator có seed |
| Validation trong train | batch 128, shuffle=false, num_workers=0 |
| Calibration decoder | batch 128; k∈{1,3,5,10,20,1000}, T∈{0,5;1;2;4}; đủ 24 tổ hợp |
| Test inference | batch 64, shuffle=false, num_workers=0, model.eval(), không augmentation |
| Thread PyTorch | Train/refit:2; lối evaluate hiện tại dùng mặc định configure_runtime:4 |
| Tái lập số ngẫu nhiên | Seed Python/NumPy/Torch/CUDA; deterministic algorithms=true; cuDNN benchmark=false |
| CUBLAS_WORKSPACE_CONFIG | setdefault `:4096:8`; nếu biến đã tồn tại thì không ghi đè |
| Dtype | Feature/đích chuẩn hóa float32; tọa độ gốc và metric sau đảo chuẩn hóa float64 |
| Early stopping | Checkpoint đổi khi validation mean2D cải thiện hơn min_delta=0,01 m; patience đếm theo checkpoint đã lưu |

Horizon cosine của các lượt chính là 100 epoch, eta_min=0,001×0,05=0,00005. Refit v1 chạy 31 epoch nhưng giữ horizon 100. Trong biến thể consistency, mỗi mẫu dataset tạo hai view nên batch forward có thể gấp đôi; không được coi chi phí tính toán là bằng nhau chỉ vì batch_size trong JSON giống nhau.

<a id="muc-6"></a>

## 6. Các cấu hình đã sàng lọc và mọi lượt train đã hoàn tất

Có 24 cấu hình v1 được phép và tám cấu hình v2. Ba JSON metric-memory của v1 không thuộc danh sách được phép, không có điểm validation/test hợp lệ. Bảng dưới là seed 42 sau chọn decoder bằng validation. Không dùng cột này để tuyên bố test dưới 6/7 m.

### 6.1 Sàng lọc v1

| ID | Val mean2D m | k/T | Best epoch | Epoch chạy | Mọi override so với baseline input |
|---|---|---|---|---|---|
| aligned_aux_relative | 5,748362 | 1000/0.5 | 46 | 66 | mixer.features_per_token=60; loss.coordinate_source="topk"; loss.lambda_auxiliary_class=0.5; positioning.mask_scale=1.0; positioning.top_k=20; positioning.input_mode="absolute_relative_mask"; positioning.auxiliary_classifier=true |
| aligned_topk | 5,866303 | 1000/0.5 | 29 | 49 | loss.coordinate_source="topk"; positioning.top_k=20 |
| auxiliary | 5,854196 | 20/0.5 | 13 | 33 | loss.lambda_auxiliary_class=0.5; positioning.auxiliary_classifier=true; positioning.class_logit_weight=1.0 |
| budget100 | 6,032527 | 1000/0.5 | 21 | 100 | training.patience=100 |
| cross_phone_consistency | 5,727462 | 20/0.5 | 36 | 100 | training.patience=100; training.same_point_consistency_weight=1.0; training.same_point_partner_policy="different_phone_first"; training.consistency_temperature=1.0 |
| deeper_same_point | 5,697390 | 20/0.5 | 71 | 100 | mixer.blocks=3; training.patience=100; training.same_point_mixing=0.5 |
| ema | 5,979415 | 3/0.5 | 54 | 74 | training.ema_decay=0.995 |
| fine_tokens | 5,862838 | 20/1.0 | 36 | 56 | mixer.tokens=52; mixer.features_per_token=20; mixer.token_hidden=64 |
| kan_head | 5,918121 | 10/0.5 | 21 | 100 | training.patience=100; positioning.head_transform="kan" |
| learned_pool | 6,344939 | 20/0.5 | 36 | 56 | positioning.pooling="learned" |
| mask | 6,167652 | 20/1.0 | 13 | 33 | positioning.mask_scale=1.0 |
| metric_centered | Không có; đã loại | — | — | — | positioning.mask_scale=1.0; positioning.head_type="metric_memory"; positioning.metric_dimension=128; positioning.metric_temperature=0.1; positioning.input_mode="relative_mask"; positioning.metric_centering=true |
| metric_centered_ema | Không có; đã loại | — | — | — | training.ema_decay=0.995; positioning.mask_scale=1.0; positioning.head_type="metric_memory"; positioning.metric_dimension=128; positioning.metric_temperature=0.1; positioning.input_mode="relative_mask"; positioning.metric_centering=true |
| metric_memory | Không có; đã loại | — | — | — | positioning.mask_scale=1.0; positioning.head_type="metric_memory"; positioning.metric_dimension=128; positioning.metric_temperature=0.1; positioning.input_mode="relative_mask" |
| regularized | 5,904866 | 20/1.0 | 36 | 56 | mixer.dropout=0.2; training.weight_decay=0.01 |
| relative | 5,905599 | 20/0.5 | 13 | 33 | mixer.features_per_token=60; positioning.mask_scale=1.0; positioning.input_mode="absolute_relative_mask" |
| relative_ema | 6,185306 | 3/0.5 | 12 | 32 | mixer.features_per_token=60; training.ema_decay=0.995; positioning.mask_scale=1.0; positioning.input_mode="absolute_relative_mask" |
| relative_only | 5,896253 | 20/1.0 | 13 | 33 | positioning.mask_scale=1.0; positioning.input_mode="relative_mask" |
| robust_aug | 6,017944 | 20/0.5 | 36 | 56 | positioning.rssi_noise_db=4.0; positioning.device_shift_db=6.0; positioning.ap_dropout=0.1 |
| same_point | 5,462713 | 10/0.5 | 87 | 100 | training.same_point_mixing=0.5 |
| same_point_aux_topk | 5,658655 | 10/0.5 | 31 | 51 | loss.coordinate_source="topk"; loss.lambda_auxiliary_class=0.5; training.same_point_mixing=0.5; positioning.top_k=20; positioning.auxiliary_classifier=true |
| same_point_consistency | 5,915659 | 10/0.5 | 66 | 100 | training.patience=100; training.same_point_mixing=0.5; training.same_point_consistency_weight=1.0; training.same_point_partner_policy="different_phone_first"; training.consistency_temperature=1.0 |
| same_point_cross_phone | 5,484101 | 20/0.5 | 87 | 100 | training.patience=100; training.same_point_mixing=0.5; training.same_point_partner_policy="different_phone_first" |
| same_point_kan_head | 5,505775 | 20/1.0 | 77 | 100 | training.patience=100; training.same_point_mixing=0.5; positioning.head_transform="kan" |
| same_point_relative | 5,582541 | 10/0.5 | 26 | 46 | training.same_point_mixing=0.5; positioning.mask_scale=1.0; positioning.input_mode="relative_mask" |
| sigma_tight | 6,200502 | 20/1.0 | 11 | 31 | loss.reference_sigma_normalized=0.025 |
| sigma_wide | 6,017176 | 1000/0.5 | 40 | 60 | loss.reference_sigma_normalized=0.08 |

### 6.2 Sàng lọc v2

| ID | Val mean2D m | k/T | Best epoch | Epoch chạy | Mọi override so với baseline input |
|---|---|---|---|---|---|
| budget100 | 6,032527 | 1000/0.5 | 21 | 100 | training.patience=100 |
| nearby_10 m | 5,667143 | 20/0.5 | 40 | 60 | training.same_point_mixing=0.0; training.nearby_point_mixing=0.5; training.nearby_point_radius_m=10.0; positioning.mask_scale=1.0; positioning.input_mode="relative_mask" |
| nearby_5 m | 5,774471 | 10/0.5 | 19 | 39 | training.same_point_mixing=0.0; training.nearby_point_mixing=0.5; training.nearby_point_radius_m=5.0; positioning.mask_scale=1.0; positioning.input_mode="relative_mask" |
| nearby_residual | 5,556056 | 20/1.0 | 26 | 46 | training.same_point_mixing=0.0; training.nearby_point_mixing=0.5; training.nearby_point_radius_m=10.0; positioning.mask_scale=1.0; positioning.input_mode="relative_mask"; positioning.coordinate_residual_scale=0.05; positioning.coordinate_residual_width=32 |
| rank_relative | 5,756392 | 20/0.5 | 26 | 46 | mixer.features_per_token=60; training.same_point_mixing=0.5; positioning.mask_scale=1.0; positioning.input_mode="relative_rank_mask" |
| relative_dropout | 5,803370 | 1000/0.5 | 26 | 46 | training.same_point_mixing=0.5; positioning.mask_scale=1.0; positioning.rssi_noise_db=4.0; positioning.ap_dropout=0.15; positioning.input_mode="relative_mask" |
| residual_axis5 m | 5,552420 | 20/1.0 | 45 | 65 | training.same_point_mixing=0.5; positioning.mask_scale=1.0; positioning.input_mode="relative_mask"; positioning.coordinate_residual_scale=0.05; positioning.coordinate_residual_width=32 |
| v1_relative | 5,582541 | 10/0.5 | 26 | 46 | training.same_point_mixing=0.5; positioning.mask_scale=1.0; positioning.input_mode="relative_mask" |

### 6.3. Chọn kiến trúc bằng ba seed

| Ứng viên v2 | Val toàn bộ m | Tọa độ đã có m | Tọa độ chưa có m | Score cân bằng |
|---|---|---|---|---|
| nearby_residual | 5,444497 | 5,093031 | 6,693653 | 5,893342 |
| residual_axis5 m | 5,534391 | 5,225615 | 6,631822 | 5,928719 |
| v1_relative | 5,657807 | 5,296138 | 6,943224 | 6,119681 |
| budget100 | 5,976738 | 5,525653 | 7,579954 | 6,552804 |

`nearby_residual` được chọn theo score cân bằng trước test. Chọn kiến trúc không có nghĩa chứng minh cải thiện có ý nghĩa thống kê. Ba seed xác nhận ít, early stopping/epoch và dung lượng model có thể khác nhau. Các biểu đồ đã sinh và kiểm số liệu nằm tại [runs/kan_joint_under6_20261021_v2/confirmation_figures_v2/evidence.json](../../../../runs/kan_joint_under6_20261021_v2/confirmation_figures_v2/evidence.json).

### 6.4. Danh sách đầy đủ 59 lượt train/calibration chính đã hoàn tất

Có 39 lượt ở v1 và 20 lượt ở v2; đây là số lượt hoàn tất trong hai thư mục `trials`, không bao gồm smoke CPU hoặc lượt bị ngắt. V1 còn 13 lượt refit riêng (10 chính +3 đối chứng), đã đóng trước vòng v2. V2 không có refit. Danh sách này giúp tránh gọi35 file cấu hình thành 35 lượt huấn luyện, hoặc gọi toàn bộ lượt train thành các model đã test.

| Vòng/ID | Seed | Epoch chạy/best | Val sau calibration m | k/T | Tham số | Train giây |
|---|---|---|---|---|---|---|
| v1/aligned_aux_relative | 42 | 66/46 | 5,748362 | 1000/0.5 | 1037493 | 560,290 |
| v1/aligned_topk | 42 | 49/29 | 5,866303 | 1000/0.5 | 1014568 | 239,990 |
| v1/auxiliary | 42 | 33/13 | 5,854196 | 20/0.5 | 1036213 | 159,771 |
| v1/budget100 | 42 | 100/21 | 6,032527 | 1000/0.5 | 1014568 | 767,355 |
| v1/budget100 | 43 | 100/27 | 5,986778 | 20/0.5 | 1014568 | 470,512 |
| v1/budget100 | 44 | 100/37 | 5,910910 | 20/1.0 | 1014568 | 467,927 |
| v1/cross_phone_consistency | 42 | 100/36 | 5,727462 | 20/0.5 | 1014568 | 1122,024 |
| v1/deeper_same_point | 42 | 100/71 | 5,697390 | 20/0.5 | 1195304 | 1245,715 |
| v1/ema | 42 | 74/54 | 5,979415 | 3/0.5 | 1014568 | 359,591 |
| v1/fine_tokens | 42 | 56/36 | 5,862838 | 20/1.0 | 1539112 | 1010,421 |
| v1/kan_head | 42 | 100/21 | 5,918121 | 10/0.5 | 6856384 | 988,874 |
| v1/learned_pool | 42 | 56/36 | 6,344939 | 20/0.5 | 607081 | 267,715 |
| v1/mask | 42 | 33/13 | 6,167652 | 20/1.0 | 1014568 | 155,727 |
| v1/regularized | 42 | 56/36 | 5,904866 | 20/1.0 | 1014568 | 512,954 |
| v1/relative | 42 | 33/13 | 5,905599 | 20/0.5 | 1015848 | 155,264 |
| v1/relative_ema | 42 | 32/12 | 6,185306 | 3/0.5 | 1015848 | 265,928 |
| v1/relative_only | 42 | 33/13 | 5,896253 | 20/1.0 | 1014568 | 154,909 |
| v1/robust_aug | 42 | 56/36 | 6,017944 | 20/0.5 | 1014568 | 263,439 |
| v1/same_point | 42 | 100/87 | 5,462713 | 10/0.5 | 1014568 | 602,171 |
| v1/same_point | 43 | 48/28 | 5,798075 | 10/0.5 | 1014568 | 291,603 |
| v1/same_point | 44 | 85/65 | 5,638771 | 20/0.5 | 1014568 | 504,298 |
| v1/same_point_aux_topk | 42 | 51/31 | 5,658655 | 10/0.5 | 1036213 | 545,991 |
| v1/same_point_consistency | 42 | 100/66 | 5,915659 | 10/0.5 | 1014568 | 1213,679 |
| v1/same_point_cross_phone | 42 | 100/87 | 5,484101 | 20/0.5 | 1014568 | 1078,193 |
| v1/same_point_cross_phone | 43 | 100/73 | 5,667324 | 20/0.5 | 1014568 | 558,482 |
| v1/same_point_cross_phone | 44 | 100/57 | 5,633135 | 20/0.5 | 1014568 | 557,029 |
| v1/same_point_kan_head | 42 | 100/77 | 5,505775 | 20/1.0 | 6856384 | 1218,645 |
| v1/same_point_relative | 100 | 80/60 | 5,740441 | 10/0.5 | 1014568 | 446,144 |
| v1/same_point_relative | 2024 | 56/36 | 5,652697 | 1000/0.5 | 1014568 | 311,712 |
| v1/same_point_relative | 42 | 46/26 | 5,582541 | 10/0.5 | 1014568 | 467,499 |
| v1/same_point_relative | 43 | 48/28 | 5,660620 | 10/0.5 | 1014568 | 267,985 |
| v1/same_point_relative | 44 | 48/28 | 5,730258 | 1000/0.5 | 1014568 | 266,043 |
| v1/same_point_relative | 45 | 54/34 | 5,619543 | 20/0.5 | 1014568 | 300,773 |
| v1/same_point_relative | 46 | 55/35 | 5,665361 | 20/1.0 | 1014568 | 305,426 |
| v1/same_point_relative | 47 | 57/37 | 5,756613 | 10/1.0 | 1014568 | 320,563 |
| v1/same_point_relative | 777 | 41/21 | 5,802873 | 10/1.0 | 1014568 | 227,678 |
| v1/same_point_relative | 999 | 37/17 | 5,737755 | 20/1.0 | 1014568 | 206,807 |
| v1/sigma_tight | 42 | 31/11 | 6,200502 | 20/1.0 | 1014568 | 234,220 |
| v1/sigma_wide | 42 | 60/40 | 6,017176 | 1000/0.5 | 1014568 | 502,406 |
| v2/budget100 | 42 | 100/21 | 6,032527 | 1000/0.5 | 1014568 | 895,198 |
| v2/budget100 | 43 | 100/27 | 5,986778 | 20/0.5 | 1014568 | 895,946 |
| v2/budget100 | 44 | 100/37 | 5,910910 | 20/1.0 | 1014568 | 904,035 |
| v2/nearby_10 m | 42 | 60/40 | 5,667143 | 20/0.5 | 1014568 | 707,171 |
| v2/nearby_5 m | 42 | 39/19 | 5,774471 | 10/0.5 | 1014568 | 446,984 |
| v2/nearby_residual | 100 | 35/15 | 5,606522 | 1000/1.0 | 1555368 | 433,731 |
| v2/nearby_residual | 2024 | 56/36 | 5,589130 | 20/1.0 | 1555368 | 462,822 |
| v2/nearby_residual | 42 | 46/26 | 5,556056 | 20/1.0 | 1555368 | 489,966 |
| v2/nearby_residual | 43 | 100/90 | 5,395632 | 10/1.0 | 1555368 | 1806,276 |
| v2/nearby_residual | 44 | 48/28 | 5,381803 | 10/1.0 | 1555368 | 1056,098 |
| v2/nearby_residual | 777 | 41/21 | 5,542629 | 1000/1.0 | 1555368 | 383,372 |
| v2/nearby_residual | 999 | 70/50 | 5,443265 | 10/1.0 | 1555368 | 787,376 |
| v2/rank_relative | 42 | 46/26 | 5,756392 | 20/0.5 | 1015848 | 443,390 |
| v2/relative_dropout | 42 | 46/26 | 5,803370 | 1000/0.5 | 1014568 | 492,765 |
| v2/residual_axis5 m | 42 | 65/45 | 5,552420 | 20/1.0 | 1555368 | 797,966 |
| v2/residual_axis5 m | 43 | 83/63 | 5,485247 | 20/1.0 | 1555368 | 2016,597 |
| v2/residual_axis5 m | 44 | 53/33 | 5,565507 | 10/1.0 | 1555368 | 1261,811 |
| v2/v1_relative | 42 | 46/26 | 5,582541 | 10/0.5 | 1014568 | 265,037 |
| v2/v1_relative | 43 | 48/28 | 5,660620 | 10/0.5 | 1014568 | 999,628 |
| v2/v1_relative | 44 | 48/28 | 5,730258 | 1000/0.5 | 1014568 | 457,744 |

<a id="muc-7"></a>

## 7. Toàn bộ kết quả test và model có sai số thấp nhất

| Nhóm | Số model | Mean2D ± sample SD m | Median2D trung bình m | P90 trung bình m | RMSE2D trung bình m |
|---|---|---|---|---|---|
| v1/budget100/full_development_refit | 3 | 7,529234 ± 0,281162 | 4,744206 | 16,660591 | 11,821497 |
| v1/budget100/train_only | 3 | 8,058153 ± 0,306071 | 5,359214 | 17,489558 | 12,241589 |
| v1/same_point_relative/full_development_refit | 10 | 7,424524 ± 0,293599 | 4,522050 | 15,562094 | 13,521332 |
| v1/same_point_relative/train_only | 10 | 7,918479 ± 0,378493 | 4,890631 | 16,495823 | 14,718835 |
| v2/nearby_residual/train-only | 7 | 7,724358 ± 0,401037 | 5,015593 | 15,987343 | 13,452544 |

Median/P90/RMSE trong bảng nhóm là trung bình chỉ số của từng model, không phải chỉ số tính trên một ensemble. Những nhóm khác số seed hoặc khác chế độ dùng dữ liệu không phải phép so sánh cô lập một thành phần thuật toán.


### Cách tính các chỉ số trong bảng

Với N=1.111 mẫu, sai số mẫu i là khoảng cách giữa hai tọa độ gốc:

$$e_i=\sqrt{(\hat x_i-x_i)^2+(\hat y_i-y_i)^2},\qquad Mean2D=\frac{1}{N}\sum_i e_i,\qquad RMSE2D=\sqrt{\frac{1}{N}\sum_i e_i^2}.$$

Median là phân vị50%; P90 là phân vị90% của e_i, không phải cam kết rằng mọi mẫu đều dưới ngưỡng đó. Building/floor accuracy đánh giá riêng; joint accuracy đòi đúng cả hai. Macro-F1 là trung bình F1 không trọng số giữa các lớp theo mapping đã lưu. Độ lệch chuẩn giữa K model độc lập dùng mẫu số K−1; tập test vẫn có 1.111 bản ghi truy vấn, không trở thành K×1.111 mẫu độc lập.

### 7.1. Bảng từng model đã đánh giá

| Vòng | Cấu hình/chế độ | Seed | Mean2D m | Median m | P90 m | RMSE m | Acc tòa | Acc tầng |
|---|---|---|---|---|---|---|---|---|
| v1 | same_point_relative/train_only | 42 | 8,163832 | 4,892168 | 16,883798 | 17,399981 | 99,640% | 93,519% |
| v1 | same_point_relative/full_development_refit | 42 | 7,449044 | 4,574903 | 15,483584 | 14,195091 | 99,910% | 92,619% |
| v1 | same_point_relative/train_only | 43 | 7,910469 | 4,779866 | 16,668309 | 15,046522 | 99,640% | 93,339% |
| v1 | same_point_relative/full_development_refit | 43 | 8,064226 | 4,377281 | 16,545237 | 19,444340 | 99,640% | 92,619% |
| v1 | same_point_relative/train_only | 44 | 7,381963 | 4,747034 | 16,194250 | 12,491300 | 99,910% | 93,969% |
| v1 | same_point_relative/full_development_refit | 44 | 7,442032 | 4,419748 | 16,761535 | 12,622657 | 99,730% | 93,789% |
| v1 | same_point_relative/train_only | 100 | 7,965642 | 4,661950 | 16,785067 | 15,226532 | 99,640% | 93,249% |
| v1 | same_point_relative/full_development_refit | 100 | 7,115416 | 4,274927 | 15,686102 | 11,997557 | 99,820% | 93,699% |
| v1 | same_point_relative/train_only | 2024 | 7,612690 | 4,595780 | 16,395587 | 13,425784 | 99,730% | 92,979% |
| v1 | same_point_relative/full_development_refit | 2024 | 7,539543 | 4,568395 | 16,341100 | 13,048534 | 99,820% | 93,969% |
| v1 | same_point_relative/train_only | 777 | 7,758807 | 5,052319 | 16,481394 | 13,274865 | 99,820% | 93,429% |
| v1 | same_point_relative/full_development_refit | 777 | 7,343180 | 4,609105 | 15,094198 | 12,381331 | 99,820% | 93,609% |
| v1 | same_point_relative/train_only | 999 | 8,255823 | 5,244053 | 15,971672 | 18,231636 | 99,460% | 93,789% |
| v1 | same_point_relative/full_development_refit | 999 | 7,162227 | 4,753063 | 14,122220 | 12,012057 | 99,730% | 93,879% |
| v1 | same_point_relative/train_only | 45 | 8,655397 | 4,682187 | 17,624252 | 17,386828 | 99,460% | 92,889% |
| v1 | same_point_relative/full_development_refit | 45 | 7,361981 | 4,381026 | 15,158806 | 13,649687 | 99,640% | 93,159% |
| v1 | same_point_relative/train_only | 46 | 7,963103 | 5,153513 | 15,636142 | 13,337538 | 99,820% | 92,439% |
| v1 | same_point_relative/full_development_refit | 46 | 7,678969 | 4,619524 | 15,145914 | 15,258509 | 99,550% | 93,429% |
| v1 | same_point_relative/train_only | 47 | 7,517063 | 5,097438 | 16,317765 | 11,367360 | 100,000% | 91,989% |
| v1 | same_point_relative/full_development_refit | 47 | 7,088622 | 4,642525 | 15,282247 | 10,603556 | 100,000% | 93,429% |
| v1 | budget100/train_only | 42 | 8,314885 | 5,544079 | 18,380216 | 12,260579 | 100,000% | 91,809% |
| v1 | budget100/full_development_refit | 42 | 7,221588 | 4,577115 | 16,325224 | 11,014860 | 99,820% | 92,799% |
| v1 | budget100/train_only | 43 | 8,140132 | 5,374211 | 17,904189 | 12,060076 | 99,910% | 90,279% |
| v1 | budget100/full_development_refit | 43 | 7,772878 | 4,814917 | 17,264429 | 12,777104 | 99,910% | 91,719% |
| v1 | budget100/train_only | 44 | 7,719440 | 5,159353 | 16,184270 | 12,404112 | 99,820% | 91,809% |
| v1 | budget100/full_development_refit | 44 | 7,593236 | 4,840585 | 16,392120 | 11,672526 | 100,000% | 91,809% |
| v2 | nearby_residual/original_train_only_no_refit | 42 | 7,421295 | 4,838045 | 15,409383 | 12,041657 | 99,910% | 92,709% |
| v2 | nearby_residual/original_train_only_no_refit | 43 | 7,722875 | 4,873466 | 16,196284 | 13,370404 | 99,910% | 93,519% |
| v2 | nearby_residual/original_train_only_no_refit | 44 | 7,721421 | 5,132299 | 15,669803 | 13,230503 | 99,820% | 93,609% |
| v2 | nearby_residual/original_train_only_no_refit | 100 | 8,579556 | 5,394503 | 16,824554 | 18,902041 | 99,550% | 93,249% |
| v2 | nearby_residual/original_train_only_no_refit | 2024 | 7,409123 | 4,938470 | 15,565260 | 11,263057 | 100,000% | 93,339% |
| v2 | nearby_residual/original_train_only_no_refit | 777 | 7,520465 | 5,064733 | 16,466304 | 11,079885 | 100,000% | 92,259% |
| v2 | nearby_residual/original_train_only_no_refit | 999 | 7,695771 | 4,867635 | 15,779816 | 14,280264 | 99,820% | 92,979% |

### 7.2. Bộ chỉ số đầy đủ của model thấp nhất v1 và thấp nhất v2

| Chỉ số | V1 same_point_relative seed 47 refit | V2 nearby_residual seed 2024 train-only |
|---|---|---|
| mean_2d | 7,088622 | 7,409123 |
| median_2d | 4,642525 | 4,938470 |
| p90_2d | 15,282247 | 15,565260 |
| rmse_2d | 10,603556 | 11,263057 |
| building_accuracy | 100,000% | 100,000% |
| floor_accuracy | 93,429% | 93,339% |
| joint_accuracy | 93,429% | 93,339% |
| building_macro_f1 | 1,000000 | 1,000000 |
| floor_macro_f1 | 0,927651 | 0,929900 |
| joint_macro_f1 | 0,924311 | 0,922750 |

Ma trận dưới đây đọc từ metrics gốc của đúng hai model. Hàng là nhãn thật; cột là dự đoán. Mỗi ma trận tổng đúng **1.111 mẫu**. Nhãn ghép dùng `tòa/tầng`.

#### V1 thấp nhất — seed 47 refit

**Tòa nhà**

| Thật ↓ / Dự đoán → | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 536 | 0 | 0 |
| 1 | 0 | 307 | 0 |
| 2 | 0 | 0 | 268 |

**Tầng**

| Thật ↓ / Dự đoán → | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | 122 | 8 | 2 | 0 | 0 |
| 1 | 6 | 427 | 27 | 2 | 0 |
| 2 | 1 | 6 | 295 | 4 | 0 |
| 3 | 0 | 0 | 10 | 161 | 1 |
| 4 | 2 | 0 | 0 | 4 | 33 |

**Tòa/tầng**

| Thật ↓ / Dự đoán → | 0/0 | 0/1 | 0/2 | 0/3 | 1/0 | 1/1 | 1/2 | 1/3 | 2/0 | 2/1 | 2/2 | 2/3 | 2/4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0/0 | 75 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0/1 | 4 | 204 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0/2 | 1 | 5 | 159 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0/3 | 0 | 0 | 5 | 80 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1/0 | 0 | 0 | 0 | 0 | 25 | 3 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1/1 | 0 | 0 | 0 | 0 | 2 | 114 | 25 | 2 | 0 | 0 | 0 | 0 | 0 |
| 1/2 | 0 | 0 | 0 | 0 | 0 | 1 | 84 | 2 | 0 | 0 | 0 | 0 | 0 |
| 1/3 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 43 | 0 | 0 | 0 | 0 | 0 |
| 2/0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 22 | 2 | 0 | 0 | 0 |
| 2/1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 109 | 2 | 0 | 0 |
| 2/2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 52 | 2 | 0 |
| 2/3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 38 | 1 |
| 2/4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 | 4 | 33 |

#### V2 thấp nhất — seed 2024 train-only

**Tòa nhà**

| Thật ↓ / Dự đoán → | 0 | 1 | 2 |
|---|---|---|---|
| 0 | 536 | 0 | 0 |
| 1 | 0 | 307 | 0 |
| 2 | 0 | 0 | 268 |

**Tầng**

| Thật ↓ / Dự đoán → | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| 0 | 120 | 9 | 3 | 0 | 0 |
| 1 | 8 | 424 | 29 | 1 | 0 |
| 2 | 2 | 5 | 294 | 4 | 1 |
| 3 | 0 | 0 | 7 | 164 | 1 |
| 4 | 1 | 0 | 0 | 3 | 35 |

**Tòa/tầng**

| Thật ↓ / Dự đoán → | 0/0 | 0/1 | 0/2 | 0/3 | 1/0 | 1/1 | 1/2 | 1/3 | 2/0 | 2/1 | 2/2 | 2/3 | 2/4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0/0 | 76 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0/1 | 5 | 202 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0/2 | 2 | 3 | 159 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0/3 | 0 | 0 | 3 | 82 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1/0 | 0 | 0 | 0 | 0 | 22 | 5 | 3 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1/1 | 0 | 0 | 0 | 0 | 3 | 113 | 26 | 1 | 0 | 0 | 0 | 0 | 0 |
| 1/2 | 0 | 0 | 0 | 0 | 0 | 2 | 84 | 1 | 0 | 0 | 0 | 0 | 0 |
| 1/3 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 44 | 0 | 0 | 0 | 0 | 0 |
| 2/0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 22 | 2 | 0 | 0 | 0 |
| 2/1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 109 | 2 | 0 | 0 |
| 2/2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 51 | 2 | 1 |
| 2/3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 38 | 1 |
| 2/4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 3 | 35 |

<a id="muc-8"></a>

## 8. Vì sao có mức giảm — bằng chứng nào đủ mạnh, điều gì chưa chứng minh

### 8.1. Tách thay đổi phương pháp khỏi refit trong v1

| Cùng seed 42/43/44 | Mean2D test m |
|---|---|
| A_control_train_only | 8,058153 |
| B_selected_train_only | 7,818755 |
| C_control_refit | 7,529234 |
| D_selected_refit | 7,651767 |

| Phép đối chiếu | Mức giảm m |
|---|---|
| A_minus_B_method_and_decoder_on_original_train | 0,239398 |
| B_minus_D_selected_refit_protocol | 0,166987 |
| A_minus_C_control_refit_protocol | 0,528919 |
| C_minus_D_method_and_decoder_after_refit | -0,122533 |

Dương là giảm sai số; âm là tăng. Gói thay đổi phương pháp+decoder giúp khoảng 0,239398 m trên train nội bộ của ba seed, nhưng sau cùng giao thức refit, cấu hình được chọn cao hơn đối chứng 0,122533 m. Vì vậy chưa có bằng chứng khẳng định riêng RSSI tương đối hoặc same-point mixing luôn cải thiện test sau refit.

Ở cùng đủ 10 seed của cấu hình v1, train-only7,918479 m chuyển sang refit 7,424524 m, giảm 0,493955 m. Refit khởi tạo mới, thêm 3.912 bản ghi, catalogue 872→933, head lớn hơn và batch/optimizer step thay đổi, chạy 31 epoch cố định. Đây là hiệu ứng quan sát của **cả giao thức refit**, chưa tách riêng hiệu ứng số mẫu hay số điểm tham chiếu.

### 8.2. V2 thay đổi thuật toán nội bộ nhưng test vẫn trên 7 m

**Thêm head KAN**, ghép cặp42/43/44: giảm toàn validation 0,052324 m với cùng decoder k5/T1; 0,123416 m sau calibration. Ở nhóm tọa độ chưa có trong train, mức giảm là 0,311402 m (âm nghĩa là tăng lỗi).

**Đổi sang trộn vị trí gần nhau**, ghép cặp42/43/44: giảm toàn validation 0,105311 m với cùng decoder k5/T1; 0,089894 m sau calibration. Ở nhóm tọa độ chưa có trong train, mức giảm là -0,061831 m (âm nghĩa là tăng lỗi).

Trên cùng bảy seed và cùng chế độ train-only, v1 có mean test 7,864175 m; v2 có 7,724358 m, chênh lệch giảm 0,139817 m. Đây là so sánh hậu nghiệm của gói head/augmentation/decoder/huấn luyện; không tách tác dụng riêng B-spline hoặc nearby mixing.

V2 có validation tốt hơn không đảm bảo test dưới 7 m. Seed 100 test8,579556 m và RMSE18,902041 m cho thấy lỗi lớn còn ảnh hưởng mean; không loại seed này khỏi tổng hợp. Không đủ phép đo để quy nguyên nhân các lỗi đó cho tường, multipath hoặc một điện thoại cụ thể. V2 chưa refit nên không thể lấy 7,724358 m so với 7,424524 m của 10 refit v1 rồi kết luận head KAN mới gây tăng lỗi0,300 m.

### 8.3. Kết luận về thuật toán

**Có thay đổi thuật toán ở mức kiến trúc và học:** v2 thêm hai KANLinear cho hiệu chỉnh tọa độ; nearby mixing đổi cách tạo input và nhãn train. **Không thay công thức cơ sở/vendor KAN:** vẫn B-spline+SiLU, không grid adaptation trong các model test. Mức thấp nhất7,088622 m thuộc v1 không có head residual; không được dùng nó làm bằng chứng head v2 đã đạt mức đó. Chưa thể viết rằng đã “tìm ra nguyên nhân xuống6 m” vì chưa có kết quả6 m cần giải thích.

<a id="muc-9"></a>

## 9. Thời gian, bộ nhớ và điều kiện chạy

| Model/lượt train | Epoch chạy | Thời gian train giây | VRAM peak allocated MiB | RAM peak RSS MiB |
|---|---|---|---|---|
| V1 seed 47 refit | 31 | 491,449 | 757,02 | 2012,16 |
| V2 seed 2024 train-only | 56 | 462,822 | 808,93 | 1776,49 |

Phần cứng đo: RTX4060 Ti8 GB. Môi trường ghi trong lượt: {"python":"3.12.1","torch":"2.9.0+cu128","numpy":"2.2.6","scikit_learn":"1.6.1","platform":"Windows-11-10.0.26200-SP0","device":"cuda","device_name":"NVIDIA GeForce RTX 4060 Ti","threads":2}. Đây là phiên bản môi trường đã chạy, không phải khuyến nghị phiên bản mới nhất.

Train-only có validation và checkpoint I/O, refit không có validation. Peak allocated là số cấp phát của PyTorch, không phải toàn bộ VRAM desktop. Peak RSS gồm Python, thư viện và dữ liệu, là đỉnh lấy mẫu nên có thể bỏ lỡ xung ngắn. Tải nền đã làm thời gian epoch thay đổi đáng kể; không dùng một lượt nhanh hơn để chứng minh kiến trúc tính toán nhanh hơn. Không có benchmark suy luận v2 cuối theo kế hoạch đầy đủ vì đã dừng trước refit; không tự điền thời gian suy luận chưa đo.

<a id="muc-10"></a>

## 10. Mọi file thêm/sửa trong phần bàn giao

Có 20 file public mới/sửa so với HEAD ban đầu; không tính các artifact cá nhân bị ignore. Những thay đổi thuật toán đã có trước yêu cầu viết báo cáo, không được phát sinh huấn luyện mới trong lần đổi tên. Bảng này bao phủ cả sửa CLI/tests và tài liệu, không chỉ model.

### 10.1 scripts/kan_under6_final.py

**Thực thi cuối theo manifest bất biến.** Kiểm tất cả refit trước đọc test; đánh giá đủ 1.111 mẫu/model, audit tọa độ, tổng hợp đủ seed và benchmark seed 42 đã cố định. Không tự tuyên bố hoàn thành mục tiêu.

### 10.2 src/ips/algorithms/kan/README.md

**Hướng dẫn tái lập.** Lệnh nghiên cứu KAN thuần, vai trò các tùy chọn và giới hạn của refit/validation/test.

### 10.3 src/ips/algorithms/kan/UPDATE_VERSION_3.md

**Báo cáo tổng hợp Update Version 3.** Đổi tên từ NGHIEN_CUU_DUOI_SAU_MET.md; bổ sung đủ thuật toán, cấu hình, kết quả đã kiểm và giữ nguyên nhật ký cũ trong phụ lục.

### 10.4 src/ips/algorithms/kan/accuracy.py

**Chọn decoder bằng validation.** Giữ đúng logits classifier phụ và residual tọa độ khi calibration; tách cấu hình train khỏi config đã đổi k/T bằng deepcopy; CLI nghiên cứu dùng pure-only, không truy xuất/kNN.

### 10.5 src/ips/algorithms/kan/algorithm.py

**Thông tin kiến trúc cho Factory chung.** Bổ sung mô tả input mode, head nội bộ KAN B-spline và residual có giới hạn; không đổi interface Algorithm.

### 10.6 src/ips/algorithms/kan/augmentation.py

**Tăng cường chỉ trên train.** Lấy mẫu khác cùng lớp và tọa độ gốc; trộn tín hiệu với mask nhị phân; tùy chọn ưu tiên điện thoại khác và cặp consistency, giữ nhãn ở same-point. NearbyPointMixing chỉ lấy điểm train khác cùng lớp trong bán kính, nội suy cả tín hiệu và nhãn XY cùng alpha; từ chối tổ hợp augmentation mơ hồ và không truy xuất khi inference.

### 10.7 src/ips/algorithms/kan/confirmation.py

**Xác nhận nhiều seed và khóa lựa chọn.** Kế hoạch v3 chọn kiến trúc theo hai nhóm tọa độ; đủ ba seed xác nhận, mười seed cho ứng viên thắng; khóa nguồn, decoder, checkpoint và epoch refit trước test.

### 10.8 src/ips/algorithms/kan/final_refit.py

**Huấn luyện lại sau khi khóa.** Khởi tạo mới trên 19.937 dòng, mở rộng nhãn RP, giữ scale gốc và lịch LR. Khôi phục cấu hình train xác thực trước calibration; hỗ trợ các tùy chọn đã chọn, kiểm provenance và không dùng validation/test chọn checkpoint refit.

### 10.9 src/ips/algorithms/kan/improvement.py

**Điều phối sàng lọc KAN thuần.** Tạo cấu hình, design, protocol, hash và snapshot; train/calibration có log riêng. Giữ cả trial không cải thiện; không có thao tác đánh giá test.

### 10.10 src/ips/algorithms/kan/improvement_analysis.py

**Phân tích validation theo mức hỗ trợ tọa độ.** Tái tính sai số và kiểm thứ tự mẫu; tách vị trí đã có/chưa có trong train. Không fit hoặc hiệu chỉnh model bằng nhãn query.

### 10.11 src/ips/algorithms/kan/models/networks.py

**Kiểm tra loại head trong factory nội bộ.** Từ chối head bộ nhớ/truy xuất; chỉ dùng point_classifier cho nhánh tham chiếu.

### 10.12 src/ips/algorithms/kan/models/reference.py

**Các lựa chọn bên trong một KAN-Mixer.** Thêm RSSI tương đối, rank có xử lý tie, mask, pooling và head/classifier tùy chọn; head residual hai KANLinear khởi tạo0, giới hạn từng trục và nằm trong cùng mạng. Giữ mặc định cũ, cấm suy luận nhiều view; catalogue refit mở rộng sau khóa.

### 10.13 src/ips/algorithms/kan/spatial_research.py

**Vòng nghiên cứu không gian tiếp sau kết quả v1.** Kiểm audit vòng cũ rồi tạo8 cấu hình bất biến gồm đối chứng, rank, spatial mixup, residual và dropout. Ghi nhận test đã được xem; chỉ sàng lọc bằng validation, từ chối mã/cấu hình đổi hoặc kết quả cũ bị gán lại cho phiên bản mới.

### 10.14 src/ips/algorithms/kan/train.py

**Vòng huấn luyện và loss.** Thêm same-point augmentation, consistency, EMA và lựa chọn dense/top-k cho loss tọa độ. Lưu cấu hình train độc lập, số optimizer step và nguồn dữ liệu. Dùng chung hàm chọn augmentation với refit, kể cả spatial mixup.

### 10.15 src/ips/cli.py

**Sửa tương thích CLI dùng chung.** Hiển thị backend đúng registry thay vì mặc định LLM chưa triển khai; giữ dataset tham chiếu trong lệnh check. Không thay thuật toán của thành viên khác.

### 10.16 tests/test_factory.py

**Tương thích Factory hiện có.** Cập nhật assertions theo các backend thật đã có trên main; không tạo backend giả.

### 10.17 tests/test_kan_improvement.py

**Kiểm thử các tùy chọn KAN mới.** Biểu diễn, mask, gradient, loss, calibration, augmentation, consistency, giới hạn KAN thuần và chọn cấu hình nhiều seed.

### 10.18 tests/test_kan_locked_final.py

**Kiểm thử giao thức cuối.** Từ chối thiếu seed/control/refit hoặc thay nguồn; audit đủ dòng, sai số tọa độ và quy tắc kết quả chính không chọn seed thuận lợi.

### 10.19 tests/test_kan_refit.py

**Kiểm thử refit và checkpoint.** Fixture cho nguồn train xác thực, catalog/head, RNG khởi tạo, LR, split/lock/hash và tương thích các tùy chọn train.

### 10.20 tests/test_kan_spatial.py

**Bất biến khoa học của các tùy chọn không gian v2.** Kiểm rank/tie/AP thiếu, giới hạn cùng tầng/bán kính và alpha của XY khi spatial mixup, không đọc held-out, residual bằng 0 ban đầu/gradient/giới hạn/calibration/state restore, và từ chối source thay đổi hay gán lại kết quả cũ.

`runtime.py`, `dataset.py`, `metrics.py`, data_access/ và CSV/split tiếp tục được dùng lại; không có diff public ở chúng trong đợt bàn giao này. `models/vendor/efficient_kan.py` còn nguyên hash theo provenance. `reference.py` hiện tại có phần đuôi dòng trắng khác bản thực thi đã khóa; bản khôi phục đã kiểm AST tương đương và dùng đúng byte của 42 file nguồn đã khóa trong thư mục frozen execution, không âm thầm thay checkpoint lịch sử.

### 10.21. Artifact và công cụ cá nhân ngoài phần public

Các nhóm sau vẫn ở máy trong `runs/` hoặc `personal_learning/`: config/design/protocol/manifest, snapshot mã, log epoch, checkpoint, CSV dự đoán, audit, hình validation/test, V3.docx và script tạo báo cáo. Các tệp này phục vụ tái lập hoặc trình bày, không phải những thuật toán dự đoán mới. Repo Git dùng chung không tự mang theo các artifact bị ignore.

| Nhóm | Công cụ hoặc thư mục chính | Vai trò/trạng thái |
|---|---|---|
| Bằng chứng v1 đã đóng | runs/kan_joint_under6_20261020_v1/ | 26 trường hợp test,13 refit, nguồn báo cáo đóng; giữ nguyên |
| Bằng chứng v2 | runs/kan_joint_under6_20261021_v2/ | 20 trial hoàn tất; dừng sau 999, không đủ 10 seed/refit |
| Test sớm theo yêu cầu | personal_learning/evaluate_kan_v2_after999.py | Plan riêng trước test, khóa7 checkpoint, predict.evaluate cũ, audit độc lập |
| Khôi phục gián đoạn | audit_kan_v2_recovery_20261022.py; interrupted_trials_20261022/ | Giữ70 file lượt dở; replay5 epoch khớp0 sai khác số học |
| Cách ly mã đã khóa | prepare_kan_v2_frozen_execution.py; frozen_execution_20261021/ | Giữ thay đổi workspace, thực thi đúng nguồn và dữ liệu junction cũ |
| Tổng hợp/kiểm kết quả | assemble_kan_v2_report_data.py; audit_kan_locked_test_v2.py; assess_kan_user_goal_under7.py | Lối đủ 10 seed/refit chưa hoàn tất; không dùng đánh tráo cho test7 seed |
| Giải thích nhân tố | kan_matched_protocol_decomposition.py; build_kan_v2_confirmation_figures.py | Tách ghép cặp chế độ và validation, có ghi giới hạn |
| Báo cáo Word trước | references/V3.docx; prepare_report_v3.py; build_report_v3.py; audit_report_v3.py | 24 trang giải thích v1; giữ nguyên, chưa chứa test sớm mới |
| Lần viết MD này | personal_learning/build_update_version3_report.py | Chỉ đọc artifact, kiểm33 CSV/metrics và tạo báo cáo; không forward model |
| Sơ đồ mới đang dở | selected_train_seed42_flow_20261022/ | Trace tensor checkpoint train 42 thật; mới validate2 SVG, chưa bàn giao đủ PNG |
| Thiết kế tokenization | runs/kan_tokenization_design_20261021_v0/ | Sáu config thiết kế; CPU smoke không thay kết quả các vòng chính |
| Thăm dò spline/grid | separate_train_activation_grid_reprojection.json và probe_kan_train_grid_reprojection.py | Khả thi CPU/train activation, không áp dụng vào model test |

<a id="muc-11"></a>

## 11. Tính toàn vẹn, gián đoạn và quyết định dừng

Cấu hình, split, checkpoint và source snapshot được lưu hash. Sửa lỗi alias giữa training_config và config sau calibration bằng deepcopy; refit đọc lại cấu hình train gốc có bằng chứng hash, không lấy k/T suy luận làm siêu tham số train. Refit reset RNG sau khi dựng tạm model nguồn, giữ cosine horizon 100 thay vì nén thành 31; giữ transform gốc. Các sửa này bảo vệ việc tái lập, không tự được kể thành một nguyên nhân vật lý giảm sai số.

Lượt2024 bị ngắt sau 5 epoch. Đã kiểm không còn worker, lưu nguyên 70 file, giữ 17 trial hoàn tất tại thời điểm khôi phục, chạy lại đúng seed/config từ đầu. Năm epoch đầu khớp loss, validation, LR và optimizer step; thời gian epoch không dùng đối chiếu. Mã thoát1073807364 không đủ để kết luận OOM hoặc lỗi phần cứng.

Sau yêu cầu dừng ở999, tạo artifact chốt lịch trước 45 để guard có sẵn dừng controller, không đổi42 file nguồn hoặc cấu hình giữa lượt chạy. Worker test sớm chờ999 hoàn tất và dừng controller rồi mới khóa7 checkpoint/test. Traceback “Incomplete trial exists … seed 45” ở log gốc là chốt dừng chủ động này, không phải45 đã train thất bại. Không tự tạo confirmation_summary/selection_lock đủ 10 seed hoặc giả kết quả refit v2.

Log kiểm thử nguồn đã khóa ghi **80 tests đạt trong10,817 giây** tại [runs/kan_joint_under6_20261021_v2/tests_frozen.log](../../../../runs/kan_joint_under6_20261021_v2/tests_frozen.log). Đây là kiểm thử phần mềm lịch sử của bản mã tương ứng, không phải80 lần đánh giá độ chính xác, và không được gọi là kiểm thử vừa chạy lại trong lần viết MD. Các test mới có ý nghĩa về mask/rank, gradient/head, biên residual, augmentation cùng lớp/train-only, calibration, provenance/refit, đủ seed/control và metric/row order.

Trong lần lập báo cáo này đã kiểm lại **33×1.111 dòng dự đoán**, đối chiếu tọa độ/nhãn thật, Euclidean errors, mean/median/P90/RMSE, accuracy, macro-F1 và ba ma trận nhầm lẫn mỗi model, cùng checkpoint hash. Không thực hiện suy luận mới. Bằng chứng kiểm nội dung lưu ở `runs/update_version_3_report_20261022/` sau khi xuất báo cáo.

Test đã được xem trong các vòng trước và vừa được đánh giá sớm theo yêu cầu; không được gọi phần nghiên cứu này là benchmark hoàn toàn chưa từng thấy. Chỉ dùng validation chọn cấu hình/decoder cho mỗi vòng trước lần test của vòng đó; không có khẳng định dữ liệu đã chứng minh không có mọi dạng ảnh hưởng gián tiếp từ lịch sử test.

<a id="muc-12"></a>

## 12. Các tệp để kiểm lại và tài liệu tham khảo

- [Tổng hợp v1](../../../../runs/kan_joint_under6_20261020_v1/official_test_summary.json).

- [Audit đủ26 model v1](../../../../runs/kan_joint_under6_20261020_v1/independent_full_test_audit_20261021.json).

- [Nguồn báo cáo v1 đã đóng](../../../../runs/kan_joint_under6_20261020_v1/report_inputs_closed_v1_20261021.json).

- [Plan test sớm theo yêu cầu](../../../../runs/kan_joint_under6_20261021_v2/early_test_after_seed999_20261022/user_requested_early_test_plan.json).

- [Khóa bảy checkpoint trước test](../../../../runs/kan_joint_under6_20261021_v2/early_test_after_seed999_20261022/locked_checkpoint_manifest.json).

- [Kết quả test mới](../../../../runs/kan_joint_under6_20261021_v2/early_test_after_seed999_20261022/summary.json).

- [Bảng CSV từng seed mới](../../../../runs/kan_joint_under6_20261021_v2/early_test_after_seed999_20261022/per_seed_metrics.csv).

- [Đối chiếu ba seed validation v2](../../../../runs/kan_joint_under6_20261021_v2/confirmation_figures_v2/evidence.json).

- [Báo cáo Word V3 trước đó](../../../../references/V3.docx).

Các đường dẫn trên trỏ artifact có thật trên máy. Muốn tái lập ở máy khác cần chuyển riêng dữ liệu/checkpoint/artifact local; file Markdown không chứa weights. Các lệnh dưới chỉ đọc kết quả, không khởi động lại thực nghiệm:

```powershell
git branch --show-current
Get-Content -LiteralPath runs/kan_joint_under6_20261021_v2/early_test_after_seed999_20261022/summary.json -Encoding UTF8
Get-Content -LiteralPath runs/kan_joint_under6_20261020_v1/official_test_summary.json -Encoding UTF8
```

Nguồn ngoài đã kiểm ngày 22/10/2026: [KAN](https://arxiv.org/abs/2404.19756), [KAN-Mixers v1](https://arxiv.org/abs/2503.08939v1), [UCI UJIIndoorLoc](https://archive.ics.uci.edu/dataset/310/ujiindoorloc), [efficient-kan commit ghim](https://raw.githubusercontent.com/Blealtan/efficient-kan/7b6ce1c87f18c8bc90c208f6b494042344216b11/src/efficient_kan/kan.py). UCI xác nhận 520 AP, sentinel100,19.937/1.111 bản ghi. Các số test7,088622/7,424524/7,724358 ở báo cáo này là kết quả dự án đã kiểm, không phải số trong bài báo.

<a id="phu-luc-a"></a>

## Phụ lục A. Nguyên văn 35 cấu hình input đã lưu của hai vòng

Các JSON dưới là input thực tế, không thêm khóa diễn giải. Khóa không đặt dùng mặc định trong mã ở thời điểm chạy; bảng mục5 và thuật toán mục3–4 giải thích các mặc định liên quan. Catalogue được fit sau input; cấu hình train đã resolved và config inference sau calibration lưu riêng trong từng checkpoint. Ba thiết kế metric-memory được đánh dấu loại, không có số liệu chạy hợp lệ.

<details>
<summary>v1 / aligned_aux_relative</summary>

[runs/kan_joint_under6_20261020_v1/configs/aligned_aux_relative.json](../../../../runs/kan_joint_under6_20261020_v1/configs/aligned_aux_relative.json) · SHA256 `19306dae7743a8f0197e10048214842b2c55d85e74bd2c12e8ca8d9aea02aa17`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 60,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05,
    "coordinate_source": "topk",
    "lambda_auxiliary_class": 0.5
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 20,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "absolute_relative_mask",
    "auxiliary_classifier": true
  }
}
```

</details>

<details>
<summary>v1 / aligned_topk</summary>

[runs/kan_joint_under6_20261020_v1/configs/aligned_topk.json](../../../../runs/kan_joint_under6_20261020_v1/configs/aligned_topk.json) · SHA256 `43d79b9275a2bf1b4b1e1899a1b97171acdd9ebdac9ded1f5bba30120b2d55d4`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05,
    "coordinate_source": "topk"
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 20,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05
  }
}
```

</details>

<details>
<summary>v1 / auxiliary</summary>

[runs/kan_joint_under6_20261020_v1/configs/auxiliary.json](../../../../runs/kan_joint_under6_20261020_v1/configs/auxiliary.json) · SHA256 `a716bf10a62950be2942aed137faf0198f0fcb3afffe0ad41fcbd226cbb9f7c8`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05,
    "lambda_auxiliary_class": 0.5
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "auxiliary_classifier": true,
    "class_logit_weight": 1.0
  }
}
```

</details>

<details>
<summary>v1 / budget100</summary>

[runs/kan_joint_under6_20261020_v1/configs/budget100.json](../../../../runs/kan_joint_under6_20261020_v1/configs/budget100.json) · SHA256 `fa58e64152716c8948871ca8dfe67857d03eb8301956882cdf2009ed55e75011`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 100,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05
  }
}
```

</details>

<details>
<summary>v1 / cross_phone_consistency</summary>

[runs/kan_joint_under6_20261020_v1/configs/cross_phone_consistency.json](../../../../runs/kan_joint_under6_20261020_v1/configs/cross_phone_consistency.json) · SHA256 `bec93cb32a9d9392e2fecd3c4248ee281bf77a70f4a664f4bcaa040fb842590b`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 100,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_consistency_weight": 1.0,
    "same_point_partner_policy": "different_phone_first",
    "consistency_temperature": 1.0
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05
  }
}
```

</details>

<details>
<summary>v1 / deeper_same_point</summary>

[runs/kan_joint_under6_20261020_v1/configs/deeper_same_point.json](../../../../runs/kan_joint_under6_20261020_v1/configs/deeper_same_point.json) · SHA256 `76fb7d90f30fff1424885c818968b05678621d5220834d15f9828b8ecd1e68e5`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 3,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 100,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05
  }
}
```

</details>

<details>
<summary>v1 / ema</summary>

[runs/kan_joint_under6_20261020_v1/configs/ema.json](../../../../runs/kan_joint_under6_20261020_v1/configs/ema.json) · SHA256 `a153d0659b42bb10e2322101c410c852c8b26ccb83fc277b926bc1ed0307e9d0`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "ema_decay": 0.995
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05
  }
}
```

</details>

<details>
<summary>v1 / fine_tokens</summary>

[runs/kan_joint_under6_20261020_v1/configs/fine_tokens.json](../../../../runs/kan_joint_under6_20261020_v1/configs/fine_tokens.json) · SHA256 `cc096579eee81f66226407f58490822e05e1decb95db4c62d5892bca7fb336a2`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 52,
    "features_per_token": 20,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 64,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05
  }
}
```

</details>

<details>
<summary>v1 / kan_head</summary>

[runs/kan_joint_under6_20261020_v1/configs/kan_head.json](../../../../runs/kan_joint_under6_20261020_v1/configs/kan_head.json) · SHA256 `57ae8cd2a7b61bb966d0e9b5b99e29a5bbf60568f3c5d1bee57cafd6d3ef9fe5`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 100,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "head_transform": "kan"
  }
}
```

</details>

<details>
<summary>v1 / learned_pool</summary>

[runs/kan_joint_under6_20261020_v1/configs/learned_pool.json](../../../../runs/kan_joint_under6_20261020_v1/configs/learned_pool.json) · SHA256 `51d55050e638de9a74bcbfdc520dac08a7cbd786e45c6764ed520f496f3e990d`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "learned",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05
  }
}
```

</details>

<details>
<summary>v1 / mask</summary>

[runs/kan_joint_under6_20261020_v1/configs/mask.json](../../../../runs/kan_joint_under6_20261020_v1/configs/mask.json) · SHA256 `0ffc94cc35c684c18132ec8dee5f0037d1056da0ab0eb58df91ec0b819783cec`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05
  }
}
```

</details>

<details>
<summary>v1 / metric_centered — ĐÃ LOẠI, KHÔNG PHẢI MODEL TEST</summary>

[runs/kan_joint_under6_20261020_v1/configs/metric_centered.json](../../../../runs/kan_joint_under6_20261020_v1/configs/metric_centered.json) · SHA256 `435028b85a9ef54beee6aba565cb73486ed98722bdd6a69ae3ac2292a4d107bc`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "head_type": "metric_memory",
    "metric_dimension": 128,
    "metric_temperature": 0.1,
    "input_mode": "relative_mask",
    "metric_centering": true
  }
}
```

</details>

<details>
<summary>v1 / metric_centered_ema — ĐÃ LOẠI, KHÔNG PHẢI MODEL TEST</summary>

[runs/kan_joint_under6_20261020_v1/configs/metric_centered_ema.json](../../../../runs/kan_joint_under6_20261020_v1/configs/metric_centered_ema.json) · SHA256 `7bd696cf9589075cbfb79703720ffd99bd36fef6d50bdb8c18860deb8ccb007c`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "ema_decay": 0.995
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "head_type": "metric_memory",
    "metric_dimension": 128,
    "metric_temperature": 0.1,
    "input_mode": "relative_mask",
    "metric_centering": true
  }
}
```

</details>

<details>
<summary>v1 / metric_memory — ĐÃ LOẠI, KHÔNG PHẢI MODEL TEST</summary>

[runs/kan_joint_under6_20261020_v1/configs/metric_memory.json](../../../../runs/kan_joint_under6_20261020_v1/configs/metric_memory.json) · SHA256 `9c6989654cf301135f536cbecea1dbcff75492363cd3d98e8ab9c8c15851884e`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "head_type": "metric_memory",
    "metric_dimension": 128,
    "metric_temperature": 0.1,
    "input_mode": "relative_mask"
  }
}
```

</details>

<details>
<summary>v1 / regularized</summary>

[runs/kan_joint_under6_20261020_v1/configs/regularized.json](../../../../runs/kan_joint_under6_20261020_v1/configs/regularized.json) · SHA256 `3e76fafb48426114793064cd9311826fa902f67e2a8e392e344996533cab6af2`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.2
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05
  }
}
```

</details>

<details>
<summary>v1 / relative</summary>

[runs/kan_joint_under6_20261020_v1/configs/relative.json](../../../../runs/kan_joint_under6_20261020_v1/configs/relative.json) · SHA256 `b043b11844662be53cd7b950a9ce34bf1398b7cb31b8b1c190f11e8eb824f6f3`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 60,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "absolute_relative_mask"
  }
}
```

</details>

<details>
<summary>v1 / relative_ema</summary>

[runs/kan_joint_under6_20261020_v1/configs/relative_ema.json](../../../../runs/kan_joint_under6_20261020_v1/configs/relative_ema.json) · SHA256 `610b801dfd1ca46eac3031baf5c04f152eb4c5eb58308ff39dfc288e41026b28`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 60,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "ema_decay": 0.995
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "absolute_relative_mask"
  }
}
```

</details>

<details>
<summary>v1 / relative_only</summary>

[runs/kan_joint_under6_20261020_v1/configs/relative_only.json](../../../../runs/kan_joint_under6_20261020_v1/configs/relative_only.json) · SHA256 `b5b80eeb1b5b7787df768889f74c01e21f9369c09f2c6af862c5c692bad4921d`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "relative_mask"
  }
}
```

</details>

<details>
<summary>v1 / robust_aug</summary>

[runs/kan_joint_under6_20261020_v1/configs/robust_aug.json](../../../../runs/kan_joint_under6_20261020_v1/configs/robust_aug.json) · SHA256 `798790d08604aefbd65929521924394a93605a47985578069dbe39451fd83723`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 4.0,
    "device_shift_db": 6.0,
    "ap_dropout": 0.1
  }
}
```

</details>

<details>
<summary>v1 / same_point</summary>

[runs/kan_joint_under6_20261020_v1/configs/same_point.json](../../../../runs/kan_joint_under6_20261020_v1/configs/same_point.json) · SHA256 `698ac1e18dc53e024345097083bfd7d80b161690c42a1b124556845ee1db9a64`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05
  }
}
```

</details>

<details>
<summary>v1 / same_point_aux_topk</summary>

[runs/kan_joint_under6_20261020_v1/configs/same_point_aux_topk.json](../../../../runs/kan_joint_under6_20261020_v1/configs/same_point_aux_topk.json) · SHA256 `498bcdeef91031532dc019725ee3d4ecff06df5ef448f8ab31d715a86ea8d99b`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05,
    "coordinate_source": "topk",
    "lambda_auxiliary_class": 0.5
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 20,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "auxiliary_classifier": true
  }
}
```

</details>

<details>
<summary>v1 / same_point_consistency</summary>

[runs/kan_joint_under6_20261020_v1/configs/same_point_consistency.json](../../../../runs/kan_joint_under6_20261020_v1/configs/same_point_consistency.json) · SHA256 `b8f1ee6a5ecc174dc0eca58189617b2b66b84afe43e72d5de04b3b3410de62a2`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 100,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5,
    "same_point_consistency_weight": 1.0,
    "same_point_partner_policy": "different_phone_first",
    "consistency_temperature": 1.0
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05
  }
}
```

</details>

<details>
<summary>v1 / same_point_cross_phone</summary>

[runs/kan_joint_under6_20261020_v1/configs/same_point_cross_phone.json](../../../../runs/kan_joint_under6_20261020_v1/configs/same_point_cross_phone.json) · SHA256 `2bb686d504f97a64018f9af5876d72946d57c0b9cb1da7d36421acc283291e9a`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 100,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5,
    "same_point_partner_policy": "different_phone_first"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05
  }
}
```

</details>

<details>
<summary>v1 / same_point_kan_head</summary>

[runs/kan_joint_under6_20261020_v1/configs/same_point_kan_head.json](../../../../runs/kan_joint_under6_20261020_v1/configs/same_point_kan_head.json) · SHA256 `06522aec405d5840c6dbaecc84f43ff0a67d9689095cd66b0d41b066e827027a`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 100,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "head_transform": "kan"
  }
}
```

</details>

<details>
<summary>v1 / same_point_relative</summary>

[runs/kan_joint_under6_20261020_v1/configs/same_point_relative.json](../../../../runs/kan_joint_under6_20261020_v1/configs/same_point_relative.json) · SHA256 `269a553c0b173d582de09c517e6630415860af2d17ce4ee54c565e4fc75ab7c4`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "relative_mask"
  }
}
```

</details>

<details>
<summary>v1 / sigma_tight</summary>

[runs/kan_joint_under6_20261020_v1/configs/sigma_tight.json](../../../../runs/kan_joint_under6_20261020_v1/configs/sigma_tight.json) · SHA256 `bb28edfd328bc3d63b4650231c6c13a15f1390487a3dace57c7fb276def5bbb5`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.025
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05
  }
}
```

</details>

<details>
<summary>v1 / sigma_wide</summary>

[runs/kan_joint_under6_20261020_v1/configs/sigma_wide.json](../../../../runs/kan_joint_under6_20261020_v1/configs/sigma_wide.json) · SHA256 `eebb4dbe66434241f78a951c18dfb05d056410314fe2c2f4d317b249f1b3c2a8`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.08
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05
  }
}
```

</details>

<details>
<summary>v2 / budget100</summary>

[runs/kan_joint_under6_20261021_v2/configs/budget100.json](../../../../runs/kan_joint_under6_20261021_v2/configs/budget100.json) · SHA256 `fa58e64152716c8948871ca8dfe67857d03eb8301956882cdf2009ed55e75011`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 100,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05
  }
}
```

</details>

<details>
<summary>v2 / nearby_10m</summary>

[runs/kan_joint_under6_20261021_v2/configs/nearby_10m.json](../../../../runs/kan_joint_under6_20261021_v2/configs/nearby_10m.json) · SHA256 `7398143985f34d9736a2410f6c61eddaba17a243c47ba5d812a6906845c27686`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.0,
    "nearby_point_mixing": 0.5,
    "nearby_point_radius_m": 10.0
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "relative_mask"
  }
}
```

</details>

<details>
<summary>v2 / nearby_5m</summary>

[runs/kan_joint_under6_20261021_v2/configs/nearby_5m.json](../../../../runs/kan_joint_under6_20261021_v2/configs/nearby_5m.json) · SHA256 `d3d11328f9e0ad4f64ff9bf24fab2d429733e444b97339b75204dd60305f4306`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.0,
    "nearby_point_mixing": 0.5,
    "nearby_point_radius_m": 5.0
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "relative_mask"
  }
}
```

</details>

<details>
<summary>v2 / nearby_residual</summary>

[runs/kan_joint_under6_20261021_v2/configs/nearby_residual.json](../../../../runs/kan_joint_under6_20261021_v2/configs/nearby_residual.json) · SHA256 `9d8efa177f1a61953c8f44ac2b0c41e9ebde78c4b737a650932abceab64c8584`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.0,
    "nearby_point_mixing": 0.5,
    "nearby_point_radius_m": 10.0
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "relative_mask",
    "coordinate_residual_scale": 0.05,
    "coordinate_residual_width": 32
  }
}
```

</details>

<details>
<summary>v2 / rank_relative</summary>

[runs/kan_joint_under6_20261021_v2/configs/rank_relative.json](../../../../runs/kan_joint_under6_20261021_v2/configs/rank_relative.json) · SHA256 `eb4287823f5f854d457453b91a0c9be5b065526a3dac3bfb8b74445fdb91a13f`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 60,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "relative_rank_mask"
  }
}
```

</details>

<details>
<summary>v2 / relative_dropout</summary>

[runs/kan_joint_under6_20261021_v2/configs/relative_dropout.json](../../../../runs/kan_joint_under6_20261021_v2/configs/relative_dropout.json) · SHA256 `fa9f64d926b560b07f0fbe83a3e1de5971f895ee5b217c1a733e66145f7def52`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 4.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.15,
    "input_mode": "relative_mask"
  }
}
```

</details>

<details>
<summary>v2 / residual_axis5m</summary>

[runs/kan_joint_under6_20261021_v2/configs/residual_axis5m.json](../../../../runs/kan_joint_under6_20261021_v2/configs/residual_axis5m.json) · SHA256 `596e2afc5050bbb722ab60f1d648e7ae8e456871921dcf8706e2262f52b56615`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "relative_mask",
    "coordinate_residual_scale": 0.05,
    "coordinate_residual_width": 32
  }
}
```

</details>

<details>
<summary>v2 / v1_relative</summary>

[runs/kan_joint_under6_20261021_v2/configs/v1_relative.json](../../../../runs/kan_joint_under6_20261021_v2/configs/v1_relative.json) · SHA256 `269a553c0b173d582de09c517e6630415860af2d17ce4ee54c565e4fc75ab7c4`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "relative_mask"
  }
}
```

</details>

<a id="phu-luc-b"></a>

## Phụ lục B. Sáu cấu hình tokenization thiết kế/CPU, chưa chạy nghiên cứu GPU đầy đủ

Không dùng kết quả smoke trên subset để so với model train đầy đủ, không gán kết quả test cho các cấu hình này. Chúng không phải các ứng viên được chọn trong hai vòng v1/v2. Thí nghiệm CPU một epoch chỉ kiểm pipeline/shape/reload, không chứng minh chất lượng định vị.

<details>
<summary>ap1_c16_h128 — thiết kế/CPU, không có kết quả test</summary>

[runs/kan_tokenization_design_20261021_v0/configs/ap1_c16_h128.json](../../../../runs/kan_tokenization_design_20261021_v0/configs/ap1_c16_h128.json) · SHA256 `a2661ad346a33010ee86f81b994ab7bf0c4a310751fb4c6d7a9640f1ce38bae9`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 520,
    "features_per_token": 2,
    "channels": 16,
    "blocks": 2,
    "token_hidden": 16,
    "channel_hidden": 32,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 128,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "relative_mask"
  }
}
```

</details>

<details>
<summary>ap1_c32_h64 — thiết kế/CPU, không có kết quả test</summary>

[runs/kan_tokenization_design_20261021_v0/configs/ap1_c32_h64.json](../../../../runs/kan_tokenization_design_20261021_v0/configs/ap1_c32_h64.json) · SHA256 `08b0d52c8ceb0d58262dffe509d8e4bf531a3678e7c60d37b8f384d1121a7705`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 520,
    "features_per_token": 2,
    "channels": 32,
    "blocks": 2,
    "token_hidden": 16,
    "channel_hidden": 64,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 64,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "relative_mask"
  }
}
```

</details>

<details>
<summary>ap5_c32_h312 — thiết kế/CPU, không có kết quả test</summary>

[runs/kan_tokenization_design_20261021_v0/configs/ap5_c32_h312.json](../../../../runs/kan_tokenization_design_20261021_v0/configs/ap5_c32_h312.json) · SHA256 `51b8edd25b4e3bc366a23635d594011931aed052fff4d4a547496c80cbdddd4c`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 104,
    "features_per_token": 10,
    "channels": 32,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 64,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 312,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "relative_mask"
  }
}
```

</details>

<details>
<summary>budget100 — thiết kế/CPU, không có kết quả test</summary>

[runs/kan_tokenization_design_20261021_v0/configs/budget100.json](../../../../runs/kan_tokenization_design_20261021_v0/configs/budget100.json) · SHA256 `fa58e64152716c8948871ca8dfe67857d03eb8301956882cdf2009ed55e75011`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 100,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine"
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 0.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05
  }
}
```

</details>

<details>
<summary>relative_control — thiết kế/CPU, không có kết quả test</summary>

[runs/kan_tokenization_design_20261021_v0/configs/relative_control.json](../../../../runs/kan_tokenization_design_20261021_v0/configs/relative_control.json) · SHA256 `269a553c0b173d582de09c517e6630415860af2d17ce4ee54c565e4fc75ab7c4`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 256,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "relative_mask"
  }
}
```

</details>

<details>
<summary>relative_head464 — thiết kế/CPU, không có kết quả test</summary>

[runs/kan_tokenization_design_20261021_v0/configs/relative_head464.json](../../../../runs/kan_tokenization_design_20261021_v0/configs/relative_head464.json) · SHA256 `42725b25b19880f762b48c7717b61a0965d36a0548558c2a006f45df83c70207`.

```json
{
  "model": "kan_mixer",
  "split_dir": "data/processed/uji_group_split_v1",
  "architecture_reference": "https://arxiv.org/abs/2503.08939v1",
  "mixer": {
    "tokens": 26,
    "features_per_token": 40,
    "channels": 64,
    "blocks": 2,
    "token_hidden": 32,
    "channel_hidden": 128,
    "dropout": 0.1
  },
  "kan": {
    "basis": "B_spline",
    "grid_size": 5,
    "spline_order": 3,
    "base_activation": "SiLU",
    "enable_standalone_scale_spline": true,
    "grid_range": [
      -3,
      3
    ],
    "scale_noise": 0.1,
    "scale_base": 1.0,
    "scale_spline": 1.0,
    "grid_eps": 0.02,
    "update_grid": false,
    "extra_regularization_loss_coefficient": 0,
    "required_diagnostic": "input_distribution_and_out_of_knot_support_fraction_per_KANLinear",
    "note": "RSSI_variant_design_not_claimed_to_be_Canuto_paper_hyperparameters"
  },
  "loss": {
    "smooth_l1_beta": 1.0,
    "lambda_xy": 3.0,
    "lambda_class": 0.5,
    "lambda_reference": 1.0,
    "label_smoothing": 0.0,
    "reference_sigma_normalized": 0.05
  },
  "training": {
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 128,
    "max_epochs": 100,
    "patience": 20,
    "min_delta": 0.01,
    "gradient_clip_l2": 1.0,
    "threads": 2,
    "kan_diagnostic_batches": 0,
    "scheduler": "cosine",
    "same_point_mixing": 0.5
  },
  "positioning": {
    "variant": "reference",
    "pooling": "flatten",
    "head_width": 464,
    "mask_scale": 1.0,
    "top_k": 5,
    "temperature": 1.0,
    "rssi_noise_db": 2.0,
    "device_shift_db": 3.0,
    "ap_dropout": 0.05,
    "input_mode": "relative_mask"
  }
}
```

</details>

<a id="phu-luc-c"></a>

## Phụ lục C. Decoder đã chọn và nguồn metric của từng seed chính

| Vòng | Seed | Epoch checkpoint train | k suy luận | T suy luận | Val sau calibration m | Phạm vi fit khi chọn |
|---|---|---|---|---|---|---|
| v1 | 42 | 26 | 10 | 0.5 | 5,582541 | train nội bộ; refit sau khóa giữ decoder đã chọn |
| v1 | 43 | 28 | 10 | 0.5 | 5,660621 | train nội bộ; refit sau khóa giữ decoder đã chọn |
| v1 | 44 | 28 | 1000 | 0.5 | 5,730258 | train nội bộ; refit sau khóa giữ decoder đã chọn |
| v1 | 100 | 60 | 10 | 0.5 | 5,740441 | train nội bộ; refit sau khóa giữ decoder đã chọn |
| v1 | 2024 | 36 | 1000 | 0.5 | 5,652697 | train nội bộ; refit sau khóa giữ decoder đã chọn |
| v1 | 777 | 21 | 10 | 1.0 | 5,802874 | train nội bộ; refit sau khóa giữ decoder đã chọn |
| v1 | 999 | 17 | 20 | 1.0 | 5,737755 | train nội bộ; refit sau khóa giữ decoder đã chọn |
| v1 | 45 | 34 | 20 | 0.5 | 5,619544 | train nội bộ; refit sau khóa giữ decoder đã chọn |
| v1 | 46 | 35 | 20 | 1.0 | 5,665362 | train nội bộ; refit sau khóa giữ decoder đã chọn |
| v1 | 47 | 37 | 10 | 1.0 | 5,756613 | train nội bộ; refit sau khóa giữ decoder đã chọn |
| v2 | 42 | 26 | 20 | 1.0 | 5,556056 | train nội bộ, không refit |
| v2 | 43 | 90 | 10 | 1.0 | 5,395632 | train nội bộ, không refit |
| v2 | 44 | 28 | 10 | 1.0 | 5,381803 | train nội bộ, không refit |
| v2 | 100 | 15 | 1000 | 1.0 | 5,606522 | train nội bộ, không refit |
| v2 | 2024 | 36 | 20 | 1.0 | 5,589130 | train nội bộ, không refit |
| v2 | 777 | 21 | 1000 | 1.0 | 5,542629 | train nội bộ, không refit |
| v2 | 999 | 50 | 10 | 1.0 | 5,443265 | train nội bộ, không refit |

V1 bảng này dùng validation kiểm độc lập trên CPU; v2 đọc đúng giá trị GPU calibration đã lưu trước test. Sai khác CPU/GPU nhỏ ở chữ số cuối không phải kết quả của một checkpoint hoặc decoder mới. Ví dụ6,032527 là validation baseline sau calibration; không được gọi nó là kết quả test6 m.

<a id="phu-luc-d"></a>

## Phụ lục D. Nhật ký nghiên cứu gốc trước khi đổi tên

Phần dưới được giữ nguyên văn để không mất123 mục thay đổi và diễn biến. Ngày, trạng thái “đang chạy/chưa test”, tên nhánh và tên file bên trong là **lịch sử tại thời điểm ghi**, không phải chỉ thị khởi động lại thực nghiệm hoặc trạng thái mới nhất. Kết luận hiện tại nằm ở mục1 và7 của báo cáo này.

<details>
<summary>Mở toàn bộ nhật ký cũ (giữ nguyên nội dung)</summary>

# Cải tiến KAN Mixer huấn luyện chung UJIIndoorLoc

Nhánh `feature/kan-duoi-sau-met`, bắt đầu ngày 20 tháng 10 năm 2026 từ `adef7d7`.
                                                                                                                                                    
## Mục tiêu và quy ước

**Mục tiêu hiện tại, cập nhật ngày 21/10/2026: mean sai số tọa độ 2D dưới 7 m
trên official test**, với một mô hình dùng chung cả ba tòa và toàn bộ 13 tổ hợp
tòa tầng. Ngưỡng ban đầu là 6 m; tên nhánh, thư mục và protocol đã khóa trước đó
vẫn giữ nguyên để bảo toàn lịch sử. Không nhận building, floor hoặc tọa độ thật
của truy vấn làm đầu vào. Không chia thành các mạng riêng từng tòa/tầng.

Giữ nguyên split `uji_group_split_v1`, AP order và metric. Fit từ train; dùng
validation chọn kiến trúc, siêu tham số và decoder. Chốt trước khi đánh giá test của
lượt nghiên cứu này. Official test đã được xem trong lịch sử nên các kết luận
phải thừa nhận giới hạn đó. Không thay kết quả hoặc checkpoint cũ.

**Cập nhật theo yêu cầu mới: chỉ dùng một KAN-Mixer.** Không thêm bộ dự đoán
khác, kNN/truy xuất fingerprint, bộ nhớ embedding, ensemble hay trung bình nhiều
lượt suy luận. Có thể chỉnh các khối KAN-Mixer, pooling/head nội bộ, loss, dữ
liệu tăng cường train và optimizer. Decoder tham chiếu sẵn có chỉ biến điểm số
do mạng dự đoán thành tọa độ, không tính láng giềng từ RSSI. Không được gọi một
mô hình thêm bộ nhớ hoặc kNN là KAN-Mixer thuần để đạt ngưỡng mục tiêu.

Baseline là KAN tham chiếu efficient-kan B spline: mean test ba seed bằng
8,058152 m, sample SD 0,306071 m. Baseline không phải KAN lai kNN và không phải
thí nghiệm ba model riêng từng tòa. Môi trường KAN tiếp tục dùng `.venv-cuda`
Python 3.12 và PyTorch 2.9; hướng dẫn `.venv39` trong AGENTS dành cho phần LLM
không thay đổi giao thức KAN do người dùng đã yêu cầu giữ và cải tiến.

**Trạng thái ngày 22/10/2026:** theo yêu cầu dừng sau seed999, vòng v2 đã
đánh giá sớm bảy checkpoint train-only trên đủ1.111 mẫu test/model, đạt
**7,724358 ± 0,401037 m**, chưa dưới7 m. Seed999 riêng đạt7,695771 m.
Chưa refit hoặc hoàn tất kế hoạch mười seed. Người dùng yêu cầu đánh giá xong
và dừng; không tiếp tục huấn luyện hoặc đánh giá thêm.

Vòng v1 đã đánh giá test và kiểm độc lập đủ
1.111 dòng cho mỗi mô hình. Mười seed refit của cấu hình chính đạt
**7,424524 ± 0,293599 m**, chưa dưới ngưỡng mới 7 m. Các mục ghi “chưa test” ở
phần lịch sử dưới đây mô tả thời điểm chạy tương ứng, không phải trạng thái
hiện tại của toàn bộ nghiên cứu.

## Sổ thay đổi

1. Thêm `improvement.py` để chuẩn bị protocol và chạy trial train/calibrate
   chỉ trên train/validation. Mỗi trial lưu config, snapshot Python, commands,
   terminal log, metadata và hash. Controller không có lệnh đánh giá test.
2. Trial `mask`: chỉ đổi `positioning.mask_scale` từ 0 lên 1. Giả thuyết:
   tín hiệu “không phát hiện” khác “phát hiện rất yếu” có thể giúp tổng quát.
3. Trial `robust_aug`: chỉ đổi nhóm augmentation gồm RSSI noise 2→4 dB,
   device shift 3→6 dB và AP dropout 0,05→0,10. Đây là thay đổi ba tham số
   cùng nhóm, chưa tách nguyên nhân của từng thành phần.

4. `models/reference.py` thêm `represent_signal`: mặc định giữ nguyên stack
   cũ; `absolute_relative_mask` thêm RSSI trừ trung bình các AP được phát
   hiện trong chính fingerprint, chia 0,2 (20,8 dB), rồi giữ thêm absolute và
   mask. Số đặc trưng/token 40→60; tensor dataset vẫn [B,520,2]. Mode
   `relative_mask` chỉ giữ relative và mask, 40 số/token. Tất cả AP thiếu có
   biểu diễn 0 và count được chặn dưới 1, không fit thống kê từ held-out.
5. Thêm pooling `learned`: Linear(64,32), Tanh, Linear(32,1), softmax trên
   token và tổng có trọng số. Các checkpoint flatten/mean cũ không sinh thêm
   weights hoặc đổi phép tính. Đây là lựa chọn đối chứng, chưa được chọn.
6. Thêm classifier phụ tùy chọn trên representation; cộng logits của nó với
   logsumexp reference bằng class_logit_weight. Loss thêm CE phụ với
   lambda_auxiliary_class. Mặc định tắt, không đổi weights/checkpoint cũ.
7. `train.py` thêm `loss.coordinate_source=topk` để thử dùng cùng tọa độ
   conditional top k cho loss. Mặc định dense vẫn giữ nguyên. Nhãn lớp thật
   chỉ dùng tính loss, forward không nhận nhãn. Log train_class_loss vẫn là
   class+reference cũ; auxiliary CE góp trong train_loss khi được bật.
8. `accuracy.py` giữ nguyên logits classifier phụ khi hiệu chỉnh decoder,
   tránh calibrate một model khác với forward thật. Không có classifier phụ
   thì hành vi cũ được giữ nguyên. Bộ 24 lựa chọn k/temperature không đổi.
9. Thêm tests cho offset invariance của kênh relative, AP thiếu và fingerprint
   rỗng, gradients của pooling/classifier/loss mới, calibration khớp forward,
   checkpoint self-contained và biểu diễn mặc định khớp tuyệt đối với cũ.
10. `improvement.py extend` thêm các design có overrides và hash, không ghi đè
    config đã đăng ký; chụp hash mã trước từng stage để phân biệt code lúc
    train/calibrate. Lượt mask đầu chạy bằng phiên bản controller ban đầu,
    có snapshot train và hashes của train trong metadata; calibration lúc đó
    đã có hỗ trợ classifier phụ nhưng option này tắt cho mask.
11. Sau khi cập nhật main, factory đã đăng ký hierarchical/llm/rf_autoloc
    nhưng CLI và hai tests còn cho rằng chỉ có KAN. Sửa danh sách CLI theo
    registry và cập nhật assertions; không thay thuật toán của các thành viên.
    Cài pandas 2.2.3 và requests 2.32.3 trong venv để import các backend main
    mới; giữ NumPy/PyTorch và môi trường train KAN. Đây là sửa tương thích
    kiểm thử, không được coi là nguyên nhân cải thiện sai số KAN.
12. Thêm EMA tùy chọn trong train: sau mỗi optimizer step, trọng số shadow
    nhận `d*shadow+(1-d)*live`, với `d=min(ema_decay,(1+steps)/(10+steps))`.
    Copy buffer từ live, chỉ đánh giá và chọn shadow trên validation. Checkpoint
    lưu shadow trong model_state_dict, đồng thời giữ live trong
    training_model_state_dict và số lần EMA update để không nhầm trạng thái
    optimizer. Mặc định tắt; trial `ema` dùng decay=0,995. Thêm unit test
    tính tay và smoke CPU riêng, không thay weights baseline.
13. Thêm `augmentation.py::SamePointMixing`, chỉ nhận dataset tên train.
    Khóa nhóm là (class_id,X,Y) gốc; chọn partner khác chính mẫu, cùng vị trí
    và lớp. Với xác suất 0,5, rút alpha đều trong [0,2;0,8]; AP cùng phát
    hiện được nội suy signal, AP chỉ một phía phát hiện chọn một parent theo
    alpha để mask vẫn nhị phân. Không thay nhãn, không sửa tensor gốc, không
    dùng validation/test. Singleton giữ nguyên. Augmentation chỉ ở DataLoader
    train, sau fit catalogue từ dữ liệu train gốc. Có unit test và smoke CPU.
14. Thêm marginalization offset tùy chọn khi suy luận bằng
    `inference_shifts_db`. Mỗi view chỉ dịch RSSI các AP phát hiện, clamp về
    [0,1]; trung bình phân bố xác suất reference bằng logsumexp/log softmax.
    Với classifier phụ, cộng bằng chứng lớp vào reference trước khi lấy
    softmax mỗi view. Giải mã phân bố hợp nhất theo decoder bình thường.
    Không có setting thì forward cũ giữ nguyên; train không bật các view.
    Test kiểm tra công thức trung bình, tính lặp lại và độc lập batch.
    Tính năng đã cài nhưng chưa được chọn hoặc hiệu chỉnh cho pipeline cuối.
15. Controller thêm `summarize`, xuất screening.json/CSV và
    exact_configuration_changes.json cho toàn bộ config đã đăng ký. Bảng
    chỉ lấy trial completed, chỉ chứa validation; không trộn score đang train
    với score sau calibration. Các design EMA/same_point/relative_ema đã
    chuẩn bị, chưa được coi đã train cho đến khi có summary/checkpoint thật.
16. Chạy đối chứng phi neural trên validation bằng script cá nhân
    `personal_learning/kan_under6_prototype_diagnostic.py`: gom fingerprint
    theo RP train bằng mean, median hoặc medoid; thử L1/L2, power và k.
    Tốt nhất mới 8,171686 m validation, kém KAN; đây không phải kết quả KAN
    và không được dùng thay cho mốc dưới 6 m. Bảng control giữ trong
    `prototype_diagnostic_validation.json`, chưa dùng official test.
17. Đăng ký design bổ sung trước khi chạy: `regularized` đổi dropout 0,1→0,2
    và AdamW weight decay 0,0001→0,01; `sigma_tight` đổi sigma 0,05→0,025;
    `sigma_wide` đổi sigma 0,05→0,08; `fine_tokens` đổi token 26→52,
    features/token 40→20 và token hidden 32→64; `aligned_aux_relative` kết
    hợp ba thay đổi đã có (relative ba kênh, classifier phụ, loss top k20).
    Không được gán kết quả của tổ hợp cho từng thành phần nếu chưa đối chứng.
18. Sửa câu log calibration từ “test unopened” thành “test not evaluated”.
    Repository có thể đọc byte raw test để kiểm hash toàn vẹn nguồn, nhưng
    không parse nhãn test hoặc dự đoán test trong train/calibrate. Thay câu
    log không thay số liệu hoặc thuật toán.
19. Đánh giá checkpoint baseline seed42 cũ trên đúng 3.912 mẫu validation
    bằng code mới và CPU: mean2D 6,032526866 m, joint accuracy
    0,991564417. Sai khác baseline GPU 6,032527022 nhỏ hơn 0,000001 m.
    Đây là kiểm chứng tương thích đường cũ, không phải một kết quả test mới.
20. Sau yêu cầu chỉ KAN-Mixer, dừng hàng đợi có `metric_centered`,
    `metric_centered_ema`, `metric_memory` khi nó còn đang chờ; chưa có trial
    GPU hoàn chỉnh cho các hướng bộ nhớ này. Gỡ module metric, các hook memory
    trong train/CLI/catalogue và tests riêng khỏi source hiện hành. Giữ bản
    sao mã có kiểm hash trong `runs/kan_joint_under6_20261020_v1/retired_nonpure_source`
    để truy nguyên công việc đã thử. Smoke CPU của hướng này không phải kết quả
    chất lượng và không được dùng chứng minh mục tiêu.
21. Trước khi phạm vi được thu hẹp đã thử trung bình xác suất các bản dịch
    RSSI trên validation, chưa đánh giá test. Các kết quả đó giữ trong
    `view_calibration/`, loại khỏi bảng chọn mô hình hiện tại. Gỡ
    `accuracy_views.py` và nhánh nhiều forward trong model; constructor từ chối
    `inference_shifts_db` khác rỗng. Bản sao mã lưu cùng archive ở mục 20.
22. `improvement.validate_kan_only` chặn head bộ nhớ, multi-view, refinement
    và ensemble trước khi tạo thư mục/run process. Không đăng ký design memory
    mới trong `extend`; các JSON design cũ bất biến được giữ nhưng đánh dấu
    excluded trong screening. Thêm `kan_only_scope.json` riêng, không sửa
    protocol ban đầu. Các trial còn lại chỉ dùng một mạng chung 13 lớp.
23. CLI `accuracy` mặc định chỉ calibration mạng thuần; `--pure-only` vẫn
    được nhận để tương thích lệnh cũ. Các API/công cụ lai lịch sử vẫn được giữ
    nhằm đọc checkpoint và tái lập thí nghiệm cũ, không dùng trong controller
    hiện tại. Thêm kiểm thử từ chối hướng bị loại và kiểm tra đúng lớp model.
24. Xác minh một lỗi provenance: checkpoint train dùng chung đối tượng dict
    giữa `config` và `metadata.config`; deepcopy toàn checkpoint vẫn giữ quan
    hệ alias. Khi calibration đổi temperature 1→0,5, bản cũ vô tình đổi cả
    metadata.config. Đã xác minh điều này trên pure_selected của same_point.
    Sửa bằng bản sao config độc lập và `metadata.training_config`; thêm test
    qua torch.save/load. Đây là sửa lưu vết, không đổi dự đoán và không được
    coi là nguyên nhân giảm sai số. Artifact cũ giữ nguyên; cấu hình train gốc
    vẫn có trong training/resolved_config.json và metadata JSON của lượt đó.
25. Thêm đối chứng `budget100` giữ cấu hình baseline nhưng patience=100.
    Lý do: same_point chạy đủ 100 epoch, best ở 87, trong khi baseline cũ dừng
    ở 41. Cần tách lợi ích augmentation khỏi lịch cosine đi đến cuối. Chưa được
    gán toàn bộ mức giảm validation cho SamePointMixing trước đối chứng này.
26. Sau khi thu hẹp phạm vi, chạy lại toàn bộ **56 tests đạt**, smoke một epoch
    CPU và `ips check` đạt. Nạp checkpoint same_point thật và đánh giá lại đủ
    3.912 mẫu validation trên CPU bằng code đã bỏ memory/multi-view: mean
    vẫn **5,462713 m**, joint accuracy 0,995143. Xác nhận lớp model là
    ReferenceKANMixer, 1.014.568 tham số, 872 điểm tham chiếu và 13 lớp,
    không có refinement/ensemble. Bằng chứng/hash nằm trong
    `single_kan_scope_verification.json`; đây vẫn là validation, không phải test.
27. `final_refit.py` trước đây chưa đọc tùy chọn augmentation/EMA mới. Đã bổ
    sung SamePointMixing cho dataset mang tên development với cờ final_refit
    rõ ràng, và giữ EMA cùng trạng thái live riêng. Chỉ chạy sau selection lock,
    không tính validation/test trong refit. Mặc định augmentation vẫn chỉ nhận
    train; validation không được truyền vào để fit hoặc tăng cường.
28. Refit phục hồi cấu hình train từ `metadata.training_config` hoặc JSON gốc
    có hash đã khóa. Với checkpoint calibration cũ có lỗi alias ở mục 24, bắt
    buộc cung cấp JSON train gốc; không đoán temperature. Chỉ cho phép khác
    top_k/temperature suy luận giữa cấu hình gốc và checkpoint được chọn.
29. Tách `training_top_k` và `training_temperature` khi refit: loss top-k phải
    dùng k/T như lúc chọn mô hình, còn eval dùng k/T đã calibration. `_decode`
    chỉ thay nhánh khi các tùy chọn mới này được khai báo; mặc định cũ không
    đổi. Refit giữ lịch cosine trên max_epochs gốc và đọc minimum_learning_rate
    nếu có, không nén lịch 100 epoch vào số epoch refit đã chọn.
30. Refit reset RNG sau khi dựng tạm model nguồn để không tiêu thụ ngẫu nhiên
    của khởi tạo mới; ghi rõ điều này trong metadata. Đây là thay đổi của quy
    trình refit, không áp dụng ngược lên checkpoint/kết quả baseline lịch sử.
    Train đầy đủ development chỉ được thực hiện sau khi chốt, là giao thức
    khác train-only 16.025 mẫu và phải được báo riêng khi thực sự chạy.
31. Thêm tùy chọn `head_transform=kan` bên trong ReferenceKANMixer: thay
    Linear(1664,256)→GELU→Dropout→Linear(256,N_RP) bằng
    KANLinear(1664,256)→Dropout→KANLinear(256,N_RP). Cùng efficient-kan,
    cùng grid5/order3/SiLU như mixing blocks; không chèn GELU giữa hai KAN.
    Mặc định linear_gelu giữ nguyên phép tính, thứ tự khởi tạo và key cũ.
    Đây là biến thể nội bộ mới của dự án, không phải khẳng định cấu trúc gốc
    của bài báo có sẵn head IPS này. Chưa được coi tốt hơn trước thực nghiệm.
32. Đăng ký trước khi chạy: `kan_head` đổi head và patience100 để đối chiếu
    budget100; `same_point_kan_head` thêm augmentation cùng vị trí; còn
    `deeper_same_point` giữ head cũ và tăng blocks2→3. Tất cả là một mạng chung,
    không thêm phương pháp truy xuất hay mô hình thứ hai. Giữ các trial không
    cải thiện và ghi thời gian/bộ nhớ vì head KAN làm tăng tham số/chi phí.
33. Kiểm thử mới xác minh gradient/checkpoint head KAN, decoder train/eval,
    hash cấu hình train cũ, chặn thay đổi ngoài T/k, cờ development rõ ràng,
    EMA/refit và không đọc split held-out. Sau các sửa này **61 tests đạt**;
    smoke CPU head KAN một epoch đạt (chỉ kiểm kỹ thuật, không phải chất lượng).
34. Kiểm tra train/validation thực tế: 872 tọa độ tham chiếu train và 933 tọa
    độ trong toàn development; 499 điểm train có nhiều PHONEID, tương ứng
    11.899/16.025 mẫu (74,2527%). Validation có 3.053/3.912 mẫu (78,0419%)
    ở tọa độ/lớp đã xuất hiện trong train. Split vẫn không trùng nhóm đo; nó
    không phải split người dùng/điện thoại/vị trí hoàn toàn mới. Bằng chứng ở
    train_validation_support_diagnostic.json, chỉ đọc train/validation.
35. Thêm chính sách partner `different_phone_first` cho augmentation: ưu tiên
    mẫu cùng lớp và tọa độ nhưng khác PHONEID; nếu không có thì giữ các partner
    cùng vị trí khác dòng. Mặc định any_same_point không đổi. PHONEID chỉ dùng
    chọn cặp train, không trở thành input forward hoặc điều kiện chọn model.
36. Thêm SamePointPairs và loss nhất quán điểm tham chiếu: ghép hai fingerprint
    cùng vị trí vào một batch 2B, chạy CÙNG mạng rồi tính supervised loss trên
    cả hai. Với p=softmax(a/Tc), q=softmax(b/Tc), m=(p+q)/2, dùng
    JS(p,q)=0,5 KL(p||m)+0,5 KL(q||m). Thêm lambda_consistency × JS vào loss.
    Các design mới đặt lambda=1 và Tc=1; không teacher hoặc bộ dự đoán khác.
    Tại điểm chỉ có một dòng, partner là chính dòng đó với nhiễu/dropout train
    độc lập. Lúc suy luận không cần cặp, PHONEID hay phép trung bình nhiều output.
37. Batch gốc 128 ở design consistency tạo 256 view trong mỗi bước. Số bước
    optimizer/epoch không tăng nhưng số view và chi phí tính tăng; ghi
    training_views_per_dataset_row=2, thêm train_reference_consistency_loss
    vào history. Không tuyên bố ngân sách FLOPs bằng baseline chỉ vì đều tối
    đa 100 epoch. Refit áp dụng đúng cùng wrapper/loss và ghi cùng metadata.
38. Đăng ký ba đối chứng trước train: same_point_cross_phone (trộn mẫu, ưu tiên
    khác điện thoại), cross_phone_consistency (cặp + JS, không trộn),
    same_point_consistency (cặp + JS + trộn p0,5). Cùng mạng/head chuẩn, patience
    100, không thêm module lúc inference. Unit tests và CPU smoke xác minh cặp
    không đổi vị trí/nhãn, chặn held-out, JS đối xứng/gradient hữu hạn và refit
    giữ loss. Smoke ghi JS 0,005874, không được coi là kết quả định vị tốt.
39. So sánh trên cùng 3.912 dòng validation, seed42, CPU: baseline 6,032527→
    same_point 5,462714 m. Tọa độ đã thấy: 5,615303→4,987396 m; 859 mẫu tọa
    độ chưa thấy: 7,515396→7,152057 m. Trên 3.877 mẫu cả hai cùng phân loại
    đúng: 5,924031→5,368895 m. Cải thiện không chỉ đến từ phân loại tòa/tầng,
    nhưng chưa tách augmentation khỏi thời lượng train. Nhóm tọa độ chưa thấy
    vẫn trên 6 m; phải nêu giới hạn này, không chỉ chọn nhóm có số đẹp.
40. Thêm improvement_analysis.py để đánh giá validation từ checkpoint thật,
    kiểm row order, nhãn nguồn và tái tính error từ tọa độ. Xuất phân tích tọa
    độ đã/chưa thấy, giữ nguyên mean2D trên toàn validation làm chỉ số chính.
    Controller summary bổ sung các nhóm này và số view train nếu có; ô chưa
    chạy là null, không điền số giả. Full suite 64 tests đạt trước test phân
    tích mới; riêng test phân tích mới và 13 test improvement còn lại đã đạt.
41. Thêm confirmation.py và khóa trước 24 cấu hình sàng lọc cùng hash mã.
    Chọn hai cấu hình tốt nhất ở seed42 cộng đối chứng budget100, xác nhận cả
    ba seed42/43/44, chọn bằng mean toàn validation rồi chạy cấu hình thắng
    với 10 seed cố định: 42,43,44,100,2024,777,999,45,46,47. Không chọn subset
    seed thuận lợi và không ghép dự đoán của nhiều mạng. Chưa được coi đã hoàn
    thành 10 seed chỉ vì plan/hàng đợi đã tồn tại.
42. Sau đủ 10 seed, controller mới được tạo selection_lock: số epoch refit là
    median best_epoch được làm tròn, tối đa100; giữ config/decoder/checkpoint
    hash và JSON train gốc từng seed. Nguồn refit gồm 10 model chính và ba seed
    budget100 đối chứng, cùng số epoch refit để có thể so sánh thay đổi model
    với việc thêm dữ liệu development sau khi chốt. Test sẽ báo riêng train-only
    và refit, mean/SD giữa các run; seed42 được cố định làm run minh họa/deploy
    trước khi xem test. Chưa có test hoặc refit dữ liệu thật ở thời điểm viết.
43. Bổ sung guard refit đối chiếu manifest, classes, AP order và preprocessing
    với checkpoint/lựa chọn đã khóa, trước khi fit catalogue. Kiểm cả hash mã
    đã khóa. Khi phát hiện còn thiếu guard này, bảo toàn plan ban đầu thành
    confirmation_plan_initial_source_audit.json và tạo plan v2 TRƯỚC xác nhận
    đa seed/test. Kiểm chứng danh sách cấu hình, seed và chính sách test giữa
    hai plan không đổi; chỉ cập nhật phiên bản/hash mã cho guard dữ liệu.
44. Sau các sửa trên, toàn bộ **67 tests đạt**; log tests_refit_manifest_guard.log.
    Kiểm thêm baseline_budget100_prefix_audit.json: 41 epoch đầu đối chứng
    budget100 khớp baseline cũ tuyệt đối ở train loss, các thành phần loss,
    validation mean/accuracy/F1, learning rate và optimizer steps (max diff=0).
    Không so thời gian. Điều này xác nhận mã mặc định không âm thầm đổi đường
    train cũ; đối chứng tiếp tục qua mốc dừng sớm cũ để kiểm tra lợi ích train lâu.
45. Tạo sáu hình validation từ số thật: cột toàn bộ/đã thấy/chưa thấy tọa độ,
    CDF sai số, ma trận gộp và hai ma trận riêng, đường học trước calibration.
    Builder cá nhân personal_learning/build_kan_under6_validation_figures.py;
    bản kiểm hình ở runs/kan_joint_under6_20261020_v1/validation_figures_seed42_v2.
    Giữ source snapshot và hash đầu vào/hình, kiểm trực quan. Mọi hình ghi rõ
    validation; không đưa số test chưa chạy hoặc biến số đường học thành số
    sau calibration. Bản v1 được giữ để truy nguyên lần chỉnh bố cục nhãn.
46. Đối chứng budget100 seed42 đã hoàn tất đủ100 epoch/12.600 optimizer steps,
    vẫn chọn epoch21, mean validation sau calibration **6,032527 m**, k1000/T0,5.
    Toàn bộ tensor state_dict của best khớp bitwise baseline cũ. So với
    same_point cùng100 epoch, mean giảm **0,569814 m (9,4457%)**. Như vậy chỉ
    kéo dài train không giải thích được mức giảm ở seed này; augmentation cho
    phép cải thiện trong điều kiện chạy lâu. Chưa suy rộng sang seed khác/test.
    Bằng chứng: budget100_completed_control_audit.json. Hash hai checkpoint
    khác nhau khi payload config/metadata khác, dù các tensor model được chọn
    giống nhau; không được coi hash file khác là bằng chứng weights khác.
47. Đánh giá lại checkpoint budget100 trên toàn validation bằng CPU: mean và
    toàn bộ phân nhóm khớp baseline cũ. Tạo bộ hình đối chứng cùng100 epoch ở
    validation_figures_seed42_matched100, dùng checkpoint/control thật và dữ
    liệu matched_budget100_vs_same_point_validation_comparison.json. Giữ bộ
    hình trung gian trước đây; chú thích mới đã bổ sung kết quả đối chứng và
    giới hạn một seed. Kiểm trực quan biểu đồ/ma trận/nhãn; chưa có hình test.
48. Thêm scripts/kan_under6_final.py làm entrypoint sau xác nhận, đặt ngoài
    source thuật toán đã khóa. Nó không thêm mô hình. Plan riêng chốt trước
    test: kết quả chính là mean10 seed của refit toàn development, train-only
    báo riêng; không chọn chế độ/seed tốt hơn sau khi xem test. Tất cả refit phải
    hoàn tất và hash khớp mới được ghi manifest đánh giá rồi đọc split test.
49. Driver cuối kiểm lại CSV dự đoán theo row index và tọa độ nguồn, tái tính
    Euclidean mean trên đủ1111 dòng, không dùng mean của riêng mẫu phân loại
    đúng. Báo SD giữa các run; 10 lần dự đoán cùng1111 mẫu không biến thành
    11110 mẫu kiểm thử độc lập. Benchmark seed42 với batch1/64, warmup10 và
    repeats100 trong process riêng, cùng RSSI train, tách forward/end-to-end.
    Đây là các bước đã chuẩn bị/kiểm thử, chưa được coi đã có kết quả test.
50. Thêm bốn tests cho layout10seed, chặn đọc test trước đủ refit, chặn mean
    có điều kiện/đảo thứ tự mẫu và không chọn best seed. Toàn bộ **71 tests đạt**,
    log tests_locked_final_pipeline.log. Hash source thuật toán và plan xác nhận
    v2 vẫn giữ nguyên; chỉ thêm driver/test/tài liệu ngoài phần đã khóa.
51. Đối chiếu lại nguồn KAN-Mixers v1, KAN, UCI và provenance efficient-kan.
    Ghi report_source_notes.md trong run để tách ý tưởng công bố khỏi thiết kế
    IPS riêng. Chưa chốt EPSG chỉ từ bài thứ cấp; giữ metric tọa độ gốc và
    ghi giới hạn CRS, không thay hệ số khoảng cách để đạt mục tiêu dưới6m.
52. Phân rã ablation seed42 từ 24 ứng viên decoder đã thực sự chạy ở mỗi
    checkpoint: cùng k5/T1, budget100=6,048640 m và same_point=5,526747 m,
    giảm0,521892 m. Calibration giúp baseline0,016113 m và augmentation0,064034 m;
    phần lợi ích khác biệt là0,047921 m. Tổng0,521892+0,047921=0,569814 m.
    Lưu validation_augmentation_decoder_ablation.json cùng source hash và hình
    validation_ablation_detail/07_weights_and_decoder_validation.png đã kiểm
    trực quan. Đây là phân rã số học từ số đo, không chứng minh cơ chế latent
    hoặc kết quả đa seed/test. Không gán toàn bộ mức giảm cho augmentation.
53. Chuẩn bị report_draft_core.md trong run, giải thích dữ liệu, KAN/mixing,
    head điểm tham chiếu, decoder, loss, augmentation, quy trình khóa test và
    cách đọc tài nguyên bằng ngôn ngữ cho người mới. Đánh dấu rõ là nháp trước
    kết quả cuối, chưa tạo Word kết luận đạt mục tiêu. Nháp nêu cả những trường
    không hoạt động ở nhánh hiện tại (mask_scale0, SmoothL1/label smoothing),
    phân biệt validation trong vòng train với metric train-set và tách phần
    giữ nguyên từ baseline khỏi nguyên nhân cải thiện mới.
54. Lập worktree_change_manifest_draft.json với18 file public đang thêm/sửa,
    branch/HEAD, SHA trước/sau và kiểm lại vendor efficient-kan không đổi byte.
    Đây là snapshot làm việc, chưa phải manifest cuối hoặc commit; các builder
    cá nhân/artifact bị ignore vẫn được ghi riêng trong nhật ký và receipt hình.

55. Sửa kế hoạch xác nhận thành v3 trước mọi seed43/44 và trước test mới.
    Validation có3053 mẫu ở tọa độ đã có trong train và859 mẫu ở tọa độ mới.
    same_point đạt5,462713 m toàn tập nhưng7,152057 m nhóm mới; relative kết hợp
    đạt5,582541 m toàn tập nhưng6,637284 m nhóm mới. Vì vậy thêm ứng viên tốt
    nhất nhóm mới vào pool hai ứng viên toàn tập và budget100; xác nhận42/43/44
    rồi chọn theo trung bình đều mean hai nhóm. Chỉ thay tiêu chí chọn kiến
    trúc; mean2D báo cáo, checkpoint/decoder theo toàn validation,24 config,
    10 seed và test1111 dòng giữ nguyên. Lưu kế hoạch cũ với hash, hủy hai queue
    đang đợi để tránh chạy nhầm v2; không hủy hoặc khởi động lại GPU trial.
56. Thêm test chọn kiến trúc theo nhóm hỗ trợ tọa độ và từ chối thiếu seed;
    toàn bộ **72 tests đạt trong15,139 giây**, log tests_balanced_confirmation.log.
    Kế hoạch v3 là quyết định dựa validation, không được mô tả là tiền đăng ký
    từ đầu nghiên cứu hoặc là chứng minh đã đạt test dưới6m.

57. Tạo bốn flow từ checkpoint same_point seed42 thật: input→output, một khối
    Mixer, decoder và giao thức validation→test. Chạy forward CPU chỉ để xác
    minh tensor, lưu trace/config/checkpoint SHA; không đo accuracy từ dữ liệu
    tổng hợp. Render Mermaid bằng Chrome headless tại máy, kiểm trực quan và
    sửa bố cục tổng quan thành ba cột để đọc được trên trang ngang. Bản dùng
    cho nháp nằm ở validation_model_flow_seed42_v2, có giải thích đánh số trong
    README. Sẽ dựng lại từ checkpoint thắng cuối nếu cấu hình thay đổi; không
    dùng hình seed42 tạm làm bằng chứng kiến trúc cuối khi chưa xác nhận.

58. Thêm phân tích cá nhân build_kan_validation_slices.py cho ba checkpoint
    budget100/same_point/same_point_relative seed42. Kiểm cùng3912 dòng, thứ tự
    và nhãn; tính lại Euclidean bằng tọa độ CSV, đối chiếu metric. Xuất bảng
    theo tòa, tòa/tầng, điện thoại, người đo và nhóm phân loại đúng/sai; thêm
    hai hình theo3 tòa và13 lớp. Đã kiểm trực quan. Tất cả ở
    validation_slices_seed42_v1, không dùng nhóm đúng lớp thay mean toàn tập.
59. Chuẩn bị assemble_kan_under6_report_data.py để gom nguồn báo cáo từ artifact
    thật và ghi hash. Snapshot validation hiện có21/24 trial; chế độ --final
    yêu cầu đủ screening, xác nhận10seed, mọi refit/test audit1111 dòng và
    benchmark seed42. Chưa tạo Word cuối hoặc khẳng định đã đạt ngưỡng từ
    snapshot này. Helper cá nhân bị ignore, không thay model hoặc training.

60. Kiểm đếm nhãn train/validation cho báo cáo refit, không fit model hoặc
    catalogue trên validation lúc sàng lọc. Có872 vị trí train,329 vị trí val,
    61 vị trí chỉ có trong val; hợp development933 vị trí. Refit sau chọn mô
    hình vì vậy sẽ đổi cả dữ liệu, số đơn vị head và số step/epoch126→156.
    Giữ nguyên center/scale train, seed khởi tạo mới và epoch đã khóa. Bổ sung
    cách giải thích này vào nháp, để không gán mọi lợi ích refit cho thuật toán.
    Bằng chứng refit_data_and_catalogue_count_audit.json chưa phải refit thật.

61. Chuẩn bị helper cá nhân build_kan_under6_report.py cho Word từ dữ liệu cuối
    đã kiểm tra, bảng đầy đủ24 trial, cấu hình, flow checkpoint đúng và nhật ký.
    Có native Word math, bảng có header lặp và số trang. Hiện chỉ kiểm cú pháp;
    chưa chạy authoring hoặc render, chưa tạo một báo cáo Word hoàn chỉnh.
    Builder từ chối snapshot validation, hình không cùng manifest, flow không
    đúng checkpoint cuối hoặc phần văn chưa được đối chiếu sau test. Trước
    khi dùng vẫn phải cập nhật lời văn theo winner thật và render/kiểm mọi trang.

62. Chuẩn bị phân rã đối chứng sau test cho báo cáo: trên cùng42/43/44, đặt
    A=budget100 train-only, B=phương pháp được chọn train-only, C=budget100
    refit, D=phương pháp được chọn refit. So A−B, B−D, A−C và C−D, có tính
    tương tác. Chỉ đọc kết quả đã chạy, không chọn lại model/seed/decoder.
    Helper từ chối thiếu/trùng đối chứng và giữ chỉ tiêu chính10seed. Kiểm
    bằng fixture nhỏ trong bộ nhớ cho thấy dù D của3seed dưới6m nhưng mean
    10seed trên6m vẫn báo chưa đạt; fixture không ghi thành số liệu thực nghiệm.
    Chưa có kết quả test thật để tính các hiệu ứng này. Không thay số lịch sử
    8,058m vào A nếu chưa kiểm chứng checkpoint tương đương, và không gọi hiệu
    ứng refit gồm nhiều thay đổi là lợi ích riêng của thuật toán KAN.

63. Thêm audit cá nhân đối chiếu checkpoint pure joint lịch sử với budget100
    hoàn tất theo seed. Dùng hash trong lock cũ, so raw byte mọi tensor state,
    mapping/AP/preprocessing/manifest và cấu hình suy luận; so prefix lịch sử
    số học, không so thời gian hoặc optimizer. Seed42 giữ best21, seed43 giữ
    best27; cả trọng số, cấu hình suy luận và prefix đều trùng. Chưa suy ra
    seed44 khi nó đang train. Collector sẽ kiểm đủ3 audit trước báo cáo cuối
    và kiểm hash checkpoint thực tế khớp checkpoint đã đánh giá. Đây không
    phải chạy test; các hash file có thể khác dù tensor giống vì metadata khác.

64. budget100 seed43 hoàn tất100 epoch, best27, validation5,986778 m;
    seed44 hoàn tất100 epoch, best37, validation5,910910 m. Audit mới xác nhận
    cả42/43/44 giữ nguyên raw byte state, cấu hình suy luận và prefix lịch sử
    số học của checkpoint pure joint cũ. Ba mean validation có trung bình
    5,976738 m và sample SD0,061427 m; không phải score test mới. Chạy lâu hơn
    đơn lẻ chưa đổi mô hình được chọn ở ba seed này. Bằng chứng ở
    baseline_equivalence_seed42/43/44.json và budget100_three_seed_validation_audit.json.
    Các file checkpoint vẫn khác hash do metadata; optimizer không được so sánh.

65. Máy khởi động lại ngày21/10 làm lượt `same_point_cross_phone`, seed43 dừng
    ở epoch40. Thư mục dở dang được chuyển vào `interrupted_attempts`, kiểm tra
    đủ69 hash file trước và sau; không xóa hoặc ghi đè. Checkpoint chưa lưu RNG
    và trạng thái generator của DataLoader nên chạy lại từ seed43 thay vì tự
    nhận là tiếp tục chính xác từ epoch41. Audit40 epoch đầu của lượt mới cho
    thấy mọi chỉ số số học trùng khớp hoàn toàn; chỉ bỏ qua thời gian chạy.
    Giữ nguyên nguồn, cấu hình và kế hoạch v3. Runner mới ghi log trực tiếp vào
    `resume_stdout_20261021.log`; không chạy lặp lại những trial đã hoàn thành.
66. Đối chiếu augmentation cùng vị trí trên đủ42/43/44: mean validation của
    đối chứng là5,976738±0,061427 m, bản tăng cường5,633186±0,167751 m. Mức giảm
    ghép cặp0,343552 m, tương ứng5,748%. Ở cùng decoder k5/T1, chênh lệch do
    trọng số/checkpoint là0,350539 m; lợi ích decoder khác biệt là−0,006987 m.
    Hai thành phần cộng lại đúng mức giảm thực tế. Không thể gán toàn bộ giảm
    cho calibration. Đây là validation3912 mẫu trên một split, chưa phải test;
    patience đối chứng100, tăng cường20; số epoch thật được ghi trong bảng.
    Hình phiên bản2 trong `validation_augmentation_three_seed_ablation_v2` đã
    sửa bố cục chú giải, giữ nguyên số đo và được kiểm tra trực quan.

67. Chuẩn bị bộ xuất hình cuối trong file cá nhân
    `personal_learning/build_kan_under6_final_figures.py`. Chỉ nhận bằng chứng
    test đầy đủ đã được audit; thử bằng snapshot validation bị từ chối đúng
    và không sinh thư mục hình giả. Khi có dữ liệu thật, bộ xuất sẽ kiểm lại
    đủ1111 dòng, cùng thứ tự/nhãn giữa model, sai số Euclidean từ tọa độ CSV,
    rồi vẽ mean/SD, từng seed, CDF, tòa/tầng, confusion, thời gian và bộ nhớ.
    Sơ đồ input được bổ sung các bước tính trung bình AP đang nghe, trừ trung
    bình và chia0,2 khi chọn chế độ RSSI tương đối. Các helper này chưa tạo báo
    cáo cuối hoặc thay đổi thuật toán đang được huấn luyện.

68. Thêm công cụ cá nhân `collect_kan_change_inventory.py` để đối chiếu mọi
    file mới/sửa với Git, ghi vai trò, giải thích và hash, không stage/commit.
    Snapshot thử ghi nhận đủ18 file public. Bộ gom dữ liệu báo cáo cuối yêu
    cầu inventory cuối, kết quả test thật, audit đủ dòng và benchmark; bộ tạo
    Word kiểm lại hash trước khi đưa bảng file vào báo cáo. Đây là tài liệu
    và kiểm tính toàn vẹn, không được xem là thay đổi làm giảm sai số model.

69. Hoàn tất đủ12 lượt xác nhận ban đầu. `same_point_cross_phone` có mean
    validation GPU qua42/43/44 là5,594853±0,097426 m; seed43/44 chạy đủ100 epoch,
    best73/57. `same_point_relative` seed43/44 lần lượt5,660620 và5,730258 m,
    cả hai dừng48 epoch, best28. Các số này đều chưa phải test.
70. Bảng xếp hạng CPU theo ba seed chọn `same_point_relative`: full mean
    5,657807 m; nhóm đã có tọa độ5,296138 m, nhóm chưa có6,943224 m; score cân
    bằng6,119681. Các score của cross_phone, same_point và budget100 lần lượt
    6,179792;6,213387;6,552804. Cross_phone có full mean thấp hơn nhưng không
    thắng tiêu chí đã khóa. Chênh lệch nhỏ giữa ba ứng viên không được tự coi
    là ưu thế có ý nghĩa thống kê. Score cân bằng chỉ chọn kiến trúc, không
    thay mean2D toàn tập hoặc ngưỡng mục tiêu test. Khác biệt cỡ10⁻⁷ giữa bảng
    CPU và calibration GPU là sai số số học của hai đường tính, không phải
    hai cấu hình khác nhau. Lưu bảng thật ở `confirmed_architecture.json`.
    Ứng viên thắng tiếp tục bảy seed đã định trước, bắt đầu100; không chọn
    lại kiến trúc hoặc seed bằng test. Chưa có selection_lock cuối hoặc refit.

71. Xuất hai hình xác nhận thật trong `confirmation_figures_v2`: so sánh các
    nhóm tọa độ và đối chiếu score chọn kiến trúc với mean toàn validation.
    Đã kiểm tra trực quan cả hai hình; nhãn không đè vạch SD/chấm seed, giữ
    nguyên số đo. Bộ xuất cuối chỉ sao chép khi winner/hash khớp. Bổ sung bảng
    median/P90/RMSE, classification, từng lượt tài nguyên, và đọc cấu hình suy
    luận trực tiếp từ checkpoint cho báo cáo; cấu hình train được lưu riêng.
    Bản Word cuối vẫn chưa được tạo vì chưa hoàn tất10 seed/refit/test.

72. Báo cáo dùng helper cá nhân `kan_report_math.py` để tạo công thức OMML
    có phân số, tổng, căn, số mũ và chỉ số thật, thay bộ diễn giải chuỗi đơn
    giản trước đó. Kiểm cấu trúc đủ23 công thức của bản thảo, từ chối công
    thức chưa được mô tả; chưa tạo DOCX. Việc render Word và đọc từng trang
    vẫn bắt buộc trước bàn giao. Đây là thay đổi cách trình bày, không đổi
    công thức mà model train hoặc dự đoán. Dùng font thân bài11pt và kích
    thước trang Letter theo skill tài liệu đang có hiệu lực.
73. `same_point_relative`, seed100 hoàn tất80 epoch, best60, calibration
    validation5,740441 m với k10/T0,5. CPU kiểm lại đủ3912 dòng: mean5,740441 m,
    nhóm đã có3053 dòng đạt5,250162 m, nhóm mới859 dòng đạt7,482958 m. Không
    loại seed này hoặc đổi cấu hình để giữ kết luận thuận lợi. Seed2024 đang
    chạy tiếp; chưa có kết quả test mới. CPU và GPU lệch khoảng1,5×10⁻⁷ m.
74. Bộ gom báo cáo cuối giữ riêng bảng validation của model train-only và
    kết quả test; model refit không được gán validation của checkpoint nguồn.
    Thêm bảng đủ10 seed với mean từng nhóm, best epoch và epoch thật đã chạy.
    Đây là bổ sung khả năng đối chiếu kết quả và giới hạn kết luận, không
    tác động các trọng số, AP order, split hoặc metric đang dùng.

75. Seed2024 của cấu hình đã chọn hoàn tất56 epoch, best36, validation
    calibration5,652697 m với k1000/T0,5, thời gian train311,712 giây. Đã có
    đủ42/43/44/100/2024, còn777/999/45/46/47; GPU tiếp tục777. Mean validation
    tạm của năm model là5,673312 m, chưa phải mean test hoặc kết quả cuối
    mười seed. Không đổi kế hoạch, cấu hình hay seed vì các kết quả tạm này.

76. Hoàn tất777/999/45/46/47, khóa đủ10 seed. Mean validation GPU là
    5,694870428752297 m, sample SD0,068927765186653 m. Best epoch theo thứ tự
    seed đã định là26/28/28/60/36/21/17/34/35/37; median làm tròn cho31 epoch
    refit. Giữ cả seed777 có mean5,802873 m và seed47 là5,756613 m; không chọn
    một tập seed thuận lợi. Test vẫn chưa được dùng cho vòng nghiên cứu này.
77. Audit độc lập `locked_confirmation_integrity_audit_20261021.json` kiểm
    đúng13 checkpoint nguồn, đủ3912 dự đoán validation mỗi model, row order,
    nhãn gốc và sai số Euclidean. Kiểm lại10 mean/SD và31 epoch từ lịch sử;
    hash mã và24 cấu hình trùng kế hoạch đã khóa. Phân nhóm dùng khóa chính
    xác gồm class tòa/tầng và XY gốc, không dùng bán kính gần nhau.
78. Refit thật seed42 hoàn tất31 epoch,19937 mẫu,209,500 giây. Nạp checkpoint
    v2 thành công trên CPU, kiểm tham số thực1.030.245, head cuối933×256, tám
    KANLinear và catalogue933 nhãn. Giữ đủ872 điểm cũ, thêm61 điểm, padding
    mỗi lớp dài nhất113. Preprocessing/AP order/classes trùng nguồn. CSV có
    4836 optimizer step, LR cuối0,0008041979948389244 theo lịch100 epoch gốc.
    Validation của refit làNone, không lấy điểm của nguồn làm score mới.
79. Sinh bốn sơ đồ `.mmd/.svg/.png` cho checkpoint refit thật trong
    `final_model_flow_seed42_v1`. Các hình dùng933 nhãn và padding113 đúng
    model, kèm README giải thích từng bước. Kiểm trực quan hình input/decoder;
    block/protocol trùng byte với hai hình đã kiểm ở preview v4. Mermaid CLI
    thực tế11.15.0, chỉnh font20, wrapping260 và khoảng cách30 để dễ đọc.
    Đây là kiểm kiến trúc/tensor, không tạo một kết quả test hoặc chứng nhận6m.
80. Bổ sung bảng/biểu đồ đối chiếu validation, test train-only và test refit
    theo từng seed cho báo cáo cuối, chỉ sinh khi có đầy đủ dữ liệu test thật.
    Tăng kiểm hash CSV validation và checkpoint trong bộ gom báo cáo; không
    thay metric, model hoặc code huấn luyện đang khóa. Word vẫn chưa tạo;
    cần hoàn tất mọi refit, test và kiểm trang trước bàn giao.

81. Trước khi có kết quả test, lưu bản sao nguyên byte của103 file mã nguồn,
    cấu hình và giao thức trong `frozen_source_snapshot_v1`. Manifest kiểm lại
    SHA-256 của từng file đã sao chép và ghi hash selection lock. Việc này
    giúp tái dựng đúng phiên bản của vòng nghiên cứu ngay cả khi sau đó cần
    thử kiến trúc khác; không thay mô hình, trọng số, dữ liệu hoặc metric.
82. Rà bản nháp giải thích theo cấu hình đã chọn: tách rõ baseline/same_point
    dùng mask_scale0 với same_point_relative dùng mask_scale1. Bổ sung công
    thức đếm AP quan sát, trung bình có mask và RSSI tương đối; kiểm ví dụ
    CPU -52/-78 dBm cho0,625/-0,625, bất biến khi cộng chung13 dB trong điều
    kiện không đổi mask/clipping, và trường hợp mọi AP thiếu cho đầu ra0.
    Kiểm XML của đủ26 công thức Word dạng native thành công; chưa tạo hoặc
    render DOCX. Đây là làm rõ tài liệu, không phải cải thiện metric mới.
83. Đóng vòng nghiên cứu v1 sau13 refit,26 đánh giá test và8 benchmark thật.
    Audit độc lập đọc lại đủ1111 dòng mỗi CSV, kiểm tọa độ/nhãn gốc, thứ tự,
    hash checkpoint, mean/median/P90/RMSE, accuracy, macro-F1 và ba ma trận.
    Kết quả chính10 seed refit là7,424523946 ±0,293599286 m: CHƯA đạt6 m.
    Đối chứng refit3 seed là7,529234037 ±0,281162273 m; đối chứng train-only
    khớp kết quả lịch sử8,058152571 ±0,306070513 m. Không thay bằng median
    4,522050 m hoặc validation5,694870 m để tuyên bố đạt mục tiêu. Giữ vòng
    này làm bằng chứng đóng; nghiên cứu mới cần kế hoạch validation mới và
    ghi nhận test đã được xem, không tuyên bố test hoàn toàn chưa biết.
84. Mở vòng v2 riêng trong `kan_joint_under6_20261021_v2`, giữ nguyên split,
    AP order, tọa độ, metric và chính sách khóa trước lượt test mới. Tám cấu
    hình gồm đối chứng budget100 gốc, lặp lại v1_relative, rank_relative,
    nearby_5m, nearby_10m, residual_axis5m, nearby_residual và relative_dropout.
    Kế hoạch ghi rõ đây là nghiên cứu tiếp sau khi đã biết kết quả v1, không
    phải một test chưa từng được xem. Chọn mới vẫn bằng validation; không
    đánh giá test từng ứng viên để tìm cấu hình thuận lợi.
85. `relative_rank_mask` thêm rank trung bình khi trùng RSSI trong chính
    fingerprint, ánh xạ [-1,1], giữ vị trí AP. AP thiếu và trường hợp chỉ nghe
    tối đa1 AP có rank0. Ba kênh relative/rank/mask thành1560 số, reshape
    26×60. Chỉ rank bất biến với biến đổi tăng đơn điệu giữ thứ tự/tie; không
    khẳng định toàn bộ biểu diễn hoặc mạng bất biến với mọi loại đổi thiết bị.
86. `NearbyPointMixing` chỉ chọn điểm train khác cùng lớp trong bán kính
    tọa độ gốc, chọn đều điểm rồi đều fingerprint của điểm đó. Nội suy XY với
    cùng alpha dùng trộn RSSI; AP thiếu vẫn có mask nhị phân. Đây là nhãn
    tổng hợp, chưa xác nhận quan hệ vật lý trong nhà. Không truy xuất khi
    inference. Bán kính5/10 có partner cho73,042%/96,480% dòng train. Dùng
    `training_representation()` chung cho train/refit; từ chối bật đồng thời
    nearby với same-point/consistency thay vì âm thầm bỏ qua một tùy chọn.
87. Head residual tùy chọn gồm hai KANLinear trong cùng mạng, xuất13×2 giá
    trị, tanh rồi nhân scale0,05. Giới hạn là±4,956 m mỗi trục ở scale hiện
    tại, không phải bán kính2D5 m. Khởi tạo hai weight cuối bằng0, giữ spline
    scaler để học được gradient. Cộng vào cả decoder dense/top-k và giữ phần
    này trong calibration k/T. Mặc định không có head mới. Checkpoint refit
    thật v1 đối chiếu mã snapshot cũ cho năm tensor đầu ra khớp chính xác.
88. Toàn bộ80 kiểm thử đạt (log `tests_frozen.log` ở vòng v2); smoke thật
    CPU/GPU một epoch cho ba tính năng mới cùng hoạt động đã hoàn tất. Các
    sai số smoke không dùng đánh giá nghiên cứu. Đã khởi chạy GPU tuần tự
    v2 từ v1_relative, in từng epoch; chưa có kết quả test cho ứng viên mới.
89. Đối chứng v1_relative seed42 đã hoàn tất46 epoch. Cả50 tensor state
    giống nguyên byte checkpoint được chọn của vòng v1; toàn bộ lịch sử
    số học46 epoch cũng trùng, trừ thời gian. Inference config và validation
    calibration5,5825414325 m, k10/T0,5 trùng chính xác. Hash cả file mới
    khác vì metadata/mã nguồn đã đổi; không diễn giải đó là trọng số đổi.
    Audit lưu tại `v1_control_complete_audit.json`. GPU chuyển sang rank_relative.
90. rank_relative seed42 hoàn tất46 epoch,1.015.848 tham số (thêm1.280 do
    embedding60→64 thay40→64). Validation GPU sau calibration là5,756392 m,
    k20/T0,5. CPU kiểm đủ3912 dòng: mean5,756393 m; nhóm3053 tọa độ đã có
    đạt5,313028 m và859 tọa độ mới7,332170 m. Đối chứng v1_relative tương
    ứng5,582541/5,285776/6,637284 m. Cấu hình rank chưa cải thiện cả mean
    chung lẫn nhóm mới ở seed này; không suy rộng thành kết luận cho mọi
    mô hình rank. Chưa đánh giá test ứng viên mới. GPU tiếp tục nearby_5m.
91. nearby_5m seed42 hoàn tất39 epoch, best19,1.014.568 tham số; calibration
    validation5,774471 m với k10/T0,5. CPU kiểm3912 dòng cho5,774471 m;
    nhóm đã có5,306389 m, nhóm mới7,438096 m. Chưa tốt hơn đối chứng ở seed
    này. Giữ kết quả và tiếp tục nearby_10m đúng kế hoạch. Không đổi bán kính
    hay chuyển sang test để chọn cấu hình theo kết quả thuận lợi.
92. nearby_10m seed42 hoàn tất60 epoch, best40,1.014.568 tham số; calibration
    validation5,667143 m, k20/T0,5, train707,171 giây. Chưa vượt đối chứng
    theo mean toàn tập; giữ để phân tích nhóm tọa độ mới theo kế hoạch.
    GPU chuyển sang residual_axis5m, không thay decoder/seed theo test.
93. Kiểm khả thi CPU riêng cho tokenization, chưa thêm vào tám ứng viên đã
    khóa: một AP/token với16/32 channel và nhóm5 AP/token đều forward/
    backward hữu hạn, đầu ra13 lớp và2 tọa độ. Tham số thật tương ứng
    1.531.064/1.536.840/754.312. Dùng batch tổng hợp2 mẫu, không có kết quả
    accuracy hoặc test; chỉ chứng minh kiến trúc hỗ trợ mà không sửa mã.
94. Thăm dò decoder riêng bằng cùng điểm số KAN v1_relative seed42 trên đủ
    3912 validation, một forward/mẫu, không truy xuất fingerprint. Trung tâm
    không gian robust tính bằng30 bước lặp có làm trơn cho mean tốt nhất
    5,654001 m, k1000/T0,5, so với decoder trung bình hiện có5,582541 m.
    Không tốt hơn nên chưa đưa vào code mô hình hoặc danh sách v2. Đây là
    cách ước lượng tọa độ khác, không đổi metric mean sai số thành median.
    Không dùng test, không tuyên bố thuật toán lặp hội tụ chính xác.
95. CPU kiểm đủ3912 validation cho nearby_10m: mean5,667143 m, nhóm3053
    tọa độ đã có trong train5,138827 m, nhóm859 tọa độ chưa có7,544848 m.
    So với đối chứng5,582541/5,285776/6,637284 m, nhóm đã có tốt hơn nhưng
    toàn tập và nhóm chưa có đều kém hơn. Không dùng cải thiện một nhóm để
    tuyên bố augmentation mới đã tăng khả năng tổng quát. Bốn cấu hình đầu
    của v2 đã có phân tích CPU; residual_axis5m đang train, chưa kết luận.
96. Đối chiếu câu hỏi “đã thay đổi thuật toán chưa”: đã thử thay kiến trúc
    pooling, số block, head KAN; cách biểu diễn tín hiệu; loss/EMA và cách
    tạo cặp train ở v1. V2 thêm rank, trộn tọa độ gần và head hiệu chỉnh
    hai KANLinear bên trong cùng mạng. Đây gồm cả thay kiến trúc và cách học,
    không chỉ đổi learning rate. Tuy vậy công thức KANLinear B-spline, grid5,
    spline bậc3 và cơ chế lưới của efficient-kan vẫn giữ nguyên; chưa thử
    thay hàm cơ sở hoặc cập nhật lưới thích nghi. Không nhận những phần đó
    là đã triển khai. Mục tiêu test dưới6 m vẫn chưa đạt.
97. residual_axis5m seed42 hoàn tất65 epoch, best45,1.555.368 tham số,
    train797,966 giây; calibration k20/T1 cho validation5,552420 m. CPU kiểm
    đủ3912 dòng: mean5,552420 m, nhóm đã có5,209257 m và nhóm mới6,772065 m.
    Mean chung tốt hơn đối chứng5,582541 m khoảng0,030 m, nhưng nhóm mới
    vẫn kém6,637284 m. Chưa đủ bằng chứng chọn cấu hình này; tiếp tục đủ tám
    ứng viên và quy tắc xác nhận đa seed. GPU chuyển sang nearby_residual.
98. Audit head đã train trên đủ3912 validation: forward thật và decoder dùng
    logits/residual cache khớp chính xác cả logits, xy và conditional_xy.
    Head thêm540.800 tham số, tổng10 KANLinear; residual shape3912×13×2.
    Giới hạn mỗi trục4,955751 đơn vị tọa độ gốc; độ dịch2D trung bình1,286516,
    không thành phần nào vượt99% giới hạn trong các lớp được dự đoán. Bỏ
    residual ở inference của chính checkpoint này, giữ k20/T1, cho5,697597 m
    thay vì5,552420 m. Đây là chẩn đoán một checkpoint, không phải đối chứng
    huấn luyện lại hoặc bằng chứng nguyên nhân của mọi thay đổi kiến trúc.
    Helper audit lần đầu dùng nhầm khóa `scale`, đã sửa đúng `shared_scale`
    theo preprocessing.json và chạy lại thành công; không đổi phép chuẩn hóa.
99. Chẩn đoán read-only trên1024 mẫu train chọn cố định seed42 của đối chứng:
    cả tám KANLinear có tỷ lệ input ngoài toàn miền hỗ trợ knots bằng0;
    ngoài miền nominal cao nhất1,458%. RMS nhánh spline/base khoảng0,156–0,400.
    Không coi RMS là độ quan trọng nhân quả, không kết luận đổi lưới chắc sẽ
    tốt hơn. State model trước/sau giống nhau. Receipt dùng trường mô tả đúng
    việc loader đọc CSV development chung nhưng chỉ train đi qua forward:
    `train_spline_activity_control_v2.json`. Bản receipt đầu giữ lại như lịch sử.
100. Bảo toàn74 file nguồn/cấu hình/split metadata của vòng v2 tại
    `frozen_source_snapshot_v2`: kiểm từng byte/hash, gồm42 file Python đã khóa.
    Snapshot được chụp trước selection lock và trước test v2; tài liệu Markdown
    là bản chụp tại thời điểm đó, có thể được cập nhật sau. Không nhân bản dữ
    liệu CSV gốc hoặc checkpoint; không nhận snapshot là kết quả đạt mục tiêu.
101. nearby_residual seed42 hoàn tất46 epoch, best26,1.555.368 tham số,
    train489,966 giây. Calibration k20/T1 cho5,556056 m; CPU đủ3912 dòng
    cho5,556056 m, nhóm đã có5,164270 m và nhóm mới6,948518 m. Tổ hợp này
    chưa vượt head residual đơn lẻ hoặc đối chứng ở nhóm mới; giữ bằng chứng
    và tiếp tục relative_dropout, rồi budget100 theo danh sách đã khóa.
102. Xuất hai hình và bảng CSV validation thực tế của sáu cấu hình đầu vào
    `interim_six_validation_figures`, kèm bản chụp JSON nguồn và hash hình.
    Đã mở kiểm trực quan cả hai PNG: nhãn, chú thích và số liệu đọc được.
    Ghi rõ một seed, chưa phải test; hình bỏ residual là chẩn đoán cùng một
    checkpoint, không phải hai mô hình train độc lập. Đồng thời đã mở kiểm
    13 hình còn lại của v1, đối chiếu byte với manifest hình gốc; hai hình
    đầu/mục phân rã cặp đã xem trước đó. Đây chưa phải kiểm trang DOCX.
103. relative_dropout seed42 hoàn tất46 epoch, best26,1.014.568 tham số,
    train492,765 giây. Calibration k1000/T0,5 cho5,803370 m; CPU đủ3912
    validation cho5,803370 m, nhóm đã có5,344960 m và nhóm mới7,432619 m.
    Tăng nhiễu2→4 dB và AP dropout0,05→0,15 chưa cải thiện ở seed này.
    Hoàn tất7/8 sàng lọc; lượt budget100 đối chứng đang chạy, không bỏ lượt
    hoặc đổi shortlist trước khi có đủ kết quả theo kế hoạch.
104. Thăm dò decoder riêng trên checkpoint đối chứng seed42, một forward/mẫu:
    trộn tọa độ có điều kiện giữa các tầng chỉ trong tòa được mạng dự đoán.
    Không dùng tòa/tầng thật, không thêm bộ dự đoán hoặc đổi nhãn phân loại.
    Tốt nhất ở k10/T0,5, hệ số trộn0,25 và nhiệt độ tầng2: mean validation
    5,568650 m, nhóm mới6,590866 m; decoder cũ tương ứng5,582541/6,637284 m.
    Mức giảm mean chỉ0,013891 m, chưa có xác nhận đa seed hoặc bằng chứng test.
    Giữ tại `separate_floor_decoder_validation_feasibility.json`, không sửa
    mô hình, cấu hình hoặc danh sách v2 đang khóa. Các tổ hợp được thử đều lưu.
105. Kiểm toán CPU tổng hợp trước khi cân nhắc lưới thích nghi cho vòng sau:
    `update_grid` của bản efficient-kan được ghim tính mục tiêu từ trọng số
    spline đã nhân scaler cạnh, rồi chép hệ số fit vào trọng số chưa scale;
    forward sau đó lại nhân scaler. Với hàm spline hằng được dựng để tách
    riêng hiệu ứng này, sai khác đầu ra lớn nhất0,21/0,49 khi scaler không
    bằng1. Các số này là đơn vị tổng hợp của lớp, không phải mét định vị.
    Wrapper CPU tạm đưa scaler về1 trong lúc fit rồi phục hồi scaler giữ sai
    khác dưới3e-7 ở hai fixture đó, đồng thời kiểm scaler1 và không có scaler.
    Không sửa vendor hoặc checkpoint thật; chưa huấn luyện IPS bằng lưới động.
    KAN hiện tại không gọi cập nhật lưới, nên nhánh không hoạt động này không
    làm thay đổi kết quả v1/v2. Trước triển khai thực cần kiểm hàm phi tuyến
    thật, chỉ lấy activation train và xử lý optimizer state của hệ số đổi cơ
    sở. Receipt: `separate_adaptive_grid_scaler_cpu_audit.json`.
106. Thống kê riêng16025 dòng train:872 vị trí tham chiếu có2–80 bản ghi mỗi
    vị trí, median20, hệ số biến thiên0,488189. Có499/872 vị trí được đo bởi
    hơn một PHONEID; số điện thoại mỗi vị trí1–4. Với sampler theo dòng hiện
    tại, điểm nhiều bản ghi xuất hiện nhiều hơn trong mỗi epoch. Đây là cơ sở
    cân nhắc lấy mẫu cân bằng vị trí trong vòng sau, chưa phải phương pháp đã
    triển khai hoặc bằng chứng rằng nó sẽ giảm sai số. Không dùng validation/
    test để tính thống kê; lưu `train_point_sampling_distribution.json`.
107. budget100 seed42 hoàn tất đủ100 epoch, best21,12600 optimizer step,
    1.014.568 tham số, train895,198 giây. Mean validation trước calibration
    6,048640 m và sau calibration k1000/T0,5 là6,032527 m. Hai số thuộc cùng
    trọng số, khác decoder; không gọi số validation là test. Audit đối chiếu
    checkpoint lịch sử xác nhận50 tensor giống nguyên byte và schema/inference
    tương đương. So với budget100 v1, đủ100 dòng lịch sử số học trùng, chỉ thời
    gian khác. Hai receipt `baseline42_historical_equivalence.json` và
    `baseline42_full100_replay_audit.json`; không suy luận tốc độ cố hữu từ
    lượt chạy có desktop/CPU probe đang hoạt động cùng lúc.
108. Hoàn tất8/8 cấu hình v2 seed42, CPU phân tích đủ3912 dòng cho từng cấu
    hình. Shortlist thật tại `confirmation_candidates.json` gồm budget100,
    nearby_residual, residual_axis5m và v1_relative: hai mean chung tốt nhất,
    ứng viên tốt nhất ở nhóm tọa độ mới, cộng đối chứng theo kế hoạch. GPU
    đã chuyển sang budget100 seed43. Chưa có kiến trúc thắng đa seed, selection
    lock hoặc test v2; không chọn riêng seed42 để công bố đạt mục tiêu.
109. Kiểm khả thi cập nhật lưới trên bản sao độc lập của tám KANLinear đã học:
    lấy512 truy vấn từ train bằng seed42, mỗi lớp thu2048 hàng activation,
    dùng1638 hàng fit lưới và410 hàng kiểm. Mọi activation đều từ train, không
    phải validation; hai nhóm activation có thể thuộc cùng fingerprint nên
    không xem là đánh giá khái quát hóa. Với wrapper xử lý scaler, đầu ra và
    gradient đều hữu hạn, knots tăng nghiêm ngặt ở cả tám lớp. RMS thay đổi
    tương đối sau đổi cơ sở khoảng0,0025%–1,016%, không bằng0; đây là sai khác
    activation, không phải mét IPS. Chưa áp dụng đồng thời lên toàn mạng, chưa
    kiểm optimizer state hoặc huấn luyện lại. Model gốc không đổi state.
    Receipt: `separate_train_activation_grid_reprojection.json`.
110. Chạy thật CLI train ở CPU một epoch cho cách chia một AP/token, tách riêng
    tại `runs/kan_ap_token_cpu_smoke_20261021`, không thêm vào tám trial v2.
    Giữ input520×2 và RSSI tương đối; token520×2, embedding520×16, hai block
    KAN, token_hidden16, channel_hidden32, head_width128. Smoke dùng256 train/
    128 validation, hai optimizer step,1.446.956 tham số và220 nhãn RP. Lưu và
    nạp lại checkpoint thành công, logits2×13/xy2×2 hữu hạn; source42 file
    đang khóa vẫn trùng. Điểm sau một epoch không dùng so sánh accuracy; cấu
    hình này chưa train đầy đủ, chưa đo VRAM GPU và không được đánh giá test.
111. Đối chứng budget100 seed43 hoàn tất100 epoch, best27,12600 optimizer
    step, train895,946 giây. Validation trước calibration6,116763 m; sau chọn
    k20/T0,5 là5,986778 m. Audit checkpoint lịch sử xác nhận50 tensor giống
    byte, schema/inference tương đương và prefix lịch sử số học khớp. Không
    gọi việc khớp checkpoint là một kết quả test mới. GPU tiếp tục seed44;
    chưa có đủ kết quả xác nhận ba seed cho các kiến trúc mới.
112. Lưu thiết kế cấu hình dự kiến trước khi có test v2 tại
    `runs/kan_tokenization_design_20261021_v0`, chưa tạo nghiên cứu huấn luyện
    mới hoặc chạy thêm GPU. Giữ RSSI tương đối/mask, hai block B-spline và cách
    học hiện có; ba cách chia token có1.531.064,1.536.840 và1.527.296 tham số
    với872 nhãn RP. Đối chứng head rộng464 có1.542.264 tham số để giảm nhiễu
    do chênh lệch dung lượng; đối chứng cũ1.014.568 và budget100 được giữ riêng.
    Đây là số đếm từ model thật và forward tổng hợp hữu hạn, chưa phải kết quả
    accuracy. Các cấu hình đầy đủ đều giữ520 AP×2 số, khác cách nhóm token và
    phân bổ channel/head. Chỉ cân nhắc sau khi đóng vòng v2 nếu mục tiêu vẫn
    chưa đạt; không dùng kết quả test tương lai để sửa lựa chọn đã lưu.

113. Người dùng đổi tiêu chí hiện tại thành mean2D test dưới7 m ngày21/10/2026.
    Giữ nguyên một KAN-Mixer cho cả ba tòa/13 lớp, mười seed định trước, mọi mẫu
    test và công thức khoảng cách. Ghi amendment riêng, không sửa protocol hoặc
    hash đã khóa theo mục tiêu6 m trước đây. Cờ đạt6 m của thí nghiệm lịch sử khác
    với việc xét yêu cầu mới7 m; báo cáo phải nêu cả hai rõ ràng. Kết quả v1
    7,424524 ±0,293599 m vẫn chưa đạt. Không giải thích rằng đã xuống6 m khi chưa
    có kết quả thật.

114. Khôi phục hàng đợi sau khi guard phát hiện sáu khoảng trắng cuối một dòng
    trong `models/reference.py`. AST không đổi; không ghi đè file đang sửa trên
    workspace. Tạo bản thực thi riêng từ42 file Python đã khóa và controller cũ,
    liên kết tới cùng thư mục dữ liệu, tiếp tục cùng đường dẫn kết quả. Kiểm tra
    hash42 file, manifest/số mẫu và nạp checkpoint/forward tổng hợp thành công.
    Đây là thay đổi cách vận hành, không phải thay đổi thuật toán hoặc metric.
    Đối chứng budget100 seed44 hoàn tất100 epoch; validation5,910910 m sau
    calibration. Audit xác nhận50 tensor/checkpoint suy luận giống đối chứng
    lịch sử; không tạo thêm một kết quả test khi đối chiếu trọng số.
115. Chuẩn bị bộ tổng hợp báo cáo v2 đọc đúng bản mã thực thi đã khóa, giữ riêng
    mục tiêu6 m trong protocol và yêu cầu hiện tại7 m. Snapshot báo cáo validation
    đã lấy đủ tám cấu hình seed42 và kiểm lại98 hash artifact; nó không phải báo
    cáo cuối và không khẳng định đạt test. Sáu kiểm thử bằng fixture tổng hợp
    kiểm tra ngưỡng7 m nghiêm ngặt, không thay cờ6 m, từ chối thiếu test, thay
    seed, đổi protocol hoặc thiếu dòng audit đều đạt. Các số trong fixture chỉ
    phục vụ kiểm thử logic, không phải số liệu IPS. Hàng đợi vẫn chỉ chạy danh
    sách đã chốt; không mở thêm cấu hình trong vòng này.

116. Ngày 22/10, đã hoàn thành báo cáo `references/V3.docx` theo yêu cầu
     giải thích kết quả 7,4 m. Đối chiếu lại 401 hash nguồn của vòng v1;
     tài liệu có 24 trang, 15 bảng, 7 hình và 11 công thức Word chỉnh sửa được.
     Kiểm số liệu từng hàng và xem toàn bộ trang sau render đều đạt. Báo cáo
     giữ rõ số 7,424524 m là mean qua mười model, còn seed42 là 7,449044 m.
     Phép ghép cặp ba seed cho thấy cấu hình được chọn sau refit cao hơn đối
     chứng refit 0,122533 m; không quy mọi mức giảm cho cải tiến lõi B-spline.
117. Kiểm tra tiến trình thật xác nhận vòng v2 đã dừng, mã thoát 1073807364;
     chưa có bằng chứng kết luận nguyên nhân phần cứng hoặc thiếu VRAM. Giữ
     nguyên toàn bộ 17 trial đã hoàn tất. Lượt `nearby_residual/seed2024` dở
     sau epoch5 được chuyển sang `interrupted_trials_20261022`, kiểm toàn bộ
     70 file lưu trữ cùng hash các trial hoàn tất trước khi chạy lại. Xác minh
     42 file mã thực thi và tám cấu hình còn đúng bản khóa; không sửa kế hoạch,
     thuật toán hoặc cấu hình giữa hàng đợi. Chạy lại từ đầu cùng seed2024
     vì chưa có hợp đồng khôi phục đầy đủ optimizer/RNG giữa epoch trong mã.
118. Lượt chạy lại ngày 22/10 khớp chính xác năm epoch đầu: loss, validation,
     learning rate và optimizer step có độ lệch số học bằng0; loại thời gian
     epoch khỏi đối chiếu vì tải desktop đã đổi. Tiếp tục cùng danh sách mười
     seed đã chốt, không chọn lại seed theo test. Thêm ghi nhận khôi phục vào
     bộ tổng hợp báo cáo riêng; snapshot validation mới kiểm101 hash đạt,
     chưa đánh giá test. Bộ theo dõi riêng chỉ kiểm độc lập và tổng hợp sau
     khi đủ refit/test theo kế hoạch; không chạy thêm mô hình GPU. Chi tiết
     ở `runs/kan_joint_under6_20261021_v2/recovery_seed2024_20261022*.json`.

119. Tổng hợp ba biểu đồ đối chiếu xác nhận v2 trong
     `confirmation_figures_v2`, kèm CSV đủ12 dòng cho bốn cấu hình và ba seed.
     Đối chiếu42 hash nguồn và xem trực tiếp cả ba PNG đều đạt. So cùng seed,
     thêm head KAN giảm mean validation0,052324 m với decoder k5/T1, hoặc
     0,123416 m sau calibration; nhóm tọa độ chưa có trong train giảm0,311402 m
     sau calibration. Thay same-point bằng nearby-point mixing giảm mean toàn
     validation0,089894 m sau calibration, nhưng nhóm tọa độ chưa có trong
     train tăng0,061831 m. Vì vậy không khẳng định augmentation mới giúp mọi
     vị trí mới, hoặc những chênh lệch này là tác động riêng trên test. Epoch,
     dung lượng mạng và decoder được chọn có thể khác giữa các model.
120. Seed2024 chạy lại hoàn tất56 epoch, best36,7056 optimizer step và
     1.555.368 tham số; train462,822482 giây theo log. Validation trước chọn
     decoder5,725185 m, sau chọn k20/T1 là5,589130 m. Đã có5/10 seed của
     `nearby_residual`; hàng đợi chuyển sang seed777. Chưa có kết quả test
     mới hoặc bằng chứng đạt mục tiêu7 m ở vòng v2.

121. Seed777 hoàn tất41 epoch, best21,5166 optimizer step, train383,371931
     giây. Validation trước chọn decoder5,737425 m; sau chọn k1000/T1 là
     5,542629 m. Con số1000 là cấu hình yêu cầu của decoder; số điểm hợp lệ
     thực tế vẫn bị giới hạn bởi catalogue của lớp dự đoán, không phải truy
     xuất1000 fingerprint RSSI. Đã xong6/10 seed; GPU chuyển sang seed999,
     sau đó còn45,46,47. Vẫn chưa đánh giá test vòng v2.

122. Người dùng đổi yêu cầu thực thi: chạy xongseed999 rồi đánh giá test ngay.
     Lần đánh giá sớm dùng bảy checkpoint train-only của42,43,44,100,2024,
     777,999; không chạy45,46,47 hoặc refit trong giai đoạn này. Giữ nguyên
     protocol mười seed/refit đã khóa, ghi kế hoạch riêng trước test tại
     `early_test_after_seed999_20261022/user_requested_early_test_plan.json`.
     Đặt chốt lịch chạy trước45 bằng artifact có ghi rõ không phải trial đã
     huấn luyện, để controller tự dừng qua guard sẵn có sau khi999 hoàn tất.
     Không sửa mã hoặc cấu hình đang chạy, không dừng ngangseed999. Bảy model
     được chốt theo thứ tự seed cũ và yêu cầu của người dùng, trước khi biết
     kết quả test. Kết quả này sẽ được ghi riêng, không gán thành kết quả
     mười seed refit của kế hoạch ban đầu.

123. Theo yêu cầu chạy test sau999, đã khóa bảy checkpoint train-only rồi đánh
     giá42,43,44,100,2024,777,999 trên cùng đủ1.111 mẫu/model. Mean2D từng
     seed lần lượt7,421295;7,722875;7,721421;8,579556;7,409123;7,520465;
     7,695771 m. Trung bình7,724358 m, sample SD0,401037 m; chưa đạt dưới7 m.
     Không loại seed100 có lỗi cao. Bảy audit độc lập kiểm đủ thứ tự dòng,
     tọa độ, nhãn, metric, ma trận và macro-F1 đều đạt. Kết quả lưu riêng tại
     `early_test_after_seed999_20261022`, không gán thành kết quả mười seed
     refit. Hàng đợi đã dừng trước45; người dùng yêu cầu đánh giá xong và dừng,
     nên không tiếp tục huấn luyện hoặc thử nghiệm khác sau khi báo kết quả.

## Kết quả sàng lọc seed42 của vòng v1 (lịch sử)

- `mask`, seed 42: validation sau calibration **6,167652 m**, k20/T1.
  Baseline cùng seed: **6,032527 m**. Mask đơn lẻ chưa cải thiện.
- `robust_aug`: **6,017944 m**, k20/T0,5; 56 epoch, best epoch 36.
- `relative`: **5,905599 m**, k20/T0,5; 33 epoch, best epoch 13.
- `relative_only`: **5,896253 m**, k20/T1; 33 epoch, best epoch 13.
- `aligned_topk`: **5,866303 m**, k1000/T0,5; 49 epoch, best epoch 29.
- `auxiliary`: **5,854196 m**, k20/T0,5; 33 epoch, best epoch 13.
- `learned_pool`: **6,344939 m**, k20/T0,5; 56 epoch, best epoch 36;
  chưa tốt hơn flatten baseline.
- `ema`: **5,979415 m**, k3/T0,5; chưa tốt hơn same_point.
- `relative_ema`: **6,185306 m**, k3/T0,5; không cải thiện so với relative.
- `regularized`: **5,904866 m**, k20/T1.
- `sigma_tight`: **6,200502 m**, k20/T1.
- `sigma_wide`: **6,017176 m**, k1000/T0,5; 60 epoch, best epoch 40.
- `fine_tokens`: **5,862838 m**, k20/T1; 56 epoch, best epoch 36;
  1.539.112 tham số, chưa tốt hơn same_point.
- `aligned_aux_relative`: **5,748362 m**, k1000/T0,5; tổ hợp nhiều thay đổi,
  không gán riêng mức giảm cho classifier phụ hoặc relative khi chưa tách.
- `budget100`: **6,032527 m**, k1000/T0,5; đủ100 epoch, best21; weights được
  chọn giống hệt baseline cũ. Tăng ngân sách đơn lẻ chưa cải thiện ở seed42.
- `same_point`: **5,462713 m**, k10/T0,5; 100 epoch, best epoch 87.
  Một mạng, không thêm bộ truy xuất hoặc trộn đầu ra mô hình. Đối chứng
  budget100 đã hoàn tất ở mục46; vẫn cần đa seed và đánh giá test.
- `same_point_relative`: **5,582541 m**, k10/T0,5; 46 epoch, best26; chưa vượt same_point
  ở chỉ số validation toàn bộ; cần xem riêng khả năng tổng quát và lặp seed.
- `same_point_aux_topk`: **5,658655 m**, k10/T0,5; chưa vượt same_point.
- `kan_head`: **5,918121 m**, k10/T0,5; đủ100 epoch, 6.856.384 tham số;
  thêm KAN ở head đơn lẻ chưa tốt hơn augmentation cùng vị trí với head cũ.
- `same_point_kan_head`: **5,505775 m**, k20/T1; đủ100 epoch; vẫn chưa tốt hơn
  same_point theo mean toàn validation. Cả hai là một KAN-Mixer, head KAN nội
  bộ không tạo mô hình dự đoán thứ hai.
- `deeper_same_point`: **5,697390 m**, k20/T0,5; đủ100 epoch, best71;
  1.195.304 tham số. Ba block chưa tốt hơn hai block ở seed42. Trên859 mẫu
  tọa độ chưa có trong train, CPU đánh giá7,467960 m; không cải thiện nhóm này.
- `same_point_cross_phone`: **5,484101 m**, k20/T0,5; đủ100 epoch, best87;
  1.014.568 tham số,1078,193 giây. Nhóm tọa độ mới7,081397 m, chưa vượt
  same_point_relative6,637284 m ở nhóm này. Đây vẫn là thay sampler train,
  không đưa PHONEID vào mạng khi dự đoán.
- `cross_phone_consistency`: **5,727462 m**, k20/T0,5; đủ100 epoch, best36;
  1.014.568 tham số,1122,024 giây. Dùng hai view train trên mỗi mẫu và JS
  weight1, không đổi số optimizer step. Nhóm tọa độ mới7,434849 m; chưa tốt
  hơn các cấu hình augmentation đang dẫn đầu. Không coi hai view train là
  hai model hoặc ensemble khi dự đoán.
- `same_point_consistency`: **5,915659 m**, k10/T0,5; đủ100 epoch, best66;
  1.014.568 tham số,1213,679 giây. Kết hợp mixing0,5 và JS weight1 chưa tốt
  hơn mixing đơn lẻ. CPU mean nhóm tọa độ mới8,000126 m. Hoàn tất24 cấu hình
  seed42; chuyển sang xác nhận budget100, same_point, same_point_cross_phone
  và same_point_relative trên42/43/44 theo kế hoạch v3 đã khóa trước test.
- Tại thời điểm sàng lọc v1, chưa dùng official test để chọn các trial trên.
  Sau khi chốt cấu hình và refit, v1 đã đánh giá26 trường hợp train-only/refit
  theo kế hoạch. Kết quả test chính và trạng thái v2 được ghi ở đầu tài liệu;
  không diễn giải các số validation trong danh sách này thành số test.

Mọi thử nghiệm thất bại hoặc không cải thiện vẫn giữ trong log. Chỉ kết luận
nguyên nhân khi có đối chứng tương ứng; không gán mọi mức giảm cho KAN hoặc
một thay đổi chưa được tách riêng. Sổ này tiếp tục được cập nhật khi có dữ liệu.

</details>
