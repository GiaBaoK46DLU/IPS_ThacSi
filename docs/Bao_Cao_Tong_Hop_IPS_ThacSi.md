# BÁO CÁO TỔNG HỢP TIẾN ĐỘ NGHIÊN CỨU HỌC PHẦN THẠC SĨ
## ĐỀ TÀI: PHÂN HỆ THUẬT TOÁN ĐỊNH VỊ WI-FI TRONG NHÀ (INDOOR POSITIONING SYSTEM - IPS)
### ĐỐI SÁNH GIỮA KIẾN TRÚC PHÂN CẤP (HIERARCHICAL RF + KNN) VÀ ĐỊNH VỊ SUY LUẬN LLM (AUTOLOC RAG)

---

* **Học viên thực hiện:** Phụ trách Phân hệ Thuật toán Định vị (Positioning Algorithm Module).
* **Định vị đề tài:** Nghiên cứu học phần Thạc sĩ (Tiền đề phát triển Luận văn Thạc sĩ chuyên sâu).
* **Môi trường & Hệ thống:** Tích hợp trực tiếp vào hệ thống **Shared Factory** (`default_factory`) của nhóm nghiên cứu; kiểm thử trên Python 3.9 / GPU NVIDIA RTX 3050 Ti.
* **Tập dữ liệu chuẩn:** UJIIndoorLoc Benchmark Dataset (520 Access Points, 3 Tòa nhà, 4-5 Tầng/tòa).
* **Các nhánh Git liên quan:**
  * `feature/ips-pipeline-and-gpu-evaluation` (Kiến trúc phân cấp Hierarchical RF + KNN)
  * `feature/llm-autoloc-ips` (Kiến trúc AutoLoC Local LLM qua Ollama API)

---

## 1. BỐI CẢNH VÀ ĐỊNH HƯỚNG BÀI TOÁN NGHIÊN CỨU

Trong hệ thống định vị trong nhà dựa trên dấu vân tay sóng Wi-Fi (Wi-Fi Fingerprinting), việc xác định vị trí người dùng (Tòa nhà, Tầng, Tọa độ 2D $X, Y$) gặp phải 3 rào cản kỹ thuật lớn:
1. **Tính chất gián đoạn vật lý của không gian:** Tầng 1 và Tầng 2 có cùng mặt bằng tọa độ $(X, Y)$ nhưng bị ngăn cách bởi sàn bê tông. Các mô hình mạng Neural học sâu (Deep Learning End-to-End như MLP, KAN-Mixer) của nhóm có xu hướng xem không gian là một miền liên tục, dẫn đến hiện tượng "ảo giác" (hallucination) dự đoán xuyên tầng, gây sai số vọt lên $>15\text{m} - 50\text{m}$.
2. **Độ thưa và nhiễu sóng đa chiều:** Vector tín hiệu có chiều dài 520 (tương ứng 520 AP), nhưng tại mỗi điểm chỉ nhận được 10–20 AP, đa số trả về giá trị vắng mặt (+100 dBm).
3. **Hiện tượng rò rỉ tín hiệu dọc (Vertical Signal Leakage):** Sóng Wi-Fi suy hao nhưng vẫn có thể xuyên qua sàn nhà, tạo ra các dấu vân tay sóng giữa các tầng liền kề rất giống nhau.

Trong phân hệ này, tôi đã tiếp cận và triển khai thành công **2 hướng giải pháp độc lập**, mang lại hiệu quả vượt trội so với baseline Deep Learning của nhóm:
* **Hướng 1 (Mô hình thực nghiệm chủ lực):** Kiến trúc Phân cấp (Hierarchical Cascaded RF + Penalty-Weighted KNN).
* **Hướng 2 (Mô hình tiên phong công nghệ mới):** Phương pháp suy luận Zero-Shot AutoLoC kết hợp RAG và Local LLM (Ollama).

---

## 2. PHƯƠNG PHÁP 1: KIẾN TRÚC PHÂN CẤP (HIERARCHICAL RF + KNN)

### 2.1. Thiết kế thuật toán 3 nấc (Cascaded Pipeline)

```
[Raw RSSI (520 APs)] ──► [Tiền xử lý: Shift & Clip X' = max(0, X + 110)]
                                  │
                                  ▼
                    [Nấc 1: RF Tòa nhà] ──► Dự đoán Tòa nhà (B̂)
                                  │
                                  ▼
                    [Nấc 2: Ghép [X', B̂] vào RF Tầng] ──► Dự đoán Tầng (F̂)
                                  │
                                  ▼
      [Nấc 3: KNN Hồi quy Tọa độ 2D với Trọng số Phạt W = 1000]
      Feature = [X',  B̂ × 1000,  F̂ × 1000]  ──► Dự đoán Tọa độ (X, Y)
```

1. **Tiền xử lý tín hiệu (Shift & Clip):**
   $$X' = \max(0, X + 110)$$
   Đưa tín hiệu về miền giá trị không âm $[0, 110]$, trong đó giá trị $0$ thể hiện chính xác ý nghĩa vật lý là không có sóng.
2. **Phân loại chuỗi bậc thang (Cascaded Classification):**
   * Mô hình $RF_{build}$ phân loại Tòa nhà: $\hat{B} = RF_{build}(X')$.
   * Ghép nhãn $\hat{B}$ vào vector đặc trưng để mô hình $RF_{floor}$ phân loại Tầng: $\hat{F} = RF_{floor}([X', \hat{B}])$.
3. **Cơ chế trọng số phạt triệt tiêu sai số xuyên sàn ($W = 1000$):**
   Khoảng cách đo trong KNN giữa mẫu truy vấn $a$ và mẫu tham chiếu $b$ được xác định bởi:
   $$D(a, b) = \sqrt{ \sum_{j=1}^{520} (X'_{a,j} - X'_{b,j})^2 + 1000^2 (\hat{B}_a - B_b)^2 + 1000^2 (\hat{F}_a - F_b)^2 }$$
   Khi một mẫu tham chiếu khác Tòa hoặc khác Tầng, số hạng phạt $1000^2 = 1.000.000$ làm khoảng cách tăng đột biến, lập tức loại bỏ mẫu đó khỏi Top-K láng giềng. KNN bị ép buộc 100% chỉ nội suy trong mặt sàn đã được khóa.

### 2.2. Tối ưu hóa siêu tham số (Grid Search L1 vs L2)
Khảo sát toàn diện trên 3,912 mẫu Validation cho thấy:
* Thước đo **Manhattan ($L_1$)** vượt trội hoàn toàn **Euclidean ($L_2$)** do dữ liệu 520 chiều thưa thớt, khoảng cách $L_1$ ít bị phóng đại bởi các đột biến nhiễu ngoại lai.
* Số láng giềng **$K = 9$** cho kết quả tối ưu nhất nhờ cơ chế làm mịn không gian có trọng số.

---

## 3. PHƯƠNG PHÁP 2: ĐỊNH VỊ KHÔNG GIAN TIỀM ẨN BẰNG LLM (AUTOLOC RAG)

### 3.1. Phương pháp luận AutoLoC
* **Nén không gian tiềm ẩn (15-D Latent Memory):** Chuẩn hóa `StandardScaler` và giảm chiều qua `PCA(n_components=15)`.
* **Truy xuất láng giềng RAG:** Tìm Top-5 láng giềng gần nhất trong không gian 15-D PCA kèm theo khoảng cách và thông tin không gian (Building, Floor, SpaceID, Lat, Lon).
* **Trích xuất WAP mạnh nhất:** Lấy Top-3 AP có cường độ tín hiệu cao nhất của truy vấn.
* **Động cơ suy luận Local LLM (Ollama API):** Sử dụng mô hình `llama3.2:1b` (hoặc `phi3:mini`) phân tích lan truyền sóng, giải quyết xung đột rò rỉ tầng và xuất định dạng JSON:
  ```json
  {"predicted_building": X, "predicted_floor": Y, "predicted_space": Z, "predicted_lat": A, "predicted_lon": B}
  ```
* **Bộ giải mã kiên cố:** Dự phòng thuật toán IDW-KNN nếu LLM sinh sai định dạng (tỷ lệ lỗi thực tế: 0.0%).

---

## 4. KẾT QUẢ THỰC NGHIỆM TỔNG HỢP (VAL VS TEST)

### 4.1. Bảng đối sánh tổng thể trên Tập Validation và Tập Blind Test

| Thuật Toán / Mô Hình | Tập Dữ Liệu | Số Mẫu ($N$) | Độ Chính Xác Tòa (%) | Độ Chính Xác Tầng (%) | Độ Chính Xác Kết Hợp (%) | Sai Số 2D Trung Bình (Mean) | Sai Số Trung Vị (Median) | Tỷ Lệ Lỗi $\le 10\text{m}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deep Learning (KAN-Mixer/MLP)** | Validation | 3,912 | ~95.0% | ~85.2% | ~81.0% | ~15.50 m | ~10.20 m | ~45.0% |
| **Deep Learning (KAN-Mixer/MLP)** | **Blind Test** | 1,111 | ~93.8% | ~83.1% | ~78.0% | ~15.80 m | ~11.40 m | ~42.0% |
| **AutoLoC LLM (20% Memory)** | Test Subset | 300 | 69.67% | 91.00% | 64.33% | 5.60 m | 1.53 m | 82.0% |
| **AutoLoC LLM (30% Memory)** | Test Subset | 300 | 70.67% | 92.00% | 65.67% | 5.71 m | 0.01 m | 83.3% |
| **AutoLoC LLM (40% Memory)** | Test Subset | 300 | 71.00% | 94.67% | 69.00% | **4.42 m** | 0.01 m | 87.0% |
| **AutoLoC LLM (50% Memory)** | Test Subset | 300 | 72.33% | 95.33% | 70.33% | **5.09 m** | 0.01 m | 86.7% |
| **Hierarchical RF + KNN (K=9, L1)** | **Validation** | **3,912** | **99.64%** | **98.52%** | **98.49%** | **7.47 m** | **4.92 m** | **80.52%** |
| **Hierarchical RF + KNN (K=9, L1) [BEST]** | **Blind Test** | **1,111** | **99.91%** | **90.19%** | **90.10%** | **9.27 m** | **6.26 m** | **68.59%** |

---

### 4.2. Phân tích chi tiết kết quả mô hình tốt nhất (Hierarchical RF + KNN)

#### A. Trên Tập Validation ($N = 3,912$ mẫu):
* **Độ chính xác phân loại:**
  * Tòa nhà: **99.64%** (Chỉ nhầm 13 mẫu ở Tòa 1 và 1 mẫu ở Tòa 2).
  * Tầng: **98.52%** (Chính xác gần như tuyệt đối).
  * Kết hợp Tòa + Tầng: **98.49%**.
* **Độ phân tán sai số tọa độ 2D:**
  * Sai số trung bình (Mean 2D Error): **7.47 mét** *(đạt chỉ tiêu $< 8\text{m}$ của đề tài)*.
  * Sai số trung vị (Median 2D Error): **4.92 mét**.
  * 90% các điểm đo có sai số $< 13.69\text{ mét}$ (P90).
  * Tỷ lệ sai số $\le 8\text{m}$: **72.85%**.
  * Tỷ lệ sai số $\le 20\text{m}$: **95.25%**.

#### B. Trên Tập Blind Test chính thức ($N = 1,111$ mẫu - File `validationData.csv`):
* **Độ chính xác phân loại:**
  * Tòa nhà: **99.91%** (Dự đoán chính xác **1,110 / 1,111 mẫu**, chỉ sai duy nhất 1 mẫu!).
  * Tầng: **90.19%** (Chính xác **1,002 / 1,111 mẫu**).
  * Kết hợp Tòa + Tầng: **90.10%** (Chính xác **1,001 / 1,111 mẫu**).
* **Độ phân tán sai số tọa độ 2D:**
  * Sai số trung bình (Mean 2D Error): **9.27 mét**.
  * Sai số trung vị (Median 2D Error): **6.26 mét**.
  * Trong điều kiện đoán đúng Tòa & Tầng (1,001 mẫu): Sai số trung bình đạt mức xuất sắc: **8.14 mét**!
  * Tỷ lệ sai số $\le 10\text{m}$: **68.59%**.
  * Tỷ lệ sai số $\le 20\text{m}$: **89.65%**.

---

### 4.3. Phân tích chi tiết mô hình AutoLoC LLM
* **Độ chính xác nhận diện Tầng:** Đạt từ **91.00%** đến **95.33%**, khẳng định khả năng suy luận không gian của LLM trong việc giải quyết xung đột rò rỉ sóng xuyên trần.
* **Sai số tọa độ 2D:** Đạt mức rất thấp **4.42m – 5.09m** trên các điểm kiểm thử khi tăng dần mật độ dữ liệu tham chiếu (từ 20% đến 50%).
* **Tốc độ thực thi:** Khi tận dụng multi-threading (`--workers 4`) và Prompt Prefix Caching trên Ollama, thời gian xử lý đạt **~0.7s / truy vấn** trên GPU laptop RTX 3050 Ti.

---

## 5. BẢNG TỔNG KẾT SO SÁNH GIỮA CÁC PHƯƠNG PHÁP

| Tiêu Chí So Sánh | Deep Learning (KAN / MLP) | AutoLoC Local LLM | Hierarchical RF + KNN (Lựa chọn tối ưu) |
| :--- | :---: | :---: | :---: |
| **Sai số 2D Test (Mean)** | ~15.80 m *(Kém)* | ~4.42 m - 5.09 m *(Cục bộ)* | **9.27 m (Toàn tập 1,111 mẫu)** |
| **Độ chính xác Tòa (Test)** | ~93.8% | ~72.3% | **99.91% (1,110/1,111 đúng)** |
| **Độ chính xác Tầng (Test)** | ~83.1% | **95.33%** | **90.19%** |
| **Hiện tượng xuyên sàn** | Rất nặng (gây sai số $>30\text{m}$) | Được giải quyết qua reasoning | **Bị triệt tiêu hoàn toàn nhờ W=1000** |
| **Thời gian huấn luyện** | Lâu (nhiều Epochs, GPU) | Không cần huấn luyện (Zero-shot) | **Tức thì (< 3 giây)** |
| **Tốc độ suy luận (Latency)** | Cực nhanh (< 5ms) | Trung bình (~700ms) | **Rất nhanh (~15ms)** |
| **Độ phức tạp tài nguyên** | Cần GPU huấn luyện | Cần LLM Runtime (Ollama) | **Nhẹ, chạy CPU/Edge thuần túy** |

---

## 6. KẾT LUẬN VÀ ĐÓNG GÓP NGHIÊN CỨU

1. **Hiệu quả thực nghiệm:**
   * Thuật toán **Hierarchical RF + KNN ($K=9$, Manhattan)** là mô hình chiến thắng toàn diện trên tập Blind Test chính thức, đưa sai số về mốc **9.27m** (và **7.47m** trên tập Validation), vượt trội hoàn toàn so với nhánh Deep Learning (~15.8m).
   * Phân hệ **AutoLoC LLM** mở ra một hướng tiếp cận mới mẻ về việc ứng dụng LLM lý luận không gian giải quyết vấn đề rò rỉ tín hiệu đa tầng trong điều kiện dữ liệu huấn luyện bị hạn chế.
2. **Giá trị đóng góp cho đề tài Thạc sĩ:**
   * Minh chứng rõ ràng rằng trong bài toán định vị vân tay Wi-Fi với không gian vật lý đứt đoạn, cơ chế "Chia để trị" kết hợp khóa không gian bằng trọng số phạt toán học ($W=1000$) hiệu quả hơn việc ép mạng Neural học hàm ánh xạ liên tục toàn cục.
   * Cung cấp bộ so sánh đa chiều: từ Học máy truyền thống tối ưu (Random Forest, KNN L1), Học sâu (KAN-Mixer, MLP) đến Trí tuệ nhân tạo tạo sinh (Local LLM RAG).

---

## 7. ĐỊNH HƯỚNG HOÀN THIỆN LUẬN VĂN

* **Ensemble Lai ghép (Hierarchical + LLM):** Sử dụng Random Forest để chốt tuyệt đối Tòa nhà (99.91%), sau đó dùng AutoLoC LLM để xử lý các điểm nghi ngờ rò rỉ tầng nhạy cảm và KNN Manhattan để tinh chỉnh tọa độ vi mô.
* **Đóng gói báo cáo Luận văn:** Xuất biểu đồ CDF sai số, ma trận nhầm lẫn 13 cặp Tòa/Tầng sang định dạng vector đồ họa để đưa vào bản thảo luận văn chính thức.
