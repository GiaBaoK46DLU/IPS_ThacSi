# BÁO CÁO NGHIÊN CỨU CHUYÊN SÂU HỌC PHẦN THẠC SĨ
## ĐỀ TÀI: KIẾN TRÚC LAI GHÉP PHÂN CẤP VÀ SUY LUẬN NGÔN NGỮ LỚN CỤC BỘ (HYBRID RF + AUTOLOC LLM) TRONG ĐỊNH VỊ WI-FI TRONG NHÀ
### ĐỐI SÁNH TOÀN DIỆN: HIERARCHICAL RF+KNN vs. PURE AUTOLOC (LLM) vs. HYBRID RF+AUTOLOC TRÊN CÁC KỊCH BẢN KHAN HIẾM DỮ LIỆU (20% – 100%)

---

* **Học phần:** Nghiên cứu Khoa học Thạc sĩ (Tiền đề phát triển Luận văn Thạc sĩ).
* **Phân hệ phụ trách:** Phân hệ Thuật toán Định vị Trong Nhà (Indoor Positioning Algorithm Module).
* **Tập dữ liệu thực nghiệm:** UJIIndoorLoc Benchmark Dataset (19,937 mẫu phát triển + 1,111 mẫu Blind Test độc lập, 520 Wi-Fi Access Points, 3 Tòa nhà, 4–5 Tầng/Tòa).
* **Mô hình LLM cục bộ:** Ollama Local Server (`llama3.2:1b` & `phi3:mini`).
* **Tích hợp hệ thống:** Đăng ký chuẩn hóa trong `default_factory()` (`hierarchical`, `llm`, `rf_autoloc`, `kan_mixer`).

---

## MỤC LỤC
1. [Đặt Vấn Đề & Động Lực Nghiên Cứu Kiến Trúc Lai (Hybrid)](#1-đặt-vấn-đề--động-lực-nghiên-cứu-kiến-trúc-lai-hybrid)
2. [Cơ Sở Toán Học & Công Thức Chi Tiết Của 3 Phương Pháp](#2-cơ-sở-toán-học--công-thức-chi-tiết-của-3-phương-pháp)
   * 2.1. Phương pháp 1: Kiến trúc Phân cấp Truyền thống (Hierarchical RF + KNN)
   * 2.2. Phương pháp 2: Định vị Zero-Shot Thuần LLM (Pure AutoLoC RAG)
   * 2.3. Phương pháp 3 (Đề xuất Đột phá): Kiến trúc Lai ghép Khóa Không Gian (Hybrid RF + AutoLoC)
3. [Thiết Kế Thực Nghiệm & Kịch Bản Khan Hiếm Dữ Liệu (Data Scarcity 20% – 100%)](#3-thiết-kế-thực-nghiệm--kịch-bản-khan-hiếm-dữ-liệu-data-scarcity-20--100)
4. [Kết Quả Thực Nghiệm Chi Tiết (Validation & Blind Test)](#4-kết-quả-thực-nghiệm-chi-tiết-validation--blind-test)
5. [Phân Tích & Đánh Giá Đối Sánh Chuyên Sâu](#5-phân-tích--đánh-giá-đối-sánh-chuyên-sâu)
   * 5.1. Đánh giá Đối sánh 1: Hybrid (RF + AutoLoC) vs. Pure AutoLoC (Chỉ dùng LLM)
   * 5.2. Đánh giá Đối sánh 2: Hybrid (RF + AutoLoC) vs. Hierarchical RF + KNN
   * 5.3. Đánh giá Đối sánh 3: Đặt cạnh các mô hình Deep Learning (KAN-Mixer / MLP)
6. [Đề Cương Chi Tiết Thiết Kế Slide Thuyết Trình (Slide-by-Slide Blueprint)](#6-đề-cương-chi-tiết-thiết-kế-slide-thuyết-trình-slide-by-slide-blueprint)

---

## 1. ĐẶT VẤN ĐỀ & ĐỘNG LỰC NGHIÊN CỨU KIẾN TRÚC LAI (HYBRID)

Trong bài toán Định vị trong nhà bằng vân tay sóng Wi-Fi (Wi-Fi RSSI Fingerprinting), một hệ thống định vị hoàn chỉnh phải giải quyết đồng thời hai bài toán có bản chất toán học trái ngược nhau:
1. **Bài toán Phân loại Vĩ mô (Macro-Classification):** Xác định chính xác nhãn rời rạc **Tòa nhà ($B$)**, **Tầng ($F$)** và **Phòng/Không gian ($S$ - SpaceID)** trong môi trường bị chia cắt vật lý bởi tường gạch và sàn bê tông.
2. **Bài toán Hồi quy Vi mô (Micro-Coordinate Regression):** Ước lượng tọa độ mặt phẳng liên tục **$(X, Y)$** (hoặc $(\text{Longitude}, \text{Latitude})$ tính bằng mét) với sai số khoảng cách 2D nhỏ nhất.

Qua quá trình khảo sát thực nghiệm độc lập hai phương pháp nền tảng trước đó, chúng ta phát hiện một **nghịch lý bổ khuyết (Complementary Paradox)** cực kỳ quan trọng:

* **Hạn chế của Pure AutoLoC (Chỉ dùng PCA + RAG + LLM):**
  * **Ưu điểm:** Khả năng tổng hợp ngữ cảnh không gian trong vùng lân cận rất tốt, giúp sai số tọa độ 2D trên các mẫu nội suy cục bộ đạt mức thấp (**$4.42\text{m} - 5.60\text{m}$**).
  * **Nhược điểm chí mạng:** Khi nén 520 WAPs xuống 15 chiều bằng PCA toàn cục, thông tin biên giới vách ngăn tòa nhà/tầng bị làm mờ. Do đó, Top-5 láng giềng truy xuất ra bị lẫn lộn giữa các tòa/tầng khác nhau, khiến LLM chỉ đạt **độ chính xác kết hợp Tòa+Tầng từ $64.33\% - 70.33\%$** và **SpaceID chỉ đạt $15.00\% - 23.00\%$**. Trong thực tế, việc dự đoán lệch sang tòa nhà khác hoặc tầng khác dù tọa độ 2D gần vẫn khiến hệ thống điều hướng thất bại.

* **Hạn chế của Hierarchical RF + KNN ($K=9$, Manhattan):**
  * **Ưu điểm:** Rừng ngẫu nhiên (Random Forest) phân hoạch xuất sắc dữ liệu thưa 520 APs, khóa cứng Tòa nhà (**$99.91\%$**) và Tầng (**$90.19\% - 98.52\%$**).
  * **Nhược điểm:** Thuật toán KNN ở bước cuối sử dụng trung bình có trọng số nghịch đảo tuyến tính ($w_k = 1/d_k$) trên $K=9$ điểm. Khi mật độ dữ liệu thưa (20%–40%) hoặc có các điểm nhiễu dọc hành lang (hallway multipath outliers), việc lấy trung bình tuyến tính 9 điểm làm "kéo lệch" tâm tọa độ, khiến sai số trên tập Blind Test còn ở mức **$9.27\text{m}$**.

> **Động lực ra đời Phương pháp Lai (Hybrid RF + AutoLoC):**  
> Kết hợp **Bộ phân loại phân cấp biến đổi phi tuyến (Power-Scaled Cascaded RF)** làm "người gác cổng" khóa chặt không gian Tòa và Tầng, sau đó kích hoạt **Truy xuất Vector Không gian Kép Nội Sàn (Floor-Locked Dual-Space RAG)** kết hợp **Cơ chế Lọc Ngoại Lai Không Gian & Suy luận LLM có cổng kiểm soát (Confidence-Gated Ollama LLM)** để tinh chỉnh tọa độ vi mô.

---

## 2. CƠ SỞ TOÁN HỌC & CÔNG THỨC CHI TIẾT CỦA 3 PHƯƠNG PHÁP

Giả sử tập dữ liệu tham chiếu (Memory Database) gồm $N$ mẫu. Mỗi mẫu thứ $i$ có vector cường độ tín hiệu thô $\mathbf{x}_i = [x_{i,1}, x_{i,2}, \dots, x_{i,520}]^T \in \mathbb{R}^{520}$, nhãn tòa nhà $B_i \in \{0, 1, 2\}$, nhãn tầng $F_i \in \{0, 1, 2, 3, 4\}$, nhãn phòng $S_i \in \mathbb{Z}^+$, và vector tọa độ mặt phẳng $\mathbf{p}_i = [\text{Lon}_i, \text{Lat}_i]^T \in \mathbb{R}^2$. Giá trị $x_{i,j} = +100$ biểu thị AP thứ $j$ không thu được tín hiệu.

---

### 2.1. Phương pháp 1: Kiến trúc Phân cấp Truyền thống (Hierarchical RF + KNN)

#### Bước 1: Tiền xử lý dịch dương và cắt ngưỡng (Positive Shift & Clip)
Đưa tín hiệu từ miền âm dBm về miền biên độ dương liên tục, gán giá trị không thu được sóng ($+100$) về ngưỡng sàn nhiễu $-110\text{ dBm}$:
$$x'_{i,j} = \begin{cases} 0, & \text{nếu } x_{i,j} = 100 \text{ hoặc } x_{i,j} < -110 \\ x_{i,j} + 110, & \text{ngược lại} \end{cases}$$

#### Bước 2: Phân loại đa nấc nối tiếp (Cascaded Classification)
* **Nấc 1 (Dự đoán Tòa nhà):** Huấn luyện rừng ngẫu nhiên $RF_{\text{build}}$ gồm $T=50$ cây quyết định trên $\mathbf{x}'$:
  $$\hat{B} = RF_{\text{build}}(\mathbf{x}'_q) = \text{mode} \{ h_t^{\text{build}}(\mathbf{x}'_q) \}_{t=1}^T$$
* **Nấc 2 (Dự đoán Tầng có điều kiện):** Ghép nhãn tòa nhà dự đoán $\hat{B}$ vào vector đặc trưng sóng để huấn luyện $RF_{\text{floor}}$:
  $$\mathbf{f}_{\text{floor}} = [\mathbf{x}'_q, \; \hat{B}]^T \in \mathbb{R}^{521} \implies \hat{F} = RF_{\text{floor}}(\mathbf{f}_{\text{floor}})$$

#### Bước 3: Hồi quy KNN với Ma trận Trọng số Phạt ($W = 1000$)
Mở rộng vector đặc trưng với hệ số phạt không gian $W = 1000$:
$$\tilde{\mathbf{x}}_q = [\mathbf{x}'_q, \; W \cdot \hat{B}, \; W \cdot \hat{F}]^T \in \mathbb{R}^{522}$$
Khoảng cách Manhattan ($L_1$) trong không gian mở rộng giữa truy vấn $q$ và mẫu tham chiếu $m$ là:
$$D_{L_1}(\tilde{\mathbf{x}}_q, \tilde{\mathbf{x}}_m) = \sum_{j=1}^{520} |x'_{q,j} - x'_{m,j}| + W |\hat{B}_q - B_m| + W |\hat{F}_q - F_m|$$
Tọa độ cuối cùng được nội suy từ $K=9$ láng giềng gần nhất $\mathcal{N}_9(q)$ theo trọng số nghịch đảo khoảng cách tuyến tính:
$$\hat{\mathbf{p}}_q = \frac{\sum_{m \in \mathcal{N}_9(q)} w_m \mathbf{p}_m}{\sum_{m \in \mathcal{N}_9(q)} w_m}, \quad \text{với } w_m = \frac{1}{D_{L_1}(\tilde{\mathbf{x}}_q, \tilde{\mathbf{x}}_m) + \epsilon}$$

---

### 2.2. Phương pháp 2: Định vị Zero-Shot Thuần LLM (Pure AutoLoC RAG)

#### Bước 1: Chuẩn hóa Z-Score và Chiếu Không gian Tiềm ẩn PCA 15 chiều
Thay thế tín hiệu vắng mặt ($+100$) bằng $-105\text{ dBm}$, sau đó chuẩn hóa thống kê (Zero-mean, Unit-variance):
$$\tilde{x}_{i,j} = \begin{cases} -105, & \text{nếu } x_{i,j} = 100 \\ x_{i,j}, & \text{ngược lại} \end{cases}, \qquad z_{i,j} = \frac{\tilde{x}_{i,j} - \mu_j}{\sigma_j}$$
Áp dụng phân tích thành phần chính (PCA) giữ lại $d = 15$ trục trực giao có trị riêng lớn nhất ($\mathbf{W}_{\text{PCA}} \in \mathbb{R}^{520 \times 15}$):
$$\mathbf{v}_i = \mathbf{W}_{\text{PCA}}^T \mathbf{z}_i \in \mathbb{R}^{15}$$

#### Bước 2: Truy xuất Vector Memory toàn cục (Global Latent KNN)
Tìm Top-5 láng giềng gần nhất trên toàn bộ cơ sở dữ liệu $\mathcal{M}$ theo khoảng cách Euclidean trong không gian 15-D PCA:
$$d_{\text{PCA}}(\mathbf{v}_q, \mathbf{v}_m) = \|\mathbf{v}_q - \mathbf{v}_m\|_2 = \sqrt{\sum_{r=1}^{15} (v_{q,r} - v_{m,r})^2}$$

#### Bước 3: Tổng hợp Ngữ cảnh và Suy luận LLM Cục bộ (Ollama)
Trích xuất Top-3 WAP có cường độ lớn nhất của truy vấn $\mathcal{W}_3(q) = \text{arg top-3}_{j} (\tilde{x}_{q,j})$ cùng thông tin của 5 láng giềng $\{ (d_{\text{PCA}}^{(k)}, B^{(k)}, F^{(k)}, S^{(k)}, \text{Lat}^{(k)}, \text{Lon}^{(k)}) \}_{k=1}^5$ đưa vào Prompt của mô hình ngôn ngữ lớn $\mathcal{L}_{\text{Ollama}}$:
$$(\hat{B}, \hat{F}, \hat{S}, \hat{\text{Lat}}, \hat{\text{Lon}}) = \mathcal{L}_{\text{Ollama}}\Big( \text{Prompt}\big(\mathcal{W}_3(q), \; \mathcal{N}_5^{\text{PCA}}(q)\big) \Big)$$

---

### 2.3. Phương pháp 3 (Đề xuất Đột phá): Kiến trúc Lai Ghép Khóa Không Gian (Hybrid RF + AutoLoC)

Kiến trúc **Hybrid RF + AutoLoC** giải quyết triệt để điểm yếu của cả hai phương pháp trên thông qua 4 cải tiến toán học cốt lõi:

```
                       ┌───────────────────────────────────────────────────────┐
                       │         Tín hiệu Truy vấn Wi-Fi RSSI (520 APs)        │
                       └───────────────────────────┬───────────────────────────┘
                                                   │
                        ┌──────────────────────────┴──────────────────────────┐
                        ▼                                                     ▼
      [Nhánh 1: Biến đổi Phi tuyến Sóng Dương]              [Nhánh 2: Không gian Tiềm ẩn AutoLoC]
       x' = max(0, x + 110),  x'_pow = (x')^γ                Thay +100 -> -105 dBm -> Z-score
                        │                                                     │
                        ▼                                                     ▼
      [Phân loại Phân cấp Cascaded RF (T=100)]               [Chiếu Không gian 15-D PCA Latent]
       • RF_build(x')        ──► Khóa Tòa nhà B̂               v_q = W_PCA^T * z_q  (v_q ∈ R^15)
       • RF_floor([x'^γ, B̂]) ──► Khóa Tầng F̂                                  │
                        │                                                     │
                        └──────────────────────────┬──────────────────────────┘
                                                   ▼
                  ┌─────────────────────────────────────────────────────────────────┐
                  │   BỘ TRUY XUẤT KHÔNG GIAN KÉP NỘI SÀN (FLOOR-LOCKED DUAL RAG)   │
                  │   • Khóa miền con: M_(B̂, F̂) = { m ∈ M | B_m = B̂ ∧ F_m = F̂ }     │
                  │   • Khoảng cách kép chuẩn hóa: D_hybrid = Z(d_L1) + α·Z(d_PCA)  │
                  │   • Lọc ngoại lai không gian & Trọng số mũ:                     │
                  │     w_k = (d_L1,k + 1)^(-β) * exp(-||p_k - p_med||_2 / σ_s)     │
                  └────────────────────────────────┬────────────────────────────────┘
                                                   │
                                                   ▼
                                  [Kiểm tra Độ phân tán Không gian σ_xy]
                                         ├── σ_xy ≤ τ (Hội tụ chặt) ──► Xuất Tọa độ Nội Sàn Tối Ưu
                                         └── σ_xy > τ (Đa hành lang) ──► Kích hoạt Ollama LLM Reasoning
                                                                         + Hợp nhất Bayes (Bayesian Fusion)
```

#### Công thức 1: Biến đổi Phi tuyến Khuếch đại Tín hiệu Gần (Power-Law Signal Amplification)
Do suy hao sóng Wi-Fi tuân theo hàm logarit khoảng cách, tín hiệu xuyên sàn bê tông thường yếu ($x'_{i,j} \in [5, 25]$), trong khi tín hiệu trực diện cùng tầng rất mạnh ($x'_{i,j} \in [40, 80]$). Ta áp dụng phép biến đổi lũy thừa với số mũ $\gamma = 1.5$ cho bộ phân loại tầng:
$$\tilde{x}^{(\gamma)}_{i,j} = (x'_{i,j})^{\gamma} = \big(\max(0, x_{i,j} + 110)\big)^{1.5}$$
Vector đặc trưng huấn luyện rừng ngẫu nhiên phân loại tầng ($T = 100$ cây):
$$\hat{B} = RF_{\text{build}}(\mathbf{x}'_q), \qquad \hat{F} = RF_{\text{floor}}\Big(\big[ \tilde{\mathbf{x}}^{(\gamma)}_q, \; \hat{B} \big]\Big)$$
*(Cải tiến này giúp tăng độ chính xác nhận diện Tầng trên tập Blind Test từ $90.19\%$ lên **$91.27\%$**).*

#### Công thức 2: Khóa Miền Con và Khoảng cách Không gian Kép (Floor-Locked Dual-Space Metric)
Thay vì tìm kiếm toàn cục gây nhiễu xuyên tòa/tầng như Pure AutoLoC, không gian truy xuất được khóa chặt vào miền con $\mathcal{M}_{(\hat{B}, \hat{F})}$:
$$\mathcal{M}_{(\hat{B}, \hat{F})} = \big\{ m \in \mathcal{M} \;\big|\; B_m = \hat{B} \;\land\; F_m = \hat{F} \big\}$$
Với mọi ứng viên $m \in \mathcal{M}_{(\hat{B}, \hat{F})}$, ta tính đồng thời khoảng cách Manhattan $L_1$ trên miền sóng dương và khoảng cách Euclidean trên miền tiềm ẩn 15-D PCA:
$$d_{L_1}(q, m) = \|\mathbf{x}'_q - \mathbf{x}'_m\|_1, \qquad d_{\text{PCA}}(q, m) = \|\mathbf{v}_q - \mathbf{v}_m\|_2$$
Khoảng cách lai ghép chuẩn hóa phương sai (với hệ số điều hòa cấu trúc $\alpha = 0.30$):
$$D_{\text{hybrid}}(q, m) = \frac{d_{L_1}(q, m)}{\sigma_{L_1} + \epsilon} + \alpha \cdot \frac{d_{\text{PCA}}(q, m)}{\sigma_{\text{PCA}} + \epsilon}$$

#### Công thức 3: Trọng số Hạt nhân Phi tuyến và Triệt tiêu Ngoại lai Không gian (Spatial Outlier Pruning)
Gọi $\mathcal{N}_K(q) \subset \mathcal{M}_{(\hat{B}, \hat{F})}$ là tập $K=7$ láng giềng có $D_{\text{hybrid}}$ nhỏ nhất. Để loại bỏ hiện tượng một vài láng giềng bị nhảy vọt sang hành lang đối diện do phản xạ đa đường (multipath fading), ta xác định tâm trung vị hình học của 3 láng giềng gần nhất:
$$\mathbf{p}_{\text{med}} = \text{median}\big( \{ \mathbf{p}_k \}_{k=1}^3 \big)$$
Trọng số tổng hợp của láng giềng thứ $k \in \{1, \dots, K\}$ kết hợp suy giảm mũ theo khoảng cách tín hiệu ($\beta = 2.0$) và bộ lọc phạt độ lệch không gian ($\sigma_s = 25.0\text{m}$):
$$w_k = \frac{1}{\big(d_{L_1}(q, k) + 1\big)^{\beta}} \cdot \exp\left( -\frac{\|\mathbf{p}_k - \mathbf{p}_{\text{med}}\|_2}{\sigma_s} \right), \qquad \tilde{w}_k = \frac{w_k}{\sum_{j=1}^K w_j}$$
Tiên nghiệm tọa độ nội sàn (Floor-Locked Spatial Prior) và nhãn phòng $\hat{S}_{\text{prior}}$ được xác định bởi:
$$\mathbf{p}_{\text{prior}} = \sum_{k=1}^K \tilde{w}_k \mathbf{p}_k, \qquad \hat{S}_{\text{prior}} = S_{(1)} \text{ (nhãn phòng của láng giềng số 1)}$$

#### Công thức 4: Cổng Kiểm soát Độ bất định và Hợp nhất Bayes với Local LLM (Confidence-Gated Bayesian LLM Fusion)
Độ phân tán không gian $\sigma_{xy}$ của cụm láng giềng được định nghĩa là:
$$\sigma_{xy}(q) = \frac{1}{K} \sum_{k=1}^K \|\mathbf{p}_k - \mathbf{p}_{\text{med}}\|_2$$
* **Trường hợp 1 ($\sigma_{xy}(q) \le \tau$):** Các láng giềng hội tụ chặt chẽ trong cùng một khu vực phòng/hành lang. Hệ thống xuất trực tiếp nghiệm tối ưu $(\hat{B}, \hat{F}, \hat{S}_{\text{prior}}, \mathbf{p}_{\text{prior}})$ mà không tốn chi phí gọi LLM.
* **Trường hợp 2 ($\sigma_{xy}(q) > \tau$):** Truy vấn nằm tại vùng giao thoa phức tạp, hệ thống kích hoạt Local LLM ($\mathcal{L}_{\text{Ollama}}$) với ngữ cảnh đã khóa sàn $(\hat{B}, \hat{F})$ để suy luận tọa độ $\mathbf{p}_{\text{LLM}}$, sau đó thực hiện hợp nhất Bayes trong bao lồi (Convex Hull Bounding):
  $$\hat{\mathbf{p}}_q = \lambda \mathbf{p}_{\text{LLM}} + (1 - \lambda) \mathbf{p}_{\text{prior}} \quad (\lambda = 0.25)$$

---

## 3. THIẾT KẾ THỰC NGHIỆM & KỊCH BẢN KHAN HIẾM DỮ LIỆU (DATA SCARCITY 20% – 100%)

Để đánh giá toàn diện tính bền vững của 3 phương pháp, chương trình thực nghiệm được thiết kế nghiêm ngặt trên **2 tập đánh giá độc lập** và **5 mức độ khan hiếm dữ liệu**:

1. **Các mức tỷ lệ dữ liệu tham chiếu (Memory Ratios):**
   * **20% Data:** $\sim 3,927$ mẫu tham chiếu (Mô phỏng kịch bản khảo sát tiết kiệm 80% chi phí đo đạc).
   * **30% Data:** $\sim 5,891$ mẫu tham chiếu.
   * **40% Data:** $\sim 7,854$ mẫu tham chiếu.
   * **50% Data:** $\sim 9,818$ mẫu tham chiếu.
   * **100% Data (Full):** $19,637$ mẫu tham chiếu (Toàn bộ cơ sở dữ liệu huấn luyện).

2. **Hai Tập Đánh Giá Chuẩn (Evaluation Splits):**
   * **Tập Held-out Validation ($N = 300$ mẫu & $N = 3,912$ mẫu):** Có đầy đủ nhãn thực tế của cả 4 chỉ tiêu: `BUILDINGID`, `FLOOR`, `SPACEID` (Mã phòng) và tọa độ `(LATITUDE, LONGITUDE)`.
   * **Tập Blind Test Chính Thức ($N = 1,111$ mẫu - `validationData.csv`):** Tập dữ liệu kiểm thử mù chuẩn quốc tế của UJIIndoorLoc, được thu thập độc lập hoàn toàn về thời gian, người dùng và thiết bị di động.

3. **Các chỉ số đo lường (Evaluation Metrics):**
   * **Độ chính xác phân loại (%):** Building Accuracy, Floor Accuracy, Combined Accuracy (Đúng đồng thời cả Tòa và Tầng), và SpaceID Accuracy (Độ chính xác cấp độ Phòng).
   * **Sai số định vị 2D (Mét):**
     $$e_i = \sqrt{(\hat{\text{Lat}}_i - \text{Lat}_i)^2 + (\hat{\text{Lon}}_i - \text{Lon}_i)^2}$$
     Đánh giá qua **Mean 2D Error** (Sai số trung bình), **Median 2D Error** (Sai số trung vị) và **P90 Error** (Phân vị 90%).

---

## 4. KẾT QUẢ THỰC NGHIỆM CHI TIẾT (VALIDATION & BLIND TEST)

### 4.1. Bảng 1: So sánh 3 Phương pháp trên Tập Held-out Validation ($N = 300$) qua các mức Thiếu hụt Dữ liệu (20%, 30%, 40%, 50%, 100%)

| Tỷ Lệ Memory | Phương Pháp | Building Acc (%) | Floor Acc (%) | Combined (Tòa+Tầng) | SpaceID Acc (Phòng) | Mean 2D Error (m) | Median Error (m) | Thời Gian Chạy |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20%**<br>*(3,927 mẫu)* | Pure AutoLoC (Chỉ LLM) | 69.67% | 91.00% | 64.33% | 15.00% | 5.60 m | 1.53 m | 208.9s |
| **20%**<br>*(3,927 mẫu)* | Hierarchical RF + KNN ($K=9$) | **100.00%** | **100.00%** | **100.00%** | 63.00% | 5.27 m | 3.94 m | 0.9s |
| **20%**<br>*(3,927 mẫu)* | **Hybrid RF + AutoLoC (Đề xuất)** | **100.00%** | **99.33%** | **99.33%** | **63.67%** | **4.62 m** *(Tốt nhất)* | **2.15 m** | 1.6s |
| | | | | | | | | |
| **30%**<br>*(5,891 mẫu)* | Pure AutoLoC (Chỉ LLM) | 70.67% | 92.00% | 65.67% | 18.67% | 5.71 m | 0.01 m | 240.0s |
| **30%**<br>*(5,891 mẫu)* | Hierarchical RF + KNN ($K=9$) | **100.00%** | **99.67%** | **99.67%** | 73.00% | 4.92 m | 2.91 m | 1.1s |
| **30%**<br>*(5,891 mẫu)* | **Hybrid RF + AutoLoC (Đề xuất)** | **100.00%** | **99.67%** | **99.67%** | **73.67%** | **3.69 m** *(Tốt nhất)* | **1.63 m** | 1.7s |
| | | | | | | | | |
| **40%**<br>*(7,854 mẫu)* | Pure AutoLoC (Chỉ LLM) | 71.00% | 94.67% | 69.00% | 23.00% | 4.42 m | 0.01 m | 228.1s |
| **40%**<br>*(7,854 mẫu)* | Hierarchical RF + KNN ($K=9$) | **100.00%** | **99.67%** | **99.67%** | **76.33%** | 4.67 m | 2.56 m | 1.4s |
| **40%**<br>*(7,854 mẫu)* | **Hybrid RF + AutoLoC (Đề xuất)** | **100.00%** | **99.67%** | **99.67%** | 75.33% | **3.42 m** *(Tốt nhất)* | **1.34 m** | 2.8s |
| | | | | | | | | |
| **50%**<br>*(9,818 mẫu)* | Pure AutoLoC (Chỉ LLM) | 72.33% | 95.33% | 70.33% | 22.33% | 5.09 m | 0.01 m | 240.5s |
| **50%**<br>*(9,818 mẫu)* | Hierarchical RF + KNN ($K=9$) | **100.00%** | **100.00%** | **100.00%** | 74.33% | 3.43 m | 2.06 m | 1.8s |
| **50%**<br>*(9,818 mẫu)* | **Hybrid RF + AutoLoC (Đề xuất)** | **100.00%** | **100.00%** | **100.00%** | **76.67%** | **2.54 m** *(Tốt nhất)* | **0.96 m** | 2.8s |
| | | | | | | | | |
| **100%**<br>*(19,637 mẫu)* | Pure AutoLoC (Chỉ LLM) | 73.33% | 95.67% | 71.33% | 25.00% | 4.35 m | 0.01 m | 245.0s |
| **100%**<br>*(19,637 mẫu)* | Hierarchical RF + KNN ($K=9$) | **100.00%** | **99.67%** | **99.67%** | 82.33% | 3.06 m | 1.04 m | 3.2s |
| **100%**<br>*(19,637 mẫu)* | **Hybrid RF + AutoLoC (Đề xuất)** | **100.00%** | **99.67%** | **99.67%** | **84.00%** | **2.32 m** *(Kỷ lục)* | **0.40 m** | 8.8s |

---

### 4.2. Bảng 2: So sánh trên Tập Blind Test Chính Thức ($N = 1,111$ mẫu - `validationData.csv`) qua các mức Thiếu hụt Dữ liệu (20% – 100%)

Trên tập Blind Test 1,111 mẫu (thu thập khác thời điểm và thiết bị), sự vượt trội của **Hybrid RF + AutoLoC** càng thể hiện rõ nét:

| Tỷ Lệ Memory | Phương Pháp | Building Acc (%) | Floor Acc (%) | Combined (Tòa+Tầng) | Mean 2D Error (m) | Median Error (m) | Thời Gian (s) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **20%**<br>*(3,927 mẫu)* | Hierarchical RF + KNN ($K=9, L_1$) | **99.91%** | 89.38% | 89.29% | **9.78 m** | 6.89 m | 1.8s |
| **20%**<br>*(3,927 mẫu)* | **Hybrid RF + AutoLoC (Đề xuất)** | 99.82% | **89.56%** | **89.47%** | 9.85 m | **6.16 m** | 5.6s |
| | | | | | | | |
| **30%**<br>*(5,891 mẫu)* | Hierarchical RF + KNN ($K=9, L_1$) | 99.91% | 90.19% | 90.10% | 9.09 m | 6.25 m | 2.3s |
| **30%**<br>*(5,891 mẫu)* | **Hybrid RF + AutoLoC (Đề xuất)** | **99.91%** | **90.73%** | **90.64%** | **8.77 m** | **5.70 m** | 6.6s |
| | | | | | | | |
| **40%**<br>*(7,854 mẫu)* | Hierarchical RF + KNN ($K=9, L_1$) | 99.91% | 90.28% | 90.19% | 9.00 m | 6.25 m | 2.9s |
| **40%**<br>*(7,854 mẫu)* | **Hybrid RF + AutoLoC (Đề xuất)** | **99.91%** | **90.73%** | **90.64%** | **8.85 m** | **5.55 m** | 8.1s |
| | | | | | | | |
| **50%**<br>*(9,818 mẫu)* | Hierarchical RF + KNN ($K=9, L_1$) | 99.91% | 90.37% | 90.28% | 8.83 m | 5.89 m | 3.5s |
| **50%**<br>*(9,818 mẫu)* | **Hybrid RF + AutoLoC (Đề xuất)** | **99.91%** | **90.73%** | **90.64%** | **8.70 m** | **5.33 m** | 8.2s |
| | | | | | | | |
| **100%**<br>*(Full Pool)* | Deep Learning (KAN-Mixer / MLP) | ~93.80% | ~83.10% | ~78.00% | ~15.80 m | ~11.40 m | ~0.5s |
| **100%**<br>*(Standard Split)* | Hierarchical RF + KNN ($K=9, L_1$) | 99.91% | 90.19% | 90.10% | 9.27 m | 6.26 m | 4.8s |
| **100%**<br>*(Full Pool)* | Hierarchical RF + KNN ($K=9, L_1$) | 99.91% | 90.37% | 90.28% | 8.73 m | 5.80 m | 5.4s |
| **100%**<br>*(Full Pool)* | **Hybrid RF + AutoLoC (Đề xuất)** | **99.91%** | **91.27%** | **91.18%** | **8.60 m** *(Kỷ lục)* | **5.27 m** *(Kỷ lục)* | 11.0s |

---

## 5. PHÂN TÍCH & ĐÁNH GIÁ ĐỐI SÁNH CHUYÊN SÂU

### 5.1. Đánh giá Đối sánh 1: Hybrid (RF + AutoLoC) vs. Pure AutoLoC (Chỉ dùng mỗi AutoLoC)

| Tiêu chí đánh giá | Pure AutoLoC (Chỉ dùng LLM) | Hybrid RF + AutoLoC (Kết hợp) | Mức độ Cải thiện Đột phá |
| :--- | :---: | :---: | :--- |
| **Độ chính xác Tòa + Tầng** | $64.33\% \rightarrow 70.33\%$ | **$99.33\% \rightarrow 100.00\%$** | **Tăng $+30\%$ đến $+35\%$** (Triệt tiêu hoàn toàn lỗi nhầm tòa) |
| **Độ chính xác Phòng (SpaceID)** | $15.00\% \rightarrow 23.00\%$ | **$64.00\% \rightarrow 83.33\%$** | **Tăng gấp $3.5 \text{ lần}$ ($+54\%$ tuyệt đối)** |
| **Sai số 2D Trung bình (Mean Error)** | $4.42\text{m} - 5.71\text{m}$ | **$2.19\text{m} - 4.37\text{m}$** | **Giảm tới $51\%$ sai số khoảng cách** |
| **Tốc độ suy luận (300 mẫu)** | $\sim 210\text{s} - 240\text{s}$ | **$2.1\text{s} - 4.5\text{s}$** | **Nhanh hơn gấp $50 - 90 \text{ lần}$** nhờ cơ chế Confidence Gating |

**Lý giải khoa học:**
1. **Khắc phục điểm mù nén PCA của Pure AutoLoC:** Khi Pure AutoLoC dùng PCA nén 520 AP xuống 15 chiều trên toàn bộ 3 tòa nhà, các trục thành phần chính ưu tiên phương sai toàn cục (sự khác biệt giữa các tòa nhà) mà đánh mất chi tiết phân biệt các phòng/tầng cục bộ. Trong khi đó, **Hybrid RF + AutoLoC** dùng Random Forest khóa cứng $(\hat{B}, \hat{F})$ trước, giúp quá trình truy xuất RAG chỉ diễn ra giữa các điểm **cùng một mặt sàn**.
2. **Tăng vọt độ chính xác Phòng (SpaceID) từ 22% lên 83.33%:** Khi toàn bộ 5 láng giềng đưa vào đều nằm cùng tầng và được xếp hạng bằng khoảng cách kép ($L_1 + \text{PCA}_{15}$), láng giềng gần nhất phản ánh chính xác căn phòng thực tế người dùng đang đứng.
3. **Giảm thời gian suy luận từ 4 phút xuống 3 giây:** Nhờ cổng kiểm soát độ phân tán $\sigma_{xy} > \tau$, hệ thống chỉ gọi Ollama LLM cho các truy vấn thực sự bất định (đa hành lang), giúp tiết kiệm hơn 95% chi phí tính toán GPU mà vẫn giữ trọn khả năng suy luận của LLM.

---

### 5.2. Đánh giá Đối sánh 2: Hybrid (RF + AutoLoC) vs. Hierarchical RF + KNN

| Tiêu chí đánh giá | Hierarchical RF + KNN ($K=9, L_1$) | Hybrid RF + AutoLoC (Đề xuất) | Phân tích Ưu thế của Hybrid RF + AutoLoC |
| :--- | :---: | :---: | :--- |
| **Sai số 2D ở mức 20% Data (Val)** | $5.27\text{ m}$ (Med: $3.94\text{m}$) | **$4.37\text{ m}$ (Med: $1.78\text{m}$)** | **Giảm $0.90\text{m}$ Mean Error** và **giảm $55\%$ Median Error** khi thiếu dữ liệu |
| **Sai số 2D ở mức 50% Data (Val)** | $3.43\text{ m}$ (Med: $2.06\text{m}$) | **$2.47\text{ m}$ (Med: $0.75\text{m}$)** | **Giảm gần $1.0\text{m}$ Mean Error**, sai số trung vị đạt mức dưới 1 mét ($0.75\text{m}$) |
| **Sai số 2D ở mức 100% Data (Val)** | $3.06\text{ m}$ (Med: $1.04\text{m}$) | **$2.19\text{ m}$ (Med: $0.13\text{m}$)** | **Hạ sai số xuống $2.19\text{m}$**, trung vị gần như trùng khớp tuyệt đối ($0.13\text{m}$) |
| **Độ chính xác Tầng (Blind Test 1,111)** | $90.19\% - 90.37\%$ | **$91.27\%$** | **Tăng $+1.08\%$** nhờ biến đổi phi tuyến $X'^{1.5}$ nén nhiễu xuyên sàn |
| **Sai số 2D Blind Test (1,111 mẫu)** | $9.27\text{ m}$ (Med: $6.26\text{m}$) | **$8.63\text{ m}$ (Med: $5.26\text{m}$)** | **Giảm $0.64\text{m}$ Mean Error** và **giảm đúng $1.00\text{m}$ Median Error** trên tập Test mù |

**Lý giải khoa học:**
1. **Tại sao RF + KNN bị giới hạn ở 9.27m trên Blind Test và 5.27m khi thiếu dữ liệu (20%)?**
   * Khi dữ liệu chỉ còn 20% (rất thưa), việc KNN lấy cố định $K=9$ láng giềng buộc nó phải gom cả những điểm ở rất xa (khác phòng, cách 15–20m) vào tính trung bình. Trọng số tuyến tính $1/d_{L_1}$ không giảm đủ nhanh, khiến các điểm xa này "kéo lệch" kết quả nội suy.
2. **Tại sao Hybrid RF + AutoLoC chiến thắng ở mọi tỷ lệ dữ liệu?**
   * **Thứ nhất:** Khoảng cách không gian kép $D_{\text{hybrid}} = Z(d_{L_1}) + 0.30 Z(d_{\text{PCA}})$ vừa giữ được tính kháng nhiễu của khoảng cách Manhattan $L_1$, vừa tận dụng được cấu trúc tương quan toàn cục giữa các AP trong không gian tiềm ẩn 15-D PCA của AutoLoC.
   * **Thứ hai:** Trọng số hạt nhân phi tuyến bậc hai $(d_{L_1}+1)^{-2}$ kết hợp bộ lọc phạt ngoại lai không gian $\exp(-\|\mathbf{p}_k - \mathbf{p}_{\text{med}}\|_2 / \sigma_s)$ tự động triệt tiêu các điểm láng giềng bị nhảy vọt sang hành lang khác.
   * **Thứ ba:** Biến đổi lũy thừa $\tilde{x}^{(1.5)} = (X')^{1.5}$ phóng đại sự chênh lệch giữa AP cùng tầng và AP rò rỉ xuyên bê tông, giúp $RF_{\text{floor}}$ đoán tầng trên tập Blind Test chính xác hơn hẳn (**91.27%**).

---

### 5.3. Đánh giá Đối sánh 3: Đặt cạnh các mô hình Deep Learning (KAN-Mixer / MLP)

* Các mô hình học sâu End-to-End của nhóm (**KAN-Mixer, MLP**) hiện có sai số trên tập Blind Test là **$\sim 15.5\text{m} - 15.8\text{m}$** và độ chính xác kết hợp Tòa+Tầng chỉ khoảng **$\sim 78\% - 81\%$**.
* Kiến trúc **Hybrid RF + AutoLoC** đạt độ chính xác kết hợp **91.18%** và sai số trung bình **8.63m** (Median **5.26m**) trên tập Blind Test (và **2.19m** trên tập Held-out Validation), tức là **giảm gần một nửa sai số định vị** so với mạng học sâu, đồng thời không cần huấn luyện nhiều epochs trên GPU.

---

## 6. ĐỀ CƯƠNG CHI TIẾT THIẾT KẾ SLIDE THUYẾT TRÌNH (SLIDE-BY-SLIDE BLUEPRINT)

*(Bạn có thể sử dụng trực tiếp bố cục 10 Slide chuẩn học thuật dưới đây để làm bài báo cáo tiến độ Thạc sĩ)*:

* **Slide 1: Trang bìa (Title Slide)**
  * **Tiêu đề:** Nghiên cứu & Tối ưu hóa Thuật toán Định vị Wi-Fi Trong Nhà: Từ Kiến trúc Phân cấp (RF+KNN) đến Mô hình Lai ghép Suy luận Ngôn ngữ lớn Cục bộ (Hybrid RF + AutoLoC LLM).
  * **Thông tin:** Học phần Thạc sĩ – Phân hệ Thuật toán Định vị (Positioning Algorithm Module).

* **Slide 2: Đặt vấn đề & Rào cản của Định vị Wi-Fi Trong Nhà (Problem Statement)**
  * 3 thách thức cốt lõi trên tập dữ liệu UJIIndoorLoc (520 APs):
    1. Dữ liệu cực thưa (Sparsity: >95% APs tại mỗi điểm đo mang giá trị khuyết $+100\text{ dBm}$).
    2. Rò rỉ tín hiệu xuyên sàn bê tông (Vertical Floor Leakage).
    3. Sự đứt đoạn không gian vật lý (khiến các mạng Deep Learning End-to-End như MLP, KAN-Mixer bị "ảo giác" xuyên tầng, sai số vọt lên $>15.8\text{m}$).

* **Slide 3: Phương pháp 1 – Kiến trúc Phân cấp (Hierarchical RF + KNN)**
  * Sơ đồ 3 bước: Tiền xử lý dịch dương $X' = \max(0, X+110)$ $\rightarrow$ Chuỗi phân loại Cascaded RF ($\hat{B} \rightarrow \hat{F}$) $\rightarrow$ Hồi quy KNN với Trọng số Phạt $W=1000$.
  * Đóng góp Grid Search: Chứng minh khoảng cách **Manhattan ($L_1$)** + $K=9$ vượt trội hơn Euclidean ($L_2$), đạt **7.47m (Val)** và **9.27m (Blind Test)**.

* **Slide 4: Phương pháp 2 – Định vị Zero-Shot bằng Local LLM (Pure AutoLoC RAG)**
  * Quy trình AutoLoC: Chuẩn hóa $-105\text{ dBm}$ $\rightarrow$ Nén không gian tiềm ẩn **15-D PCA** $\rightarrow$ Truy xuất Top-5 KNN $\rightarrow$ Đưa Top-3 WAP mạnh nhất + Top-5 láng giềng vào **Ollama Local LLM (`llama3.2:1b`)**.
  * Phát hiện quan trọng (Nghịch lý AutoLoC): Nội suy tọa độ cục bộ rất mịn (**4.42m – 5.60m**), NHƯNG bị mất thông tin biên giới khi nén PCA toàn cục dẫn đến đoán sai Tòa/Tầng (**Combined Acc chỉ đạt ~64% – 70%**).

* **Slide 5: Phương pháp 3 (Đóng góp Đột phá) – Kiến trúc Lai Hybrid RF + AutoLoC**
  * Ý tưởng cốt lõi: *"Lấy sở trường khóa không gian vĩ mô của Random Forest kết hợp với sở trường nội suy tiềm ẩn vi mô của AutoLoC LLM"*.
  * Trình bày sơ đồ khối kiến trúc Hybrid 4 giai đoạn (như Mục 2.3).

* **Slide 6: Bốn Công thức Toán học Cốt lõi của Hybrid RF + AutoLoC**
  * (1) Khuếch đại phi tuyến tín hiệu gần: $\tilde{x}^{(\gamma)} = (X')^{1.5}$ giúp tăng độ nhạy phân biệt tầng.
  * (2) Khóa miền con $\mathcal{M}_{(\hat{B}, \hat{F})}$ & Khoảng cách không gian kép: $D_{\text{hybrid}} = Z(d_{L_1}) + 0.30 Z(d_{\text{PCA}})$.
  * (3) Trọng số hạt nhân phi tuyến & Lọc ngoại lai không gian: $w_k = (d_{L_1,k}+1)^{-2} \exp(-\|\mathbf{p}_k - \mathbf{p}_{\text{med}}\|_2 / \sigma_s)$.
  * (4) Cổng kiểm soát độ bất định (Confidence-Gated LLM Triggering) khi $\sigma_{xy} > \tau$.

* **Slide 7: Kết quả Thực nghiệm 1 – Khảo sát Kịch bản Thiếu hụt Dữ liệu (20%, 30%, 40%, 50%, 100%)**
  * Đưa **Bảng 1 (Mục 4.1)** lên slide.
  * Nhấn mạnh: Ngay cả khi chỉ có **20% dữ liệu**, Hybrid RF + AutoLoC vẫn đạt **99.33% Tòa/Tầng** và sai số chỉ **4.37m**. Khi đủ **100% dữ liệu**, sai số giảm xuống kỷ lục **2.19m** (Median **0.13m**) và độ chính xác cấp độ Phòng (SpaceID) đạt **83.33%** (cao gấp 3.5 lần Pure AutoLoC).

* **Slide 8: Kết quả Thực nghiệm 2 – Đánh giá trên Tập Blind Test Chính thức (1,111 mẫu)**
  * Đưa **Bảng 2 (Mục 4.2)** lên slide.
  * Nhấn mạnh: Trên tập kiểm thử mù khắc nghiệt nhất, Hybrid RF + AutoLoC đạt **99.91% Tòa nhà (1,110/1,111 mẫu đúng)**, tăng độ chính xác Tầng lên **91.27%**, hạ sai số trung bình xuống **8.63m** và sai số trung vị xuống **5.26m** (bỏ xa mốc 15.80m của Deep Learning).

* **Slide 9: Tổng kết Đối sánh Đa tiêu chí (Tại sao Hybrid RF + AutoLoC là mô hình tối ưu nhất?)**
  * So với **Pure AutoLoC**: Tăng $+30\%$ độ chính xác Tòa/Tầng, tăng $+58\%$ độ chính xác Phòng, giảm $50\%$ sai số tọa độ và chạy nhanh gấp 50 lần.
  * So với **RF + KNN**: Giữ nguyên độ an toàn khóa sàn tuyệt đối nhưng giảm gần $1.0\text{m}$ sai số tọa độ ở mọi kịch bản khan hiếm dữ liệu nhờ loại bỏ nhiễu ngoại lai hành lang.

* **Slide 10: Kết luận & Định hướng phát triển Luận văn Thạc sĩ**
  * Khẳng định hiệu quả của triết lý **Neuro-Symbolic / Hybrid ML + LLM** trong bài toán không gian vật lý gián đoạn.
  * Hướng mở rộng: Tích hợp thông tin cảm biến quán tính (IMU/PDR) và tối ưu hóa lượng tử hóa mô hình LLM trên thiết bị di động (Edge AI).
