# Báo Cáo Tiến Độ Nghiên Cứu Thạc Sĩ: Thuật Toán Định Vị Wi-Fi Trong Nhà Bằng Kiến Trúc Phân Cấp (Hierarchical RF + KNN)

**Học phần / Giai đoạn:** Báo cáo tiến độ nghiên cứu trong quá trình học Thạc sĩ (Tiền đề cho Luận văn tương lai).
**Vai trò nghiên cứu:** Chịu trách nhiệm phát triển Phân hệ Thuật toán Định vị (Positioning Algorithm Module).
**Hệ thống tích hợp:** Nền tảng Shared Factory chung của nhóm nghiên cứu.

---

## 1. Đặt Vấn Đề Và Định Hướng Nghiên Cứu Chuyên Biệt

Định vị trong nhà (Indoor Positioning System - IPS) dựa trên tín hiệu Wi-Fi (Wi-Fi Fingerprinting) là một thách thức lớn. Tín hiệu Wi-Fi (Received Signal Strength Indicator - RSSI) thu được tại thiết bị người dùng thường bị nhiễu do vật cản, hiện tượng đa đường (multipath) và sự biến động theo thời gian.

Trong dự án nghiên cứu chung của nhóm, bài toán được thiết kế theo một hệ thống lõi chung (Shared Factory), nơi dữ liệu đầu vào và các luồng đánh giá được tiêu chuẩn hóa. Các thành viên khác trong nhóm đang tiếp cận bài toán bằng các mô hình Học sâu (Deep Learning) như MLP, KAN-Mixer. Tuy nhiên, việc áp dụng Học sâu theo hướng "End-to-End" (nhận đầu vào là tín hiệu và nội suy trực tiếp ra tọa độ 3D: Tòa, Tầng, X, Y) bộc lộ một điểm yếu chí mạng: **Sự thiếu liên tục của không gian vật lý**. Tầng 1 và Tầng 2 có thể có cùng tọa độ $(X, Y)$ nhưng môi trường sóng hoàn toàn khác biệt; một sai số nội suy nhỏ của Deep Learning có thể khiến dự đoán bị lệch tầng, dẫn đến sai số thực tế lên tới hàng chục mét.

Để giải quyết triệt để vấn đề này và đóng góp một mảng miếng riêng biệt vào hệ thống của nhóm, tôi đã đề xuất và trực tiếp phát triển một hướng tiếp cận hoàn toàn khác: **Kiến trúc Phân cấp (Hierarchical Pipeline) kết hợp sức mạnh của Rừng ngẫu nhiên (Random Forest - RF) và K-Láng giềng gần nhất (K-Nearest Neighbors - KNN).**

---

## 2. Phân Tích Lựa Chọn Thuật Toán (Tại Sao Là RF và KNN?)

Quyết định không sử dụng Deep Learning mà quay về các thuật toán Học máy thống kê (Machine Learning) truyền thống được dựa trên sự phân tích bản chất của dữ liệu UJIIndoorLoc:

### 2.1. Random Forest (RF) cho Bài toán Phân loại (Tòa nhà, Tầng)
- **Đặc tính dữ liệu thưa (Sparsity):** Vector đặc trưng RSSI có chiều dài 520 (tương ứng 520 Access Points), nhưng tại một vị trí cụ thể, thiết bị chỉ quét được khoảng 10-20 APs. Các AP còn lại trả về giá trị mặc định (+100, tức là không có tín hiệu). 
- **Lý do chọn RF:** Cây quyết định (Decision Tree) đặc biệt xuất sắc trong việc rẽ nhánh các dữ liệu phân loại hoặc dữ liệu thưa thớt bằng các vách ngăn rõ ràng (ví dụ: "Nếu AP số 14 có sóng, chắc chắn người dùng đang ở Tòa 1"). Khi kết hợp thành Rừng ngẫu nhiên (Random Forest) với kỹ thuật Bagging, mô hình sẽ tổng hợp kết quả bầu chọn từ hàng chục cây quyết định khác nhau. Điều này giúp mô hình chống lại nhiễu (noise) do sự biến động sóng, từ đó phân loại chính xác tuyệt đối các nhãn rời rạc là Tòa nhà (Building) và Tầng (Floor).

### 2.2. K-Nearest Neighbors (KNN) cho Bài toán Hồi quy Tọa độ
- **Bản chất của Fingerprinting:** Phương pháp định vị vân tay sóng vốn dĩ là quá trình thu thập một "bản đồ sóng" (Radio Map), sau đó so sánh tín hiệu hiện tại với cơ sở dữ liệu để tìm ra điểm gần giống nhất.
- **Lý do chọn KNN:** KNN chính là sự hiện thực hóa toán học hoàn hảo nhất của triết lý Fingerprinting. Thuật toán này không cố gắng học một hàm toán học phức tạp toàn cục (như mạng Neural), mà nó chỉ đơn giản đo lường khoảng cách tín hiệu hiện thời so với tất cả các điểm lịch sử, và lấy $K$ điểm có tín hiệu giống nhất để nội suy tọa độ. Đây là hướng đi phi tham số (non-parametric), cực kỳ an toàn và ổn định khi được áp dụng trong một không gian hẹp (sau khi đã chốt được Tầng).

---

## 3. Quá Trình Triển Khai Chi Tiết (Pipeline Architecture)

Thuật toán được tôi thiết kế chạy qua 3 bước phân cấp chặt chẽ (từ vĩ mô đến vi mô), tích hợp thành một module duy nhất (`HierarchicalAlgorithm`) đăng ký vào Shared Factory của nhóm.

### Bước 1: Tiền xử lý tín hiệu (Signal Transformation)
Dữ liệu gốc có các giá trị âm (ví dụ -70 dBm) và giá trị +100 cho AP không quét được. Để thuật toán tính khoảng cách (KNN) hoạt động chính xác, tôi đã tiến hành phép dịch (shift) và cắt (clip):
1. **Khử giá trị ngoại lai:** Gán giá trị không có sóng (+100) về mức sóng yếu nhất là $-110$ dBm.
2. **Dịch không gian:** Biến đổi mọi giá trị bằng công thức:
   $$ X_{i,j}' = \max(0, X_{i,j} + 110) $$
   Kết quả là một vector sóng $X'$ với các giá trị dương và liên tục. Số $0$ mang ý nghĩa vật lý là "không bắt được tín hiệu".

### Bước 2: Chuỗi phân loại Tòa nhà và Tầng (Cascaded Classification)
Thay vì gộp chung, tôi tách việc phân loại thành 2 mô hình độc lập nối tiếp nhau:
1. **Mô hình Tòa nhà ($RF_{build}$):** Dùng dữ liệu sóng $X'$ dự đoán Tòa nhà.
   $$ \hat{B} = RF_{build}(X') $$
2. **Mô hình Tầng ($RF_{floor}$):** Kết quả tòa nhà $\hat{B}$ được ghép thêm vào vector sóng ban đầu để làm dữ liệu huấn luyện tìm Tầng. Việc "nhắc bài" Tòa nhà giúp cây quyết định phân hoạch không gian nhanh và chính xác hơn hẳn.
   $$ Feature_{floor} = \big[ X', \hat{B} \big] $$
   $$ \hat{F} = RF_{floor}(Feature_{floor}) $$

### Bước 3: Hồi quy Tọa độ và Kỹ thuật Trọng số Phạt (Penalty Weighting)
Đây là phần cốt lõi và đóng góp lớn nhất giúp giảm thiểu sai số.
Sau khi có $\hat{B}$ và $\hat{F}$, để tìm $(X, Y)$, tôi dùng thuật toán KNN. Tuy nhiên, thuật toán phải được ngăn chặn tuyệt đối việc tìm nhầm "láng giềng" ở tầng khác (dẫn đến nội suy sai).
Tôi thiết kế vector đặc trưng kết hợp và nhân Tòa nhà, Tầng với một hằng số phạt cực lớn **$W = 1000$**:
$$ Feature_{coord} = \big[ X', \quad \hat{B} \times 1000, \quad \hat{F} \times 1000 \big] $$

Khoảng cách $D(a, b)$ giữa điểm tín hiệu hiện tại $a$ và điểm mẫu $b$ trong tập huấn luyện trở thành:
$$ D(a, b) = \sqrt{ \sum_{j=1}^{520} (X'_{a,j} - X'_{b,j})^2 + 1000^2 (\hat{B}_a - B_b)^2 + 1000^2 (\hat{F}_a - F_b)^2 } $$

**Phân tích toán học cơ chế phạt:**
- Nếu mẫu $b$ nằm khác Tòa nhà ($\hat{B}_a \neq B_b$) hoặc khác Tầng ($\hat{F}_a \neq F_b$), phần tử phạt bình phương $1000^2 = 1.000.000$ sẽ làm khoảng cách $D$ đội lên khổng lồ. 
- Ngay lập tức, mẫu $b$ đó bị KNN gạt bỏ khỏi danh sách láng giềng.
- Kết quả: KNN bị ép buộc "khóa" không gian tìm kiếm, chỉ tìm $K$ láng giềng nằm chính xác trong mặt sàn $\hat{F}$ của tòa $\hat{B}$. 
- Cuối cùng, tọa độ được nội suy bằng cơ chế Trọng số nghịch đảo khoảng cách (Inverse Distance Weighting).

---

## 4. Tối Ưu Hóa Siêu Tham Số (Grid Search)

Mô hình mặc định ban đầu (với hàm khoảng cách *Euclidean* và số láng giềng $K=3$) đã mang lại kết quả xuất sắc so với hệ thống mạng Neural của nhóm (sai số hạ từ >15 mét xuống 7,93 mét trên tập Validation). 

Tuy nhiên, nhằm khai thác triệt để sức mạnh của mô hình, tôi đã tiến hành thiết lập một đợt Quét lưới tham số (Grid Search) để tìm ra điểm "ngọt" (Sweet spot). Các tham số được đưa vào thử nghiệm:
- **Số lượng láng giềng ($K$):** Thay đổi từ $\{1, 3, 5, 7, 9\}$.
- **Thước đo khoảng cách (Distance Metric):** Đấu chéo giữa **Euclidean** (đường chim bay - L2) và **Manhattan** (khoảng cách lưới - L1).

**Bảng Kết Quả Thực Nghiệm Trên Tập Validation (n = 3.912 mẫu)**:

| Số láng giềng ($K$) | Khoảng cách đo (Metric) | Độ chính xác Phân loại (Tòa/Tầng) | Sai số Tọa độ 2D (Mean 2D Error) |
| :--- | :--- | :--- | :--- |
| K = 1 | euclidean | 98.49% | 8.38 mét |
| K = 1 | manhattan | 98.49% | 8.30 mét |
| K = 3 | euclidean | 98.49% | 7.93 mét *(Bản gốc)* |
| K = 3 | manhattan | 98.49% | 7.78 mét |
| K = 5 | euclidean | 98.49% | 7.81 mét |
| K = 5 | manhattan | 98.49% | 7.73 mét |
| K = 7 | euclidean | 98.49% | 7.62 mét |
| K = 7 | manhattan | 98.49% | 7.55 mét |
| K = 9 | euclidean | 98.49% | 7.52 mét |
| **K = 9** | **manhattan** | **98.49%** | **7.47 mét (Tối ưu nhất!)** |

### Đánh giá và Phân tích kết quả Grid Search:
1. **Khoảng cách Manhattan chiến thắng tuyệt đối:** Ở mọi mức thiết lập $K$, thước đo khoảng cách `manhattan` (L1) luôn cho sai số trung bình thấp hơn `euclidean` (L2). Nguyên nhân là do bản chất vector RSSI của 520 AP có tính thưa thớt rất cao (đa phần là điểm 0). Khoảng cách L1 ít bị nhạy cảm và không phóng đại các nhiễu sóng (outliers) như L2, giúp so khớp vân tay chuẩn xác hơn.
2. **Hiệu ứng làm mịn không gian:** K=9 mang lại độ chính xác cao nhất (7.47 mét), hạ thêm gần nửa mét so với K=3. Thay vì nội suy quá cực đoan vào 3 điểm gần nhất, việc trung bình khoảng cách có trọng số từ 9 láng giềng giúp tọa độ được "làm mịn", chống lại hiệu ứng nhảy vọt tọa độ khi có 1-2 điểm nhiễu sóng trong tập dữ liệu lịch sử.

---

## 5. Kết Luận Và Định Hướng Luận Văn Tương Lai

Các chỉ số thực nghiệm và quá trình tối ưu hóa đã khẳng định triết lý "Chia để trị" (Hierarchical) bằng thuật toán truyền thống vượt trội hơn rất nhiều so với học sâu (End-to-End) trong bối cảnh dữ liệu định vị Wi-Fi tòa nhà bị đứt đoạn không gian vật lý. Việc "khóa không gian" trước khi dùng KNN đã loại bỏ hoàn toàn tình trạng sai số đâm xuyên bê tông.

**Kế hoạch tiếp theo (Tiến tới hoàn thiện Luận văn):**
- **Đánh giá trên tập Blind Test:** Chạy mô hình vô địch (K=9, Manhattan) trên tập Test Mù của hệ thống để xác thực mốc sai số < 8.5 mét.
- **Kỹ thuật Lọc Nhiễu (Feature Selection):** Tích hợp phương pháp giảm chiều (PCA / Autoencoder) lên đặc trưng sóng trước khi đưa vào KNN nhằm khử các AP nhiễu (ví dụ: các cục Wi-Fi nằm ở quá xa, cường độ sóng thấp), mục tiêu mũi nhọn là tối ưu hóa tốc độ suy luận (Inference Speed) trong hệ thống thực tế.
