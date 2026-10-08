# Nhánh HUIM — hợp đồng, chưa có triển khai IPS

Kế thừa HUIMAlgorithm ở adapter.py. Triển khai prepare(train), execute(validation), predict(inputs),
encode_transactions(inputs), itemsets_to_position(inputs); get_diagram tùy chọn. execute được khai phá
itemset, không cần giả lập epoch/optimizer.

Fit mức RSSI/item vocabulary/utility/index trên train. Validation chỉ chọn quy tắc/ngưỡng; test dự đoán
sau khi chốt. Không coi itemset là IPSPredictions nếu chưa có bộ chuyển thành vị trí. Đầu ra giữ
source_id/row_indices và hệ tọa độ/cặp nhãn chung.

Nhóm cần chốt binning RSSI/missing, utility, thuật toán/ngưỡng khai phá, khớp pattern, suy ra tòa/tầng/xy,
không khớp/nhiều khớp và thời gian/bộ nhớ mine/predict. Có concrete Algorithm thật và tests rồi mới đăng ký.
Adapter này không được đăng ký trong default_factory. Xem [kiến trúc chung](../../../../docs/architecture/shared_factory.md).
