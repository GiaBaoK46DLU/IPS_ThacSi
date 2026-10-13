# Nhánh LLM — hợp đồng, chưa có triển khai IPS

Kế thừa LLMAlgorithm ở adapter.py. Triển khai prepare(train), execute(validation), predict(inputs),
encode_rssi(inputs), decode_position(response, inputs); get_diagram tùy chọn. Fine-tune hoặc chốt pipeline
chỉ suy luận đều trả AlgorithmExecution, không bắt buộc optimizer/epoch.

Chỉ fit vocabulary/retrieval/ví dụ prompt học từ train; không dùng truth vị trí của fingerprint đang dự đoán.
IPSPredictions giữ source_id/row_indices và quy ước tọa độ/cặp nhãn chung. Chuỗi văn bản hay embedding chưa
phải dự đoán IPS nếu thiếu decoder vị trí.

Nhóm cần chốt model, RSSI/AP/missing serialization, few-shot/retrieval, fine-tune/inference, decoding,
phản hồi không hợp lệ và model revision/seed/token/runtime/memory. Sau khi có concrete Algorithm thật
và tests, đăng ký qua registry riêng. Adapter này không được đăng ký trong default_factory.
Xem [hợp đồng Algorithm](../base.py) và [kiểu dữ liệu chung](../../experiments/contracts.py).
