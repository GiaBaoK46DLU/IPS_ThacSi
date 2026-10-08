# Ví dụ đầu vào cho dự đoán

`wifi_scans.csv` có năm lần quét và đúng 520 cột `WAP001`…`WAP520`, không có nhãn vị trí. Các dòng lấy từ tập phát triển để kiểm tra định dạng inference; xem `source_rows.json`. Đây không phải test độc lập hoặc dữ liệu đo trực tiếp mới.

```powershell
.\.venv\Scripts\python.exe -m ips predict --checkpoint runs/my_kan_direct/best.pt --csv examples/wifi_scans.csv --output runs/example_predictions_new.csv --device cpu
```

Mỗi dòng output có tòa, tầng, tọa độ đã giải chuẩn hóa và xác suất softmax lớn nhất. Xác suất đó chưa được hiệu chỉnh để bảo đảm độ tin cậy. `row_index` ở output trỏ vào CSV input này, không phải chỉ số ở CSV nguồn ban đầu.
