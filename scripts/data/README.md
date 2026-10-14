# Công cụ dữ liệu dùng chung

Các script này dùng `ips.data`; không phụ thuộc tài liệu cá nhân trong `docs/` hoặc `references/`.

```powershell
python scripts/data/download_data.py --help
python -m ips prepare --verify-only
python -m ips prepare
```

`prepare` audit CSV, tạo split khi chưa có manifest và kiểm chứng split. Nếu manifest đã tồn tại,
chỉ kiểm chứng, không ghi lại metadata hoặc đổi hash của split dùng với checkpoint lịch sử.
Audit mới nằm trong `runs/data_audit/`, không đưa vào Git.

Muốn tái tạo riêng để so sánh, dùng một thư mục mới:

```powershell
python scripts/data/make_splits.py --output-dir runs/split_reproduction_new
python scripts/data/verify_splits.py --split-dir runs/split_reproduction_new
```

`split_policy.json` giữ cùng StratifiedGroupKFold, seed 2026, khóa nhóm, mapping và train-only
coordinate fit của giao thức v1. Các artifact chỉ số/preprocessing/mapping phải khớp bản đã lưu;
manifest tái tạo ghi đường dẫn policy và hash công cụ hiện tại nên có provenance khác manifest lịch sử.
Không thay manifest cũ bằng manifest tái tạo để nạp checkpoint có hash cũ.
