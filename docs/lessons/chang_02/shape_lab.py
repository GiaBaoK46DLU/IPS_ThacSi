"""Bài thực hành chặng 2: nhìn các trục của một khối Mixer.

Dữ liệu là đặc trưng minh họa; không phải dữ liệu UJIIndoorLoc.
Chạy: python docs/lessons/chang_02/shape_lab.py
"""

import torch
from torch import nn


def show(label, value):
    print(f"{label}: {list(value.shape)}")


def main():
    torch.manual_seed(42)
    print(f"PyTorch: {torch.__version__}")

    # Phần 1: B=2 mẫu; S=4 token/AP; C=3 channel ẩn.
    x = torch.tensor([
        [[0.2, 0.7, -0.1], [0.5, -0.3, 0.4],
         [-0.2, 0.8, 0.6], [0.1, 0.2, -0.5]],
        [[0.4, 0.1, -0.2], [0.3, -0.1, 0.8],
         [0.2, 0.5, 0.3], [-0.4, 0.2, 0.1]],
    ], dtype=torch.float32)
    show("Đầu vào", x)

    norm_token = nn.LayerNorm(3)
    norm_channel = nn.LayerNorm(3)
    token_fc1, token_fc2 = nn.Linear(4, 8), nn.Linear(8, 4)
    channel_fc1, channel_fc2 = nn.Linear(3, 6), nn.Linear(6, 3)
    activation = nn.GELU()

    # Chỉ quan sát lượt tính xuôi, không tính gradient hoặc huấn luyện.
    with torch.no_grad():
        # Phần 2: chuẩn hóa C trước, sau đó đưa S ra trục cuối.
        q = norm_token(x)
        show("Chuẩn hóa trước token-mixing", q)
        q = q.transpose(1, 2)
        show("Chuyển vị", q)
        q = activation(token_fc1(q))
        show("Lớp ẩn token", q)
        q = token_fc2(q)
        show("Đầu ra MLP token", q)
        q = q.transpose(1, 2)
        show("Chuyển vị trở lại", q)
        u = x + q
        show("Residual thứ nhất", u)

        # Phần 3: C đã ở trục cuối; dùng MLP chung cho từng token.
        r = norm_channel(u)
        show("Chuẩn hóa trước channel-mixing", r)
        r = activation(channel_fc1(r))
        show("Lớp ẩn channel", r)
        r = channel_fc2(r)
        show("Đầu ra MLP channel", r)
        y = u + r
        show("Residual thứ hai", y)

        # Phần 4: chuẩn hóa cuối rồi lấy trung bình theo token.
        pooled = nn.LayerNorm(3)(y).mean(dim=1)
        show("Sau pooling token", pooled)
        assert y.shape == x.shape
        assert pooled.shape == (2, 3)

        single_channel = torch.tensor([[[-0.4], [-0.8]]])
        normalized_single = nn.LayerNorm(1)(single_channel)
        print("Minh họa LayerNorm khi C=1:", normalized_single.tolist())
        torch.testing.assert_close(
            normalized_single, torch.zeros_like(normalized_single)
        )

    print("Đã kiểm tra kích thước. Chưa huấn luyện hoặc đánh giá IPS.")


if __name__ == "__main__":
    main()
