"""Create the local Vietnamese result report from independently audited files."""
import collections
import csv
import statistics
from pathlib import Path

from .....data import ROOT
from .audit import METRICS


def write_report(output):
    from .experiment import read,SEEDS
    result=read(output/'official_test_summary.json');protocol=read(output/'protocol.json')
    destination=ROOT/'docs/Bao/KAN_SKAN'
    destination.mkdir(parents=True,exist_ok=True)
    baseline=result['baseline_v4']['per_seed']
    selections=[read(output/'activation_search'/f'seed{s}'/'selection.json') for s in SEEDS]
    resources=[read(output/'final_refit'/f'seed{s}'/'resources.json') for s in SEEDS]
    parameters=[read(output/'final_refit'/f'seed{s}'/'run_metadata.json')['trainable_parameters'] for s in SEEDS]
    candidate_counts=collections.Counter(i for selection in selections
        for values in selection['selected_activations'].values() for i in values)
    lines=['# S-KAN-Mixer trên UJIIndoorLoc — đối chiếu V4','',
        'Đây là triển khai thích nghi cho IPS theo cơ chế chọn không gian hàm của S-KAN. '
        'Danh sách 18 ứng viên được khai báo riêng; không khẳng định tái lập chính xác mã tác giả chưa được cung cấp.',
        '',f"Nhánh: `{protocol['branch']}`. Kết quả dùng 10 mô hình độc lập, mỗi mô hình dự đoán đủ 1.111 mẫu Test chính thức. "
        'Các chỉ số là trung bình ± độ lệch chuẩn mẫu giữa 10 seed; không phải ensemble.',
        '', '## Kết quả Test so với V4','',
        '| Chỉ số | V4 | S-KAN-Mixer |','|---|---:|---:|']
    for key in METRICS:
        a=statistics.mean(r[key] for r in baseline);s=statistics.stdev(r[key] for r in baseline)
        b=result['aggregate'][key]
        lines.append(f"| {key} | {a:.6f} ± {s:.6f} | {b['mean']:.6f} ± {b['sample_std']:.6f} |")
    delta=result['paired_mpe_delta_skan_minus_v4']
    lines+=['',f"Chênh lệch MPE ghép theo seed (S-KAN − V4): **{delta['mean']:+.6f} m**. "
        'Giá trị dương nghĩa là S-KAN có sai số cao hơn. Không điều chỉnh mô hình sau khi xem Test.',
        '', '| Seed | V4 MPE (m) | S-KAN MPE (m) | Chênh lệch (m) |', '|---:|---:|---:|---:|']
    for a,b in zip(baseline,result['per_seed']):
        lines.append(f"| {a['seed']} | {a['mean_2d']:.6f} | {b['mean_2d']:.6f} | {b['mean_2d']-a['mean_2d']:+.6f} |")
    lines+=['','## Cấu hình và quy trình đối chiếu','',
        'Giữ nguyên cấu hình ngoài của V4: đầu vào relative RSSI + mask (520 AP); 26 token × 40 feature; '
        '64 channel, 2 block, token hidden 32, channel hidden 128, dropout 0,1; flatten và head 256 → RP. '
        'Batch 128; AdamW lr 0,001, weight decay 0,0001; cosine horizon 100; patience 20; min_delta 0,01; clip L2 = 1. '
        'SamePointMixing 0,5 và lấy mẫu cân bằng điểm tham chiếu. Nhiễu RSSI 2 dB; dịch thiết bị 3 dB; AP dropout 0,05.',
        '', 'Loss giữ nguyên: 0,5 × cross entropy lớp tầng/tòa + 1 × soft-target RP cross entropy '
        '+ 3 × khoảng cách Euclid tọa độ chuẩn hóa trên lớp thật. Sigma RP = 0,05. '
        'Trường smooth_l1_beta có trong JSON kế thừa nhưng không được dùng trong nhánh reference.',
        '', 'Chỉ thay tám lớp KAN trong hai nhánh token/channel bằng lớp chọn hàm theo node đầu vào. '
        'Tổng cộng 500 node được chọn hàm ở mỗi seed. Tất cả ứng viên có nhánh nền SiLU; không thêm LayerNorm trong ứng viên.',
        '', 'Chọn hàm trên 1% internal Train (160 mẫu, phân tầng 13 lớp); catalogue RP được fit trên toàn bộ internal Train. '
        'Mỗi vòng: train toàn bộ 3 epoch → chỉ train logits lựa chọn 3 epoch → bỏ một ứng viên/node và chuẩn hóa lại. '
        '17 vòng từ 18 xuống 1 ứng viên/node. Batch chọn hàm 16; AdamW lr 0,001; logits không weight decay; '
        'optimizer khởi tạo lại mỗi pha; không scheduler trong pha chọn hàm.',
        '', 'Sau chọn hàm, bỏ toàn bộ trọng số đã học; khởi tạo lại theo seed rồi train 16.025 mẫu, '
        'chọn checkpoint bằng 3.912 mẫu Validation. Decoder chọn riêng mỗi seed trên Validation từ '
        'k = [1,3,5,10,20,1000], T = [0,5;1;2;4]. '
        f"Số epoch refit là round(median(10 best epoch)) = **{result['refit_epochs']}**; "
        'huấn luyện mới trên toàn bộ 19.937 mẫu development, giữ nguyên cấu trúc hàm và tiền xử lý train-only. '
        'Khóa đủ 10 checkpoint trước khi đánh giá Test.',
        '', '## Chi phí, số tham số và giới hạn','',
        f"Số tham số trainable sau refit: trung bình {statistics.mean(parameters):,.0f}, khoảng {min(parameters):,}–{max(parameters):,}. "
        f"Tổng thời gian pha chọn hàm của 10 seed: {sum(s['seconds'] for s in selections)/60:.2f} phút. "
        'Chi tiết thời gian/VRAM train và refit nằm trong resources.json từng seed.',
        '', 'So sánh giữ nguyên kiến trúc ngoài và ngân sách train chính; không cân bằng tổng compute hay số tham số, '
        'vì S-KAN có thêm pha chọn hàm. Đã biết kết quả lịch sử V4 trên Test trước thí nghiệm này; '
        'Test này là đối chiếu benchmark đã có lịch sử tiếp xúc, không phải một holdout hoàn toàn chưa từng được xem.',
        '', '## Tần suất hàm được chọn (5.000 node của 10 seed)','',
        '| Ứng viên | Số node |','|---|---:|']
    for index,spec in enumerate(protocol['config']['kan']['candidates']):
        lines.append(f"| {spec['name']} | {candidate_counts[index]} |")
    lines+=['','## Tệp để review','',f'- Kết quả và audit: `{output}`.',
        '- Cấu hình chung: `src/ips/algorithms/kan/configs/selectable/uji_v4_matched.json`.',
        '- Cấu hình đầy đủ từng seed sau chọn hàm: `activation_search/seed*/selected_config.json`.',
        '- Cấu hình refit và decoder cuối: `final_refit/seed*/resolved_config.json`.',
        '- Mỗi seed có `official_test/seed*/predictions.csv`, `metrics.json`, `independent_audit.json`.',
        '', 'Nguồn phương pháp: [Yang et al., S-KAN](https://arxiv.org/abs/2408.08338v1). '
        'Công thức, thay đổi thích nghi và nguồn tham khảo bổ sung được ghi ở `variants/selectable/README.md` và `provenance.json`.']
    (destination/'S_KAN_UJI_V4_Comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for name in ('official_test_summary.json','completion_audit.json'):
        (destination/name).write_bytes((output/name).read_bytes())
    with (destination/'per_seed_test.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['seed']+METRICS)
        writer.writeheader();writer.writerows(result['per_seed'])
