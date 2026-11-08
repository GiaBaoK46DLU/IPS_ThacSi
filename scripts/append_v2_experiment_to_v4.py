"""Append the audited V2 experiment to the V4 report without rebuilding old pages."""
from pathlib import Path
from copy import deepcopy
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import statistics

from lxml import etree as E
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runs/kan_v4_v2_addendum_20261027'
SOURCE = ROOT / 'docs/Bao/KAN_V4/IPS_KAN_V4.docx'
BASE_SHA = 'bb69ca48c2760b346ce660f98d21cc04e5bf2272407fb6b7a9ab5dae169f39dd'
TITLE = '31 Chạy thử nghiệm 10 seed, refit với cấu hình V2'
BOOKMARK = '_KanV2Refit10Seeds'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
NS = {'w': W}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
fmt = lambda v, n=6: f'{v:.{n}f}'.replace('.', ',')
text = lambda p: ''.join(p.xpath('.//w:t/text()', namespaces=NS))

WORK.mkdir(parents=True, exist_ok=True)
backup = WORK / 'original_V4_34pages.docx'
if not backup.exists():
    assert sha(SOURCE) == BASE_SHA, 'V4 report changed externally; preserve the edited file.'
    backup.write_bytes(SOURCE.read_bytes())
assert sha(backup) == BASE_SHA
public = ROOT / 'docs/Bao/KAN_V2_Refit_10Seeds'
v2 = read(public / 'official_test_summary.json')
lock = read(public / 'selection_lock.json')
audit = read(public / 'audit_summary.json')
v3 = next(g for g in read(ROOT / 'runs/kan_joint_under6_20261020_v1/official_test_summary.json')['groups']
          if g['role'] == 'primary' and g['regime'] == 'full_development_refit')
v4 = read(ROOT / 'docs/Bao/KAN_V4/official_test_summary.json')
assert audit['status'] == 'passed' and audit['distinct_state_dicts_verified']
assert len(v2['per_seed']) == 10 and v2['refit_epochs'] == 28
assert all(r['reused_historical_validation'] is False for r in lock['all_ten_validation_results'])
for name, digest in audit['receipt_sha256'].items():
    assert sha(public / name) == digest

doc = Document(backup)
body = doc._element.body
for child in list(body):
    if child.tag != qn('w:sectPr'):
        body.remove(child)
tables_expected = []

def p(value, style=None, size=None):
    paragraph = doc.add_paragraph(style=style)
    paragraph.add_run(value)
    paragraph.paragraph_format.widow_control = True
    if size:
        for run in paragraph.runs:
            run.font.size = Pt(size)
    return paragraph


def heading(value, major=False, new_page=False):
    result = p(value, 'Heading 1' if major else 'Heading 2')
    result.paragraph_format.page_break_before = new_page
    return result


def table(number, title, headers, rows, widths):
    caption = p(f'Bảng {number}. {title}', 'Caption')
    caption.paragraph_format.keep_with_next = True
    result = doc.add_table(rows=1, cols=len(headers))
    result.alignment = WD_TABLE_ALIGNMENT.CENTER
    result.autofit = False
    for j, width in enumerate(widths):
        result.columns[j].width = Cm(width)
        result.rows[0].cells[j].text = headers[j]
    for row in rows:
        cells = result.add_row().cells
        for j, value in enumerate(row):
            cells[j].text = str(value)
    for i, row in enumerate(result.rows):
        properties = row._tr.get_or_add_trPr()
        properties.append(OxmlElement('w:cantSplit'))
        if i == 0:
            properties.append(OxmlElement('w:tblHeader'))
        for j, cell in enumerate(row.cells):
            cell.width = Cm(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            properties = cell._tc.get_or_add_tcPr()
            margins = OxmlElement('w:tcMar')
            for side in ('top', 'left', 'bottom', 'right'):
                element = OxmlElement('w:' + side)
                element.set(qn('w:w'), '65' if side in ('top', 'bottom') else '95')
                element.set(qn('w:type'), 'dxa')
                margins.append(element)
            properties.append(margins)
            borders = OxmlElement('w:tcBorders')
            for side in ('top', 'left', 'bottom', 'right'):
                element = OxmlElement('w:' + side)
                for key, value in [('val', 'single'), ('sz', '4'), ('color', 'D9D9D9')]:
                    element.set(qn('w:' + key), value)
                borders.append(element)
            properties.append(borders)
            fill = OxmlElement('w:shd')
            fill.set(qn('w:fill'), '24465C' if i == 0 else ('F2F5F7' if i % 2 == 0 else 'FFFFFF'))
            properties.append(fill)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(1)
                paragraph.paragraph_format.space_before = Pt(1)
                paragraph.paragraph_format.line_spacing = 1.02
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 or len(headers) == 2 else WD_ALIGN_PARAGRAPH.CENTER
                for run in paragraph.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(10.5)
                    run.bold = i == 0
                    run.font.color.rgb = RGBColor(255, 255, 255) if i == 0 else RGBColor(0, 0, 0)
    spacer = p('', size=2)
    spacer.paragraph_format.space_after = Pt(1)
    spacer.paragraph_format.line_spacing = 1
    tables_expected.append({'number': number, 'headers': headers, 'rows': [[str(x) for x in row] for row in rows]})


heading(TITLE, major=True, new_page=True)
p('Đợt đối chứng bổ sung dùng lại cấu hình ReferenceKANMixer V2 từng đạt 8,058 ± 0,306 m trên ba seed train nội bộ. '
  'Lần này huấn luyện mới toàn bộ 10 lượt Validation, sau đó khởi tạo lại 10 model để refit trên toàn bộ development. '
  'Không đưa ba checkpoint V2 lịch sử vào nhóm chạy mới.')
p('Kết quả chính sau refit là Mean2D Test 7,624741 ± 0,208742 m trên đủ 1.111 mẫu/model. '
  'Mục này được bổ sung sau nghiên cứu V4; kết quả V4 và so sánh V3/V4 ở các chương trước giữ nguyên.')
heading('31.1 Cấu hình V2 được sử dụng')
config_rows = [
    ('Phạm vi mạng', 'Một ReferenceKANMixer học chung ba tòa nhà, 13 lớp tòa–tầng; không ensemble hoặc kNN.'),
    ('Đầu vào', 'RSSI tuyệt đối và kênh mask nhân 0; 520 AP, 26 token × 40 feature. Không dùng RSSI tương đối.'),
    ('Mixer và head', 'Hai block; channel 64, token_hidden 32, channel_hidden 128; flatten 1.664 → 256 → RP; dropout 0,1.'),
    ('Catalogue', '872 RP trong train nội bộ; 933 RP sau refit development. Chuẩn hóa tọa độ từ train nội bộ được giữ nguyên.'),
    ('KAN B-spline', 'Grid size 5, spline order 3, SiLU, grid [−3;3], update_grid=false; spline scaler riêng được bật.'),
    ('Khởi tạo và điều chuẩn', 'scale_noise=0,1; scale_base=1; scale_spline=1; grid_eps=0,02; extra_regularization=0.'),
    ('Hàm mất mát', 'λ_class=0,5; λ_reference=1; λ_xy=3; σ_RP=0,05 chuẩn hóa; label_smoothing=0. Giữ công thức reference ở chương 11.'),
    ('Tối ưu', 'AdamW; learning rate ban đầu 0,001; weight decay 0,0001; batch size 128; gradient clip L2=1.'),
    ('Lịch học', 'Cosine horizon 100 epoch, learning rate tối thiểu 0,00005; không nén lịch thành 28 epoch.'),
    ('Train nội bộ', 'Tối đa 100 epoch; patience 20; min_delta=0,01 m; 2 luồng CPU; diagnostic_batches=0.'),
    ('Tăng cường RSSI', 'Gaussian noise 2 dB; dịch chung fingerprint 3 dB; AP dropout 0,05, chỉ bật khi train.'),
    ('Lấy mẫu và trộn', 'Xáo trộn đều theo fingerprint; không dùng trọng số cân bằng điểm V4, same-point mixing, nearby mixing hoặc EMA.'),
    ('Decoder lúc train', 'top_k=5, temperature=1; giá trị này được giữ riêng cho loss khi refit.'),
    ('Decoder suy luận', 'Chọn riêng từng seed bằng Validation, rồi khóa trước refit; không hiệu chỉnh lại trên Test.'),
    ('Refit thực tế', '28 epoch trên 19.937 fingerprint development; khởi tạo trọng số mới cho từng seed; không chấm Validation cũ trong refit.'),
]
table(26, 'Cấu hình V2 của đợt train mới và refit', ['Thành phần', 'Giá trị được dùng'], config_rows, [4.0, 12.8])
p('smooth_l1_beta=1 trong JSON là tham số tương thích nhánh hồi quy cũ, không thay loss tọa độ của nhánh reference. '
  'Tệp cấu hình đầy đủ: src/ips/algorithms/kan/configs/v2_reference_reproduction.json.', size=9)

heading('31.2 Quy trình train mới và kết quả từng seed', new_page=True)
p('Các seed được cố định theo thứ tự 42, 43, 44, 100, 2024, 777, 999, 45, 46, 47, giống bộ seed của V3/V4. '
  'Mỗi lượt train mới dùng cùng split 16.025 train và 3.912 Validation; Test chính thức vẫn là 1.111 dòng gốc. '
  'Checkpoint được chọn bằng Mean2D Validation với min_delta và patience của V2.')
p('Sau chọn checkpoint, decoder thuần được hiệu chỉnh trên Validation với k ∈ {1;3;5;10;20;1000} và '
  'temperature ∈ {0,5;1;2;4}, tổng 24 cấu hình/seed. k=1000 có nghĩa lấy tối đa số RP hợp lệ của lớp dự đoán, '
  'không tạo ra 1.000 láng giềng RSSI. Decoder mỗi seed được giữ nguyên sau refit.')
epochs = [r['best_epoch'] for r in lock['all_ten_validation_results']]
p(f'Mười best epoch là {", ".join(str(x) for x in epochs)}. Median bằng 28,5; quy tắc round() của Python '
  'đã khóa trước Test làm tròn trường hợp nửa đơn vị về số chẵn gần nhất, nên toàn bộ refit dùng 28 epoch. '
  'Refit dùng trọng số khởi tạo mới, không fine-tune từ checkpoint Validation, và giữ cosine horizon 100.')
val_mean = statistics.mean(r['mean_2d'] for r in lock['all_ten_validation_results'])
val_std = statistics.stdev(r['mean_2d'] for r in lock['all_ten_validation_results'])
p(f'Mean2D Validation trước refit là {fmt(val_mean)} ± {fmt(val_std)} m. Đây là score của model train nội bộ; '
  'sau refit, Validation cũ đã thuộc dữ liệu huấn luyện và không còn được xem là tập đánh giá độc lập.')
seed_rows = []
for trial, test in zip(lock['all_ten_validation_results'], v2['per_seed']):
    assert trial['seed'] == test['seed']
    seed_rows.append([test['seed'], trial['best_epoch'], trial['decoder']['top_k'], fmt(trial['decoder']['temperature'], 1),
                      fmt(trial['mean_2d']), fmt(test['mean_2d'])])
table(27, 'Mười lượt train mới và Mean2D sau refit V2',
      ['Seed', 'Epoch tốt nhất', 'k', 'Nhiệt độ', 'Validation trước refit (m)', 'Test sau refit (m)'],
      seed_rows, [1.1, 2.1, 1.7, 2.3, 4.8, 4.8])
p('Cả 10 checkpoint refit được khóa SHA256 trước lượt Test mới đầu tiên. Từng model được chấm độc lập trên '
  'đủ 1.111 dòng, rồi tổng hợp mean và sample SD với ddof=1. Không chọn seed có sai số thấp nhất để thay kết quả nhóm.')

heading('31.3 Kết quả Test và so sánh với V3 V4', new_page=True)
metrics = [('mean_2d', 'Mean2D (m)', 1), ('median_2d', 'Median (m)', 1), ('p90_2d', 'P90 (m)', 1),
           ('rmse_2d', 'RMSE (m)', 1), ('building_accuracy', 'Đúng tòa (%)', 100),
           ('floor_accuracy', 'Đúng tầng (%)', 100), ('joint_accuracy', 'Đúng tòa–tầng (%)', 100)]
metric_rows = []
for key, label, scale in metrics:
    a, c = v2['aggregate'][key], v4['aggregate'][key]
    metric_rows.append([label, f"{fmt(a['mean']*scale,4)} ± {fmt(a['sample_std']*scale,4)}",
                       f"{fmt(v3[key+'_mean']*scale,4)} ± {fmt(v3[key+'_sample_std']*scale,4)}",
                       f"{fmt(c['mean']*scale,4)} ± {fmt(c['sample_std']*scale,4)}"])
metric_rows += [['Epoch refit', '28', '31', '41'], ['Model / mẫu Test mỗi model', '10 / 1.111', '10 / 1.111', '10 / 1.111']]
table(28, 'So sánh ba nhóm refit trên cùng UJIIndoorLoc', ['Chỉ số', 'V2 refit mới', 'V3', 'V4'],
      metric_rows, [3.3, 4.5, 4.5, 4.5])
delta3 = v2['aggregate']['mean_2d']['mean'] - v3['mean_2d_mean']
delta4 = v2['aggregate']['mean_2d']['mean'] - v4['aggregate']['mean_2d']['mean']
p(f'Mean2D V2 cao hơn V3 {fmt(delta3,4)} m ({fmt(delta3/v3["mean_2d_mean"]*100,2)}%) và cao hơn V4 '
  f'{fmt(delta4,4)} m ({fmt(delta4/v4["aggregate"]["mean_2d"]["mean"]*100,2)}%). '
  'V4 vẫn có Mean2D thấp nhất trong ba nhóm. V2 có RMSE thấp hơn V3/V4, nhưng Mean2D, P90 và độ chính xác tầng '
  'kém hơn; các chỉ số phản ánh những khía cạnh khác nhau của phân bố lỗi.')
p('Đây là so sánh toàn bộ pipeline: V2 dùng RSSI tuyệt đối và không trộn cùng điểm; V3 dùng RSSI tương đối, '
  'mask hữu dụng và same-point mixing; V4 bổ sung sampling cân bằng điểm. Số epoch refit lần lượt là 28, 31, 41, '
  'nên chưa cô lập tác dụng riêng của mỗi thành phần. Con số V2 lịch sử 8,058 ± 0,306 m thuộc ba model train nội bộ, '
  'khác cả số seed lẫn phạm vi dữ liệu train so với nhóm refit mới.')
heading('31.4 Kiểm toán và nguồn bằng chứng')
p('Kiểm toán đạt trên 10 bộ trọng số refit khác nhau và 11.110 dòng dự đoán. Các metric được tính lại từ '
  'tọa độ gốc, kiểm đủ row index/thứ tự dòng, nhãn thật, hash checkpoint và số bước optimizer. Mỗi refit có '
  '4.368 bước = 156 batch/epoch × 28 epoch. Bộ kiểm thử mã sau khi hoàn tất đạt 89/89. '
  'Mười model cùng dùng 1.111 query Test; 11.110 dự đoán không phải 11.110 mẫu thực tế độc lập.')
p('Lượt refit seed 42 bị gián đoạn ở 10/28 epoch đã được lưu riêng và chạy lại từ đầu; các lượt Validation mới '
  'và khóa lựa chọn được giữ nguyên. Test đã có lịch sử được xem trong dự án; đợt V2 này là đối chứng bổ sung '
  'sau nghiên cứu V4, không phải xác nhận trên một Test chưa từng được quan sát.')
p('Nguồn: docs/Bao/KAN_V2_Refit_10Seeds; các tệp official_test_summary.json, selection_lock.json, '
  'test_checkpoint_lock.json, audit_summary.json và recovery_20261027_1923.json. Artifact gốc gồm checkpoint, '
  'CSV dự đoán và log nằm tại runs/kan_v2_refit_10seeds_fresh_20261027. '
  'Nhánh thực hiện: experiment/kan-v2-refit-10seeds.', size=9)

with ZipFile(backup) as archive:
    members = [(info, archive.read(info.filename)) for info in archive.infolist()]
original = E.fromstring(dict((i.filename, data) for i, data in members)['word/document.xml'])
original_body = original.find('w:body', NS)
original_count = len(original_body)
original_nodes = [E.tostring(element, method='c14n') for element in original_body]

replacements = {
    'Phần thuật toán đi từ RSSI đầu vào tới tọa độ đầu ra, kèm ký hiệu, công thức, kích thước tensor và ví dụ. Phần thực nghiệm tách Validation dùng chọn mô hình khỏi Test dùng xác nhận. Mọi bảng so sánh kết quả đều đối chiếu V4 với V3 same_point_relative; tài liệu IPS_KAN_V2_thuat_toan chỉ định hướng cách trình bày thuật toán.':
    'Phần thuật toán đi từ RSSI đầu vào tới tọa độ đầu ra, kèm công thức, tensor và ví dụ. Các chương 1–30 đối chiếu V4 với V3 same_point_relative; tài liệu V2 định hướng cách trình bày thuật toán. Mục 31 ở cuối báo cáo bổ sung thử nghiệm 10 lượt train mới và refit cấu hình V2, cùng bảng so sánh ba nhóm.',
    '[M5] IPS_KAN_V2_thuat_toan.docx trong references/KAN-mixer: mẫu cách giải thích thuật toán và các công thức nền được kế thừa. Không dùng kết quả V2 làm baseline trong báo cáo này.':
    '[M5] IPS_KAN_V2_thuat_toan.docx trong references/KAN-mixer: mẫu giải thích thuật toán và công thức nền. Thử nghiệm refit cấu hình V2 được bổ sung tại mục 31; các chương 1–30 giữ baseline chính là V3.',
}
changed_nodes = []
for index, element in enumerate(list(original_body)):
    if text(element) in replacements:
        value = replacements[text(element)]
        runs = element.xpath('.//w:t', namespaces=NS)
        runs[0].text = value
        for run in runs[1:]:
            run.text = ''
        changed_nodes.append(index)
assert len(changed_nodes) == 2

new_nodes = [deepcopy(element) for element in body if element.tag != qn('w:sectPr')]
major = new_nodes[0]
bookmark_id = str(max(int(b.get(qn('w:id'))) for b in original.findall('.//w:bookmarkStart', NS)) + 1)
start = E.Element(qn('w:bookmarkStart'));start.set(qn('w:id'), bookmark_id);start.set(qn('w:name'), BOOKMARK)
end = E.Element(qn('w:bookmarkEnd'));end.set(qn('w:id'), bookmark_id)
major.insert(1, start);major.append(end)
for element in new_nodes:
    original_body.insert(len(original_body) - 1, element)

toc_last = next(element for element in original_body if element.tag == qn('w:p')
                and element.find('.//w:hyperlink', NS) is not None and text(element).endswith('tiếp theo34'))
toc_entry = deepcopy(toc_last)
for key in list(toc_entry.attrib):
    del toc_entry.attrib[key]
hyperlink = toc_entry.find('w:hyperlink', NS)
hyperlink.set(qn('w:anchor'), BOOKMARK)
toc_texts = hyperlink.findall('.//w:t', NS)
assert len(toc_texts) == 2
toc_texts[0].text = TITLE;toc_texts[1].text = '35'
hyperlink.find('.//w:instrText', NS).text = ' PAGEREF ' + BOOKMARK + ' ' + chr(92) + 'h '
original_body.insert(original_body.index(toc_last) + 1, toc_entry)

unchanged = 0
for index, old in enumerate(original_nodes):
    current_index = index + (1 if index > original_body.index(toc_last) else 0)
    if index not in changed_nodes and index < original_count - 1:
        assert E.tostring(original_body[current_index], method='c14n') == old, index
        unchanged += 1
output = WORK / 'IPS_KAN_V4.docx'
with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
    for info, data in members:
        if info.filename == 'word/document.xml':
            data = E.tostring(original, xml_declaration=True, encoding='UTF-8', standalone=True)
        archive.writestr(info, data)

expected = {'source_sha256': BASE_SHA, 'draft_sha256': sha(output), 'title': TITLE,
            'tables': tables_expected, 'original_body_elements_unchanged': unchanged,
            'original_paragraphs_updated': changed_nodes, 'new_elements': len(new_nodes),
            'source_receipt_sha256': {name: sha(public / name) for name in ('official_test_summary.json','selection_lock.json','audit_summary.json')},
            'output': str(output)}
(WORK / 'authoring.json').write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: expected[k] for k in ('title','original_body_elements_unchanged','original_paragraphs_updated','new_elements','output')}, ensure_ascii=False))
