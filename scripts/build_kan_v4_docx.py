"""Build the Vietnamese V4 technical report from frozen local evidence."""
from pathlib import Path
from copy import deepcopy
import json, math, statistics, zipfile, hashlib, shutil
from lxml import etree
from PIL import Image
from docx import Document
from docx.shared import Cm, Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'runs/kan_v4_docx_20261026'
OUT=ROOT/'references/KAN-mixer/IPS_KAN_V4.docx'
if OUT.exists():
    prior_path=WORK/'authoring.json'
    if not prior_path.exists():raise ValueError('Existing Word report has no ownership receipt; preserve it.')
    prior=json.loads(prior_path.read_text(encoding='utf-8'))
    digest=hashlib.sha256(OUT.read_bytes()).hexdigest()
    if digest!=prior['output_sha256']:raise ValueError('Word report changed externally; preserve the edited document.')
    (WORK/'versions').mkdir(exist_ok=True)
    shutil.copy2(OUT,WORK/'versions'/('IPS_KAN_V4_'+digest[:12]+'.docx'))
D=json.loads((WORK/'evidence.json').read_text(encoding='utf-8'))
D['source_sha256']={k.replace('\\','/'):v for k,v in D['source_sha256'].items()}
with zipfile.ZipFile(ROOT/D['V2_reference']['path']) as z:
    source_xml=etree.fromstring(z.read('word/document.xml'))
MATHS=source_xml.findall('.//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath')
XSL=etree.XSLT(etree.parse('C:/Program Files/Microsoft Office/root/Office16/MML2OMML.XSL'))
doc=Document(); sec=doc.sections[0]
sec.page_width=Cm(21);sec.page_height=Cm(29.7)
sec.top_margin=Cm(1.9);sec.bottom_margin=Cm(1.9);sec.left_margin=Cm(2.2);sec.right_margin=Cm(2.0)
sec.header_distance=Cm(.7);sec.footer_distance=Cm(.7)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3','Caption']:
    s=doc.styles[name];s.font.name='Times New Roman';s.font.color.rgb=RGBColor(0,0,0)
    s.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Times New Roman')
    for border in list(s.element.xpath('.//w:pBdr')):border.getparent().remove(border)
normal=doc.styles['Normal'];normal.font.size=Pt(12);normal.paragraph_format.line_spacing=1.1;normal.paragraph_format.space_after=Pt(6)
for name,size in [('Title',23),('Subtitle',14),('Heading 1',16),('Heading 2',13),('Heading 3',12)]:
    doc.styles[name].font.size=Pt(size)
    doc.styles[name].font.bold=name not in ('Subtitle',)
for name in ('Heading 1','Heading 2','Heading 3'):
    doc.styles[name].paragraph_format.keep_with_next=True
    doc.styles[name].paragraph_format.space_before=Pt(0 if name=='Heading 1' else 8)
    doc.styles[name].paragraph_format.space_after=Pt(9 if name=='Heading 1' else 5)
doc.styles['Caption'].font.size=Pt(10.5);doc.styles['Caption'].font.italic=True
doc.core_properties.title='Báo cáo KAN Mixer V4 trên UJIIndoorLoc'
doc.core_properties.subject='Thuật toán cấu hình thực nghiệm và so sánh V4 với V3'
doc.core_properties.author='Dự án IPS Thạc sĩ'
doc.core_properties.keywords='KAN-Mixer; V4; V3; UJIIndoorLoc; cân bằng điểm tham chiếu'
head=sec.header.paragraphs[0];head.text='KAN Mixer V4 trên UJIIndoorLoc';head.runs[0].font.size=Pt(9)
foot=sec.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.RIGHT
foot.add_run('Trang ').font.size=Pt(9)
def field(p,instruction):
    r=OxmlElement('w:r');b=OxmlElement('w:fldChar');b.set(qn('w:fldCharType'),'begin');r.append(b);p._p.append(r)
    r=OxmlElement('w:r');v=OxmlElement('w:instrText');v.set(qn('xml:space'),'preserve');v.text=' '+instruction+' ';r.append(v);p._p.append(r)
    r=OxmlElement('w:r');e=OxmlElement('w:fldChar');e.set(qn('w:fldCharType'),'end');r.append(e);p._p.append(r)
field(foot,'PAGE');settings=doc.settings.element
update=OxmlElement('w:updateFields');update.set(qn('w:val'),'true');settings.append(update)
COUNTS={'table':0,'figure':0,'equation':0}; HEADINGS=[]
def fmt(x,n=4):return f'{x:.{n}f}'.replace('.',',')
def para(text,bold=False,style=None):
    p=doc.add_paragraph(style=style);r=p.add_run(text);r.bold=bold
    p.paragraph_format.widow_control=True
    return p
def page(title):
    p=doc.add_paragraph(title,style='Heading 1');p.paragraph_format.page_break_before=True
    if title=='Mục lục':
        p.style='TOC Heading';p.runs[0].font.color.rgb=RGBColor(0,0,0);p.runs[0].font.size=Pt(16)
    HEADINGS.append(title);return p
def h2(text):return doc.add_paragraph(text,style='Heading 2')
def code(text):
    p=para(text);p.paragraph_format.line_spacing=1.0;p.paragraph_format.space_after=Pt(5)
    for r in p.runs:r.font.name='Consolas';r.font.size=Pt(9)
    return p
def source(text):
    p=para('Nguồn đối chiếu: '+text)
    for r in p.runs:r.italic=True;r.font.size=Pt(9)
    return p
def table(title,headers,rows,widths=None):
    COUNTS['table']+=1;n=COUNTS['table']
    p=para(f'Bảng {n}. {title}',style='Caption');p.paragraph_format.keep_with_next=True
    t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    widths=widths or [16.8/len(headers)]*len(headers)
    for j,w in enumerate(widths):t.columns[j].width=Cm(w)
    for j,txt in enumerate(headers):t.rows[0].cells[j].text=str(txt)
    for row in rows:
        cells=t.add_row().cells
        for j,v in enumerate(row):cells[j].text=str(v).replace('absolute_relative_context','absolute_relative_\ncontext')
    for i,row in enumerate(t.rows):
        pr=row._tr.get_or_add_trPr();nosplit=OxmlElement('w:cantSplit');pr.append(nosplit)
        if i==0:rep=OxmlElement('w:tblHeader');pr.append(rep)
        for j,cell in enumerate(row.cells):
            cell.width=Cm(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr=cell._tc.get_or_add_tcPr()
            margins=OxmlElement('w:tcMar')
            for side in ('top','left','bottom','right'):
                e=OxmlElement('w:'+side);e.set(qn('w:w'),'65' if side in ('top','bottom') else '95');e.set(qn('w:type'),'dxa');margins.append(e)
            tcpr.append(margins)
            borders=OxmlElement('w:tcBorders')
            for side in ('top','left','bottom','right'):
                e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
            tcpr.append(borders)
            shd=OxmlElement('w:shd');shd.set(qn('w:fill'),'24465C' if i==0 else ('F2F5F7' if i%2==0 else 'FFFFFF'));tcpr.append(shd)
            for p in cell.paragraphs:
                p.paragraph_format.space_after=Pt(1);p.paragraph_format.space_before=Pt(1);p.paragraph_format.line_spacing=1.02
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                if len(headers)==2 and j==1:p.alignment=WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.name='Times New Roman';r.font.size=Pt(10.5);r.bold=i==0
                    r.font.color.rgb=RGBColor(255,255,255) if i==0 else RGBColor(0,0,0)
    p=doc.add_paragraph();p.paragraph_format.space_after=Pt(1);p.paragraph_format.space_before=Pt(0);p.paragraph_format.line_spacing=1
    p.add_run().font.size=Pt(2)
    return t
def equation_node(node,explanation):
    COUNTS['equation']+=1;p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(2)
    node=deepcopy(node)
    if COUNTS['equation']==22:
        rows=[OxmlElement('m:e'),OxmlElement('m:e')];second=False
        for child in list(node):
            texts=child.findall('.//'+qn('m:t'))
            if not second and child.tag==qn('m:r') and len(texts)==1 and ';' in texts[0].text:
                before,after=texts[0].text.split(';',1)
                a,b=deepcopy(child),deepcopy(child)
                a.find(qn('m:t')).text=before.strip();b.find(qn('m:t')).text=after.strip()
                rows[0].append(a);rows[1].append(b);second=True
            else:rows[int(second)].append(child)
        array=OxmlElement('m:eqArr');array.extend(rows);node[:]=[array]
    for size in node.findall('.//'+qn('w:sz')):size.set(qn('w:val'),'23')
    p._p.append(node);p.add_run(f"    ({COUNTS['equation']})").font.size=Pt(10)
    if explanation:para(explanation)
def eq(i,explanation=''):equation_node(MATHS[i],explanation)
def mml(xml,explanation=''):
    node=XSL(etree.fromstring(('<math xmlns="http://www.w3.org/1998/Math/MathML">'+xml+'</math>').encode())).getroot()
    if node.tag.endswith('oMathPara'):node=node.find('{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath')
    equation_node(node,explanation)
def figure(name,title,width=6.15):
    COUNTS['figure']+=1;n=COUNTS['figure'];para(f'Hình {n} minh họa {title[0].lower()+title[1:]}.')
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(3)
    path=WORK/'figures'/name
    with Image.open(path) as im:w,h=im.size
    width=min(width,6.6*w/h)
    shape=p.add_run().add_picture(str(path),width=Inches(width));shape._inline.docPr.set('descr',title)
    p._p.get_or_add_pPr().append(OxmlElement('w:keepNext'))
    q=para(f'Hình {n}. {title}',style='Caption');q.alignment=WD_ALIGN_PARAGRAPH.CENTER

# The report uses V2 only for exposition and reusable native mathematical notation.
para('Báo cáo KAN Mixer V4 trên UJIIndoorLoc',style='Title')
para('Thuật toán cấu hình và so sánh với V3',style='Subtitle')
para('Ngày 27 tháng 10 năm 2026',bold=True)
para('Nhánh nghiên cứu kan-version-4. Phạm vi: một mạng KAN-Mixer học chung ba tòa nhà và 13 lớp tòa–tầng, được huấn luyện độc lập trên 10 seed cố định.')
h2('Kết quả chính')
para('V4 đạt sai số định vị trung bình Test 7,277383 ± 0,211021 m, tương đương khoảng 7,28 m. V3 đạt 7,424524 ± 0,293599 m theo cùng chế độ refit và cùng 1.111 mẫu Test. Mức giảm là 0,147141 m, tương ứng 1,9818%. Mục tiêu trung bình dưới 7 m chưa đạt.',bold=True)
para('Thay đổi thuật toán được chọn trong V4 là lấy mẫu cân bằng theo điểm tham chiếu trong huấn luyện. Kiến trúc hai khối KAN-Mixer B-spline, biểu diễn RSSI tương đối kết hợp mask, hàm mất mát ba thành phần và cách giải mã tọa độ được kế thừa từ V3. Số epoch refit tăng từ 31 lên 41 theo kết quả Validation đã khóa trước Test.')
h2('Cách đọc báo cáo')
para('Phần thuật toán đi từ RSSI đầu vào tới tọa độ đầu ra, kèm ký hiệu, công thức, kích thước tensor và ví dụ. Phần thực nghiệm tách Validation dùng chọn mô hình khỏi Test dùng xác nhận. Mọi bảng so sánh kết quả đều đối chiếu V4 với V3 same_point_relative; tài liệu IPS_KAN_V2_thuat_toan chỉ định hướng cách trình bày thuật toán.')
para('Dấu ± trong kết quả chính là độ lệch chuẩn mẫu giữa 10 sai số trung bình của 10 model. Các model cùng dùng một bộ 1.111 query; số lần dự đoán không làm tăng số đối tượng Test độc lập. Kết quả của seed thấp nhất không được chọn thay cho chỉ số 10 seed.')
source('UPDATE_VERSION_4.md; official_test_summary.json của V3 và V4; kiểm toán cuối 10 model.')

page('Mục lục')
field(doc.add_paragraph(),'TOC \\o "1-1" \\h \\z \\u')

page('1 Bài toán dữ liệu và ký hiệu')
para('Hệ thống định vị trong nhà sử dụng dấu vân tay Wi-Fi: một mẫu đo là vector cường độ RSSI của 520 điểm truy cập AP theo thứ tự WAP001 đến WAP520. Mô hình dự đoán hai tọa độ, tòa nhà và tầng. Một mạng dùng chung cho ba tòa; mỗi seed tạo một bộ trọng số riêng. [1, M1]')
table('Phạm vi dữ liệu dùng chung giữa V3 và V4',['Tập hoặc thành phần','Giá trị và vai trò'],[
('Train nội bộ','16.025 fingerprint; fit tọa độ và catalogue 872 điểm tham chiếu'),('Validation nội bộ','3.912 fingerprint; chọn checkpoint, thiết kế và tham số giải mã'),('Development sau khóa lựa chọn','19.937 fingerprint; refit mạng từ khởi tạo mới; catalogue 933 điểm'),('Test chính thức','1.111 fingerprint từ validationData.csv gốc; chỉ chấm sau khóa checkpoint'),('Không gian nhãn','13 cặp BUILDINGID và FLOOR hợp lệ; tọa độ LONGITUDE rồi LATITUDE')],[6.0,10.8])
table('Ký hiệu dùng trong công thức',['Ký hiệu','Ý nghĩa'],[('B, N','Kích thước batch và số lượt lấy mẫu trong một epoch'),('T, C','26 token và 64 kênh; không dùng T để chỉ nhiệt độ'),('R, cᵣ, pᵣ','Số điểm tham chiếu, lớp tòa–tầng và tọa độ chuẩn hóa của điểm r'),('mᵢ, zᵢ, rᵢ','Mặt nạ phát hiện AP, tín hiệu chuẩn hóa và tín hiệu tương đối'),('P, y, μ, s','Tọa độ gốc, tọa độ chuẩn hóa, tâm train và hệ số chung hai trục'),('aᵣ, ℓc, τ, k','Logit điểm, logit lớp, nhiệt độ và số điểm giữ trong giải mã')],[4.0,12.8])
para('Các nhóm đo không giao nhau giữa train và Validation không đồng nghĩa mọi tọa độ, thiết bị hoặc người đo đều mới. Trong Validation, 3.053 dòng có tọa độ đã xuất hiện trong train và 859 dòng có tọa độ chưa xuất hiện. Hai nhóm được theo dõi riêng trong bước chọn thiết kế.')
source('data/processed/uji_group_split_v1/manifest.json và các tệp indices; [M3].')

page('2 Chỉ số đánh giá và cách tổng hợp')
para('Sai số chính thức là khoảng cách Euclidean hai chiều giữa tọa độ dự đoán và tọa độ thật, sau khi đảo chuẩn hóa về hệ tọa độ gốc. Mọi dòng Test đều được giữ, gồm cả các trường hợp sai tòa, sai tầng hoặc có sai số lớn. [M1, M2]')
eq(5,'Δx và Δy là sai khác hai trục. Ví dụ Δx = 3 và Δy = 4 thì khoảng cách là 5 m. Mean2D khác với trung bình độ lệch tuyệt đối riêng từng trục.')
eq(6,'Mỗi seed tạo một Mean2D trên cùng 1.111 query. Sau đó mới lấy trung bình và độ lệch chuẩn giữa 10 Mean2D; không chọn seed thuận lợi.')
mml('<mover><mi>E</mi><mo>¯</mo></mover><mo>=</mo><mfrac><mn>1</mn><mn>10</mn></mfrac><munderover><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mn>10</mn></munderover><msub><mi>E</mi><mi>j</mi></msub><mo>;</mo><mi>SD</mi><mo>=</mo><msqrt><mfrac><mrow><munderover><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mn>10</mn></munderover><msup><mrow><mo>(</mo><msub><mi>E</mi><mi>j</mi></msub><mo>−</mo><mover><mi>E</mi><mo>¯</mo></mover><mo>)</mo></mrow><mn>2</mn></msup></mrow><mn>9</mn></mfrac></msqrt>','Eⱼ là Mean2D của model thứ j. Mẫu số 9 xác định sample SD. Chỉ số này đo biến thiên do huấn luyện trên bộ Test cố định, không phải độ lệch chuẩn khoảng cách của từng query hoặc khoảng tin cậy cho quần thể người dùng.')
table('Các chỉ số bổ sung',['Chỉ số','Cách hiểu'],[('Median2D','Phân vị 50% khoảng cách của từng model'),('P90 2D','Phân vị 90%; không phải sai số lớn nhất hoặc bảo đảm mọi mẫu'),('RMSE2D','Căn trung bình bình phương khoảng cách; nhạy với ngoại lai'),('Building accuracy','Tỷ lệ đúng tòa nhà'),('Floor accuracy','Tỷ lệ đúng tầng, xét riêng với tòa nhà'),('Joint accuracy','Tỷ lệ đúng đồng thời tòa nhà và tầng')],[4.4,12.4])
para('Mean2D ở đây không cộng một chiều cao giả định khi dự đoán sai tầng. Tên LONGITUDE và LATITUDE trong CSV được giữ, nhưng phép đo dùng trực tiếp hệ tọa độ phẳng gốc của benchmark; metadata chưa xác nhận một mã EPSG cụ thể.')

page('3 Chuẩn hóa RSSI và biểu diễn tương đối')
para('Giá trị RSSI bằng 100 biểu thị AP không được phát hiện. Mặt nạ được tạo trước biến đổi tín hiệu để phân biệt AP vắng với AP được phát hiện ở −104 dBm. Quy tắc tiền xử lý này giữ nguyên giữa V3 và V4. [M1]')
eq(1,'mᵢ nhận 1 khi AP i có phép đo, nhận 0 khi AP vắng.')
eq(2,'Phép cộng 104 và chia 104 đưa tín hiệu phát hiện về thang 0 đến 1. AP thiếu được đặt zᵢ = 0, mᵢ = 0; AP ở −104 dBm có zᵢ = 0 nhưng mᵢ = 1.')
mml('<mi>n</mi><mo>=</mo><mi>max</mi><mo>(</mo><mn>1</mn><mo>,</mo><munder><mo>∑</mo><mi>i</mi></munder><msub><mi>m</mi><mi>i</mi></msub><mo>)</mo><mo>;</mo><msub><mi>μ</mi><mi>z</mi></msub><mo>=</mo><mfrac><mrow><munder><mo>∑</mo><mi>i</mi></munder><msub><mi>m</mi><mi>i</mi></msub><msub><mi>z</mi><mi>i</mi></msub></mrow><mi>n</mi></mfrac>','Trung bình μz chỉ dùng các AP phát hiện trong chính fingerprint, không ước lượng bằng Validation hoặc Test.')
mml('<msub><mi>r</mi><mi>i</mi></msub><mo>=</mo><mfrac><mrow><msub><mi>m</mi><mi>i</mi></msub><mo>(</mo><msub><mi>z</mi><mi>i</mi></msub><mo>−</mo><msub><mi>μ</mi><mi>z</mi></msub><mo>)</mo></mrow><mn>0.2</mn></mfrac>','Mẫu số 0,2 tương đương 20,8 dB trên thang chuẩn hóa 104 dB. Mỗi AP được đưa vào mạng bằng cặp (rᵢ, mᵢ), tương ứng input_mode = relative_mask và mask_scale = 1.')
table('Ví dụ ba AP trước và sau lấy tín hiệu tương đối',['RSSI','z','m','r'],[('−52','0,50','1','+0,625'),('100','0','0','0'),('−78','0,25','1','−0,625')],[3.6,4.4,3.6,5.2])
para('Trong ví dụ, μz = 0,375. Nếu mọi AP phát hiện cùng dịch một mức và tập AP không đổi, phép trừ trung bình loại bỏ dịch chung. Tính chất này không bảo đảm mọi khác biệt giữa điện thoại đều được khử, vì thiết bị còn có thể thay đổi tập AP và độ lệch không đồng nhất theo AP. Mẫu không phát hiện AP nào vẫn được giữ; n được chặn dưới ở 1 và đầu vào trở thành toàn 0.')
source('data.py::feature_transform; models/reference.py::represent_signal.')

page('4 Chuẩn hóa tọa độ và danh mục tham chiếu')
para('Mạng học trên tọa độ đã chuẩn hóa nhưng báo cáo sai số trong hệ tọa độ gốc. Tâm μ và scale s được fit từ 16.025 dòng train nội bộ; cùng các giá trị ấy được giữ khi refit trên toàn bộ development. [M1, M3]')
eq(3,'Var dùng mẫu số N, tương ứng ddof = 0. Scale s dùng chung cho hai trục, nhờ đó phép biến đổi không làm méo tương quan khoảng cách giữa X và Y.')
eq(4,'P là tọa độ gốc và y là tọa độ chuẩn hóa. Đảo chuẩn hóa được thực hiện bằng float64 trước tính metric.')
table('Tham số tọa độ và catalogue',['Đại lượng','Giá trị'],[('Tâm X','−7.463,915495586015'),('Tâm Y','4.864.870,4533671485'),('Scale chung s','99,11502300196764'),('Catalogue train nội bộ','872 điểm theo bộ ba class_id, X, Y'),('Catalogue refit','933 điểm từ toàn bộ 19.937 dòng development'),('Tọa độ và nhãn tham chiếu','Buffer trong checkpoint; không phải tham số học bằng optimizer')],[6.1,10.7])
eq(21,'Công thức catalogue ở pha train nội bộ có R = 872. Sau khi khóa lựa chọn, refit dựng lại catalogue từ development nên R = 933. Các lần quét lặp cùng lớp và cùng tọa độ chỉ tạo một điểm; cùng X/Y nhưng khác tầng vẫn là các điểm riêng.')
para('Ví dụ minh họa với μ = (100,200), s = 10: P = (120,190) được đổi thành y = (2,−1). Nếu dự đoán chuẩn hóa là (1,5;−0,8), tọa độ khôi phục là (115,192). Khi dùng dữ liệu thật phải dùng μ và s trong bảng, không thay bằng các số minh họa.')
para('Catalogue lưu vị trí có nhãn của dữ liệu được phép huấn luyện. Trong suy luận của cấu hình hiện tại, mạng chấm điểm các mục catalogue rồi nội suy tọa độ; không tìm k láng giềng trong không gian RSSI của các hàng fingerprint.')

page('5 Thành phần KAN và các hàm B spline')
para('Thuật toán hiện tại là ReferenceKANMixer: KAN-Mixer với đầu ra điểm tham chiếu. Các nhánh trộn sử dụng backend efficient-kan B-spline. Ý tưởng KAN học hàm một biến trên các kết nối [2]; công thức thực thi dưới đây được đối chiếu trực tiếp với mã vendor của dự án. Các hệ số và lựa chọn IPS không phải cấu hình tái lập nguyên bản bài phân loại ảnh KAN-Mixers [3].')
eq(9,'Với đầu ra j, mỗi chiều vào i đóng góp một hàm φji. Nhánh cơ sở gồm base_weight nhân SiLU; nhánh spline gồm spline_weight và spline_scaler riêng. KANLinear đang dùng không có bias riêng.')
eq(10,'SiLU là hàm cố định. Ví dụ SiLU(0) = 0 và SiLU(1) xấp xỉ 0,731; optimizer học các hệ số quanh hàm này chứ không học công thức SiLU.')
eq(11,'Ở bậc 0, basis là chỉ báo một khoảng giữa hai knot. Knot là các mốc chia trục đầu vào của từng chiều.')
eq(12,'Truy hồi được áp dụng ở d = 1, 2 và 3. Mỗi basis mới là tổ hợp của hai basis lân cận ở bậc trước, với hệ số phụ thuộc x. Sau ba bước ta có spline bậc ba dùng trong KANLinear.')
para('Với một lớp I → O, grid_size = 5 và spline_order = 3 tạo 8 hệ số spline trên mỗi kết nối. Khi bật standalone scale, tổng tham số học là OI trọng số cơ sở + 8OI hệ số spline + OI scaler = 10OI. Grid là buffer cố định; không được cập nhật trong các lượt chính.')
source('models/vendor/efficient_kan.py::KANLinear; cấu hình kan trong phụ lục A.')

page('6 Grid LayerNorm và hai loại Dropout')
eq(13,'Khoảng nội [−3,3] được chia thành năm đoạn, bước knot 1,2. Ba mốc được mở rộng về mỗi phía; 12 knot trải từ −6,6 đến 6,6, tạo tám basis cuối trên mỗi chiều.')
para('grid_range không phải phép cắt mọi activation về [−3,3]. Spline có miền knot mở rộng và nhánh SiLU vẫn hoạt động. update_grid = false và hệ số regularization_loss bổ sung của vendor bằng 0; không được gán các cơ chế ấy là thành phần mới của V4.')
eq(14,'LayerNorm tính trung bình và phương sai trên 64 channel của chính token đang xét. Đây là thống kê của activation trong một mẫu, khác với μ và s của tọa độ.')
eq(15,'γ và β là hai vector học được. Mặc định ε = 10⁻⁵. Mỗi LayerNorm có bộ tham số riêng; các block không chia sẻ bộ tham số chuẩn hóa với nhau.')
eq(16,'Dropout bên trong mạng có p = 0,1. Khi train, phần được giữ chia cho 0,9 để bảo toàn kỳ vọng; khi eval, phép biến đổi là đồng nhất.')
para('AP dropout là thao tác khác: AP được bỏ ngay ở signal và mask với xác suất 0,05; các AP còn lại không chia cho 0,95. Các phép nhiễu đầu vào và Dropout trong mạng đều tắt khi model.eval(). LayerNorm tiếp tục hoạt động vì nó là phép biến đổi của mạng.')
para('Ví dụ hai channel (1,3) có trung bình 2 và phương sai 1. Với γ = 1, β = 0, bỏ qua ε để tính nhẩm, LayerNorm cho (−1,1). Trong mạng thật phép chuẩn hóa dùng đủ 64 channel của mỗi token.')

page('7 Luồng tensor và sơ đồ suy luận')
para('Mỗi fingerprint được đưa vào dưới dạng tensor [520,2]. Sau biểu diễn tương đối, 20 AP liên tiếp tạo một token 40 số. Thứ tự AP vẫn là thứ tự nguồn; cấu hình được chọn không bật nhóm co-observation. [M1]')
figure('architecture.png','Luồng suy luận của KAN Mixer tham chiếu đang sử dụng',width=5.9)
para('B là số mẫu trong batch; R = 872 khi chọn mô hình bằng Validation và R = 933 ở model refit cuối. Phép reshape không học tham số. Mạng nhận RSSI và mask; nhãn tòa–tầng thật của query chỉ dùng bên ngoài forward để đánh giá, không dùng để định tuyến suy luận.')

page('8 Embedding và hai khối trộn KAN')
eq(7,'Embedding Linear 40 → 64 dùng chung một bộ W và bias β cho mọi token. Tensor [B,26,40] được đổi thành [B,26,64]. Việc nhóm AP liên tiếp là cách chia đầu vào, không có nghĩa các AP trong nhóm ở gần nhau về vật lý.')
eq(17,'LN chuẩn hóa channel, transpose đổi tensor thành [B,64,26], rồi nhánh K_T trộn 26 token qua chuỗi KANLinear 26 → 32 → 26. Kết quả được chuyển vị lại rồi cộng residual vào đầu vào H.')
eq(18,'Nhánh K_C tác động trên trục cuối của [B,26,64], qua chuỗi KANLinear 64 → 128 → 64. Cùng phép biến đổi channel được dùng ở mọi token.')
table('Kích thước tensor trong một block',['Thành phần','Đầu vào','Trung gian','Đầu ra'],[('Trộn token','[B,64,26]','[B,64,32]','[B,64,26]'),('Cộng residual token','[B,26,64]','—','[B,26,64]'),('Trộn channel','[B,26,64]','[B,26,128]','[B,26,64]'),('Cộng residual channel','[B,26,64]','—','[B,26,64]')],[4.6,4.0,4.1,4.1])
para('Mỗi nhánh có hai KANLinear và Dropout sau từng lớp; không chèn GELU giữa hai KANLinear vì nhánh SiLU đã ở bên trong KAN. Mỗi block có bốn KANLinear; hai block tổng cộng tám lớp. Block thứ hai có trọng số riêng. Residual giữ đường truyền trực tiếp: đầu vào 0,4 cộng phần cập nhật −0,1 cho đầu ra 0,3.')
source('models/networks.py::mixing_branch và MixerBlock.forward.')

page('9 Phần đầu ra và phân loại tòa tầng')
eq(19,'Sau final LayerNorm, flatten ghép 26 × 64 thành 1.664 số. Head là Linear 1664 → 256, GELU, Dropout 0,1, Linear 256 → R. Head dùng Linear thông thường, không phải head KAN hiệu chỉnh tọa độ.')
eq(20,'GELU dùng hàm phân phối Gaussian chuẩn Φ. Nó không có tập tham số học riêng; các lớp Linear ở hai phía mới có trọng số và bias.')
table('Tham số học của cấu hình được chọn',['Thành phần','Train nội bộ','Refit cuối'],[('Embedding','2.624','2.624'),('Hai MixerBlock và LayerNorm','361.472','361.472'),('Final LayerNorm','128','128'),('Head Linear','650.344','666.021'),('Tổng tham số','1.014.568','1.030.245')],[8.1,4.35,4.35])
para('Chênh 15.677 tham số khi refit bằng 61 × (256 + 1), vì catalogue tăng từ 872 lên 933 điểm. Số lớp KAN và cấu trúc trộn giữ nguyên. Catalogue tọa độ và nhãn là buffer, không cộng vào số tham số học.')
eq(22,'Logit lớp ℓc là logsumexp các điểm thuộc lớp c. Chọn lớp có logit lớn nhất rồi tra ánh xạ lớp để lấy building và floor. Phép này không chia nhiệt độ τ và không có classifier tòa/tầng riêng trong cấu hình được chọn.')
para('Ví dụ lớp A có score 1 và 0: ℓA = log(e + 1) xấp xỉ 1,313. Lớp B có một score 1,2 nên ℓB = 1,2. Mạng chọn A nhờ tổng bằng chứng của nhiều điểm, dù B có một điểm riêng lớn hơn từng điểm của A.')

page('10 Giải mã tọa độ bằng top k và nhiệt độ')
para('Trong mỗi lớp tòa–tầng, decoder giữ những điểm tham chiếu có score cao nhất. Đây là top k theo score do mạng học, không phải k fingerprint gần query nhất theo khoảng cách RSSI. [M1]')
eq(23,'S_c là các điểm giữ lại của lớp c. Softmax tạo trọng số không âm, tổng bằng 1. τ nhỏ làm phân phối tập trung hơn; τ lớn làm phân phối phẳng hơn.')
eq(24,'Tọa độ từng lớp là trung bình có trọng số các tọa độ tham chiếu. Đầu ra lấy nhánh lớp do mạng dự đoán, rồi nhân s và cộng μ. Cấu hình hiện tại không cộng residual tọa độ, không kết hợp dự đoán từ một mạng thứ hai.')
para('Ví dụ hai điểm (0,0) và (10,0) có score 0 và log(3). Với τ = 1, trọng số là 1/4 và 3/4 nên tọa độ là (7,5;0). Với τ = 0,5, trọng số thành 1/10 và 9/10 nên tọa độ là (9;0). Thay τ có thể đổi tọa độ dù trọng số mạng giữ nguyên.')
table('Các lựa chọn giải mã và phạm vi sử dụng',['Thiết lập','Giá trị hoặc quy tắc'],[('Lưới k trên Validation','1, 3, 5, 10, 20, 1000'),('Lưới nhiệt độ τ','0,5; 1; 2; 4'),('Số tổ hợp hiệu chỉnh','24 tổ hợp; chọn riêng bằng Validation của từng seed'),('k và τ trong JSON gốc','5 và 1,0; chưa phải giá trị giải mã cuối sau calibration'),('k = 1000','Giữ mọi điểm hợp lệ của lớp, không tạo thêm điểm'),('Số lượt suy luận','Một forward pass của một mạng cho mỗi mẫu')],[6.4,10.4])
para('Mã đệm catalogue mỗi lớp đến M bằng số điểm lớn nhất của một lớp, rồi thực hiện min(k,M). Slot đệm mang score −∞ nên trọng số bằng 0. Dự đoán là tổ hợp lồi của các điểm được giữ; đó là ràng buộc của decoder hiện tại.')

page('11 Hàm mất mát ba thành phần')
para('V3 và cấu hình V4 được chọn cùng tối ưu ba mục tiêu: phân loại lớp, phân bố điểm tham chiếu và tọa độ. Các nhãn chỉ được dùng trong loss của dữ liệu được phép train. [M1]')
eq(27,'Nhãn mềm có trọng số Gaussian theo khoảng cách chuẩn hóa tới nhãn thật, chỉ trên các điểm cùng lớp. σ = 0,05 tương đương độ rộng khoảng 4,955751 m sau đổi thang; không phải bán kính cắt cứng.')
eq(28,'q là phân bố mục tiêu chuẩn hóa. Xác suất π của mạng được tính bằng softmax trên toàn bộ R điểm, nên score sai lớp vẫn cạnh tranh trong mẫu số. label_smoothing = 0 không tắt nhãn mềm không gian này.')
eq(29,'L_class là cross entropy trên 13 logits lớp, khuyến khích tòa–tầng thật có xác suất cao.')
eq(25,'Tọa độ dense của mỗi lớp dùng toàn bộ điểm hợp lệ với nhiệt độ huấn luyện τtr = 1. Đây là đường dùng để tính loss, khác với top k dùng xuất dự đoán.')
eq(26,'Loss chọn tọa độ dense của lớp thật trong batch train. Số 10⁻⁸ làm trơn khoảng cách tại 0, được áp dụng trong hệ tọa độ chuẩn hóa.')
eq(30,'Ví dụ L_class = 0,2, L_ref = 1 và L_xy = 0,05 cho L = 1,25. Hệ số lambda điều chỉnh gradient, không phải phần trăm mức quan trọng hoặc tỷ lệ đúng của nhiệm vụ.')
para('smooth_l1_beta = 1 có trong schema nhưng không được dùng ở nhánh reference. Cấu hình được chọn không bật auxiliary classifier, loss nhất quán, spatial-risk bổ sung hoặc head residual. train_class_loss trong log là L_class + L_ref; train_loss mới là tổng có các hệ số ở công thức trên.')

page('12 Thuật toán lấy mẫu cân bằng điểm tham chiếu')
para('Đây là thay đổi thuật toán được chọn trong V4. Một điểm được xác định bằng class_id và đúng tọa độ X/Y gốc. Train có từ 2 đến 80 fingerprint tại một điểm; lấy đều theo hàng khiến các vị trí đo nhiều xuất hiện nhiều hơn trong gradient. V4 giảm sự lệch này bằng cách đổi xác suất chọn hàng, còn loss của từng batch giữ nguyên. [M1]')
mml('<msub><mi>w</mi><mi>i</mi></msub><mo>=</mo><mfrac><mn>1</mn><msub><mi>n</mi><mrow><mi>g</mi><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msub></mfrac><mo>;</mo><mi>P</mi><mo>(</mo><mi>i</mi><mo>)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mi>R</mi><msub><mi>n</mi><mrow><mi>g</mi><mo>(</mo><mi>i</mi><mo>)</mo></mrow></msub></mrow></mfrac><mo>;</mo><mi>P</mi><mo>(</mo><mi>g</mi><mo>=</mo><mi>r</mi><mo>)</mo><mo>=</mo><mfrac><mn>1</mn><mi>R</mi></mfrac>','g(i) là điểm của hàng i, n_g(i) là số hàng tại điểm đó. Tổng trọng số của mỗi điểm bằng 1, nên sau chuẩn hóa mỗi điểm có xác suất 1/R.')
figure('sampling_example.png','Ví dụ minh họa cân bằng hai vị trí có số lần quét khác nhau',width=6.1)
para('Ví dụ này có 82 hàng, gồm điểm A đo 2 lần và điểm B đo 80 lần. V3 cho tổng xác suất khoảng 2,44% tại A và 97,56% tại B; V4 cho mỗi điểm 50%. Đây là ví dụ minh họa cơ chế, không phải thống kê thực tế của hai điểm cụ thể trong UJI.')
para('WeightedRandomSampler lấy có hoàn lại và vẫn tạo N lượt/epoch: 16.025 ở train nội bộ, 19.937 ở refit. Một hàng có thể lặp hoặc vắng trong epoch; không được hiểu là mỗi hàng được duyệt đúng một lần. V3 dùng shuffle, tương ứng xáo trộn các hàng không hoàn lại trong epoch.')

page('13 Mục tiêu kỳ vọng và hiện thực lấy mẫu')
mml('<mi>E</mi><mo>[</mo><mi>L</mi><mo>]</mo><mo>=</mo><mfrac><mn>1</mn><mi>R</mi></mfrac><munderover><mo>∑</mo><mrow><mi>r</mi><mo>=</mo><mn>1</mn></mrow><mi>R</mi></munderover><mfrac><mn>1</mn><msub><mi>n</mi><mi>r</mi></msub></mfrac><munder><mo>∑</mo><mrow><mi>i</mi><mo>∈</mo><mi>r</mi></mrow></munder><msub><mi>L</mi><mi>i</mi></msub>','Theo kỳ vọng của sampler, mỗi điểm góp trọng số ngang nhau, rồi các fingerprint trong điểm được lấy đều. Kỳ vọng còn bao gồm ngẫu nhiên của augmentation trong các lượt train thực tế.')
para('Đây là thay đổi phân phối rủi ro huấn luyện từ đều fingerprint sang đều vị trí, không phải cộng thêm một loss trọng số vào mỗi batch. Cân bằng theo điểm không tự làm đều các lớp tòa–tầng: lớp có nhiều điểm tham chiếu vẫn có nhiều tổng xác suất hơn lớp có ít điểm.')
h2('Giả mã một epoch')
for text in ['1. Nhận đúng tập train nội bộ hoặc development đã được cho phép refit.',
 '2. Gom các hàng theo bộ ba class_id, X gốc, Y gốc; đếm nᵣ.',
 '3. Gán mỗi hàng i trọng số 1/n_g(i).',
 '4. Khởi tạo WeightedRandomSampler với num_samples = N, replacement = true và generator theo seed.',
 '5. Với mỗi batch tối đa 128 hàng, tạo same-point mixing nếu được chọn, rồi nhiễu/AP dropout trong forward.',
 '6. Tính ba loss như V3; backward, clip gradient và AdamW; ghi đầy đủ số bước.']:
    para(text)
table('Khác biệt trong trình nạp dữ liệu',['Thuộc tính','V3','V4'],[('Đơn vị có xác suất bằng nhau','Fingerprint','Điểm tham chiếu'),('Cơ chế','Shuffle','WeightedRandomSampler'),('Lấy hoàn lại trong epoch','Không','Có'),('Số lượt lấy mỗi epoch','N','N'),('Bước train nội bộ/epoch','126','126'),('Bước refit/epoch','156','156'),('Dùng Validation/Test để đếm điểm','Không','Không')],[6.5,5.15,5.15])
para('Bản refit tính lại tần suất trên development vì có thêm dữ liệu được phép học sau khóa lựa chọn. Nó không lấy tần suất từ Test và không thay tọa độ của bất kỳ query nào.')
source('v4_methods.py::point_sampler; train.py và final_refit.py khi tạo DataLoader.')

page('14 Tăng cường dữ liệu được kế thừa từ V3')
para('Same-point mixing có xác suất 0,5. Nếu có một fingerprint khác cùng lớp và đúng cùng tọa độ gốc, chọn đều một đối tác trong các hàng hợp lệ; alpha lấy đều từ 0,2 đến 0,8. Tọa độ và lớp đích giữ nguyên. Chính sách được chọn là any_same_point, không ưu tiên điện thoại khác. [M1]')
para('Ở AP nghe được từ cả hai nguồn, tín hiệu là alpha nhân nguồn đầu cộng (1−alpha) nhân nguồn đối tác. Ở AP chỉ có ở một nguồn, mã chọn trọn cặp signal và mask từ một nguồn bằng Bernoulli(alpha). Nhờ vậy mask vẫn là 0/1. Nếu không có đối tác, mẫu được giữ nguyên.')
para('Ví dụ z₁ = 0,5, z₂ = 0,7, alpha = 0,25 và hai mask bằng 1 cho tín hiệu trộn 0,65. Nếu chỉ nguồn đầu nghe AP, mẫu trộn nghe AP với xác suất 0,25. Các đối tác đều thuộc tập đang được phép train; không lấy từ Validation hoặc Test.')
eq(31,'Nhiễu riêng từng AP có độ lệch chuẩn 2 dB; dịch chung theo fingerprint có độ lệch chuẩn 3 dB. Cả hai chia 104 để chuyển về thang z.')
eq(32,'AP dropout giữ mỗi AP đang nghe với xác suất 0,95. Clip về [0,1] và nhân mask sau cùng làm AP vắng không bị nhiễu biến thành AP nghe được. Sau đó mới tính RSSI tương đối từ signal và mask đã tăng cường.')
para('Các thao tác này chỉ dùng khi model.train(). Khi model.eval(), same-point mixing, noise, shift và Dropout đều tắt; chỉ tiền xử lý xác định và một lần forward được thực hiện. V4 được chọn không dùng nearby-point mixing, phối hợp nhiều view hoặc test-time augmentation.')
source('augmentation.py::SamePointMixing; models/reference.py::encode_features.')

page('15 Cập nhật tham số và lịch học')
para('Cả V3 và V4 dùng AdamW, learning rate ban đầu 0,001, weight decay 0,0001, batch size 128 và clip gradient L2 ở 1. Các tham số mặc định AdamW là β₁ = 0,9, β₂ = 0,999 và ε = 10⁻⁸. [M1]')
eq(33,'Clip được tính trên norm chung của gradient mọi tham số, rồi thu nhỏ đồng thời khi cần. Catalogue, AP order và grid cố định không phải đối tượng được optimizer học.')
eq(34,'m và v là các bộ nhớ trung bình có quên của gradient và bình phương gradient. Chỉ số t ở đây là bước optimizer, không phải token.')
eq(35,'Hiệu chỉnh lệch ban đầu bù việc các bộ nhớ bắt đầu tại 0.')
eq(36,'Weight decay được tách khỏi gradient của loss. Mã dùng một nhóm model.parameters(), không tách bias hoặc LayerNorm thành nhóm loại trừ decay.')
eq(37,'η₀ = 0,001, ηmin = 0,00005; e là số lần scheduler đã step. Epoch đầu dùng e = 0 và scheduler.step() gọi cuối epoch. Refit 41 epoch vẫn giữ horizon 100, không nén lịch cosine xuống 41.')
para('Trong pha chọn model, tối đa 100 epoch; chỉ lưu best khi Mean2D Validation nhỏ hơn best trước ít nhất 0,01 m. Nếu không cải thiện đủ trong 20 epoch liên tiếp thì dừng sớm. Các điểm Validation được tính ở chế độ eval, trong hệ tọa độ gốc. Cơ chế chọn best và calibration k/τ là hai bước khác nhau.')

page('16 Cấu hình kiến trúc và hàm mất mát đầy đủ')
para('Bảng dưới là các giá trị thực của v4_balanced_points.json; chúng trùng với cấu hình V3 được chọn, ngoại trừ sampling được mô tả ở chương 13. Các tham số giữ nguyên vẫn được liệt kê để tái lập được toàn bộ mạng.')
table('Kiến trúc và backend',['Nhóm tham số','Giá trị'],[('model và positioning.variant','kan_mixer; reference'),('Số token; đặc trưng/token','26; 40'),('Channel; số block','64; 2'),('Token hidden; channel hidden','32; 128'),('Dropout trong mạng','0,1'),('Pooling; head_width','flatten; 256'),('Head transform','linear_gelu mặc định; không có head residual'),('Basis; grid_size; spline_order','B_spline; 5; 3'),('Base activation','SiLU'),('Grid range; update_grid','[−3,3]; false'),('Standalone scale spline','true'),('scale_noise; scale_base; scale_spline','0,1; 1; 1'),('grid_eps','0,02; không kích hoạt cập nhật grid'),('Extra KAN regularization coefficient','0')],[8.0,8.8])
table('Loss và biểu diễn tín hiệu',['Tham số','Giá trị'],[('lambda_class; lambda_reference; lambda_xy','0,5; 1; 3'),('reference_sigma_normalized','0,05'),('label_smoothing','0'),('coordinate_source','dense mặc định'),('smooth_l1_beta','1 trong schema; không dùng cho nhánh reference'),('input_mode; mask_scale','relative_mask; 1'),('rssi_noise_db; device_shift_db; ap_dropout','2; 3; 0,05')],[8.0,8.8])

page('17 Cấu hình huấn luyện và các tùy chọn thực sự bật')
table('Huấn luyện của phương án đạt 7,2774 m',['Tham số','Giá trị và phạm vi'],[('split_dir','data/processed/uji_group_split_v1'),('training.sampling','reference_points; thay đổi được chọn trong V4'),('same_point_mixing','0,5; kế thừa V3'),('learning_rate; weight_decay','0,001; 0,0001'),('batch_size; threads','128; 2'),('max_epochs','100 trong pha Validation và horizon cosine'),('patience; min_delta','20 epoch; 0,01 m'),('gradient_clip_l2','1'),('scheduler; learning rate sàn','cosine; 0,00005'),('kan_diagnostic_batches','0'),('Refit','41 epoch, 19.937 lượt lấy mẫu mỗi epoch'),('Số bước refit/model','41 × 156 = 6.396'),('Seed cố định','42, 43, 44, 100, 2024, 777, 999, 45, 46, 47'),('Thiết bị thực nghiệm','CUDA; RTX 4060 Ti 8 GB'),('k; nhiệt độ trong JSON gốc','5; 1; hiệu chỉnh bằng Validation từng seed trước refit')],[7.2,9.6])
para('Các tùy chọn không có trong JSON được hiểu theo mặc định thực thi: source_order cho AP grouping; auxiliary classifier tắt; coordinate_residual_scale = 0; same_point_consistency_weight = 0; EMA tắt; lambda_spatial_risk = 0; không dùng inference_shifts_db. Chúng không được cộng dồn thành gói cải tiến đạt 7,2774 m chỉ vì mã có hỗ trợ.')
para('Trong refit, training_temperature = 1 và training_top_k = 5 được khôi phục từ cấu hình huấn luyện gốc, còn top_k và temperature cho eval giữ kết quả calibration của từng seed. Loss tọa độ hiện dùng dense nên không phụ thuộc training_top_k.')

page('18 Thiết kế nghiên cứu và năm hướng sàng lọc')
para('Vòng V4 có số giả thuyết hữu hạn, mỗi thay đổi được đối chiếu với cùng V3 và cùng split. Sàng lọc dùng seed 42; baseline Validation tương ứng là 5,582541 m. Các giá trị trong bảng là Validation, không phải Test. [M2, M3]')
table('Các thiết kế đăng ký trước kết quả',['Thiết kế','Thành phần thay đổi','Validation seed 42','Quyết định'],[
('ap_tokens','520 token × 16 kênh; head 128','Không hoàn tất hợp lệ','Dừng sau 11 epoch không học hiệu quả'),('coobservation_patches','Sắp AP bằng đồng xuất hiện trên train','5,801703 m','Loại'),('joint_spatial_risk','Thêm kỳ vọng khoảng cách mọi RP vào loss','5,718381 m','Loại'),('balanced_points','Lấy mẫu đều điểm tham chiếu','5,542652 m','Xác nhận nhiều seed'),('absolute_relative_context','Thêm RSSI tuyệt đối bên cạnh tương đối và mask','5,809118 m','Loại')],[4.1,5.7,3.4,3.6])
para('Ở lượt AP-token, chẩn đoán trên train ghi nhận activation trước GELU ở head âm rất lớn và đầu ra GELU bằng 0 trong float32, làm logits gần như không đổi theo query. Lượt này được giữ như một thử nghiệm dở do tính khả thi; không coi sai số chưa calibration khoảng 148,50 m là điểm accuracy cùng ngân sách với các lượt hoàn tất.')
para('Nhóm AP co-observation làm tăng tương đồng mask trong patch nhưng không cải thiện sai số. Spatial-risk loss cải thiện một số chỉ số phân loại nhưng làm Mean2D Validation xấu hơn. Cấu hình cuối không bật ba cơ chế đó hoặc phần tín hiệu tuyệt đối bổ sung.')
para('Sau sàng lọc, tối đa hai ứng viên cải thiện full Validation ít nhất 0,01 m được xác nhận trên seed 42/43/44. Ứng viên phải cải thiện cả mean toàn Validation và score cân bằng hai nhóm tọa độ so với V3 cùng ba seed. balanced_points là phương án vượt các cổng này.')

page('19 Xác nhận Validation trên đủ mười seed')
para('V4 đạt 5,585354 ± 0,038662 m trên đủ 10 lượt Validation; V3 cùng seed đạt 5,694870 ± 0,068928 m. Mức giảm trung bình là 0,109516 m; 9/10 cặp cải thiện. Seed 46 hơi xấu hơn và vẫn được giữ trong thống kê.')
table('Validation và checkpoint được chọn',['Seed','V3 Mean2D','V4 Mean2D','Best V3','Best V4','Epoch V4 đã chạy'],[(r['seed'],fmt(r['V3_mean_validation'],6),fmt(r['mean_2d'],6),r['V3_best_epoch'],r['best_epoch'],r['epochs_completed']) for r in D['selected']],[1.5,3.1,3.1,2.5,2.5,4.1])
sl=D['validation_slices']['slices']
table('Mean2D Validation theo nhóm tọa độ',['Nhóm','Dòng/model','V3','V4'],[(label,sl[key]['rows_per_seed'],fmt(sl[key]['V3'],6),fmt(sl[key]['V4'],6)) for key,label in [('seen_train_coordinate','Đã có trong train'),('unseen_train_coordinate','Chưa có trong train'),('all','Toàn bộ')]], [6.2,3.2,3.7,3.7])
para('Score cân bằng bằng trung bình hai Mean2D của nhóm đã thấy và chưa thấy, với trọng số mỗi nhóm 1/2. Nó dùng chọn thiết kế, còn chỉ số báo cáo Test vẫn là mean trên mọi query. Không đổi trọng số Test sang theo điểm tham chiếu để làm nhỏ kết quả.')
source('confirmation_three.json; selection_lock.json; paired_validation_ten_20261026T124406Z.json.')

page('20 Khóa lựa chọn và refit trên development')
para('Sau xác nhận đủ 10 seed, mean Validation phải tốt hơn V3 ít nhất 0,01 m mới được phép refit. Danh sách best epoch V4 theo thứ tự đăng ký là 37, 16, 40, 56, 27, 44, 69, 35, 47, 42. Hai phần tử giữa sau sắp xếp là 40 và 42; median là 41. Cả 10 refit vì thế dùng 41 epoch.')
table('Giải mã đã khóa trước Test',['Seed','V3 k','V3 τ','V4 k','V4 τ'],[(r['seed'],r['V3_top_k'],fmt(r['V3_temperature'],1),r['top_k'],fmt(r['temperature'],1)) for r in D['selected']],[2.4,3.6,3.6,3.6,3.6])
para('Refit khởi tạo trọng số mới từ seed tương ứng và train trên 19.937 fingerprint. Nó không fine-tune trọng số tốt nhất của pha Validation. Catalogue tăng lên 933 điểm; tâm và scale tọa độ giữ từ train nội bộ. Same-point mixing và sampler tính trên development được phép học.')
para('Trong refit không chấm lại Validation cũ để chọn checkpoint vì các mẫu ấy đã trở thành dữ liệu train. Không chấm Test giữa các epoch. Kết thúc 41 epoch là checkpoint cố định của lượt; sau khi đủ 10 checkpoint, test_checkpoint_lock.json ghi hash cả bộ trước mọi suy luận Test.')
para('V3 dùng 31 epoch theo quy tắc khóa riêng của vòng trước; V4 dùng 41 epoch theo Validation V4. So sánh cuối là so sánh hai pipeline đã chọn bằng Validation, chưa cô lập tác động trực tiếp của sampler khỏi tác động gián tiếp qua số epoch refit.')

page('21 Đối chiếu chi tiết thuật toán V3 và V4')
table('Các thành phần của hai phương án báo cáo',['Thành phần','V3 same_point_relative','V4 balanced_points'],[
('Dữ liệu và split','16.025 / 3.912 / 1.111','Giữ nguyên'),('Mạng chung','3 tòa; 13 lớp','Giữ nguyên'),('RSSI và mask','relative_mask; mask_scale 1','Giữ nguyên'),('Tám KANLinear B-spline','2 block; grid 5; bậc 3','Giữ nguyên'),('Token và channel','26 × 40 đầu vào; 64 kênh','Giữ nguyên'),('Flatten và head','1664 → 256 → R','Giữ nguyên'),('Loss','0,5 L_class + L_ref + 3 L_xy','Giữ nguyên'),('Same-point mixing','Xác suất 0,5','Giữ nguyên'),('Noise / shift / AP dropout','2 dB / 3 dB / 0,05','Giữ nguyên'),('Phân phối lấy mẫu train','Đều fingerprint, shuffle','Đều điểm, có hoàn lại'),('Batch / bước mỗi epoch','128 / 126 hoặc 156','Giữ nguyên'),('Lựa chọn k và τ','Validation từng seed','Cùng lưới; kết quả từng seed thay đổi'),('Refit development','31 epoch','41 epoch từ median best epoch'),('Số tham số refit','1.030.245','Giữ nguyên'),('Suy luận','Một mạng và decoder tham chiếu','Giữ nguyên'),('Kết hợp kNN hoặc ensemble','Không trong phương án này','Không trong phương án này')],[5.3,5.75,5.75])
para('Diff cấu trúc của hai JSON đầu vào chỉ có training.sampling: V3 không khai báo, mặc định rows; V4 đặt reference_points. Epoch refit và các decoder cuối là kết quả của bước lựa chọn sau train, nên phải đối chiếu thêm selection lock và cấu hình checkpoint, không chỉ nhìn JSON đầu vào.')
para('V4 có thay đổi thuật toán huấn luyện thông qua sampler. Không có bằng chứng một kiến trúc KAN-Mixer mới tạo ra mức 7,2774 m. Các mở rộng kiến trúc hoặc loss tồn tại trong mã dưới dạng tùy chọn thực nghiệm nhưng không hoạt động ở model được chọn.')

page('22 Kết quả Test tổng hợp so với V3')
old=D['V3'];new=D['V4']['aggregate']
metrics=[('Mean2D','mean_2d','m',1),('Median2D','median_2d','m',1),('P90 2D','p90_2d','m',1),('RMSE2D','rmse_2d','m',1),('Đúng tòa','building_accuracy','%',100),('Đúng tầng','floor_accuracy','%',100),('Đúng đồng thời','joint_accuracy','%',100)]
table('Trung bình và sample SD của 10 model',['Chỉ số','V3','V4'],[(name+' ('+unit+')',fmt(old[key+'_mean']*scale,4)+' ± '+fmt(old[key+'_sample_std']*scale,4),fmt(new[key]['mean']*scale,4)+' ± '+fmt(new[key]['sample_std']*scale,4)) for name,key,unit,scale in metrics],[5.0,5.9,5.9])
para('Mean2D giảm 0,147141 m, tương ứng 1,9818%, và sample SD giữa model giảm. Trung bình median và RMSE cũng giảm. Tuy nhiên P90 tăng từ 15,562094 lên 15,742108 m; kết quả không cho phép kết luận mọi vùng của phân bố sai số đều tốt hơn.')
figure('test_seeds.png','Mean2D Test của V3 và V4 theo toàn bộ seed đã đăng ký')
para('Đường 7 m là ngưỡng mục tiêu cho trung bình 10 seed. Việc một model đi xuống dưới đường này chưa đủ làm cả nhóm đạt mục tiêu. Không dùng kết quả Test để thay epoch, k, τ hoặc lựa chọn seed.')

page('23 Kết quả Test từng seed và phân bố sai số')
newrows=D['V4']['per_seed'];oldrows=D['V3_per_seed']
table('Mean2D Test trên 1.111 query của từng model',['Seed','V3 (m)','V4 (m)','V4 trừ V3 (m)'],[(r['seed'],fmt(oldrows[str(r['seed'])]['mean_2d'],6),fmt(r['mean_2d'],6),('+' if r['mean_2d']>=oldrows[str(r['seed'])]['mean_2d'] else '')+fmt(r['mean_2d']-oldrows[str(r['seed'])]['mean_2d'],6)) for r in newrows],[2.4,4.6,4.6,5.2])
para('6/10 cặp cải thiện Test. Seed 47 đạt 6,940479 m, là minimum mô tả sau Test. Seed 42 đã được cố định làm model minh họa/triển khai trước Test đạt 7,499968 m; không thay nó bằng seed 47 sau khi xem điểm số.')
figure('test_cdf.png','Trung bình các đường phân phối tích lũy sai số của 10 model',width=6.05)
para('Ở mỗi ngưỡng khoảng cách, tính tỷ lệ query không vượt ngưỡng cho từng model rồi lấy trung bình 10 tỷ lệ. Đây là mô tả phân bố trên cùng cohort, không phải ensemble tọa độ hoặc một bộ 11.110 người dùng độc lập.')

page('24 Sai số Test theo tòa nhà và tầng')
para('Các nhóm dưới đây được lập từ nhãn thật để phân tích sau đánh giá, không dùng nhãn thật định tuyến suy luận. Mỗi giá trị là trung bình 10 Mean2D của nhóm tương ứng; giữ đủ tất cả mẫu của nhóm.')
table('Test theo tòa nhà',['Tòa','Dòng/model','V3 (m)','V4 (m)'],[(r['label'],r['n'],fmt(r['V3'],4),fmt(r['V4'],4)) for r in D['by_building']],[2.5,4.3,5,5])
table('Test theo 13 cặp tòa và tầng',['Tòa/tầng','Dòng/model','V3 (m)','V4 (m)','Chênh lệch (m)'],[(r['label'],r['n'],fmt(r['V3'],4),fmt(r['V4'],4),('+' if r['V4']>=r['V3'] else '')+fmt(r['V4']-r['V3'],4)) for r in D['by_class']],[3.0,3.5,3.4,3.4,3.5])
para('Cả ba tòa đều giảm mean ở cấp tòa. Ở cấp tòa–tầng, các nhóm 0/1, 1/1, 2/0 và 2/1 tăng sai số. Nhóm 2/4 vẫn khó, dù giảm từ khoảng 13,58 xuống 12,55 m. Vì vậy, cải thiện tổng thể không đồng nghĩa mỗi tầng được cải thiện.')
source('Tính trực tiếp từ 20 CSV predictions.csv của V3/V4; evidence.json lưu hash đầu vào.')

page('25 Số AP và độ bất định của cải thiện')
figure('test_ap_groups.png','Mean2D Test theo số AP phát hiện và số mẫu của từng nhóm')
table('Các nhóm AP trên Test',['AP phát hiện','Dòng/model','V3 (m)','V4 (m)'],[(r['label'],r['n'],fmt(r['V3'],4),fmt(r['V4'],4)) for r in D['by_ap']],[4.1,3.7,4.5,4.5])
para('Không có dòng Test không phát hiện AP. Nhóm 1–9 AP và nhóm từ 30 AP có sai số lớn; số AP nhiều không tự bảo đảm định vị tốt. Các nhóm khác nhau còn có thể khác khu vực, thiết bị và nhiễu, nên bảng không đo tác động nhân quả của việc tăng số AP.')
para('Trong Validation, 15 dòng không AP giảm mean khá mạnh và đóng góp khoảng 20% mức giảm tổng. 365 dòng nghe 1–9 AP lại tăng mean khoảng 0,049 m. Đối chứng chỉ các dòng Validation có AP vẫn giảm khoảng 0,088 m; đây là chẩn đoán, không loại dòng khỏi chỉ số chính.')
para('Bootstrap mô tả sau lựa chọn trên 359 nhóm đo Validation, 2.000 lần với seed 20261026, cho khoảng 95% của V4−V3 là [−0,200350; −0,013554] m khi giữ cố định 10 seed. Khi lấy mẫu lại cả cặp seed và nhóm đo, khoảng là [−0,231094; +0,016836] m, vẫn chứa 0. Các khoảng chưa hiệu chỉnh selection bias và không phải bằng chứng xác nhận trên một Test mới.')

page('26 Chi phí kiểm toán và tính tái lập')
table('Môi trường và chi phí ghi nhận',['Thành phần','Giá trị'],[('Python / PyTorch','3.12.1 / 2.9.0+cu128'),('NumPy / SciPy','2.2.6 / 1.18.1'),('scikit-learn / pandas','1.6.1 / 2.2.3'),('GPU / luồng CPU','RTX 4060 Ti 8 GB / 2'),('Trial Validation hoàn tất','13 model; tổng 8.716,67 s elapsed training'),('Refit cuối','10 model; tổng 9.180,43 s elapsed training'),('Tổng hai nhóm hoàn tất','17.897,10 s; khoảng 4,97 giờ'),('Kiểm toán Validation','13 model × 3.912 dòng = 50.856 dòng dự đoán'),('Kiểm toán Test','10 model × 1.111 dòng = 11.110 dòng dự đoán'),('Kiểm thử regression và invariant','86 kiểm thử đạt')],[7.0,9.8])
para('Elapsed training gồm các khoảng chờ/chia sẻ GPU; không phải GPU compute hours. Tổng trên chưa tính lượt AP-token bị dừng, lượt seed 47 dở trước phục hồi, calibration, kiểm toán hoặc lập báo cáo. Thời gian giữa seed khác nhau nhiều do tải nền; không dùng chúng để kết luận kiến trúc V4 nhanh hơn V3.')
para('Kiểm toán cuối xác nhận đúng 10 seed và 10 trạng thái trọng số khác nhau; catalogue từ development; 41 epoch/model; preprocessing; thứ tự query, tọa độ thật, tọa độ dự đoán và mọi metric. Tất cả 1.111 mẫu/model được giữ. Hash checkpoint được đối chiếu với khóa tạo trước Test; hash nguồn và artifact baseline V3 được bảo toàn.')
para('Lượt seed 47 Validation bị gián đoạn đã được lưu riêng, sau đó chạy lại cùng mã/config/seed. Prefix hai epoch khớp mọi trường số học ngoài thời gian. Không dùng điểm số của lượt dở thay cho kết quả hoàn tất. Kiểm thử tương thích V3 trên 13 fingerprint cho logits và XY giống tuyệt đối với snapshot V3.')
source(D['final_audit']['path']+'; audits/regression_suite.log; recovery_seed47_prefix_audit.json.')

page('27 Nguồn mã và các lệnh kiểm lại')
para('Bản báo cáo phản ánh trạng thái kan-version-4 và các artifact đã hoàn tất. Các lệnh dưới đây kiểm hoặc phân tích dữ liệu đã lưu; không cần chạy thêm training để đọc kết quả.')
table('Các điểm thực thi cần đọc',['Tệp hoặc hàm','Vai trò'],[('models/reference.py','Biểu diễn RSSI, catalogue, ReferenceKANMixer và decoder'),('models/networks.py','MixerBlock và hai nhánh KAN token/channel'),('models/vendor/efficient_kan.py','KANLinear và B-spline được ghim'),('augmentation.py','SamePointMixing dùng trong train và refit'),('v4_methods.py::point_sampler','Trọng số 1/nᵣ và WeightedRandomSampler'),('train.py::combined_loss','Ba loss và các tùy chọn nghiên cứu'),('research_v4.py','Sàng lọc hữu hạn, khóa lựa chọn, refit và Test cuối'),('final_refit.py','Huấn luyện lại từ khởi tạo mới sau khóa Validation'),('scripts/audit_kan_v4_results.py','Kiểm toán độc lập CSV/checkpoint/nguồn')],[8.3,8.5])
h2('Kiểm toán Test đã hoàn tất')
code('.\\.venv\\Scripts\\python.exe -B -X utf8 scripts/audit_kan_v4_results.py --output-dir runs/kan_v4_20261026_r2 --phase final')
h2('Phân tích cặp Validation đã lưu')
code('.\\.venv-cuda\\Scripts\\python.exe -B -X utf8 scripts/analyze_kan_v4_validation.py --output-dir runs/kan_v4_20261026_r2')
para('Nếu tái lập một lượt huấn luyện mới, dùng v4_balanced_points.json và thư mục kết quả mới, giữ split và seed; không ghi đè study đã hoàn tất. Controller gốc có stop_report.json để tránh tự khởi chạy lại vòng đã dừng. Mọi thí nghiệm tiếp theo cần xác định lại giao thức trước khi sử dụng kết quả Test đã biết.')

page('28 Khóa cấu hình và checkpoint triển khai')
para('Model minh họa/triển khai cố định trước Test là seed 42, không phải seed thấp nhất sau Test. Cả 10 checkpoint đều được giữ để kiểm toán và báo cáo độ ổn định. Những hash dưới đây cho phép xác định đúng đối tượng báo cáo.')
table('Artifact chính',['Đối tượng','Đường dẫn tương đối trong repository'],[('Cấu hình công khai','src/ips/algorithms/kan/configs/v4_balanced_points.json'),('Study V4','runs/kan_v4_20261026_r2/'),('Khóa lựa chọn','selection_lock.json trong study'),('Khóa trước Test','test_checkpoint_lock.json trong study'),('Checkpoint seed 42','final_refit/seed42/best.pt trong study'),('CSV Test seed 42','official_test/seed42/predictions.csv trong study'),('Tổng hợp Test','official_test_summary.json trong study'),('Báo cáo dừng','stop_report.json trong study')],[5.6,11.2])
for label,value in [('SHA256 cấu hình công khai',D['source_sha256']['src/ips/algorithms/kan/configs/v4_balanced_points.json']),('SHA256 khóa lựa chọn',D['source_sha256']['runs/kan_v4_20261026_r2/selection_lock.json']),('SHA256 checkpoint refit seed 42','f8b9c167a56f96fcb1ec0c3cfc51a6a140e1b5177f82d256ea8d94f35946176d'),('SHA256 tổng hợp Test V4',D['source_sha256']['runs/kan_v4_20261026_r2/official_test_summary.json'])]:
    para(label,bold=True);code(value)
para('Checkpoint không chỉ gồm trọng số: cấu hình, thứ tự AP, mapping lớp, preprocessing và catalogue là thành phần của hợp đồng suy luận. Khi dùng model mới trên cùng benchmark phải giữ đúng thứ tự 520 AP. Model UJI không được suy luận zero-shot trực tiếp trên một tòa nhà khác với danh sách AP và hệ tọa độ khác.')

page('29 Kết luận và hướng kiểm chứng tiếp theo')
para('V4 đưa Mean2D Test của nhóm 10 seed từ 7,424524 xuống 7,277383 m, nhưng chưa đạt mục tiêu dưới 7 m. Kết quả dựa trên đủ 1.111 query/model và đã qua kiểm toán độc lập. Điểm thấp nhất 6,940479 m của seed 47 không thay cho mean của nhóm.')
para('Thành phần được chọn là thuật toán lấy mẫu đều điểm tham chiếu, làm thay đổi ảnh hưởng kỳ vọng của các vị trí đo nhiều hoặc ít. Backbone KAN-Mixer B-spline và loss của V3 giữ nguyên. Số epoch refit 41 thay vì 31 là kết quả chọn bằng Validation; vì vậy không quy toàn bộ mức giảm cho tác dụng riêng của sampler.')
para('Các giới hạn chính gồm: một số seed và tầng vẫn xấu hơn V3; P90 tăng nhẹ; Validation không tách hoàn toàn thiết bị/người đo/tọa độ; Test có lịch sử được xem trong dự án. Cân bằng số lần quét có thể chưa trực tiếp giải quyết đủ lệch thiết bị và thời điểm, nhưng chưa có ablation nhân quả để xác nhận cơ chế này.')
para('Vòng nghiên cứu hữu hạn được dừng để đánh giá lại theo tiêu chí tài nguyên và mức tiến triển của người dùng: bốn trong năm hướng không cải thiện Validation, hướng còn lại chỉ giảm Test khoảng 2% sau gần năm giờ elapsed training được ghi nhận. Ngưỡng năm hướng liên tiếp không cải thiện không bị kích hoạt, vì balanced_points đã cải thiện.')
para('Hướng có cơ sở tiếp theo là xử lý head sụp trong thiết kế AP-token bằng chuẩn hóa trước readout, rồi kiểm chứng mask-aware readout trong ablation riêng. Cần dùng Validation tách thiết bị và giao thức đánh giá được xác định trước cho vòng mới. Các hướng này chưa được thực nghiệm trong kết quả V4 đang báo cáo.')
h2('Phạm vi kết luận khoa học')
para('Bằng chứng hiện tại ủng hộ một cải tiến quy trình huấn luyện trên UJIIndoorLoc, chưa chứng minh một kiến trúc KAN mới hoặc lợi ích riêng của spline so với backbone khác có cùng đầu ra và ngân sách. Việc Test từng được quan sát cần được công bố, thay vì mô tả V4 là xác nhận trên một bộ blind chưa từng được tiếp cận.')

page('30 Tài liệu tham khảo và nguồn bằng chứng')
for text in [
 '[1] Torres-Sospedra, J. và cộng sự. UJIIndoorLoc. UCI Machine Learning Repository, 2014. DOI 10.24432/C5MS59. https://archive.ics.uci.edu/dataset/310/ujiindoorloc',
 '[2] Liu, Z. và cộng sự. KAN Kolmogorov-Arnold Networks. arXiv:2404.19756. https://arxiv.org/abs/2404.19756',
 '[3] Canuto, J. L. S., Aylon, L. B. R. và de Souza, R. C. T. KAN-Mixers a new deep learning architecture for image classification. arXiv:2503.08939v1, 2025. https://arxiv.org/abs/2503.08939v1',
 '[M1] Mã nguồn dự án IPS trên nhánh kan-version-4: data.py; algorithms/kan/models/reference.py; models/networks.py; models/vendor/efficient_kan.py; augmentation.py; train.py; v4_methods.py; final_refit.py. Hash nguồn được lưu trong protocol.json và kiểm bởi auditor.',
 '[M2] V3 same_point_relative: runs/kan_joint_under6_20261020_v1/official_test_summary.json; confirmation_summary.json; các CSV và checkpoint tương ứng. Chỉ dùng nhóm full_development_refit đủ 10 seed làm baseline chính.',
 '[M3] V4 balanced_points: runs/kan_v4_20261026_r2/protocol.json; screening.json; confirmation_three.json; selection_lock.json; test_checkpoint_lock.json; official_test_summary.json.',
 '[M4] Kiểm toán: runs/kan_v4_20261026_r2/audits/final_20261026T151612036225Z.json; paired_validation_ten_20261026T124406Z.json; stop_report.json; completion_audit.json.',
 '[M5] IPS_KAN_V2_thuat_toan.docx trong references/KAN-mixer: mẫu cách giải thích thuật toán và các công thức nền được kế thừa. Không dùng kết quả V2 làm baseline trong báo cáo này.',
 '[M6] UPDATE_VERSION_3.md và UPDATE_VERSION_4.md trong src/ips/algorithms/kan; báo cáo và nhật ký nghiên cứu. Khi khác biệt cách diễn đạt, cấu hình checkpoint, mã thực thi và CSV đã kiểm toán là bằng chứng định lượng ưu tiên.'
]:para(text)
para('Các nguồn [1]–[3] trình bày dữ liệu hoặc ý tưởng nền. Các số liệu V3/V4 và chi tiết vận hành IPS trong báo cáo lấy từ nguồn nội bộ [M1]–[M4], không được suy ra từ benchmark phân loại ảnh của bài KAN-Mixers. Mọi biểu đồ so sánh đều được dựng từ CSV đã lưu; không chạy thêm suy luận để lập báo cáo.')

config_lines=json.dumps(D['config'],ensure_ascii=False,indent=2).splitlines()
for idx,chunk in enumerate([config_lines[:41],config_lines[41:]]):
    page('Phụ lục A Cấu hình JSON đầy đủ' if idx==0 else 'Phụ lục A Cấu hình JSON tiếp theo')
    if idx==0:para('Đây là nội dung cấu hình đầu vào công khai. Catalogue references chỉ được fit sau khi nạp đúng tập train; các k/τ cuối theo seed nằm ở chương 20. Giá trị refit 41 epoch thuộc selection lock, không thay max_epochs = 100 của lịch cosine.')
    for line in chunk:
        p=code(line);p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=Pt(10.5)
        for r in p.runs:r.font.size=Pt(8.7)
    if idx==1:para('Chỉ một khóa khác JSON V3: training.sampling = reference_points. Các giá trị mặc định ngoài JSON và phạm vi không dùng của smooth_l1_beta đã được giải thích trong chương 16 và 17.')

OUT.parent.mkdir(exist_ok=True)
doc.save(OUT)
(WORK/'document_manifest.json').write_text(json.dumps({'output':str(OUT),'headings':HEADINGS,'counts':COUNTS,'V3_baseline':'same_point_relative full_development_refit ten seeds','V2_role':'algorithm exposition only','target_met':False},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'output':str(OUT),'counts':COUNTS,'explicit_pages':len(HEADINGS)+1,'paragraphs':len(doc.paragraphs),'tables':len(doc.tables)},ensure_ascii=False))
