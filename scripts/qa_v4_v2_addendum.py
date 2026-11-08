"""Check the appended section and render page PNGs for visual review."""
from pathlib import Path
from zipfile import ZipFile
import json
import hashlib
import unicodedata

from lxml import etree as E
from PIL import Image, ImageChops, ImageDraw
from pdf2image import convert_from_path
from pypdf import PdfReader
from docx import Document

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runs/kan_v4_v2_addendum_20261027'
RENDER = WORK / 'render'
RENDER.mkdir(exist_ok=True)
POPPLER = r'C:/Users/Maxsys/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'
docx = WORK / 'IPS_KAN_V4.docx'
expected = json.loads((WORK / 'authoring.json').read_text(encoding='utf-8'))
document = Document(docx)
assert len(document.tables) == 28
for actual, table in zip(document.tables[-3:], expected['tables']):
    assert [[c.text for c in row.cells] for row in actual.rows] == [table['headers']] + table['rows']
with ZipFile(docx) as archive:
    assert archive.testzip() is None
    root = E.fromstring(archive.read('word/document.xml'))
    assert len(root.findall('.//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath')) == 41
    assert len(root.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr')) == 5
    for name in archive.namelist():
        if name.endswith('.xml'):
            E.fromstring(archive.read(name))

reader = PdfReader(WORK / 'IPS_KAN_V4.pdf')
texts = [unicodedata.normalize('NFC', p.extract_text() or '') for p in reader.pages]
for number, value in enumerate(texts, 1):
    (RENDER / f'page-{number}.txt').write_text(value, encoding='utf-8')
assert 37 <= len(texts) <= 39, len(texts)
assert '31 Chạy thử nghiệm' in ' '.join(texts[34].split())
assert 'cấu hình V2' in ' '.join(texts[1].split())
images = convert_from_path(str(WORK / 'IPS_KAN_V4.pdf'), dpi=140, poppler_path=POPPLER, thread_count=4)
baseline = convert_from_path(str(WORK / 'known_baseline.pdf'), dpi=140, poppler_path=POPPLER, thread_count=4)
assert len(baseline) == 34
same, changed = [], []
for number, image in enumerate(images, 1):
    image.save(RENDER / f'page-{number}.png')
    if number <= 34:
        old = baseline[number-1].convert('RGB')
        if old.size == image.size and ImageChops.difference(old, image.convert('RGB')).getbbox() is None:
            same.append(number)
        else:
            changed.append(number)
sheet = Image.new('RGB', (4 * 300, ((len(images) + 3) // 4) * 445), 'white')
draw = ImageDraw.Draw(sheet)
for index, page in enumerate(images):
    page = page.copy()
    page.thumbnail((288, 412))
    x, y = (index % 4) * 300, (index // 4) * 445
    sheet.paste(page, (x + 6, y + 22))
    draw.text((x + 8, y + 4), f'Page {index+1}', fill='black')
sheet.save(RENDER / 'contact_sheet.png')
report = {'status': 'awaiting_visual_review', 'pages': len(images), 'tables': 28, 'figures': 5,
          'native_equations': 41, 'new_section_starts_page': 35, 'numeric_tables_verified': True,
          'original_pages_pixel_identical': same, 'original_pages_changed': changed,
          'new_pages': list(range(35, len(images)+1)),
          'docx_sha256': hashlib.sha256(docx.read_bytes()).hexdigest(),
          'pdf_sha256': hashlib.sha256((WORK/'IPS_KAN_V4.pdf').read_bytes()).hexdigest()}
(WORK / 'qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False))
