"""Finish only owned DOCX layout, preserving the cached Word table of contents."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from copy import deepcopy
import hashlib,json
from lxml import etree as E
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'runs/kan_v4_docx_20261026'
PATH=ROOT/'references/KAN-mixer/IPS_KAN_V4.docx'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
previous=json.loads((WORK/'qa.json').read_text(encoding='utf-8'))
assert previous['docx_sha256']==sha(PATH),'Document changed externally; preserve it.'
(WORK/'render/pre_final_png_hashes.json').write_text(json.dumps({p.name:sha(p) for p in (WORK/'render').glob('page-*.png')}),encoding='utf-8')
with ZipFile(PATH) as z:members=[(i,z.read(i.filename)) for i in z.infolist()]
w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
m='http://schemas.openxmlformats.org/officeDocument/2006/math'
ns={'w':w,'m':m};text=lambda e:''.join(e.xpath('.//w:t/text()',namespaces=ns))
for idx,(info,data) in enumerate(members):
    if info.filename not in ('word/document.xml','word/styles.xml'):continue
    root=E.fromstring(data)
    if info.filename=='word/styles.xml':
        for style in root.findall('w:style',ns):
            if style.get('{'+w+'}styleId') in ('Title','Subtitle'):
                for border in style.findall('.//w:pBdr',ns):border.getparent().remove(border)
    else:
        math=root.findall('.//m:oMath',ns)[21]
        if math.find('m:eqArr',ns) is None:
            rows=[E.Element('{'+m+'}e'),E.Element('{'+m+'}e')];second=False
            for ch in list(math):
                ts=ch.findall('.//m:t',ns)
                if not second and ch.tag=='{'+m+'}r' and len(ts)==1 and ';' in ts[0].text:
                    before,after=ts[0].text.split(';',1);a,b=deepcopy(ch),deepcopy(ch)
                    a.find('m:t',ns).text=before.strip();b.find('m:t',ns).text=after.strip()
                    rows[0].append(a);rows[1].append(b);second=True
                else:rows[int(second)].append(ch)
            assert second
            array=E.Element('{'+m+'}eqArr');array.extend(rows);math[:]=[array]
        for t in root.findall('.//w:t',ns):
            if t.text=='Ngày 26 tháng 10 năm 2026':t.text='Ngày 27 tháng 10 năm 2026'
            if t.text=='absolute_relative_context':
                t.text='absolute_relative_';r=t.getparent();pos=r.index(t)+1
                r.insert(pos,E.Element('{'+w+'}br'));new=E.Element('{'+w+'}t');new.text='context';r.insert(pos+1,new)
        for p in root.findall('.//w:p',ns):
            style=p.find('w:pPr/w:pStyle',ns)
            if style is not None and style.get('{'+w+'}val') in ('Title','Subtitle'):
                for border in p.findall('w:pPr/w:pBdr',ns):border.getparent().remove(border)
        body=root.find('w:body',ns)
        a=next(e for e in body if e.tag=='{'+w+'}p' and text(e)=='Phụ lục A Cấu hình JSON đầy đủ')
        b=next(e for e in body if e.tag=='{'+w+'}p' and text(e)=='Phụ lục A Cấu hình JSON tiếp theo')
        codes=list(body)[body.index(a)+2:body.index(b)]
        assert len(codes) in (41,48),len(codes)
        if len(codes)==48:
            moving=codes[41:]
            for e in moving:body.remove(e)
            for j,e in enumerate(moving):body.insert(body.index(b)+1+j,e)
        for e in list(body)[body.index(a)+1:]:
            if e.tag!='{'+w+'}p':continue
            fonts=e.find('.//w:rPr/w:rFonts',ns)
            if fonts is None or fonts.get('{'+w+'}ascii')!='Consolas':continue
            pp=e.find('w:pPr',ns)
            if pp is None:pp=E.Element('{'+w+'}pPr');e.insert(0,pp)
            spacing=pp.find('w:spacing',ns)
            if spacing is None:spacing=E.SubElement(pp,'{'+w+'}spacing')
            spacing.set('{'+w+'}line','210');spacing.set('{'+w+'}lineRule','exact');spacing.set('{'+w+'}after','0')
        titles=['Luồng suy luận của KAN Mixer tham chiếu','Ví dụ lấy mẫu cân bằng điểm tham chiếu','Mean2D Test của V3 và V4 theo seed','Phân phối tích lũy sai số V3 và V4','Mean2D Test theo số AP phát hiện']
        for shape,title in zip(root.findall('.//{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}docPr'),titles):shape.set('descr',title)
    members[idx]=(info,E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True))
tmp=PATH.with_suffix('.qa.tmp')
with ZipFile(tmp,'w',ZIP_DEFLATED) as z:
    for info,data in members:z.writestr(info,data)
tmp.replace(PATH)
print('Finalized native equation22, appendix, figure descriptions, title and report date.')
