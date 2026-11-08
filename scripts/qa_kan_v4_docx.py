"""Structural and numerical checks of the authored retrospective Word report."""
from pathlib import Path
import hashlib,json,zipfile
from lxml import etree
from docx import Document
from pypdf import PdfReader
from pdf2image import convert_from_path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'runs/kan_v4_docx_20261026'
DOCX=ROOT/'references/KAN-mixer/IPS_KAN_V4.docx';PDF=WORK/'render/IPS_KAN_V4.pdf'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
fmt=lambda x,n=4:f'{x:.{n}f}'.replace('.',',')
d=json.loads((WORK/'evidence.json').read_text(encoding='utf-8'))
doc=Document(DOCX)
appendix=next(i for i,p in enumerate(doc.paragraphs) if p.text=='Phụ lục A Cấu hình JSON đầy đủ')
code_lines=[p.text for p in doc.paragraphs[appendix:] if any(r.font.name=='Consolas' for r in p.runs)]
assert json.loads('\n'.join(code_lines))==d['config'],'Appendix does not reproduce the complete selected configuration.'
tables=[[[c.text for c in r.cells] for r in t.rows] for t in doc.tables]
assert len(tables)==25 and len(doc.inline_shapes)==5
with zipfile.ZipFile(DOCX) as z:
    assert z.testzip() is None
    tree=etree.fromstring(z.read('word/document.xml'))
    assert len(tree.findall('.//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath'))==41
    text=' '.join(tree.itertext())
    assert all(s in text for s in ['7,277383 ± 0,211021','7,424524 ± 0,293599','0,147141','1,9818%','6.396','41 epoch','31 epoch'])
    assert not any(s in text for s in ['8,058','7,838','15,726'])
for row,r in zip(tables[13][1:],d['selected']):
    assert row==[str(r['seed']),fmt(r['V3_mean_validation'],6),fmt(r['mean_2d'],6),str(r['V3_best_epoch']),str(r['best_epoch']),str(r['epochs_completed'])]
for row,r in zip(tables[15][1:],d['selected']):
    assert row==[str(r['seed']),str(r['V3_top_k']),fmt(r['V3_temperature'],1),str(r['top_k']),fmt(r['temperature'],1)]
metric_keys=['mean_2d','median_2d','p90_2d','rmse_2d','building_accuracy','floor_accuracy','joint_accuracy']
for row,key in zip(tables[17][1:],metric_keys):
    factor=100 if key.endswith('accuracy') else 1
    assert row[1]==fmt(d['V3'][key+'_mean']*factor)+' ± '+fmt(d['V3'][key+'_sample_std']*factor)
    assert row[2]==fmt(d['V4']['aggregate'][key]['mean']*factor)+' ± '+fmt(d['V4']['aggregate'][key]['sample_std']*factor)
for row,r in zip(tables[18][1:],d['V4']['per_seed']):
    before=d['V3_per_seed'][str(r['seed'])]['mean_2d'];delta=r['mean_2d']-before
    assert row==[str(r['seed']),fmt(before,6),fmt(r['mean_2d'],6),('+' if delta>=0 else '')+fmt(delta,6)]
for number,key in [(19,'by_building'),(20,'by_class'),(21,'by_ap')]:
    for row,r in zip(tables[number][1:],d[key]):
        assert row[:4]==[r['label'],str(r['n']),fmt(r['V3']),fmt(r['V4'])]
for key,value in d['source_sha256'].items():assert sha(ROOT/key)==value,key
for key,value in d['prediction_sha256'].items():assert sha(ROOT/key)==value,key
assert all(s._inline.docPr.get('descr') for s in doc.inline_shapes)
pdf=PdfReader(PDF); pages=[]
for i,page in enumerate(pdf.pages,1):
    text=page.extract_text() or ''
    assert len(text.strip())>90,('Empty/near-empty page',i)
    pages.append({'page':i,'characters':len(text),'start':text[:170],'end':text[-120:]})
(WORK/'render/page_text_audit.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
report={'status':'passed','docx_sha256':sha(DOCX),'pdf_sha256':sha(PDF),'pages':len(pdf.pages),'tables':25,'figures':5,'native_equations':41,'comparison':'V3 versus V4 only; V2 is exposition model','numerical_tables_verified':True,'complete_configuration_appendix_verified':True,'source_and_prediction_hashes_unchanged':True,'visual_review':'pending'}
(WORK/'qa.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
poppler=Path('C:/Users/Maxsys/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin')
paths=convert_from_path(str(PDF),dpi=140,output_folder=str(WORK/'render'),fmt='png',output_file='page',paths_only=True,thread_count=2,poppler_path=str(poppler))
for n,p in enumerate(paths,1):Path(p).replace(WORK/f'render/page-{n}.png')
print(json.dumps(report,ensure_ascii=False))
