"""Read-only model evidence preparation and charts for the V4 Word report."""
from pathlib import Path
import csv
import hashlib
import json
import statistics
import zipfile
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/kan_v4_docx_20261026'
STUDY=ROOT/'runs/kan_v4_20261026_r2'
BASE=ROOT/'runs/kan_joint_under6_20261020_v1'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def prepare():
    OUT.mkdir(exist_ok=True)
    (OUT/'figures').mkdir(exist_ok=True)
    original=ROOT/'references/KAN-mixer/IPS_KAN_V2_thuat_toan.docx'
    with zipfile.ZipFile(original) as z:
        tree=ET.fromstring(z.read('word/document.xml'))
        tags={'{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t','{http://schemas.openxmlformats.org/officeDocument/2006/math}t'}
        paragraphs=[''.join(t.text or '' for t in p.iter() if t.tag in tags) for p in tree.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p')]
    (OUT/'source_v2_text.txt').write_text('\n'.join(f'{i}: {p}' for i,p in enumerate(paragraphs)),encoding='utf-8')
    summary=read(STUDY/'official_test_summary.json')
    old=read(BASE/'official_test_summary.json')
    old_group=next(g for g in old['groups'] if g['name']=='same_point_relative' and g['regime']=='full_development_refit')
    old_rows={r['seed']:r for r in old['all_per_model_results'] if r['name']=='same_point_relative' and r['regime']=='full_development_refit'}
    protocol=read(STUDY/'protocol.json'); lock=read(STUDY/'selection_lock.json')
    assert summary['n_models']==10 and summary['target_met'] is False
    selected=[]
    for seed in protocol['seeds']:
        item=read(STUDY/f'trials/balanced_points/seed{seed}/completed.json')
        cfg=read(STUDY/f'final_refit/seed{seed}/resolved_config.json')
        old_cfg=read(BASE/f'trials/same_point_relative/seed{seed}/training/resolved_config.json')
        old_cal=read(BASE/f'trials/same_point_relative/seed{seed}/calibration/calibration.json')
        old_selected=old_cal.get('selected',old_cal.get('pure'))
        if old_selected is None: old_selected=min(old_cal['candidates'],key=lambda r:r['mean_2d'])
        old_train=read(BASE/f'trials/same_point_relative/seed{seed}/training/summary.json')
        item.update(top_k=cfg['positioning']['top_k'],temperature=cfg['positioning']['temperature'],
          V3_top_k=old_selected['top_k'],V3_temperature=old_selected['temperature'],
          V3_best_epoch=old_train['best_epoch'],V3_mean_validation=protocol['baseline_validation_ten']['per_seed'][str(seed)])
        item['epochs_completed']=read(STUDY/f'trials/balanced_points/seed{seed}/training/summary.json')['epochs_completed']
        selected.append(item)
    errors={}; predictions={}
    for method in ('V3','V4'):
        all_errors=[]; all_frames=[]
        for seed in protocol['seeds']:
            p=(BASE/f'official_test/full_development_refit/same_point_relative/seed{seed}/predictions.csv' if method=='V3' else STUDY/f'official_test/seed{seed}/predictions.csv')
            rows=list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
            assert len(rows)==1111
            xy=np.array([[float(r['pred_x']),float(r['pred_y'])] for r in rows]); truth=np.array([[float(r['true_x']),float(r['true_y'])] for r in rows])
            e=np.linalg.norm(xy-truth,axis=1)
            expected=old_rows[seed]['mean_2d'] if method=='V3' else next(r['mean_2d'] for r in summary['per_seed'] if r['seed']==seed)
            assert abs(e.mean()-expected)<1e-9
            assert np.allclose(e,[float(r['error_2d']) for r in rows],rtol=0,atol=1e-8)
            all_errors.append(e); all_frames.append(rows)
            predictions[str(p.relative_to(ROOT))]=digest(p)
        errors[method]=np.stack(all_errors)
        if method=='V3': template=all_frames[0]
        else:
            assert [r['row_index'] for r in template]==[r['row_index'] for r in all_frames[0]]
    buildings=np.array([int(r['true_building']) for r in template]); floors=np.array([int(r['true_floor']) for r in template]); aps=np.array([int(r['detected_aps']) for r in template])
    def slice_row(label,mask):
        return {'label':label,'n':int(mask.sum()),**{m:float(v[:,mask].mean()) for m,v in errors.items()}}
    by_building=[slice_row(str(b),buildings==b) for b in sorted(set(buildings))]
    by_class=[slice_row(f'{b}/{f}',(buildings==b)&(floors==f)) for b,f in sorted(set(zip(buildings,floors)))]
    by_ap=[slice_row(label,mask) for label,mask in [('1–9',(aps>0)&(aps<10)),('10–19',(aps>=10)&(aps<20)),('20–29',(aps>=20)&(aps<30)),('≥30',aps>=30)] if mask.any()]
    cfg=read(ROOT/'src/ips/algorithms/kan/configs/v4_balanced_points.json')
    oldcfg=read(BASE/'configs/same_point_relative.json')
    def differences(a,b,prefix=''):
        out=[]
        for key in sorted(set(a)|set(b)):
            aa=a.get(key);bb=b.get(key);name=prefix+key
            if isinstance(aa,dict) and isinstance(bb,dict):out+=differences(aa,bb,name+'.')
            elif aa!=bb:out.append({'key':name,'V3':aa,'V4':bb})
        return out
    audits=sorted((STUDY/'audits').glob('final_*.json'))
    evidence={'V2_reference':{'path':str(original.relative_to(ROOT)),'sha256':digest(original),'paragraphs':len(paragraphs),'purpose':'Presentation model only; all quantitative comparisons are V3 versus V4.'},
       'V3':old_group,'V3_per_seed':old_rows,'V4':summary,'selected':selected,'protocol':protocol,'selection_lock':lock,
       'config':cfg,'config_diff':differences(oldcfg,cfg),'validation_progress':read(STUDY/'ten_seed_progress.json'),
       'validation_slices':read(sorted(STUDY.glob('paired_validation_ten_*.json'))[-1]),
       'by_building':by_building,'by_class':by_class,'by_ap':by_ap,
       'refit_summaries':[read(STUDY/f'final_refit/seed{s}/summary.json') for s in protocol['seeds']],
       'final_audit':{'path':str(audits[-1].relative_to(ROOT)),'sha256':digest(audits[-1]),'status':read(audits[-1])['status']},
       'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [original,STUDY/'official_test_summary.json',BASE/'official_test_summary.json',STUDY/'selection_lock.json',ROOT/'src/ips/algorithms/kan/configs/v4_balanced_points.json',ROOT/'src/ips/algorithms/kan/models/reference.py',ROOT/'src/ips/algorithms/kan/train.py',ROOT/'src/ips/algorithms/kan/v4_methods.py']},'prediction_sha256':predictions}
    (OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez(OUT/'errors.npz',V3=errors['V3'],V4=errors['V4'])
    architecture='''%%{init: {"theme":"base","themeVariables":{"fontFamily":"Arial","fontSize":"19px","primaryColor":"#eef3f7","primaryTextColor":"#111111","primaryBorderColor":"#536b7d","lineColor":"#536b7d"}}}%%
flowchart TB
 A["RSSI của 520 AP<br/>Chuẩn hóa z và mặt nạ m"] --> B["RSSI tương đối và mask<br/>26 token × 40 số → embedding 64 kênh"]
 B --> C["Hai khối KAN Mixer<br/>Trộn token 26 → 32 → 26<br/>Trộn kênh 64 → 128 → 64"]
 C --> D["LayerNorm và flatten 1664 số<br/>Linear 1664 → 256 → R điểm tham chiếu"]
 D --> E["Chọn lớp bằng logsumexp<br/>Top k và softmax theo nhiệt độ τ<br/>Đảo chuẩn hóa → X Y và tòa tầng"]
'''
    (OUT/'figures/architecture.mmd').write_text(architecture,encoding='utf-8')
    print(json.dumps({'output':str(OUT),'config_diff':evidence['config_diff'],'validation_models':len(selected),'test_classes':len(by_class),'by_building':by_building,'by_ap':by_ap},ensure_ascii=False))

if __name__=='__main__':prepare()
