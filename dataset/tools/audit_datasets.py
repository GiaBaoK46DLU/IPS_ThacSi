"""Read-only audit of supplied CSV/TXT/XLSX fingerprints; never fit or train a model.

Run with the project's NumPy environment. lxml is optional for faster XLSX reading.
All analysis caches are separate from raw inputs. Counts use actual nonempty rows,
not Excel's formatted dimension. Sheet formulas use cached values and are flagged.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,posixpath,re,sys,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from zipfile import ZipFile
import numpy as np
try:
    from lxml import etree as XML
    FAST_XML=True
except ImportError:
    import xml.etree.ElementTree as XML
    FAST_XML=False

ROOT=Path(__file__).resolve().parents[2]
DATASET=ROOT/'dataset'
NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL='{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id'
AP_PATTERN=re.compile(r'^(?:AP|MAC|WAP)\d+$',re.I)
def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def colnum(value):
    n=0
    for c in value:
        if not c.isalpha():break
        n=n*26+ord(c.upper())-64
    return n-1
def excel_rows(path,info):
    with ZipFile(path) as z:
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            doc=XML.fromstring(z.read('xl/sharedStrings.xml'))
            shared=[''.join(s.itertext()) for s in doc.findall(f'{{{NS}}}si')]
        relationships={r.attrib['Id']:r.attrib['Target'] for r in XML.fromstring(z.read('xl/_rels/workbook.xml.rels'))}
        sheets=XML.fromstring(z.read('xl/workbook.xml')).findall(f'{{{NS}}}sheets/{{{NS}}}sheet')
        info['sheets']=[s.attrib['name'] for s in sheets]
        if len(sheets)!=1:raise ValueError(f'Expected one data sheet, found {len(sheets)}: {path}')
        target=relationships[sheets[0].attrib[REL]]
        target=target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
        with z.open(target) as stream:
            options={'events':('end',)}
            if FAST_XML:options['tag']=f'{{{NS}}}row'
            for _,row in XML.iterparse(stream,**options):
                if row.tag!=f'{{{NS}}}row':continue
                values={}
                for cell in row:
                    if cell.tag!=f'{{{NS}}}c':continue
                    value=cell.findtext(f'{{{NS}}}v')
                    if cell.find(f'{{{NS}}}f') is not None:
                        info['formula_cells']+=1
                        if value is None:info['formula_cells_without_cached_value']+=1
                    kind=cell.attrib.get('t')
                    if kind=='s' and value is not None:value=shared[int(value)]
                    elif kind=='inlineStr':
                        inline=cell.find(f'{{{NS}}}is')
                        value=''.join(inline.itertext()) if inline is not None else None
                    values[colnum(cell.attrib['r'])]=value
                result=[None]*(max(values,default=-1)+1)
                for i,v in values.items():result[i]=v
                row_number=int(row.attrib.get('r',0))
                row.clear()
                if FAST_XML:
                    while row.getprevious() is not None:del row.getparent()[0]
                yield row_number,result
def table_rows(path,info):
    if path.suffix.lower()=='.xlsx':yield from excel_rows(path,info);return
    with path.open(encoding='utf-8-sig',newline='') as stream:
        line=stream.readline()
        delimiter='\t' if '\t' in line else ',' if ',' in line else None
        info['delimiter']=delimiter or 'whitespace'
        stream.seek(0)
        for n,line in enumerate(stream,start=1):
            row=next(csv.reader([line],delimiter=delimiter)) if delimiter else line.split()
            if delimiter=='\t':
                while row and row[-1]=='':row.pop();info['trailing_empty_fields']+=1
            yield n,row
def number(v):
    try:return float(v)
    except (TypeError,ValueError):return float('nan')
def finite_range(values):
    values=np.asarray(values,dtype=float)
    finite=values[np.isfinite(values)]
    return [float(finite.min()),float(finite.max())] if len(finite) else None
def distribution(values):return {str(k):int(n) for k,n in sorted(Counter(str(v) for v in values).items())}
def digest_rows(array):
    return np.asarray([hashlib.blake2b(np.ascontiguousarray(row).tobytes(),digest_size=16).hexdigest() for row in array],dtype='U32')
def classify(path):
    parts=path.relative_to(DATASET/'raw').parts
    family=parts[1] if parts[0]=='sodindoorloc' else parts[0]
    case=f'{family}/{parts[1]}' if family in ('dlulib','uji_library') else family
    name=path.name.lower()
    role=('map' if 'map' in name else 'train_mean' if 'mean' in name else
          'train' if 'train' in name else 'test')
    return family,case,role
def audit_one(args):
    path,cache=map(Path,args)
    started=time.perf_counter()
    family,case,role=classify(path)
    info={'path':path.relative_to(ROOT).as_posix(),'family':family,'case':case,'role':role,
          'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'formula_cells':0,'formula_cells_without_cached_value':0,'trailing_empty_fields':0,
          'empty_or_style_only_rows':0,'ragged_rows':0,'ragged_examples':[]}
    iterator=iter(table_rows(path,info))
    first_n,first=next(iterator)
    if role=='map':
        headers=['RP','X','Y']
        iterator=iter([(first_n,first),*list(iterator)])
    else:headers=[str(v).strip() for v in first]
    info['columns']=len(headers)
    info['headers']=headers
    ap=[i for i,h in enumerate(headers) if AP_PATTERN.fullmatch(h)]
    meta=[i for i in range(len(headers)) if i not in ap]
    rows=[];raw_positions=[]
    for row_number,row in iterator:
        if not row or all(v is None or str(v).strip()=='' for v in row):
            info['empty_or_style_only_rows']+=1;continue
        if len(row)!=len(headers):
            info['ragged_rows']+=1
            if len(info['ragged_examples'])<5:info['ragged_examples'].append({'row':row_number,'fields':len(row)})
            row=(row+[None]*len(headers))[:len(headers)]
        rows.append(row);raw_positions.append(row_number)
    info['rows']=len(rows)
    info['duplicate_headers']=len(headers)-len(set(headers))
    meta_values={headers[i]:[r[i] for r in rows] for i in meta}
    info['metadata']={h:{'missing':sum(v is None or str(v).strip()=='' for v in values),
                         'distinct':len(set(str(v) for v in values)),
                         'range':finite_range([number(v) for v in values])} for h,values in meta_values.items()}
    categorical={'floor','floorid','buildingid','id_rp','rp','smartphones','users','userid','phoneid','direction','month','sampletimes','spaceid','relativeposition'}
    for h,values in meta_values.items():
        if h.lower() in categorical:info['metadata'][h]['counts']=distribution(values)
    def field(names):
        h=next((h for h in headers if h.lower() in names),None)
        return np.asarray([number(v) for v in meta_values[h]]) if h else np.full(len(rows),np.nan)
    rp=field({'rp','id_rp'})
    x=field({'x','longitude'});y=field({'y','latitude'})
    floor=field({'floor','floorid'})
    building=field({'buildingid'})
    if role=='map':
        points=np.column_stack([rp,x,y])
        np.savez_compressed(cache,points=points)
        info['map_duplicate_ids']=len(rp)-len(set(rp))
        info['ap_count']=0
        return info
    if np.isnan(x).all() and np.isnan(y).all() and family in ('hus','hcxy'):
        map_path=path.parent/('hus_mapXY.txt' if family=='hus' else 'mapxy_HCXY.txt')
        mapping={}
        with map_path.open(encoding='utf-8-sig') as stream:
            for line in stream:
                key,a,b=map(float,line.split());mapping[key]=(a,b)
        xy=np.asarray([mapping.get(v,(np.nan,np.nan)) for v in rp])
        x,y=xy[:,0],xy[:,1]
        info['coordinate_join']='RP -> headerless companion map (column 1 ID, columns 2/3 coordinates)'
        info['unmapped_rp_rows']=int(np.isnan(x).sum())
    xy=np.column_stack([x,y])
    labels=np.column_stack([np.nan_to_num(building,nan=-1),np.nan_to_num(floor,nan=-1),xy])
    info['coordinates']={'columns':[next((h for h in headers if h.lower() in {'x','longitude'}),None),next((h for h in headers if h.lower() in {'y','latitude'}),None)],
        'x_range':finite_range(x),'y_range':finite_range(y),'missing_xy_rows':int((~np.isfinite(xy).all(1)).sum()),
        'unique_building_floor_xy':len(set(tuple(r) for r in labels)),'units_verified_from_file':False}
    rssi=np.asarray([[number(r[i]) for i in ap] for r in rows],dtype=np.float64)
    if not ap:raise ValueError(f'No AP columns: {path}')
    info['ap_count']=len(ap)
    info['ap_columns']=[headers[i] for i in ap]
    values,counts=np.unique(rssi[np.isfinite(rssi)],return_counts=True)
    frequencies={float(v):int(n) for v,n in zip(values,counts)}
    info['rssi']={'range':finite_range(rssi),'nonfinite_cells':int((~np.isfinite(rssi)).sum()),
        'rows_with_nonfinite_rssi':int((~np.isfinite(rssi).all(1)).sum()),
        'fractional_cells':int(((rssi!=np.rint(rssi)) & np.isfinite(rssi)).sum()),
        'count_100':frequencies.get(100.,0),'count_minus100':frequencies.get(-100.,0),'count_zero':frequencies.get(0.,0),
        'most_common_values':sorted(frequencies.items(),key=lambda r:r[1],reverse=True)[:15]}
    # Diagnostic hypotheses only; meaning of a missing-value code needs provenance.
    candidate=-100 if family=='dlulib' else 100 if family in ('ujiindoorloc','uji_library','tuji','hcxy','syl') else None
    info['candidate_missing_code']=candidate
    if candidate is not None:
        seen=(rssi!=candidate)&np.isfinite(rssi)
        detected=seen.sum(1)
        info['candidate_missing_fraction']=float((rssi==candidate).mean())
        info['candidate_detected_ap_counts']={'min':int(detected.min()),'median':float(np.median(detected)),'mean':float(detected.mean()),'max':int(detected.max())}
        info['candidate_all_undetected_rows']=int((detected==0).sum())
        info['candidate_never_detected_ap_columns']=[headers[ap[i]] for i in np.flatnonzero(~seen.any(0))]
        info['candidate_detected_rssi_range']=finite_range(rssi[seen])
    fp=digest_rows(rssi)
    labelled=digest_rows(np.column_stack([rssi,labels]))
    full_keys=[hashlib.blake2b(json.dumps([None if v is None else str(v) for v in row],ensure_ascii=False,separators=(',',':')).encode(),digest_size=16).hexdigest() for row in rows]
    info['duplicate_excess']={'full_rows':len(rows)-len(set(full_keys)),'rssi':len(rows)-len(set(fp)),
        'rssi_building_floor_xy':len(rows)-len(set(labelled))}
    if np.isfinite(rp).any():
        info['rp_counts']=distribution(rp[np.isfinite(rp)])
        valid=rp[np.isfinite(rp)]
        info['rp_count']=len(set(valid))
        per=Counter(valid)
        info['samples_per_rp_count_distribution']=distribution(per.values())
        info['rps_with_multiple_coordinates']=sum(len({tuple(labels[i]) for i in np.flatnonzero(rp==p)})>1 for p in per)
    cache.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(cache,rssi=rssi,xy=xy,labels=labels,rp=rp,fp=fp,labelled=labelled,
                        raw_row_numbers=np.asarray(raw_positions))
    info['cache']=cache.relative_to(ROOT).as_posix()
    info['elapsed_seconds']=time.perf_counter()-started
    return info

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--output',type=Path,default=DATASET/'analysis/2026-10-14')
    parser.add_argument('--cache',type=Path,default=ROOT/'runs/dataset_audit_20261014/cache')
    parser.add_argument('--reuse-cache',action='store_true',help='Reconcile outputs from previously audited, hash-identical raw files.')
    args=parser.parse_args()
    args.cache.mkdir(parents=True,exist_ok=True)
    paths=sorted((DATASET/'raw').rglob('*'))
    paths=[p for p in paths if p.is_file()]
    tasks=[(str(p),str(args.cache/(hashlib.sha256(p.relative_to(ROOT).as_posix().encode()).hexdigest()[:16]+'.npz'))) for p in paths]
    records=[]
    if args.reuse_cache:
        records=json.loads((args.output/'file_audit.json').read_text(encoding='utf-8'))
        assert all(hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in records)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(audit_one,t):Path(t[0]) for t in tasks}
            for f in as_completed(futures):
                record=f.result();records.append(record)
                print(json.dumps({'file':record['path'],'rows':record['rows'],'aps':record['ap_count'],
                    'empty_style_rows':record['empty_or_style_only_rows'],'ragged':record['ragged_rows']},ensure_ascii=False),flush=True)
                write_json(args.output/'file_audit_partial.json',records)
    records.sort(key=lambda r:r['path'])
    for r in records:
        if r['role']=='map':continue
        with np.load(ROOT/r['cache']) as a:
            valid=np.isfinite(a['rssi']).all(1)&np.isfinite(a['xy']).all(1)
            signals=a['rssi'][valid];labels=a['labels'][valid]
            summary={'valid_numeric_fingerprint_rows':int(valid.sum()),'invalid_numeric_or_coordinate_rows':int((~valid).sum()),
                'unique_building_floor_xy':len({tuple(row) for row in labels}),
                'x_range':finite_range(a['xy'][valid,0]),'y_range':finite_range(a['xy'][valid,1]),
                'rssi_range':finite_range(signals),'duplicate_excess_rssi':len(signals)-len(set(a['fp'][valid])),
                'duplicate_excess_labelled_rssi':len(signals)-len(set(a['labelled'][valid]))}
            code=r['candidate_missing_code']
            if code is not None:
                seen=signals!=code;counts=seen.sum(1)
                summary.update(missing_code_hypothesis=code,missing_fraction=float((signals==code).mean()),
                    all_undetected_rows=int((counts==0).sum()),detected_mean=float(counts.mean()),
                    detected_median=float(np.median(counts)),detected_rssi_range=finite_range(signals[seen]),
                    never_detected_aps=[r['ap_columns'][j] for j in np.flatnonzero(~seen.any(0))])
            r['numeric_summary']=summary
    write_json(args.output/'file_audit.json',records)
    pairs=[]
    for train in records:
        if train['role'] not in ('train','train_mean'):continue
        for test in records:
            if test['case']!=train['case'] or test['role']!='test':continue
            if train['ap_columns']!=test['ap_columns']:
                pairs.append({'train':train['path'],'test':test['path'],'ap_order_equal':False});continue
            with np.load(ROOT/train['cache']) as a,np.load(ROOT/test['cache']) as b:
                av=np.isfinite(a['rssi']).all(1)&np.isfinite(a['xy']).all(1)
                bv=np.isfinite(b['rssi']).all(1)&np.isfinite(b['xy']).all(1)
                af,bf=set(a['fp'][av]),b['fp'][bv];al,bl=set(a['labelled'][av]),b['labelled'][bv]
                ak={tuple(r) for r in a['labels'][av]};bk=[tuple(r) for r in b['labels'][bv]]
                item={'case':train['case'],'train':train['path'],'train_role':train['role'],'test':test['path'],
                    'ap_order_equal':True,'train_rows':int(av.sum()),'test_rows':int(bv.sum()),
                    'train_invalid_rows_separately_reported':int((~av).sum()),'test_invalid_rows_separately_reported':int((~bv).sum()),
                    'distinct_rssi_intersection':len(af&set(bf)),
                    'test_rows_with_train_rssi':sum(v in af for v in bf),
                    'distinct_labelled_rssi_intersection':len(al&set(bl)),
                    'test_rows_with_train_labelled_rssi':sum(v in al for v in bl),
                    'unique_test_locations':len(set(bk)),
                    'test_locations_in_train':len(ak&set(bk)),
                    'test_rows_at_train_locations':sum(v in ak for v in bk)}
                train_xy=a['xy'][av];test_xy=b['xy'][bv]
                outside=((test_xy<train_xy.min(0))|(test_xy>train_xy.max(0))).any(1)
                item['test_rows_outside_train_xy_bounding_box']=int(outside.sum())
                never=train['candidate_never_detected_ap_columns'] if train['candidate_missing_code'] is not None else []
                if never:
                    idx=[train['ap_columns'].index(h) for h in never]
                    candidate=train['candidate_missing_code']
                    observed=(b['rssi'][bv][:,idx]!=candidate)&np.isfinite(b['rssi'][bv][:,idx])
                    item['train_unseen_aps_detected_in_test']=[never[j] for j in np.flatnonzero(observed.any(0))]
                pairs.append(item)
    write_json(args.output/'train_test_overlap.json',pairs)
    means=[]
    for mean in records:
        if mean['role']!='train_mean':continue
        raw=next(r for r in records if r['case']==mean['case'] and r['role']=='train')
        with np.load(ROOT/raw['cache']) as a,np.load(ROOT/mean['cache']) as b:
            differences=[]
            for rp,row in zip(b['rp'],b['rssi']):
                group=a['rssi'][a['rp']==rp]
                if len(group):differences.extend((row-group.mean(0)).tolist())
            delta=np.asarray(differences)
            means.append({'case':mean['case'],'raw':raw['path'],'mean':mean['path'],
                'compared_cells':len(delta),'max_absolute_difference_from_raw_mean':float(np.abs(delta).max()),
                'fraction_within_half_db_of_raw_mean':float((np.abs(delta)<=.50000001).mean()),
                'mean_absolute_difference':float(np.abs(delta).mean())})
    write_json(args.output/'mean_variant_checks.json',means)
    plan=json.loads((args.output/'relocation_manifest.json').read_text())
    assert {r['path']:r['sha256'] for r in records}=={r['new_path']:r['sha256'] for r in plan['records']}
    write_json(args.output/'verification.json',{'raw_file_count':len(records),'raw_sha256_preserved':True,
        'train_test_pairs_audited':len(pairs),'workbooks':sum(r['path'].endswith('.xlsx') for r in records),
        'models_trained':False,'missing_code_semantics_not_assumed_verified':True,'reader':'lxml' if FAST_XML else 'stdlib XML',
        'source_date':'2026-10-14'})
    print('COMPLETE: read-only dataset audit; no training or raw edits.',flush=True)

if __name__=='__main__':main()
