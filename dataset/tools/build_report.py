"""Build the Vietnamese dataset review from measured audit JSON and caches."""
from pathlib import Path
from collections import defaultdict
import csv,json,hashlib,os,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'dataset/analysis/2026-10-14'
FIG=OUT/'figures'
FIG.mkdir(exist_ok=True)
records=json.loads((OUT/'file_audit.json').read_text(encoding='utf-8'))
pairs=json.loads((OUT/'train_test_overlap.json').read_text(encoding='utf-8'))
names={'ujiindoorloc':'UJIIndoorLoc','dlulib':'DLULIB','tuji':'TUJI1','hcxy':'SOD–HCXY','syl':'SOD–SYL','uji_library':'UJI Library','hus':'HUS'}
order=['ujiindoorloc','dlulib','tuji','hcxy','syl','uji_library','hus']
def fmt(n):return f'{int(n):,}'.replace(',','.')
def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in rows)+'\n'
def rows_for(family,role):return [r for r in records if r['family']==family and r['role']==role]
def size(family,role):return sum(r['numeric_summary']['valid_numeric_fingerprint_rows'] for r in rows_for(family,role))
def link(r):return '['+Path(r['path']).name+']('+os.path.relpath(ROOT/r['path'],OUT).replace('\\','/')+')'
def savefig(name):
    plt.savefig(FIG/(name+'.png'),dpi=180,bbox_inches='tight')
    plt.savefig(FIG/(name+'.svg'),bbox_inches='tight')
    plt.close()
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})

# One dataset view per AP namespace; raw observations only, mean variants separate.
fig,ax=plt.subplots(figsize=(11,5));y=np.arange(len(order))
aps=[rows_for(f,'train')[0]['ap_count'] for f in order]
ax.barh(y,aps,color=['#517394' if f!='hus' else '#aa625c' for f in order])
ax.set_yticks(y,[names[f] for f in order]);ax.invert_yaxis();ax.set_xlabel('Số cột AP trong dữ liệu cung cấp')
for i,v in enumerate(aps):ax.text(v+5,i,str(v),va='center')
ax.set_xlim(0,700);ax.grid(axis='x',alpha=.15)
savefig('01_so_ap')
fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(order))
train=[size(f,'train') for f in order];test=[size(f,'test') for f in order]
ax.bar(x-.2,train,.4,label='Train nguyên mẫu',color='#517394');ax.bar(x+.2,test,.4,label='Test cung cấp',color='#d08a2a')
ax.set_yscale('log');ax.set_ylabel('Mẫu hợp schema (thang log)');ax.set_xticks(x,[names[f] for f in order]);ax.legend()
for i,(a,b) in enumerate(zip(train,test)):
    ax.text(i-.2,a*1.08,fmt(a),ha='center',fontsize=9);ax.text(i+.2,b*1.08,fmt(b),ha='center',fontsize=9)
ax.set_ylim(100,180000);savefig('02_so_mau')
family_pairs={f:[p for p in pairs if p['train_role']=='train' and any(r['path']==p['train'] and r['family']==f for r in records)] for f in order}
overlap=[];location=[]
for f in order:
    pp=family_pairs[f];n=sum(p['test_rows'] for p in pp)
    overlap.append(100*sum(p['test_rows_with_train_labelled_rssi'] for p in pp)/n)
    location.append(100*sum(p['test_rows_at_train_locations'] for p in pp)/n)
fig,ax=plt.subplots(figsize=(12,5))
ax.bar(x-.2,location,.4,label='Trùng tọa độ số học: có thể là đo lại',color='#517394')
ax.bar(x+.2,overlap,.4,label='Trùng RSSI + tọa độ: cần kiểm tra độc lập',color='#b65f55')
ax.set_xticks(x,[names[f] for f in order]);ax.set_ylabel('% mẫu test');ax.set_ylim(0,130);ax.legend(loc='upper center',fontsize=9)
for i,(a,b) in enumerate(zip(location,overlap)):
    ax.text(i-.2,a+2,f'{a:.1f}',ha='center',fontsize=9);ax.text(i+.2,b+2,f'{b:.1f}',ha='center',fontsize=9)
savefig('03_trung_train_test')
library=next(r for r in records if r['case']=='uji_library/floor_5' and r['role']=='train')
months=np.arange(1,26);offline=[int(library['metadata']['Month']['counts'][str(m)]) for m in months]
online=[]
for m in months:online.append(sum(int(r['metadata']['Month']['counts'].get(str(m),0)) for r in records if r['case']=='uji_library/floor_5' and r['role']=='test'))
fig,ax=plt.subplots(figsize=(12,5))
ax.bar(months-.18,offline,.36,label='Data-Training',color='#517394');ax.bar(months+.18,online,.36,label='Tổng Test-01…05',color='#d08a2a')
ax.set_xlabel('Month ghi trong file');ax.set_ylabel('Mẫu tầng 5');ax.set_xticks(months);ax.legend()
savefig('04_library_theo_thang')
panels=[('dlulib/floor_1','DLULIB tầng 1'),('tuji','TUJI1'),('hcxy','SOD–HCXY'),('syl','SOD–SYL'),('uji_library/floor_3','UJI Library tầng 3'),('hus','HUS')]
fig,axes=plt.subplots(2,3,figsize=(14,9))
for ax,(case,label) in zip(axes.flat,panels):
    for role,color,marker in [('train','#517394','o'),('test','#d08a2a','x')]:
        points=[]
        for r in records:
            if r['case']!=case or r['role']!=role:continue
            with np.load(ROOT/r['cache']) as a:
                valid=np.isfinite(a['rssi']).all(1)&np.isfinite(a['xy']).all(1)
                points.extend(a['xy'][valid].tolist())
        unique=np.unique(np.asarray(points),axis=0)
        ax.scatter(unique[:,0],unique[:,1],s=12,alpha=.7,c=color,marker=marker,label=role)
    ax.set_title(label);ax.set_xlabel('X trong file / map');ax.set_ylabel('Y trong file / map');ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.12);ax.legend(fontsize=8)
fig.tight_layout();savefig('05_vi_tri_thu_thap')

summary=[]
for f in order:
    r=rows_for(f,'train')[0]
    summary.append({'dataset_view':names[f],'family':f,'aps':r['ap_count'],'train_raw_rows':size(f,'train'),
        'train_mean_rows':size(f,'train_mean'),'test_rows':size(f,'test'),
        'test_duplicate_labelled_train_percent':overlap[order.index(f)],'test_at_exact_train_coordinates_percent':location[order.index(f)]})
(OUT/'dataset_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
with (OUT/'dataset_summary.csv').open('w',encoding='utf-8-sig',newline='') as s:
    w=csv.DictWriter(s,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
with (OUT/'file_inventory.csv').open('w',encoding='utf-8-sig',newline='') as s:
    fields=['path','family','case','role','bytes','sha256','rows','ap_count','empty_or_style_only_rows','ragged_rows','valid_rows','invalid_rows']
    w=csv.DictWriter(s,fieldnames=fields);w.writeheader()
    for r in records:
        row={k:r.get(k,'') for k in fields};v=r.get('numeric_summary',{})
        row['valid_rows']=v.get('valid_numeric_fingerprint_rows','');row['invalid_rows']=v.get('invalid_numeric_or_coordinate_rows','');w.writerow(row)

sources=[
 {'id':'UJI','title':'UCI UJIIndoorLoc','url':'https://archive.ics.uci.edu/dataset/310/ujiindoorloc','local_verification':'SHA-256 equals previously verified canonical CSV in data/raw/ujiindoorloc'},
 {'id':'DLU','title':'DLULIB data descriptor (publisher, pre-proof at review time)','url':'https://doi.org/10.1016/j.dib.2026.113292','local_verification':'Matches declared 17 AP, 650 RP, 19500 raw fingerprints; no native workbook hash obtained'},
 {'id':'TUJI','title':'TUJI1 original dataset and descriptor','url':'https://zenodo.org/records/7641701','paper':'https://doi.org/10.1016/j.dib.2024.110356','local_verification':'All RSSI values and six numeric labels exactly match native CSV in DATASET.zip; local metadata header names differ'},
 {'id':'SOD','title':'SODIndoorLoc author repository and descriptor','url':'https://github.com/bijingxue/SODIndoorLoc','paper':'https://doi.org/10.1186/s43020-022-00086-y','local_verification':'HCXY RSSI/XY joined by RP and SYL all numeric columns exactly match author CSVs'},
 {'id':'LIB','title':'Long-Term Wi-Fi dataset v2.2','url':'https://zenodo.org/records/3748719','local_verification':'Local Excel has 620 AP and Month1..25, but no byte-level/native-row comparison with the 103584-fingerprint original was performed'},
 {'id':'HUS','title':'Provenance not identified','url':None,'local_verification':'Local TXT/maps only; supplied testcase overlap invalidates an independent heldout claim'}]
(OUT/'sources.json').write_text(json.dumps({'checked_date':'2026-10-14','sources':sources},ensure_ascii=False,indent=2),encoding='utf-8')

parts=[]
def add(t):parts.append(t.strip()+'\n')
add('''# Đánh giá các bộ dữ liệu Wi-Fi IPS được cung cấp

Ngày kiểm tra: **14/10/2026**. Đây là kiểm kê và đánh giá thiết kế thực nghiệm, **không phải kết quả huấn luyện KAN-Mixer mới**.

Đã đọc đủ 34 file, khoảng 267,33 MiB. Sau khi bỏ phần dòng trống chỉ có định dạng và ghi riêng 6 dòng ngoài schema, có 193.527 bản ghi train/test hợp schema số; thêm 650 vector Mean và hai bảng map tọa độ. Vector Mean không được cộng như 650 phép đo độc lập.

Bảy nhóm được cung cấp tương ứng sáu họ nguồn: HCXY và SYL cùng thuộc SODIndoorLoc. Các tầng là các phần con, không tự xem mỗi tầng hoặc mỗi tháng là một bộ độc lập.

## 1. Kết luận để chọn bộ nghiên cứu

**Ưu tiên mở rộng DLULIB và TUJI1**, giữ UJIIndoorLoc làm đối chứng lịch sử. DLULIB bổ sung ba tầng, ít AP và metadata người/thiết bị/hướng; TUJI1 có nguồn gốc đối chiếu được và test ở lưới vị trí khác train. HCXY/SYL phù hợp đánh giá các điểm chưa đo và dữ liệu lặp; UJI Library phù hợp nghiên cứu biến đổi theo thời gian nhưng cần tách theo Month. **HUS chưa nên dùng với testcase hiện có để công bố độ chính xác độc lập.** Đây là đánh giá khả năng thiết kế thí nghiệm, không phải dự đoán bộ nào sẽ có sai số tốt nhất.

Các cảnh báo cần xử lý trước khi train:

1. HUS có 204/205 và 205/205 mẫu testcase trùng nguyên RSSI kèm tọa độ với train.
2. TUJI1 bị đặt sai tên các cột metadata so với nguồn gốc: phải ánh xạ theo nghĩa thật.
3. DLU có mã −100 rất thường gặp; ý nghĩa “không thấy AP” cần xác nhận với data descriptor/giảng viên, không áp dụng quy tắc 100 của UJI một cách tự động.
4. UJI Library có 6 dòng thống kê/ngoài schema ở cuối hai workbook, và cả train lẫn test đang chứa Month1..25.
5. Pipeline hiện tại khóa 520 AP và schema UJIIndoorLoc. Bộ khác chưa thể chạy trực tiếp chỉ bằng cách đổi tên file.
''')
add('## 2. Số liệu tổng quan\n\n'+table(['Nhóm','AP','Train nguyên mẫu','Train Mean riêng','Test cung cấp'],[[s['dataset_view'],s['aps'],fmt(s['train_raw_rows']),fmt(s['train_mean_rows']) if s['train_mean_rows'] else '—',fmt(s['test_rows'])] for s in summary]))
add('''Train UJI là 19.937 mẫu development gốc, chưa phải 16.025 train nội bộ của thực nghiệm trước. Test Library là tổng các file/tháng/tầng được cung cấp, chưa phải một holdout thời gian đã chốt. 410 test HUS là tổng hai testcase có vấn đề trùng train.

![Số AP](figures/01_so_ap.png)

![Số mẫu](figures/02_so_mau.png)

## 3. Cách đọc các kiểm tra

- **Dòng hợp schema số:** các AP và hai tọa độ đọc được số hữu hạn. Nó không tự chứng minh nhãn đúng, đơn vị đúng hay mẫu độc lập.
- **Trùng RSSI:** các giá trị trên toàn bộ AP bằng nhau chính xác. Hai vị trí có thể có cùng fingerprint; không tự loại những trường hợp đó khỏi test.
- **Trùng RSSI + nhãn vị trí:** cùng vector AP và cùng tọa độ/lớp đã có trong train. Đây là nguy cơ đánh giá lại quan sát đã học, đặc biệt nghiêm trọng ở HUS.
- **Trùng tọa độ:** so sánh số học chính xác, có kèm tầng/tòa nếu có. Đo lại ở cùng điểm không tự là rò rỉ; ngược lại, không trùng chính xác cũng không chứng minh hai điểm cách xa nhau.
- **Duplicate excess:** số dòng trừ số khóa phân biệt, không phải số cặp trùng. Thời gian/thiết bị/hướng đo có thể làm full-row khác nhau dù RSSI và vị trí không đổi.
- **AP không thấy trong train nhưng có trong test:** thay đổi hỗ trợ tín hiệu, không được dùng test để fit chuẩn hóa hay chọn AP.
- Mọi tỷ lệ “không thấy AP” dựa trên mã chỉ định trong phần kiểm tra. Mã chưa có chứng cứ nguồn được ghi là giả thuyết, không biến thành sự thật từ một histogram.

![Hai loại trùng train/test](figures/03_trung_train_test.png)

## 4. DLULIB — ba tầng của thư viện Đại học Đà Lạt

**Từ nguồn công bố:** data descriptor của DLULIB nêu 17 AP, 650 RP và 19.500 fingerprint, gồm train/test theo tầng cùng metadata thiết bị/người/hướng. Trang publisher tại thời điểm kiểm tra ghi bản pre-proof. [Nguồn DLULIB](https://doi.org/10.1016/j.dib.2026.113292).

**Từ file trên máy:** train All_20 có AP1..AP17, X, Y, ID_RP, FloorID, Smartphones, Users, Direction, SampleTimes. Test không có Users/Direction; Mean chỉ còn AP và X/Y/RP/Floor. Mỗi RP có đúng 20 train và 10 test, Mean có một vector/RP. Tổng nguyên mẫu là 13.000 train + 6.500 test, đúng 650×30.
''')
add(table(['Tầng','RP','Train All_20','Mean','Test','Dòng trống định dạng trong train/test'],[[i,rows_for('dlulib','train')[j]['rp_count'],fmt(rows_for('dlulib','train')[j]['rows']),rows_for('dlulib','train_mean')[j]['rows'],fmt(rows_for('dlulib','test')[j]['rows']),str(rows_for('dlulib','train')[j]['empty_or_style_only_rows'])+' / '+str(rows_for('dlulib','test')[j]['empty_or_style_only_rows'])] for j,i in enumerate([1,2,3])]))
add('''Các dòng trống này không phải mẫu RSSI thiếu: chúng không có dữ liệu, chỉ giữ định dạng Excel. Không dùng `max_row` làm số mẫu.

Mọi test đều ở RP train đã đo; không có trùng nguyên RSSI kèm vị trí. Tầng 3 có một fingerprint test bằng train nhưng tọa độ khác, nên không được gọi đó là sao chép nhãn. Metadata Smartphones của train và test sử dụng cùng các tập tên theo từng tầng; split hiện tại không chứng minh khả năng với thiết bị chưa thấy.

Mean đã được đối chiếu với trung bình số học của 20 mẫu từng RP: tất cả 11.050 ô AP lệch tối đa 0,5 dB, phù hợp làm tròn. Phép tính ấy tính cả −100. Nếu −100 là placeholder, trung bình này không đồng nghĩa “trung bình RSSI chỉ khi AP được phát hiện”, và mask suy từ Mean khác mask mỗi lần quét. Không gộp Mean và All_20 thành các mẫu train mới; không dùng Mean tính từ cả nhóm để dự đoán validation đã tách từ chính nhóm đó.

RSSI không có +100, có nhiều −100, các giá trị còn lại đến −25 dBm. File không kèm mô tả mã thiếu/đơn vị X,Y, nên cần xác nhận −100 và mét trước chuẩn hóa/metric. Giữ nguyên dữ liệu hiện tại; chưa impute, đổi sentinel hay sửa tên thiết bị.

**Thiết kế nên dùng:** giữ test cung cấp cho giao thức đo lại điểm; tách train/validation theo nhóm RP hoặc nhóm thu nhận được xác minh, có phân tầng FloorID. Báo thêm một giao thức giữ lại RP và một giao thức giữ thiết bị nếu đủ coverage. IDs Users có thể cần namespace theo tầng; không mặc định cùng số là cùng người ở mọi tầng. Gộp ba tầng để đánh giá floor chỉ sau khi xác nhận AP1..17 biểu diễn cùng AP ở mọi file. X/Y/RP/Floor chỉ làm nhãn; người/thiết bị/hướng chỉ làm group/audit cho baseline RSSI.

## 5. TUJI1 — cần sửa nghĩa metadata trong adapter

Đã tải `DATASET.zip` từ [Zenodo của tác giả](https://zenodo.org/records/7641701). Cả train 6.752 và test 2.147 có 310 AP. So sánh theo đúng thứ tự dòng: **tất cả RSSI và cả sáu cột nhãn đều bằng nguồn gốc chính xác**. Chứng cứ và SHA archive nằm trong `tuji_original_comparison.json`.
''')
add(table(['Tên cột trong file cung cấp','Nghĩa theo nguồn gốc'],[['LONGITUDE','x trong hệ tọa độ cục bộ'],['LATITUDE','y trong hệ tọa độ cục bộ'],['FLOOR','z, không phải floor'],['BUILDINGID','floor, không phải building'],['SPACEID','building, không phải room/RP'],['RELATIVEPOSITION','device ID, không phải vị trí tương đối']]))
add('''Trong phiên bản này z/floor/building đều 0 nên lỗi tên dễ bị che giấu. Device ID1..5 tương ứng S20, S7, POCO, tabS7, A12 theo `Device_labels.csv` gốc. Các vị trí train/test không trùng chính xác, nhưng lưới gần nhau; cả năm thiết bị đều xuất hiện ở cả hai split. Không gọi split đó là unseen-device.

Nguồn TUJI1 mô tả +100 cho AP không thấy và RSSI âm cho tín hiệu nhận được. Có **một ô −101 trong train**, tồn tại cả ở nguồn gốc, nên không tự sửa về −100 dựa trên một khoảng mô tả. [Data descriptor](https://doi.org/10.1016/j.dib.2024.110356).

**Thiết kế:** adapter giải mã sáu cột theo nghĩa thật, giữ split công bố; validation từ train theo nhóm tọa độ hoặc spatial blocks, không trộn các thiết bị ở cùng tọa độ qua train/val khi mục tiêu là unseen-location. Đánh giá riêng theo device1..5; một giao thức leave-device-out cần xây dựng mới và báo riêng. Tòa/tầng hằng số không phải bài toán phân loại có ý nghĩa; tập trung sai số X/Y và độ bền thiết bị. Không tự đổi CSV gốc trong tác vụ này.

## 6. SODIndoorLoc — HCXY và SYL là hai phần cùng nguồn

Đã tải bốn CSV từ [repo của tác giả](https://github.com/bijingxue/SODIndoorLoc). HCXY: RSSI và X/Y sau join RP→map khớp toàn bộ nguồn, theo đúng thứ tự. SYL: toàn bộ các ô số của train/test khớp nguồn. `sod_original_comparison.json` lưu URL và hash download; tên cột địa lý ở file cung cấp không có nghĩa kinh/vĩ độ tính bằng độ.

Bài gốc xác định tọa độ E/N cục bộ, đơn vị mét; HCXY ở tầng4/tòa2 và SYL ở tầng4/tòa3, +100 biểu diễn AP không thấy. [Data descriptor SODIndoorLoc](https://doi.org/10.1186/s43020-022-00086-y).
''')
add(table(['Phần','AP','Train / điểm','Test / điểm','Khóa RSSI+vị trí phân biệt trong train / test'],[['HCXY',347,'11.370 / 379','860 / 86','5.014 / 316'],['SYL',363,'8.880 / 296','1.020 / 102','792 / 140']]))
add('''HCXY có RP và 347 MAC, kèm map 465 ID: train379 và test86 không trùng tọa độ. Bản TXT đã bỏ metadata Floor/Building/Scene/User/Phone/SampleTimes, nhưng có thể khôi phục trong bản dẫn xuất từ nguồn đã đối chiếu; không đoán những cột ấy từ số RP.

SYL có LONGITUDE/LATITUDE thực chất E/N, FloorID, BuildingID, SceneID, UserID, PhoneID, SampleTimes. Có 30 mẫu/RP train và 10 mẫu/TP test; chỉ một vị trí test (10 dòng) trùng tọa độ train. Không có trùng RSSI kèm vị trí giữa split. RSSI lặp rất nhiều trong từng RP: SampleTimes thay đổi không làm những fingerprint giống nhau thành quan sát tín hiệu độc lập.

**Thiết kế:** giữ test nguồn; validation theo RP/spatial block từ train, giữ cả 30 mẫu một RP trong cùng phía. Báo metric theo fingerprint và theo điểm, tránh độ tin cậy giả từ nhiều lần lặp. Có thể đối chiếu raw30 và mean/RP fit trên train, dùng cùng dữ liệu kiểm chứng. Không tự dedup test để làm kết quả đẹp; không hợp nhất cùng tên MAC1 giữa hai tòa nếu chưa có mapping physical AP. Bộ CETC331 của SOD không được cung cấp, nên đây chưa phải toàn bộ SODIndoorLoc.

## 7. UJI Library — nhiều thời điểm và nhiều kiểu điểm test

Có hai tầng3/5, mỗi file 620 AP, schema `x,y,Floor,Month,AP1..AP620`. File hợp schema có 23.040 train và 78.000 test, tổng101.040. Nguồn [Zenodo v2.2](https://zenodo.org/records/3748719) mô tả 25 tháng, 103.584 fingerprint và thêm thiết bị ở chiến dịch cuối. Excel địa phương có Month1..25, nhưng chưa so từng hàng với bản native: không gọi chúng là bản nguyên vẹn v2.2 hoặc khẳng định mọi dòng cùng một điện thoại.

Data-Training mỗi tầng có Month1:4.320, Month2..24:288/tháng, Month25:576. Nó **không chỉ chứa tháng đầu**. Cả năm file test cũng trải đủ 25 tháng.
''')
add(table(['Test mỗi tầng','Mẫu hợp schema','Điểm test/tầng','Trùng vị trí train','Ngoài bounding box X/Y train'],[['01',7200,24,'100%','0'],['02',7200,24,'0%','2.400 dòng'],['03',7200,24,'0%','2.400 dòng'],['04',10200,34,'0%','1.200 dòng'],['05',7200,24,'100%','0']]))
add('''Hai workbook tầng3 chứa phần thống kê sau quan sát. Data-Training: dòng11522 thiếu x/y/Floor/Month, chứa giá trị AP và một số ngoài cột AP; dòng11523 chỉ có B=240. Test-01: dòng7202 có RSSI nhưng thiếu nhãn; dòng7204–7206 có các số tổng hợp ở B/C/E. Đây là **6 dòng ngoài schema**, không phải tầng150 hoặc các vị trí y=39.000. Vị trí ô và cached values được lưu ở `library_schema_issues.json`. Có 1.247 ô công thức trong phần cuối này; không recalculation hoặc sửa workbook gốc.

![Month của UJI Library](figures/04_library_theo_thang.png)

**Thiết kế:** đọc range quan sát/kiểm đủ nhãn và log các dòng không phải mẫu, giữ raw. Nếu nghiên cứu temporal drift: train dữ liệu offline Month1, chọn cấu hình bằng offline Month2..5, rồi báo online test Month6..25 theo từng tháng/tầng/loại điểm. Chốt tháng trước khi chạy; không train Data-Training tất cả25 tháng rồi tuyên bố generalization sang tháng chưa thấy. Nếu cần thích nghi theo tháng, tạo giao thức thích nghi riêng và tách cost/metric.

Head điểm tham chiếu hiện tại lấy tổ hợp lồi các điểm train. Nhiều điểm Test02/03/04 nằm ngoài bounding box train nên head thuần và hybrid catalogue train không thể ngoại suy tới mọi vị trí ấy. Đây là giới hạn hình học của thiết kế, không phải kết quả đo model. Có thể thử residual-coordinate head riêng, chọn bằng validation, đồng thời giữ baseline/reference nguyên trạng để so sánh. Bounding box chỉ là một kiểm tra cần, không thay thuật toán kiểm bao lồi đầy đủ.

## 8. HUS — đang có trùng nguyên mẫu thử với train

Train có20.500 mẫu,5 AP,205 RP (100 mẫu/RP). Map có205 ID, x từ0 đến8,5 và y từ0,5 đến6. Không có floor/building/device/time trong file; chưa xác định được data descriptor, đơn vị tọa độ hoặc ý nghĩa tên HUS. Không có ô +100/−100/0 trong RSSI quan sát, nhưng không tự suy ra một quy tắc sentinel mới từ sự vắng mặt đó.

Testcase1 có204/205 mẫu trùng đúng RSSI và tọa độ với train; testcase2 có205/205. Cả hai ở toàn bộ205 RP của train. Một model ghi nhớ hoặc nearest-neighbor có thể hưởng lợi trực tiếp; không dùng những file này làm bằng chứng sai số trên dữ liệu độc lập.

**Thiết kế:** giữ nguyên testcase để truy vết; hỏi nguồn quy trình tạo testcase và đo dữ liệu mới. Nếu chưa có, chỉ dùng làm demo/debug, hoặc tạo một giao thức giữ nguyên nhóm RP/khóa trùng khi chia từ train. Giao thức giữ RP đo unseen-location và không thay thế cho test độc lập mới. Không xóa các hàng trùng rồi gọi phần còn lại là đủ benchmark: testcase1 chỉ còn một mẫu không trùng, testcase2 không còn mẫu nào.

## 9. UJIIndoorLoc — bản bổ sung trùng bộ đang sử dụng

Hai CSV mới có SHA-256 bằng chính xác hai CSV canonical trong `data/raw/ujiindoorloc/`. Không phải một bộ mới để nhân đôi thí nghiệm. Giữ cả bản giảng viên cung cấp trong raw và bản operational cũ; không ghi đè hay hợp hai bản thành nhiều mẫu.

Vẫn là520AP,19.937development,1.111official test; split cũ16.025/3.912/1.111 và checkpoint không bị thay đổi. Development có76 dòng không thấy AP nào và các quan sát trùng; giữ chính sách lịch sử và báo lỗi theo nhóm, không loại khỏi metric để đạt mục tiêu. Tọa độ LONGITUDE/LATITUDE là các số phẳng gốc; CRS đầy đủ còn chưa xác minh trong repo. [UCI](https://archive.ics.uci.edu/dataset/310/ujiindoorloc).

## 10. Những gì cần thay để dùng KAN-Mixer cho nhiều bộ

Hiện `FingerprintInputs` yêu cầu WAP001..WAP520 và RSSI[N,520]; reader kiểm schema529 cột; `ReferenceKANMixer` kiểm input[B,520,2] và token×features=1040. Vì thế **mới UJIIndoorLoc chạy nguyên pipeline được**. Các bộ khác chưa được thêm vào Factory/CLI ở tác vụ này.
''')
add(table(['Thành phần đề xuất','Quy tắc'],[['DatasetSpec/adapter','n_aps, AP order, sentinel/range, coordinate frame/units, roles của cột, metadata và nguồn'],['Shared inputs','Cho phép D AP theo spec; giữ nguyên mặc định UJI và kiểm AP order/hash cho từng dataset'],['Neural tokenizer','T×F=2×D, hoặc padding cố định có mask; giữ tất cả AP, không cắt620 xuống520'],['Nhãn IPS','Mapping tòa/tầng theo dữ liệu; không ép13 lớp hoặc coi thiếu nhãn là lớp0 đo được'],['Preprocessing','Fit center/scale/statistics/catalogue chỉ train, cùng split cho các thuật toán'],['Reference decoder','Catalogue theo train của bộ mới; không tái dùng872 điểm UJI hay checkpoint UJI như mô hình đã train cho bộ khác'],['Chi phí','Báo parameters, optimizer steps, runtime/RAM/VRAM; các D/head/reference count khác làm budget khác']]))
add('''Ví dụ token hóa để thử sau khi có adapter: TUJI310AP→31token×20features (10AP/token); HCXY347AP→pad350 rồi35×20; SYL363AP→pad370 rồi37×20; Library620AP→31×40 (20AP/token); DLU17AP→17×2 là một lựa chọn nhỏ. Đây là phương án thiết kế, chưa có trong code model và chưa được chọn bởi thực nghiệm. Padding là AP giả không phát hiện, có mask0, không phải thêm phép đo mới.

Không gộp các bộ vào một bảng bằng cùng AP1/MAC1: tên cột ordinal không tự cho biết cùng BSSID. Tọa độ ở các tòa/nguồn không cùng frame; không concatenate rồi giả sử một không gian Euclid chung. Với một tòa/tầng cố định, accuracy tòa/tầng không chứng minh định vị tốt: báo N/A nếu bài toán không phân loại, tập trung mean2D/median/P90/RMSE và CDF. Không dùng chênh lệch FloorID3→5 làm chiều cao mét khi không có chiều cao thật.

Mục tiêu7–8m trước đó thuộc UJIIndoorLoc. Không áp nguyên mốc đó cho khu vực nhỏ hoặc bộ có cấu trúc khác; chốt chỉ số, đối chứng nearest-neighbor và mục tiêu từng bộ bằng validation trước khi xem kết quả mới. Tác vụ audit đã xem phân bố/nhãn test để kiểm chất lượng; ghi rõ bối cảnh này trong protocol tương lai.

## 11. Lộ trình thực nghiệm đề xuất

**Tối thiểu:** giữ UJI đã có; thêm adapter TUJI đã xác nhận metadata, tiếp đến DLU sau xác nhận sentinel/đơn vị/AP mapping. Mỗi bộ giữ test nguồn, tạo validation có group, chạy weighted fingerprint baseline và KAN reference thuần trên cùng split; bản hybrid báo riêng. Chạy3 seeds sau lựa chọn, lưu hash trước đánh giá, mean±sampleSD và paired row errors. Mean2D vẫn dùng đủ mẫu test thuộc giao thức đã chốt; dữ liệu không phải quan sát được tách bằng điều kiện schema và log trước train.

**Mở rộng:** thêm HCXY/SYL để đo holdout vị trí; Library theo tháng để đo temporal drift; thêm leave-device-out TUJI/DLU và raw-vs-mean cho DLU/SOD. Nhiều lần quét cùng RP/thời điểm không phải nhiều đơn vị độc lập để bootstrap; lấy group/RP làm đơn vị nếu có thiết kế phù hợp. HUS chỉ tham gia kết luận khi có test không trùng và nguồn được xác minh.

![Không gian điểm đo](figures/05_vi_tri_thu_thap.png)

Hình dùng đơn vị tọa độ nguyên bản trên từng panel, không giả định mọi frame giống nhau. Các phần chưa xác minh đơn vị không được diễn giải thành mét chỉ từ tên x/y.

## 12. Thư mục và khả năng tái lập

```text
dataset/
  README.md
  raw/
    dlulib/floor_1, floor_2, floor_3/
    sodindoorloc/hcxy/, syl/
    hus/
    tuji/
    uji_library/floor_3/, floor_5/
    ujiindoorloc/
  analysis/2026-10-14/
    BAO_CAO_DANH_GIA.md
    dataset_summary.csv, file_inventory.csv
    file_audit.json, train_test_overlap.json
    sources.json, các đối chiếu nguồn và relocation_manifest.json
    figures/                     PNG + SVG, chỉ đồ thị dữ liệu
  tools/audit_datasets.py, build_report.py
```

Giữ nguyên tên các file nguồn bên trong từng tầng/bộ, chỉ đổi thư mục. 34/34 hash và kích thước được kiểm trước/sau; old_path→new_path có trong `relocation_manifest.json`. Không đổi raw, không thay dữ liệu/split UJI operational, không sửa source KAN, không train, commit hoặc push trong tác vụ này. Thư mục dataset mới vẫn chưa được đưa vào Git; chưa tự quyết định chia sẻ dữ liệu nguyên gốc lên remote.

Tái lập audit từ root repo, dùng môi trường NumPy hiện tại:

```powershell
./.venv-cuda/Scripts/python.exe dataset/tools/audit_datasets.py --workers 2
./.venv-cuda/Scripts/python.exe dataset/tools/build_report.py
```

lxml là tùy chọn để đọc nhanh Excel, nếu không có dùng XML chuẩn; không cần Excel/LibreOffice và không thực thi công thức. Cache phân tích ở `runs/dataset_audit_20261014/cache`, nguồn tải bổ sung ở `runs/dataset_source_evidence_20261014/`. `--reuse-cache` chỉ dùng khi SHA raw vẫn khớp; metadata và đối chiếu nguồn bổ sung phải được giữ cùng báo cáo.

## 13. Danh mục từng file

Số dòng dưới đây là số dữ liệu không trống sau header; cột “hợp schema” loại riêng những dòng cuối workbook không đủ AP/nhãn. Map chỉ là lookup RP→XY, không phải fingerprint train.
''')
add(table(['File','Vai trò','Dòng không trống','Hợp schema','AP','Dòng trống định dạng'],[[link(r),r['role'],fmt(r['rows']),fmt(r['numeric_summary']['valid_numeric_fingerprint_rows']) if r['role']!='map' else 'Map',r['ap_count'],fmt(r['empty_or_style_only_rows'])] for r in records]))
add('''## 14. Chứng cứ và những điều còn cần xác nhận

- `relocation_manifest.json`: 34 SHA nguồn và đường dẫn trước/sau.
- `file_audit.json`: mọi header, RSSI, metadata, dòng trống/lệch schema, full-row và signal/label duplicate.
- `train_test_overlap.json`: 22 cặp kiểm tra, tách mean/raw và từng testcase/tháng-file.
- `mean_variant_checks.json`: đối chiếu Mean của DLU với20raw/RP; không coi đó là nguồn độc lập.
- `tuji_original_comparison.json`: số liệu gốc khớp, mapping tên cột cần sửa ở adapter.
- `sod_original_comparison.json`: bốn đối chiếu nguồn và metadata/frame gốc; HCXY metadata bị lược bỏ trong TXT.
- `uji_existing_comparison.json`: hai bản UJI có cùng SHA canonical.
- `library_schema_issues.json`: sáu dòng ngoài quan sát, vị trí ô và cached values.
- `sources.json`: phân biệt nguồn xác nhận, nguồn chỉ khớp mô tả và nguồn chưa biết.

Cần giảng viên/data owner xác nhận: mô tả/nguồn HUS và cách tạo testcase; quy ước −100 và đơn vị XY của DLU; mapping AP vật lý chung giữa các tầng; quy trình chuyển Library native→Excel và chỉ số campaign/thiết bị đã lược bỏ. Không tự bù các thông tin này bằng tên file.

Chưa có thí nghiệm model trên các bộ mới. Các ưu tiên và lộ trình trên là nhận xét thiết kế từ chất lượng dữ liệu, không phải số liệu dự đoán hay thành tích công bố của dự án.
''')
(OUT/'BAO_CAO_DANH_GIA.md').write_text('\n'.join(parts),encoding='utf-8')
readme='''# Các bộ dữ liệu IPS được cung cấp

Đọc [báo cáo đánh giá ngày 14/10/2026](analysis/2026-10-14/BAO_CAO_DANH_GIA.md) trước khi đưa dữ liệu vào pipeline.

- `raw/`: 34 file gốc, giữ nguyên byte; DLU và UJI Library theo tầng, HCXY/SYL gom dưới SODIndoorLoc.
- `analysis/2026-10-14/`: báo cáo, bảng kiểm kê, đối chiếu nguồn và biểu đồ.
- `tools/`: script đọc/audit và tổng hợp; không train mô hình hay sửa raw.

Những điểm cần chú ý: HUS test gần như trùng hoàn toàn train; TUJI có header metadata sai nghĩa; DLU −100 cần xác nhận; Library có phần thống kê cuối workbook và chứa25 tháng. KAN hiện tại chỉ đọc schema 520 AP của UJI, bộ khác cần adapter/điều chỉnh hợp đồng đầu vào.

Đường dẫn cũ→mới và SHA-256 nằm trong [relocation manifest](analysis/2026-10-14/relocation_manifest.json). Hai CSV UJI ở đây là bản sao bộ operational `data/raw/ujiindoorloc`, không hợp nhất để tăng số mẫu.
'''
(ROOT/'dataset/README.md').write_text(readme,encoding='utf-8')
plan=json.loads((OUT/'relocation_manifest.json').read_text())
assert all(hashlib.sha256((ROOT/r['new_path']).read_bytes()).hexdigest()==r['sha256'] for r in plan['records'])
assert sum(size(f,'train')+size(f,'test') for f in order)==193527
assert size('dlulib','train')+size('dlulib','test')==19500
assert all(r['formula_cells_without_cached_value']==0 for r in records)
print(json.dumps({'raw_files_preserved':34,'valid_raw_fingerprints':193527,'mean_vectors_separate':650,'figures':5,'report':str(OUT/'BAO_CAO_DANH_GIA.md')}))
