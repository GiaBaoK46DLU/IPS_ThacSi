/** Stage 8: editable slides from real, completed Stage 7 evidence.
 * Dependencies are isolated in ignored runs/stage8_build; no ML environment changes.
 * npm install --prefix runs/stage8_build pptxgenjs@4.0.1 --no-audit --no-fund
 * node scripts/build_stage8.cjs
 */
'use strict';
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const root = path.resolve(__dirname, '..');
const PptxGenJS = require(path.join(root, 'runs/stage8_build/node_modules/pptxgenjs'));
const JSZip = require(path.join(root, 'runs/stage8_build/node_modules/jszip'));
const reportDir = path.join(root, 'reports/stage7_cuda_v1');
const out = path.join(root, 'docs/lessons/chang_08/presentation');
const buildDir = path.join(root, 'runs/stage8_build');
fs.mkdirSync(out, {recursive:true}); fs.mkdirSync(buildDir, {recursive:true});
const read = p => JSON.parse(fs.readFileSync(p, 'utf8'));
const agg = read(path.join(reportDir, 'aggregate_metrics.json'));
const verify = read(path.join(reportDir, 'verification_report.json'));
const cfg = read(path.join(reportDir, 'frozen_config.json'));
const ids = ['mlp','mlp_mixer','kan_mixer'];
const names = ['MLP','MLP-Mixer','KAN-Mixer'];
const colors = ['2563EB','059669','D97706'];
const navy = '142735', muted = '526778', light = 'F1F5F9';
for (const m of ids) {
  if (agg[m].test.mean_2d.n !== 3) throw Error('Expected three genuine seeds');
  if (!(agg[m].test.mean_2d.mean > 8)) throw Error('Revisit narrative: target status changed');
}
if (cfg.training.max_epochs !== 100) throw Error('Unexpected protocol');
const fm = x => x.toFixed(2).replace('.', ',');
const stat = (m, key, split='test', factor=1) => {
  const v=agg[m][split][key]; return `${fm(v.mean*factor)} ± ${fm(v.std*factor)}`;
};
const res = (m,key,factor=1) => {
  const v=agg[m].resources[key];return `${fm(v.mean*factor)} ± ${fm(v.std*factor)}`;
};
const n = x => new Intl.NumberFormat('vi-VN').format(x);
const sources = {
  data:'https://archive.ics.uci.edu/dataset/310/ujiindoorloc',
  mixer:'https://arxiv.org/abs/2105.01601',
  kan:'https://arxiv.org/abs/2404.19756',
  kanMixer:'https://arxiv.org/abs/2503.08939v1',
  backend:'https://github.com/Blealtan/efficient-kan/tree/7b6ce1c87f18c8bc90c208f6b494042344216b11',
  report:'reports/stage7_cuda_v1/aggregate_metrics.json',
};
const pptx = new PptxGenJS();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'Nghiên cứu IPS';
pptx.subject = 'Tổng kết giao thức và thực nghiệm UJIIndoorLoc';
pptx.title = 'KAN-Mixer cho định vị trong nhà';
pptx.company = '';
pptx.lang = 'vi-VN';
pptx.theme = {headFontFace:'Arial',bodyFontFace:'Arial',lang:'vi-VN'};
const notes = [], shapes=[];
function text(s,t,x,y,w,h,size=21,opts={}) {
  s.addText(t,{x,y,w,h,fontFace:'Arial',fontSize:size,color:navy,margin:0,
    breakLine:false,valign:'middle',isTextBox:true,...opts});
}
function page(title,dark=false) {
  const s=pptx.addSlide(); s.background={color:dark?navy:'FFFFFF'};
  if(title) text(s,title,.65,.45,12,.8,32,{bold:true,color:dark?'FFFFFF':navy});
  text(s,String(pptx._slides.length),12.03,6.99,.65,.22,11,
    {align:'right',color:dark?'AFC3D0':muted});
  return s;
}
function foot(s,t) {text(s,t,.65,6.73,11.65,.39,11,{color:muted});}
function note(s,title,body,refs=[]) {
  const full=body+(refs.length?'\n\nNguồn:\n'+refs.join('\n'):'');
  s.addNotes(full);notes.push({slide:pptx._slides.length,title,body:full});
}
function table(s,rows,x,y,w,widths,size=18,rowH=.64) {
  const styled=rows.map((row,i)=>row.map((v,j)=>({text:String(v),options:{
    bold:i===0,fill:i===0?navy:(i%2===1?'F1F5F9':'FFFFFF'),
    color:i===0?'FFFFFF':(j===0&&i<=3?colors[i-1]:navy),
    align:j===0?'left':'center',valign:'middle'}})));
  s.addTable(styled,{x,y,w,colW:widths,rowH,fontFace:'Arial',fontSize:size,
    border:{type:'solid',color:'FFFFFF',pt:1.5},margin:[.10,.12,.10,.12],
    autoPage:false,verbose:false});
}
function box(s,t,x,y,w,h,color=navy,fill=light,size=20) {
  s.addShape(pptx.ShapeType.rect,{x,y,w,h,line:{color,width:1},fill:{color:fill},radius:0});
  text(s,t,x+.12,y+.07,w-.24,h-.14,size,{align:'center',color,bold:true});
}
function arrow(s,x1,y1,x2,y2,color='64748B',dashed=false) {
  // Diagrams are native slide objects, separate from decorative assets.
  s.addShape(pptx.ShapeType.line,{x:Math.min(x1,x2),y:Math.min(y1,y2),
    w:Math.abs(x2-x1),h:Math.abs(y2-y1),flipH:x2<x1,flipV:y2<y1,
    line:{color,width:1.6,beginArrowType:'none',endArrowType:'triangle',dashType:dashed?'dash':'solid'}});
}
function bar(s,values,x,y,w,h,unit,max) {
  s.addChart(pptx.ChartType.bar,ids.map((m,i)=>({name:names[i],labels:[unit],values:[values[i]]})),{
    x,y,w,h,barDir:'col',catAxisLabelFontFace:'Arial',catAxisLabelFontSize:15,
    valAxisLabelFontFace:'Arial',valAxisLabelFontSize:14,catAxisLabelColor:muted,
    valAxisLabelColor:muted,valAxisMinVal:0,valAxisMaxVal:max,
    chartColors:colors,showLegend:true,legendPos:'b',legendFontFace:'Arial',legendFontSize:14,
    showValue:true,dataLabelPosition:'outEnd',dataLabelFormatCode:'0.00',dataLabelColor:navy,
    dataLabelBkgrdColor:'FFFFFF',dataLabelFontFace:'Arial',dataLabelFontSize:18,
    showTitle:false,showBorder:false,showCatName:false,
    valGridLine:{color:'DFE7ED',width:1},catGridLine:{style:'none'},
    showCatName:false,showSerName:false,showMarker:false,
    showValue:true,showCatName:false,gapSize:80,
  });
}

// 1: Minimal cover.
{
  const s=page('',true);
  text(s,'KAN-Mixer cho\nđịnh vị trong nhà',.8,1.35,11.8,1.9,45,{bold:true,color:'FFFFFF'});
  text(s,'So sánh MLP, MLP-Mixer và KAN-Mixer\ntrên UJIIndoorLoc',.82,3.7,11.5,1.0,25,{color:'CBDEE8'});
  text(s,'Tổng kết đề tài\n08/10/2026',.82,5.75,8,.65,17,{color:'CBDEE8'});
  note(s,'Mở đầu','Đề tài dự đoán vị trí từ tín hiệu Wi-Fi bằng ba kiến trúc mạng. Bài trình bày tập trung vào cách đặt bài toán, quy trình so sánh và kết quả thật. Giao thức tối thiểu đã hoàn tất, nhưng mean sai số 2D trên test chưa đạt mục tiêu 7–8 m. Tôi sẽ đánh giá cả chất lượng, tốc độ và bộ nhớ.');
}
// 2: Problem, with a explicitly illustrative scan.
{
  const s=page('Bài toán định vị bằng Wi-Fi');
  text(s,'Một lần quét Wi-Fi tạo một “dấu vân tay” tín hiệu.',.7,1.55,11.9,.65,24);
  table(s,[['AP','RSSI'],['WAP001','−71'],['WAP002','100'],['…','…']],.8,2.55,3.1,[1.65,1.45],21,.6);
  box(s,'Mô hình\nhọc máy',5.0,3.0,2.9,1.5,navy,light,24);
  box(s,'Tòa / tầng\nTọa độ (x, y)',9.5,3.0,2.9,1.5,'14532D','DCFCE7',24);
  arrow(s,4.05,3.75,4.82,3.75);arrow(s,8.05,3.75,9.3,3.75);
  text(s,'RSSI cho biết cường độ tín hiệu. AP là điểm truy cập Wi-Fi.',.8,5.5,11.8,.7,21);
  foot(s,'Minh họa đầu vào, không phải dự đoán thực nghiệm. Mã 100: AP không được phát hiện.');
  note(s,'Bài toán IPS','IPS là hệ thống định vị trong nhà. RSSI là cường độ tín hiệu mà thiết bị thu được từ mỗi access point. Fingerprint là toàn bộ tín hiệu trong một lần quan sát. Hai giá trị trên slide chỉ minh họa, không phải một dự đoán thật. Theo tài liệu UJIIndoorLoc, 100 là mã không phát hiện AP. Mạng học liên hệ giữa tín hiệu và nhãn vị trí ở train, rồi dự đoán vị trí khi chỉ có tín hiệu.',[sources.data]);
}
// 3: Questions, without retrospective preregistration claims.
{
  const s=page('Câu hỏi nghiên cứu');
  text(s,'01',.8,1.75,1,.7,40,{color:colors[2],bold:true});
  text(s,'KAN có giảm sai số khi thay các nhánh MLP trong Mixer?',2,1.7,10.5,1.0,25,{bold:true});
  text(s,'02',.8,3.1,1,.7,40,{color:colors[0],bold:true});
  text(s,'Mức cải thiện có bù được thời gian và bộ nhớ tăng thêm?',2,3.05,10.5,1.0,25,{bold:true});
  text(s,'03',.8,4.45,1,.7,40,{color:colors[1],bold:true});
  text(s,'Split, mask và cách nhóm AP ảnh hưởng kết quả thế nào?',2,4.4,10.5,1.0,25,{bold:true});
  foot(s,'Câu 1–2 đối chiếu số đo hiện có. Câu 3 cần các thí nghiệm mới để kiểm chứng.');
  note(s,'Câu hỏi và giả thuyết','Giả thuyết định hướng của câu một là KAN-Mixer có mean2D thấp hơn MLP-Mixer khi giữ giao thức so sánh. Chúng ta không mặc định giả thuyết đúng. Câu hai xem cả chất lượng và chi phí. Câu ba cần ablation mask, cách nhóm AP và split giữ người/thiết bị ngoài train. Những tiêu chí nghiên cứu mới viết ở chặng 8 là đề xuất cho tương lai, không được gọi là đăng ký trước của chặng 7.', ['docs/lessons/chang_08/research_questions.md']);
}
// 4: Editable research diagram with validation branch and final-test branch.
{
  const s=page('Quy trình nghiên cứu');
  box(s,'Dữ liệu\nvà kiểm tra',.75,1.8,3.05,1.02);
  box(s,'Split cố định\nTiền xử lý train',5.15,1.8,3.05,1.02);
  box(s,'3 mô hình\n3 seed',9.55,1.8,3.05,1.02,'14532D','DCFCE7');
  arrow(s,3.95,2.31,4.98,2.31);arrow(s,8.35,2.31,9.38,2.31);
  box(s,'Validation\nChọn checkpoint',9.55,3.35,3.05,1.02,'14532D','DCFCE7');
  arrow(s,11.08,2.96,11.08,3.2);
  box(s,'Test cố định\nBenchmark',5.15,4.92,3.05,1.02,'1E3A8A','DBEAFE');
  box(s,'Phân tích\nKết luận',.75,4.92,3.05,1.02,'78350F','FEF3C7');
  arrow(s,11.08,4.51,11.08,5.43);arrow(s,10.93,5.43,8.36,5.43);
  arrow(s,4.99,5.43,3.96,5.43);
  arrow(s,9.4,3.85,8.75,3.85,'059669',true);
  arrow(s,8.75,3.85,8.75,2.32,'059669',true);
  arrow(s,8.75,2.32,9.4,2.32,'059669',true);
  text(s,'Chọn bằng validation',5.05,3.54,3.2,.58,17,{color:'14532D',align:'center'});
  foot(s,'Test chỉ đánh giá checkpoint đã chốt. Ý tưởng cải tiến tiếp theo cần giao thức mới.');
  note(s,'Quy trình nghiên cứu','Ta kiểm tra dữ liệu, lưu split và chỉ fit tiền xử lý trên train. Mỗi mô hình chạy ba seed. Trong vòng huấn luyện, validation chọn best checkpoint và quyết định early stopping. Khi hoàn tất chín lượt, audit và khóa checkpoint trước đánh giá test. Benchmark diễn ra sau training. Không lấy kết quả test để đổi checkpoint hoặc siêu tham số của giao thức này. Sơ đồ Mermaid chi tiết trong tài liệu còn tách riêng test và benchmark.',['docs/lessons/chang_08/research_workflow.mmd','reports/stage7_cuda_v1/protocol_lock.json']);
}
// 5: Dataset, native table and large factual numbers.
{
  const s=page('Dữ liệu UJIIndoorLoc');
  text(s,'520',.8,1.7,3.3,1.0,55,{bold:true,color:colors[0]});
  text(s,'cột AP',.85,2.74,3.0,.45,21);
  text(s,'3',5.0,1.7,3.3,1.0,55,{bold:true,color:colors[1]});
  text(s,'tòa nhà',5.05,2.74,3.0,.45,21);
  text(s,'13',9.2,1.7,3.3,1.0,55,{bold:true,color:colors[2]});
  text(s,'cặp tòa / tầng',9.25,2.74,3.0,.45,21);
  table(s,[['Tệp gốc','Số mẫu','Vai trò trong đề tài'],
    ['trainingData.csv','19.937','Train và validation nội bộ'],
    ['validationData.csv','1.111','Test cuối giữ nguyên']],.8,3.85,11.7,[4.2,1.6,5.9],20,.68);
  foot(s,'Nguồn UCI, CC BY 4.0. Nhãn/metadata không được đưa vào input RSSI.');
  note(s,'UJIIndoorLoc','Dữ liệu có 520 cột Wi-Fi và các cột nhãn, người, điện thoại, thời gian. Số mẫu trên slide lấy từ tài liệu chính thức và CSV đã kiểm tra ở chặng 5. Dataset có ba tòa và 13 tổ hợp building/floor hợp lệ. Tên validationData.csv dễ gây nhầm: trong giao thức này nó là test cuối. Nhãn tọa độ, tòa, tầng và metadata người/điện thoại/thời gian không làm đầu vào mô hình.',[sources.data,'docs/lessons/chang_05/README.md']);
}
// 6: RSSI transformation, numeric table.
{
  const s=page('Biểu diễn RSSI và giá trị không phát hiện');
  text(s,'Mỗi AP có hai số: tín hiệu chuẩn hóa z và mask m.',.8,1.65,11.7,.7,24);
  table(s,[['RSSI r','z','m','Ý nghĩa'],['100','0','0','Không phát hiện'],
    ['−104','0','1','Có quan sát'],['−52','0,5','1','Có quan sát']],.8,2.75,11.7,[2.2,1.8,1.8,5.9],21,.65);
  text(s,'AP có quan sát: z = (r + 104) / 104',.8,5.73,11.7,.55,23,{bold:true});
  foot(s,'Input mỗi mẫu: 520 × 2. Chuẩn hóa tọa độ chỉ dùng thống kê train.');
  note(s,'Tiền xử lý RSSI','Giá trị thực đã kiểm tra nằm trong khoảng −104 đến 0, và 100 biểu thị không quan sát. Với AP được quan sát, z=(r+104)/104 và m=1. Với mã100, z=0 và m=0. Mask giúp phân biệt AP không quan sát với tín hiệu −104 có quan sát vì cả hai cùng z=0. Các hàng trên là ví dụ biến đổi đúng công thức. Tọa độ dùng một center hai chiều và shared scale fit riêng từ train nội bộ, rồi inverse trước tính sai số.',[sources.data,'src/ips/data.py','data/processed/uji_group_split_v1/preprocessing.json']);
}
// 7: Split integrity and its limits.
{
  const s=page('Train, validation và test');
  table(s,[['Phần dữ liệu','Số mẫu','Công việc'],['Train','16.025','Học tham số, fit tiền xử lý'],
    ['Validation nội bộ','3.912','Chọn checkpoint, dừng sớm'],
    ['Test chính thức','1.111','Đánh giá cuối']],.8,1.85,11.7,[3.35,1.75,6.6],21,.74);
  text(s,'Nhóm đo proxy không giao nhau giữa train và validation.',.8,5.12,11.7,.65,23,{bold:true,color:'14532D'});
  text(s,'Người và vị trí vẫn có thể chung giữa hai phần dữ liệu.',.8,5.92,11.7,.6,21);
  foot(s,'SGKF 5 fold, fold 0, split seed 2026. Đây chưa phải đánh giá người hoặc vị trí mới.');
  note(s,'Cách chia dữ liệu','StratifiedGroupKFold dùng năm fold, shuffle và seed2026, chọn fold0. Proxy nhóm gồm USERID, PHONEID, building, floor, tọa độ, space và relative position. Nhóm proxy không giao nhau, nhưng còn người, vị trí và một số fingerprint chung giữa train/validation. Do đó split không đảm bảo generalization sang người mới. Test USERID0 nghĩa là chưa ghi nhận người, không phải toàn bộ mẫu do một người thực đo. Không dùng test để fit preprocessing.',['docs/lessons/chang_05/README.md','reports/stage7_cuda_v1/fairness_audit.json']);
}
// 8: Three model table, explicit architectural provenance.
{
  const s=page('Ba kiến trúc cho cùng bài toán');
  table(s,[['Mô hình','Cách xử lý fingerprint','Tham số'],
    ['MLP','Làm phẳng 1.040 số, lớp 128 và 64',n(agg.mlp.resources.trainable_parameters)],
    ['MLP-Mixer','26 token, 64 channel, 2 khối trộn',n(agg.mlp_mixer.resources.trainable_parameters)],
    ['KAN-Mixer','Thay cả hai nhánh MLP bằng KAN',n(agg.kan_mixer.resources.trainable_parameters)]],
    .8,1.95,11.7,[2.4,6.9,2.4],20,.85);
  text(s,'Cùng đầu ra: 13 logits tòa/tầng và 2 tọa độ.',.8,5.57,11.7,.58,23,{bold:true});
  foot(s,'Token/head IPS là thiết kế của đề tài. Số tham số chưa được khớp bằng nhau.');
  note(s,'Ba kiến trúc','MLP baseline nhận 1.040 số sau làm phẳng, hai lớp 128 và 64. Hai Mixer nhận 26 nhóm 20 AP, mỗi nhóm 40 số, embedding 64 channel và 2 khối. KAN-Mixer thay cả token-mixing và channel-mixing bằng các KANLinear của efficient-kan. Embedding và head vẫn tuyến tính. KAN-Mixers của Canuto v1 nghiên cứu phân loại ảnh. Đây là biến thể RSSI cho IPS do đề tài thiết kế. So sánh hiện tại giữ giao thức nhưng chưa khớp số tham số.',[sources.mixer,sources.kanMixer,sources.backend,'src/ips/models/networks.py']);
}
// 9: Mixer branches and KAN formula, editable diagram.
{
  const s=page('MLP-Mixer và biến thể KAN-Mixer');
  box(s,'B × 26 × 64',.8,2.05,2.45,1.0);
  box(s,'Token-mixing\nTrộn 26 nhóm AP',4.1,2.05,3.75,1.0,'1E3A8A','DBEAFE');
  box(s,'Channel-mixing\nTrộn 64 đặc trưng',8.7,2.05,3.75,1.0,'14532D','DCFCE7');
  arrow(s,3.4,2.55,3.95,2.55);arrow(s,8.0,2.55,8.55,2.55);
  text(s,'MLP: lớp tuyến tính và GELU',.8,3.9,11.7,.65,25,{color:colors[1],bold:true});
  text(s,'KAN: yⱼ = Σᵢ φⱼᵢ(xᵢ)',.8,4.77,11.7,.65,28,{color:colors[2],bold:true});
  text(s,'φ là hàm học được: nhánh SiLU và tổng các B-spline.',.8,5.62,11.7,.65,22);
  foot(s,'Hai nhánh dùng LayerNorm, residual và dropout. Spline grid cố định, size 5, bậc 3.');
  note(s,'Hai nhánh trộn','B là số mẫu trong batch, 26 là số token và 64 là số channel. Token-mixing áp dụng cùng phép trộn 26 token cho từng channel. Channel-mixing áp dụng cùng phép trộn 64 channel cho từng token. MLP học các trọng số tuyến tính quanh hàm GELU. KANLinear học hàm một biến trên cạnh: phi gồm trọng số base nhân SiLU và tổ hợp B-spline với hệ số học được. Cấu hình dùng grid_size=5, spline_order=3, grid_range=[−3,3] và không cập nhật grid. Không chèn GELU giữa hai KANLinear.',['src/ips/models/networks.py','reports/stage7_cuda_v1/frozen_config.json',sources.kan,sources.kanMixer]);
}
// 10: Training protocol, big experiment count with concise rows.
{
  const s=page('Giao thức thực nghiệm');
  text(s,'9 lượt GPU',.8,1.7,4,1.0,40,{bold:true,color:colors[0]});
  text(s,'3 mô hình × seed 42, 43, 44',5.1,1.8,7.3,.7,24);
  table(s,[['Thiết lập','Giá trị chung'],['Thiết bị','RTX 4060 Ti 8 GB, FP32, batch 64'],
    ['Tối ưu','AdamW, lr 0,001, weight decay 0,0001'],
    ['Loss','Cross-entropy + SmoothL1 tọa độ'],
    ['Ngân sách','Tối đa 100 epoch, patience 15'],
    ['Chọn checkpoint','Mean2D validation, min_delta 0,01 m']],.8,3.0,11.7,[3.1,8.6],19,.5);
  foot(s,'Cùng split, tiền xử lý và loss. Epoch thực chạy khác nhau do early stopping.');
  note(s,'Giao thức so sánh','Chín lượt chạy tuần tự trên cùng RTX 4060 Ti. Cấu hình dùng AdamW với lr=0,001, weight_decay=0,0001, dropout=0,1, gradient_clip=1. Loss là CE cộng SmoothL1 với lambda_xy=1, beta=1 trên tọa độ chuẩn hóa. Chọn best nếu mean2D validation cải thiện hơn min_delta=0,01 m, patience=15, tối đa 100 epoch. Cùng ngân sách tối đa không có nghĩa cùng epoch thực tế, FLOPs hoặc thời gian. Môi trường PyTorch 2.9.0+cu128, FP32, deterministic, bốn luồng CPU. Không tuning thêm sau khi xem test.',['reports/stage7_cuda_v1/frozen_config.json','reports/stage7_cuda_v1/protocol_lock.json']);
}
// 11: Metrics, explain units and sample example.
{
  const s=page('Chỉ số đánh giá');
  text(s,'Phân loại',.8,1.75,3.1,.7,28,{bold:true,color:colors[0]});
  text(s,'Accuracy, macro-F1\nConfusion matrix\nĐánh giá tòa, tầng và cặp tòa/tầng',4.0,1.75,8.4,1.7,22);
  text(s,'Tọa độ',.8,3.95,3.1,.7,28,{bold:true,color:colors[1]});
  text(s,'e = √[(x̂ − x)² + (ŷ − y)²]\nMean, median, P90 và RMSE của e',4.0,3.85,8.4,1.5,25);
  text(s,'Ví dụ: lệch x = 3 m, lệch y = 4 m thì e = 5 m.',.8,5.87,11.7,.55,22);
  foot(s,'Mục tiêu 8 m dùng mean2D trên toàn bộ test. MAE tọa độ là đại lượng khác.');
  note(s,'Đọc các chỉ số','Accuracy đo tỷ lệ đúng. Macro-F1 tính F1 mỗi lớp rồi lấy trung bình đều, giúp thấy lớp ít mẫu. Joint accuracy yêu cầu cùng đúng tòa và tầng. Sau inverse tọa độ, ta tính khoảng cách Euclid 2D của từng mẫu. Mean nhạy với lỗi lớn, median mô tả điểm giữa, P90 là ngưỡng chứa 90% lỗi. RMSE2D phạt mạnh lỗi lớn. Ví dụ lệch 3 m và 4 m cho khoảng cách 5 m, trong khi MAE trên hai trục của riêng mẫu này là 3,5 m. Mục tiêu 8 m dùng mean2D, giữ mọi mẫu sai tòa/tầng.',['src/ips/metrics.py','reports/stage7_cuda_v1/README.md']);
}
// 12: Key evidence, native chart plus uncertainty table.
{
  const s=page('Sai số test chưa đạt mục tiêu 8 m');
  table(s,[['Mô hình','Mean2D (m)'],...ids.map((m,i)=>[names[i],stat(m,'mean_2d')])],
    .8,2.0,5.6,[2.3,3.3],22,.76);
  bar(s,ids.map(m=>agg[m].test.mean_2d.mean),6.8,1.85,5.7,3.65,'Mean2D (m)',20);
  text(s,'0/9',.8,5.68,2,.72,38,{bold:true,color:colors[2]});
  text(s,'checkpoint đạt mean2D ≤ 8 m',3.0,5.77,9.2,.57,24,{bold:true});
  foot(s,'Toàn bộ 1.111 mẫu test. Bảng: mean ± SD mẫu qua 3 seed. Biểu đồ: mean.');
  note(s,'Kết quả định vị chính',`Mean2D test: MLP ${stat('mlp','mean_2d')} m, MLP-Mixer ${stat('mlp_mixer','mean_2d')} m và KAN-Mixer ${stat('kan_mixer','mean_2d')} m. Biểu đồ chỉ biểu diễn trung bình, bảng có SD mẫu giữa ba seed. Không có seed nào đạt mean≤8 m. MLP-Mixer thấp hơn MLP 0,28 m khi ghép theo seed, SD chênh lệch 1,35 m. KAN thấp hơn MLP 0,07 m, SD chênh lệch 1,42 m. Chưa đủ bằng chứng về ưu thế thống kê. Không đổi mean2D test thành median hoặc MAE validation để tuyên bố đạt mục tiêu.`,[sources.report,'reports/stage7_cuda_v1/per_seed_metrics.csv']);
}
// 13: Separate classification evidence.
{
  const s=page('Kết quả phân loại tòa và tầng');
  table(s,[['Mô hình','Tòa acc (%)','Tầng acc (%)','Joint acc (%)'],
    ...ids.map((m,i)=>[names[i],stat(m,'building_accuracy','test',100),stat(m,'floor_accuracy','test',100),stat(m,'joint_accuracy','test',100)])],
    .8,1.95,11.7,[2.4,3.1,3.1,3.1],21,.78);
  text(s,'Building accuracy trên 99% trong cả ba mô hình.',.8,5.36,11.7,.62,24,{bold:true});
  text(s,'Nhầm tầng vẫn ảnh hưởng đến kết quả định vị.',.8,6.08,11.7,.5,22);
  foot(s,'Mean ± SD qua 3 seed. Joint yêu cầu đồng thời đúng building và floor.');
  note(s,'Kết quả phân loại','MLP-Mixer có joint accuracy trung bình 92,56%, MLP 90,97% và KAN 90,70%. Building accuracy đều trên 99%, nhưng floor accuracy khoảng 91–93%. Vì vậy cần đánh giá hai đầu ra riêng: tòa đúng chưa đảm bảo tầng đúng hay tọa độ chính xác. Joint macro-F1 theo thứ tự MLP, MLP-Mixer, KAN-Mixer là 89,47%, 91,16% và 88,70%. Ba seed chưa cho phép tuyên bố một kiến trúc tốt nhất nói chung.',[sources.report,'reports/stage7_cuda_v1/figures/confusion_mlp_mixer.png']);
}
// 14: Tail risk: table plus real case, not illustrative chart.
{
  const s=page('Lỗi lớn làm thay đổi kết luận');
  table(s,[['Mô hình','Median2D (m)','P90 2D (m)','RMSE2D (m)'],
    ...ids.map((m,i)=>[names[i],stat(m,'median_2d'),stat(m,'p90_2d'),stat(m,'rmse_2d')])],
    .8,1.95,11.7,[2.4,3.1,3.1,3.1],20,.76);
  text(s,'KAN có median thấp hơn, nhưng RMSE cao hơn MLP.',.8,5.25,11.7,.65,24,{bold:true,color:colors[2]});
  text(s,'Một mẫu thật: KAN seed 44, 2 AP, lỗi 223,22 m.',.8,6.02,11.7,.56,22);
  foot(s,'Giữ cả mẫu lỗi lớn. Ví dụ là test row_index=21, chưa đại diện nguyên nhân của mọi lỗi.');
  note(s,'Phân tích lỗi lớn','Median2D của KAN thấp nhất, nhưng RMSE2D 23,32 m cao hơn MLP 19,37 m. Vì vậy cần đọc nhiều chỉ số. Ví dụ thật: KAN seed 44, row_index=21, dòng CSV 23, có hai AP quan sát, nhầm tầng 1 sang 0 và sai 223,22 m. Một ví dụ không chứng minh ít AP gây mọi lỗi lớn. Phân tích theo tầng còn cho thấy nhóm 1/0 và 1/1 khó với KAN. Nhóm test 1/0 chỉ có 30 mẫu nên phải ghi số mẫu khi so sánh.',['reports/stage7_cuda_v1/worst_test_errors.csv','reports/stage7_cuda_v1/group_metrics.csv',sources.report]);
}
// 15: Training cost.
{
  const s=page('Chi phí huấn luyện trên RTX 4060 Ti');
  table(s,[['Mô hình','Epoch thực chạy','Train + val (s)'],...ids.map((m,i)=>[names[i],res(m,'epochs_completed'),res(m,'training_seconds')])],
    .8,1.9,11.7,[3.0,3.9,4.8],23,.82);
  const total=ids.reduce((sum,m)=>sum+agg[m].resources.training_seconds.mean*3,0)/60;
  text(s,`${fm(total)} phút`,.8,5.4,4.7,.88,39,{bold:true,color:colors[0]});
  text(s,'tổng vòng train + validation của 9 lượt',5.45,5.52,7.0,.7,23);
  foot(s,'Có log, diagnostics và checkpoint. Chưa gồm startup và đánh giá cuối.');
  note(s,'Thời gian huấn luyện',`Tổng vòng train + validation là ${fm(total)} phút. Bảng là trung bình và SD qua ba seed của từng mô hình. KAN chạy nhiều epoch hơn MLP, nên tỷ lệ tổng thời gian không đại diện chi phí một optimizer step. Early stopping làm ngân sách thực tế khác nhau dù đều tối đa 100 epoch. Thời gian gồm validation, ghi log, lấy diagnostics và lưu checkpoint. Chưa gồm import/startup/tải dữ liệu trước vòng lặp hoặc evaluation/benchmark cuối. Không dùng ngoại suy pilot ba epoch làm kết quả chính.`,[sources.report,'reports/stage7_cuda_v1/per_seed_metrics.csv']);
}
// 16: Inference speed + memory scope.
{
  const s=page('Tốc độ suy luận và bộ nhớ');
  bar(s,ids.map(m=>agg[m].benchmarks['1'].end_to_end.median_ms.mean),.65,1.7,5.8,3.82,'B1 p50 (ms)',6);
  table(s,[['Mô hình','RAM peak\n(MiB)','VRAM peak\nreserved (MiB)'],...ids.map((m,i)=>[names[i],fm(agg[m].resources.train_ram_peak_mib.mean),fm(agg[m].resources.train_vram_reserved_mib.mean)])],
    6.9,2.05,5.65,[2.15,1.7,1.8],17,.77);
  text(s,'KAN B1 p50 khoảng 4,95 ms, MLP khoảng 0,46 ms.',.8,5.88,11.7,.65,23,{bold:true});
  foot(s,'Bộ nhớ đo khi train. RAM: RSS toàn tiến trình. VRAM: PyTorch. Latency chưa gồm quét Wi-Fi.');
  note(s,'Latency và bộ nhớ','Biểu đồ là trung bình p50 end-to-end B1 qua ba seed: MLP 0,46 ms, Mixer 1,19 ms, KAN 4,95 ms. End-to-end từ RSSI trong RAM qua tiền xử lý, GPU, mô hình và inverse đến mảng CPU, chưa gồm quét Wi-Fi hoặc I/O file. Mỗi benchmark có 10 warmup, 100 lần đo, đồng bộ CUDA và tiến trình riêng. Bảng là peak RAM toàn tiến trình và peak VRAM reserved khi train, lấy mean qua ba seed. VRAM allocated riêng là 67,99/76,92/414,54 MiB. Những số này khác storage trọng số. Phiên desktop chưa cô lập hoàn toàn, không ngoại suy nguyên số sang máy khác.',[sources.report,'src/ips/benchmark.py','src/ips/resources.py']);
}
// 17: Limits, distinct meaningful rows, plenty of whitespace.
{
  const s=page('Giới hạn của kết luận');
  text(s,'3 seed',.8,1.8,3,.7,32,{bold:true,color:colors[0]});
  text(s,'Mô tả biến thiên huấn luyện\nChưa đủ để khẳng định ưu thế thống kê',4.05,1.65,8.3,1.2,23);
  text(s,'Split',.8,3.3,3,.7,32,{bold:true,color:colors[1]});
  text(s,'Train và validation còn chung người, vị trí\nChưa chứng minh khái quát sang người mới',4.05,3.15,8.3,1.2,23);
  text(s,'Ngân sách',.8,4.9,3,.7,29,{bold:true,color:colors[2]});
  text(s,'Cùng tối đa 100 epoch\nChưa khớp tham số, FLOPs hay thời gian',4.05,4.75,8.3,1.2,23);
  foot(s,'Test đã được xem. Cải tiến trên cùng test cần ghi là thăm dò và thêm kiểm chứng độc lập.');
  note(s,'Giới hạn','Ba seed cùng split chỉ mô tả biến thiên huấn luyện. SD mẫu không phải CI và không chứng minh độ ổn định trên dữ liệu khác. Proxy nhóm không giao nhau vẫn chưa giữ người/vị trí ngoài train. Số tham số, FLOPs và thời gian của ba kiến trúc khác nhau. Tọa độ theo quy ước mét dataset, CRS/EPSG chưa hoàn toàn xác minh, nên chưa suy ra lỗi 3D. Vì test đã xem, cải tiến chọn sau phân tích này trên test cũ mang tính thăm dò.',['reports/stage7_cuda_v1/analysis_notes.md']);
}
// 18: Main conclusion and clearly prospective work.
{
  const s=page('Kết luận và bước tiếp theo',true);
  text(s,'Pipeline tối thiểu đã hoàn tất',.8,1.75,11.7,.9,34,{bold:true,color:'FFFFFF'});
  text(s,'MLP-Mixer có mean2D trung bình thấp nhất trong bảng.\nKAN-Mixer chưa cho thấy lợi ích rõ để bù chi phí.',.8,3.03,11.5,1.25,25,{color:'CBDEE8'});
  text(s,'Mục tiêu 8 m còn cần cải thiện',.8,4.65,11.7,.75,31,{bold:true,color:'FDBA74'});
  text(s,'Thử mask, nhóm AP và mô hình lai bằng giao thức mới.\nCần tập kiểm chứng độc lập để xác nhận cải tiến.',.8,5.65,11.5,.95,23,{color:'CBDEE8'});
  note(s,'Kết luận','Đóng góp hiện có là mô hình hóa IPS, cài đặt ba mạng và thực nghiệm tái lập, cùng đánh giá chất lượng và chi phí. Chưa có thuật toán KAN mới hay kết quả đạt mục tiêu 8 m. MLP-Mixer có mean2D trung bình thấp nhất nhưng chênh lệch nhỏ và biến thiên đáng kể. KAN-Mixer chưa có lợi ích mean2D rõ ràng trong cấu hình này. Bước tiếp theo chưa chạy: ablation mask, AP ordering, hai biến thể lai, kNN baseline, so sánh khớp ngân sách và group CV. Chọn bằng validation, lưu giao thức mới và kiểm chứng độc lập.',['docs/lessons/chang_08/completion_checklist.md','docs/lessons/chang_08/research_questions.md',sources.report]);
}
// 19: Technical appendix, editable evidence diagram.
{
  const s=page('Phụ lục: luồng kích thước tensor');
  const stages=[['RSSI + mask','B × 520 × 2'],['26 nhóm AP','B × 26 × 40'],['Embedding','B × 26 × 64'],['2 Mixer blocks','B × 26 × 64']];
  stages.forEach((a,i)=>{box(s,a.join('\n'),.8+i*3.12,1.85,2.4,1.16,navy,light,19);if(i<3)arrow(s,3.35+i*3.12,2.43,3.77+i*3.12,2.43);});
  box(s,'LayerNorm, gộp token\nB × 64',4.88,3.75,3.4,1.16,'14532D','DCFCE7',21);
  arrow(s,11.36,3.15,11.36,4.33);arrow(s,11.2,4.33,8.45,4.33);
  box(s,'Head phân loại\nB × 13 logits',.8,5.42,3.95,1.0,'1E3A8A','DBEAFE',21);
  box(s,'Head tọa độ\nB × 2',8.55,5.42,3.95,1.0,'78350F','FEF3C7',21);
  arrow(s,5.1,5.05,2.78,5.27);arrow(s,8.05,5.05,10.53,5.27);
  foot(s,'Token branch: 26–32–26. Channel branch: 64–128–64. B là batch size.');
  note(s,'Phụ lục tensor','520 AP, mỗi AP có 2 số, tạo 1.040 features. Reshape thành 26 token với 40 features mỗi token, tương đương 20 AP. Linear embedding biến 40 thành 64, dùng chung mọi token. Token branch đổi B×26×64 thành B×64×26, trộn 26–32–26 rồi đổi trục về. Channel branch trộn 64–128–64. Mỗi branch có pre-LN, residual và dropout. Sau 2 khối, LayerNorm và trung bình trên trục token cho B×64. Hai head tuyến tính cho 13 logits và 2 tọa độ chuẩn hóa.',['src/ips/models/networks.py','reports/stage7_cuda_v1/frozen_config.json']);
}
// 20: Reference slide, short readable anchors with hyperlink and full notes.
{
  const s=page('Nguồn và bằng chứng');
  const refs=[['UJIIndoorLoc','UCI Machine Learning Repository',sources.data],
    ['MLP-Mixer','Tolstikhin và cộng sự, 2021',sources.mixer],
    ['KAN','Liu và cộng sự, 2024',sources.kan],
    ['KAN-Mixers v1','Canuto và cộng sự, 2025',sources.kanMixer],
    ['efficient-kan','Backend khóa commit 7b6ce1c…',sources.backend]];
  refs.forEach((r,i)=>{
    text(s,r[0],.8,1.62+i*.78,4.15,.53,23,{bold:true,color:colors[i%3],hyperlink:{url:r[2]}});
    text(s,r[1],5.18,1.62+i*.78,7.1,.53,21);
  });
  text(s,'Kết quả và cấu hình: reports/stage7_cuda_v1/',.8,5.83,11.6,.56,22,{bold:true});
  foot(s,'Tên tài liệu có liên kết. URL đầy đủ và nguồn từng số đo nằm trong ghi chú slide.');
  note(s,'Nguồn','Các tài liệu nền giải thích kiến trúc. Những số thực nghiệm IPS trên slide lấy từ suite cục bộ đã hoàn tất. Backend efficient-kan giữ commit 7b6ce1c87f18c8bc90c208f6b494042344216b11, có provenance và giấy phép MIT. Tài liệu KAN-Mixers v1 nghiên cứu ảnh, còn token hóa và head IPS là thiết kế của đề tài. JSON, cấu hình, audit và dự đoán từng mẫu trong repo cho phép truy nguồn kết quả.',Object.values(sources));
}

if(pptx._slides.length!==20)throw Error('Expected exactly 20 slides');
const digest=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
(async()=>{
  const deck=path.join(out,'IPS_KAN_Mixer_UJIIndoorLoc.pptx');
  await pptx.writeFile({fileName:deck,compression:true});
  // PptxGenJS 4.0.1 declares default slideMaster2..N overrides although it
  // only emits slideMaster1. Remove exactly those dangling declarations.
  // Never rewrite existing XML parts or remove actual slide masters.
  const zip=await JSZip.loadAsync(fs.readFileSync(deck));
  const contentTypes=await zip.file('[Content_Types].xml').async('string');
  const repairedParts=[];
  const repaired=contentTypes.replace(/<Override\b[^>]*PartName="([^"]+)"[^>]*\/>/g,(tag,part)=>{
    if(zip.file(part.replace(/^\//,''))) return tag;
    if(!/^\/ppt\/slideMasters\/slideMaster\d+\.xml$/.test(part)) throw Error(`Unexpected missing part ${part}`);
    repairedParts.push(part);return '';
  });
  zip.file('[Content_Types].xml',repaired);
  fs.writeFileSync(deck,await zip.generateAsync({type:'nodebuffer',compression:'DEFLATE'}));
  const guide='# Ghi chú thuyết trình chặng 8\n\n18 slide chính + 2 slide phụ lục. Thời lượng gợi ý: 15–20 phút.\n\n'+notes.map(v=>`## Slide ${v.slide} — ${v.title}\n\n${v.body}\n`).join('\n');
  fs.writeFileSync(path.join(out,'speaker_notes.md'),guide,'utf8');
  fs.writeFileSync(path.join(buildDir,'build_manifest.json'),JSON.stringify({
    slide_count:20,font:'Arial',size:'16:9',pptxgenjs:require(path.join(root,'runs/stage8_build/node_modules/pptxgenjs/package.json')).version,
    deck,sha256:digest(deck),source_aggregate_sha256:digest(path.join(reportDir,'aggregate_metrics.json')),
    verification_evidence:verify,model_results:ids.map((m,i)=>({model:m,name:names[i],test_mean_2d:agg[m].test.mean_2d,
      params:agg[m].resources.trainable_parameters})),native_charts:2,native_diagrams:4,
    repaired_dangling_master_content_types:repairedParts,
  },null,2),'utf8');
  console.log(`Created ${deck} (${pptx._slides.length} slides). Source results preserved.`);
})().catch(e=>{console.error(e);process.exitCode=1;});
