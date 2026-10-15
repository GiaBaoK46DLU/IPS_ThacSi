"""Export measured tables/figures after the frozen supplied-data experiment."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ...data import ROOT, sha256
from ...data_access.supplied import read_supplied
from .runtime import save_json
from .supplied_experiment import load_prepared

NAMES = {"pure": "KAN tham chiếu", "hybrid": "KAN lai"}
COLORS = {"pure": "#2266a8", "hybrid": "#df7933"}
METRICS = ["mean_2d", "median_2d", "p90_2d", "rmse_2d", "mae_coordinates",
           "rmse_coordinates", "floor_accuracy", "floor_macro_f1", "joint_accuracy"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":13,"axes.spines.top":False,
    "axes.spines.right":False,"figure.dpi":120,"savefig.dpi":200})


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:return list(csv.DictReader(stream))


def csv_save(path,rows):
    with Path(path).open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def generate(name,output):
    base=output/name
    completed=json.loads((base/"completed.json").read_text());seeds=completed["seeds"]
    bundle,datasets=load_prepared(name,output)
    test_raw=read_supplied(name,"test")
    report=base/"report";figures=report/"figures";tables=report/"tables"
    figures.mkdir(parents=True,exist_ok=True);tables.mkdir(exist_ok=True)
    unit="m giả định" if name=="dlulib" else "m"
    records=[];predictions={};histories={};resources=[];group_records=[]
    for seed in seeds:
        training=base/f"seed{seed}/training"
        summary=json.loads((training/"summary.json").read_text());resource=json.loads((training/"resources.json").read_text())
        histories[seed]=csv_read(training/"history.csv")
        resources.append({"seed":seed,"epochs":summary["epochs_completed"],"best_epoch":summary["best_epoch"],
            "train_seconds":summary["training_seconds"],"parameters":summary["trainable_parameters"],
            "RAM_peak_MiB":resource["training_phase"]["rss_peak_sampled_bytes"]/2**20,
            "VRAM_allocated_peak_MiB":resource["cuda_training_phase"]["peak_allocated_bytes"]/2**20,
            "VRAM_reserved_peak_MiB":resource["cuda_training_phase"]["peak_reserved_bytes"]/2**20,
            "best_checkpoint_MiB":resource["checkpoint_best_bytes"]/2**20})
        for variant in NAMES:
            for split in ("train","validation","test"):
                evaluation=base/f"seed{seed}/evaluation/{variant}/{split}"
                metrics=json.loads((evaluation/"metrics.json").read_text())
                timing_path=base/f"seed{seed}/benchmark/{variant}/test/resources.json"
                if split!='test' or not timing_path.exists():timing_path=evaluation/'resources.json'
                timing=json.loads(timing_path.read_text())
                records.append({"dataset":name,"seed":seed,"variant":variant,"split":split,"n":metrics["n_samples"],
                    **{m:metrics[m] for m in METRICS},"inference_seconds":timing["inference_seconds"],
                    "inference_ms_per_sample":timing["milliseconds_per_sample_amortized"],
                    "inference_VRAM_allocated_MiB":timing["cuda"]["peak_allocated_bytes"]/2**20,
                    "inference_RAM_sampled_MiB":timing["process_ram"]["rss_peak_sampled_bytes"]/2**20})
                rows=csv_read(evaluation/"predictions.csv")
                predictions[(seed,variant,split)]=rows
                if split=="test":
                    for floor in sorted({r["true_floor"] for r in rows}):
                        part=[r for r in rows if r["true_floor"]==floor]
                        error=np.asarray([float(r["error_2d"]) for r in part])
                        group_records.append({"seed":seed,"variant":variant,"floor":int(floor),"n":len(part),
                            "mean_2d":float(error.mean()),"median_2d":float(np.median(error)),"p90_2d":float(np.quantile(error,.9)),
                            "floor_accuracy":float(np.mean([r["true_floor"]==r["predicted_floor"] for r in part]))})
    csv_save(tables/"per_seed_metrics.csv",records);csv_save(tables/"training_resources.csv",resources)
    csv_save(tables/"test_per_floor.csv",group_records)
    aggregate=[]
    for variant in NAMES:
        for split in ("train","validation","test"):
            part=[r for r in records if r["variant"]==variant and r["split"]==split]
            row={"variant":variant,"split":split,"n_each_seed":part[0]["n"],"seeds":len(seeds)}
            for metric in METRICS+["inference_ms_per_sample"]:
                values=np.asarray([r[metric] for r in part]);row[metric+"_mean"]=float(values.mean());row[metric+"_sd"]=float(values.std(ddof=1)) if len(values)>1 else None
            aggregate.append(row)
    csv_save(tables/"aggregate_metrics.csv",aggregate)
    image_names=[]
    def save(number,title):
        stem=f"{number:02d}_{title}";plt.savefig(figures/f"{stem}.png",bbox_inches="tight");plt.savefig(figures/f"{stem}.svg",bbox_inches="tight");plt.close();image_names.append(stem)

    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout="constrained")
    counts=[len(datasets['train']),len(datasets['validation']),len(test_raw)]
    bars=axes[0].bar(["Train","Validation","Test nguồn"],counts,color=["#286991","#68a3b5","#d29458"])
    for bar,count in zip(bars,counts):axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height(),f"{count:,}",ha="center",va="bottom")
    axes[0].set_ylim(0,max(counts)*1.18);axes[0].set_ylabel("Số lần quét")
    bars=axes[1].bar(["AP thật","AP trống bổ sung"],[len(test_raw.ap_order),520-len(test_raw.ap_order)],color=["#286991","#b6bec5"])
    for bar in bars:axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height(),str(int(bar.get_height())),ha="center",va="bottom")
    axes[1].set_ylim(0,560);axes[1].set_ylabel("Số vị trí AP trong đầu vào");save(1,"du_lieu_va_padding")

    fig,axes=plt.subplots(2,1,figsize=(10,6),layout="constrained")
    for seed in seeds:
        rows=histories[seed];epochs=[int(r['epoch']) for r in rows]
        axes[0].plot(epochs,[float(r['train_loss']) for r in rows],label=f"seed {seed}")
        axes[1].plot(epochs,[float(r['val_mean_2d']) for r in rows],label=f"seed {seed}")
        best=next(r for r in resources if r['seed']==seed)['best_epoch'];r=rows[best-1]
        axes[1].scatter([best],[float(r['val_mean_2d'])],marker="*",s=130,zorder=4)
    axes[0].set_ylabel("Loss train");axes[1].set_ylabel(f"Mean 2D validation ({unit})");axes[1].set_xlabel("Epoch")
    for ax in axes:ax.grid(alpha=.2);ax.legend(ncol=3)
    save(2,"duong_hoc")

    fig,ax=plt.subplots(figsize=(9,4),layout="constrained");x=np.arange(3)
    for j,variant in enumerate(NAMES):
        part=[next(r for r in aggregate if r['variant']==variant and r['split']==s) for s in ('train','validation','test')]
        bars=ax.bar(x+(j-.5)*.34,[r['mean_2d_mean'] for r in part],.34,yerr=[r['mean_2d_sd'] or 0 for r in part],capsize=4,color=COLORS[variant],label=NAMES[variant])
        for b,r in zip(bars,part):ax.text(b.get_x()+b.get_width()/2,b.get_height()+(r['mean_2d_sd'] or 0)+.07,f"{r['mean_2d_mean']:.3f}",ha='center',va='bottom',fontsize=10)
    ax.set_xticks(x,["Train chẩn đoán","Validation","Test nguồn"]);ax.set_ylabel(f"Mean 2D ({unit})");ax.legend();ax.set_ylim(0,ax.get_ylim()[1]*1.12);ax.grid(axis='y',alpha=.2);save(3,"sai_so_theo_tap")

    fig,ax=plt.subplots(figsize=(9,4),layout="constrained");x=np.arange(len(seeds))
    for j,variant in enumerate(NAMES):
        part=[next(r for r in records if r['seed']==seed and r['variant']==variant and r['split']=='test') for seed in seeds]
        bars=ax.bar(x+(j-.5)*.34,[r['mean_2d'] for r in part],.34,color=COLORS[variant],label=NAMES[variant])
        for b,r in zip(bars,part):ax.text(b.get_x()+b.get_width()/2,b.get_height(),f"{r['mean_2d']:.3f}",ha='center',va='bottom')
    ax.set_xticks(x,[f"Seed {s}" for s in seeds]);ax.set_ylabel(f"Mean 2D test ({unit})");ax.legend(loc='upper center',ncol=2);ax.set_ylim(0,ax.get_ylim()[1]*1.15);save(4,"test_theo_seed")

    fig,ax=plt.subplots(figsize=(9,4),layout="constrained")
    for variant in NAMES:
        values=np.sort([float(r['error_2d']) for seed in seeds for r in predictions[(seed,variant,'test')]])
        ax.plot(values,np.arange(1,len(values)+1)/len(values),label=NAMES[variant],color=COLORS[variant],linewidth=2)
    ax.set_xlabel(f"Sai số 2D ({unit})");ax.set_ylabel("Tỷ lệ dự đoán có sai số không vượt ngưỡng");ax.set_ylim(0,1.02);ax.grid(alpha=.2);ax.legend();save(5,"cdf_test_gop_ba_seed")

    representative=44 if 44 in seeds else seeds[-1]
    fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
    for ax,variant in zip(axes,NAMES):
        metrics=json.loads((base/f"seed{representative}/evaluation/{variant}/test/metrics.json").read_text())
        matrix=np.asarray(metrics['floor_confusion_matrix']);ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max(1,int(matrix.max())))
        labels=sorted({c['floor'] for c in bundle.classes});ax.set_xticks(range(len(labels)),[f"Tầng {f}" for f in labels]);ax.set_yticks(range(len(labels)),[f"Tầng {f}" for f in labels])
        for i,j in np.ndindex(matrix.shape):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black',fontsize=13)
        ax.set_xlabel('Dự đoán');ax.set_ylabel('Thực tế');ax.set_title(f"{NAMES[variant]} seed {representative}")
    if len(bundle.classes)==1:fig.suptitle('Một tầng cố định nên ma trận chỉ mang tính mô tả')
    save(6,"ma_tran_tang_test")

    # Known/new relative to internal TRAIN catalogue, never validation labels.
    train=datasets['train'];train_keys={tuple([int(c),*xy]) for c,xy in zip(train.classes.numpy(),train.xy_original)}
    mapping={(c['building'],c['floor']):c['class_id'] for c in bundle.classes}
    known=np.asarray([tuple([mapping[(int(b),int(f))],*xy]) in train_keys for b,f,xy in zip(test_raw.buildings,test_raw.floors,test_raw.xy)])
    nearest=np.zeros(len(test_raw));outside=np.zeros(len(test_raw),dtype=bool)
    for cls in range(len(bundle.classes)):
        positions=np.unique(train.xy_original[train.classes.numpy()==cls],axis=0)
        selected=np.asarray([mapping[(int(b),int(f))]==cls for b,f in zip(test_raw.buildings,test_raw.floors)])
        queries=test_raw.xy[selected]
        nearest[selected]=np.sqrt(((queries[:,None]-positions[None])**2).sum(-1)).min(1)
        outside[selected]=((queries<positions.min(0))|(queries>positions.max(0))).any(1)
    csv_save(tables/'test_spatial_diagnostics.csv',[{'source_file':test_raw.source_files[i],'source_row':int(test_raw.source_rows[i]),
        'nearest_internal_train_position_distance':float(nearest[i]),'outside_same_floor_train_bbox':bool(outside[i])} for i in range(len(test_raw))])
    group_rows=[]
    for seed in seeds:
        for variant in NAMES:
            rows=predictions[(seed,variant,'test')];errors=np.asarray([float(r['error_2d']) for r in rows])
            for label,mask in [('seen_train_position',known),('unseen_train_position',~known)]:
                if mask.any():group_rows.append({'seed':seed,'variant':variant,'group':label,'n':int(mask.sum()),'mean_2d':float(errors[mask].mean())})
    csv_save(tables/'test_seen_unseen_positions.csv',group_rows)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    labels=[s for s in ('seen_train_position','unseen_train_position') if any(r['group']==s for r in group_rows)];x=np.arange(len(labels))
    for j,variant in enumerate(NAMES):
        means=[];sds=[]
        for label in labels:
            v=[r['mean_2d'] for r in group_rows if r['variant']==variant and r['group']==label];means.append(np.mean(v));sds.append(np.std(v,ddof=1))
        ax.bar(x+(j-.5)*.34,means,.34,yerr=sds,capsize=4,label=NAMES[variant],color=COLORS[variant])
    ax.set_xticks(x,[('Đã có trong catalogue train' if s.startswith('seen') else 'Chưa có trong catalogue train') for s in labels]);ax.set_ylabel(f'Mean 2D test ({unit})');ax.legend(loc='upper center',ncol=2);ax.set_ylim(0,ax.get_ylim()[1]*1.20);save(7,'diem_da_do_va_chua_do')

    floors=sorted({r['true_floor'] for r in predictions[(representative,'pure','test')]})
    fig,axes=plt.subplots(len(floors),2,figsize=(11,4*len(floors)),squeeze=False,layout='constrained')
    all_errors=[float(r['error_2d']) for seed in seeds for variant in NAMES for r in predictions[(seed,variant,'test')]]
    vmax=float(np.quantile(all_errors,.95))
    point_rows=[]
    for j,variant in enumerate(NAMES):
        for i,floor in enumerate(floors):
            grouped={}
            for seed in seeds:
                for row in predictions[(seed,variant,'test')]:
                    if row['true_floor']!=floor:continue
                    key=(float(row['true_x']),float(row['true_y']))
                    grouped.setdefault(key,[]).append(float(row['error_2d']))
            coords=np.asarray(list(grouped));errors=np.asarray([np.mean(v) for v in grouped.values()])
            ax=axes[i,j];scatter=ax.scatter(coords[:,0],coords[:,1],c=errors,cmap='viridis',vmin=0,vmax=vmax,s=35)
            ax.set_aspect('equal',adjustable='box');ax.set_xlabel(f'X ({unit})');ax.set_ylabel(f'Y ({unit})');ax.set_title(f'{NAMES[variant]} tầng {floor}');fig.colorbar(scatter,ax=ax,label=f'Mean 2D tại điểm ({unit})',shrink=.8)
            for (x,y),error in zip(coords,errors):point_rows.append({'variant':variant,'floor':int(floor),'x':x,'y':y,'mean_error_across_queries_and_seeds':error})
    save(8,'ban_do_sai_so');csv_save(tables/'test_per_position.csv',point_rows)

    fig,axes=plt.subplots(1,3,figsize=(11,3.8),layout='constrained')
    for ax,key,label in zip(axes,['train_seconds','RAM_peak_MiB','VRAM_allocated_peak_MiB'],['Thời gian train (s)','RAM toàn tiến trình (MiB)','VRAM cấp phát PyTorch (MiB)']):
        ax.bar([str(s) for s in seeds],[r[key] for r in resources],color='#367995');ax.set_title(label);ax.set_xlabel('Seed');ax.grid(axis='y',alpha=.2)
    save(9,'tai_nguyen_huan_luyen_chung')

    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');x=np.arange(len(seeds))
    for j,variant in enumerate(NAMES):
        part=[next(r for r in records if r['seed']==seed and r['variant']==variant and r['split']=='test') for seed in seeds]
        ax.bar(x+(j-.5)*.34,[r['inference_ms_per_sample'] for r in part],.34,color=COLORS[variant],label=NAMES[variant])
    ax.set_xticks(x,[f'Seed {s}' for s in seeds]);ax.set_ylabel('ms/mẫu trung bình trong batch 128');ax.legend(loc='upper center',ncol=2);ax.set_ylim(0,ax.get_ylim()[1]*1.20);save(10,'thoi_gian_suy_luan_test')

    paired=[]
    for seed in seeds:
        pure=predictions[(seed,'pure','test')];hybrid=predictions[(seed,'hybrid','test')]
        for p,h in zip(pure,hybrid):
            assert (p['source_file'],p['source_row'])==(h['source_file'],h['source_row'])
            paired.append({'seed':seed,'source_file':p['source_file'],'source_row':int(p['source_row']),
                'pure_error':float(p['error_2d']),'hybrid_error':float(h['error_2d']),
                'hybrid_minus_pure':float(h['error_2d'])-float(p['error_2d'])})
    csv_save(tables/'paired_test_errors.csv',paired)
    detected=(test_raw.rssi!=test_raw.missing_code).sum(1)
    ap_rows=[]
    for seed in seeds:
        for variant in NAMES:
            errors=np.asarray([float(r['error_2d']) for r in predictions[(seed,variant,'test')]])
            for label,mask in [('0',detected==0),('1_to_4',(detected>=1)&(detected<=4)),('5_to_16',(detected>=5)&(detected<=16)),('17_or_more',detected>=17)]:
                if mask.any():ap_rows.append({'seed':seed,'variant':variant,'detected_AP_group':label,'n':int(mask.sum()),'mean_2d':float(errors[mask].mean())})
    csv_save(tables/'test_error_by_detected_AP_count.csv',ap_rows)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
    ax.hist([r['hybrid_minus_pure'] for r in paired],bins=60,color='#357b91');ax.axvline(0,color='black',linestyle='--');ax.set_xlabel(f'Sai số KAN lai trừ KAN tham chiếu ({unit})');ax.set_ylabel('Số dự đoán gộp ba seed');save(11,'chenh_lech_tung_mau')

    # AP selection for this diagnostic uses train detection frequency only.
    signal=train.features.numpy()[:,:len(test_raw.ap_order)]
    order=np.argsort(-signal[...,1].sum(0),kind='stable')[:min(24,len(test_raw.ap_order))]
    x=db_transform(signal[:,order]);centered=x-x.mean(0);den=np.sqrt((centered**2).sum(0));constant=den==0;den[constant]=1
    correlation=(centered.T@centered)/(den[:,None]*den[None,:])
    correlation[constant,:]=np.nan;correlation[:,constant]=np.nan
    labels=[test_raw.ap_order[i] for i in order]
    csv_save(tables/'ap_correlation_train.csv',[{'AP':label,**{labels[j]:float(v) for j,v in enumerate(row)}} for label,row in zip(labels,correlation)])
    fig,ax=plt.subplots(figsize=(8,7),layout='constrained');im=ax.imshow(correlation,vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(len(labels)),labels,rotation=90,fontsize=9);ax.set_yticks(range(len(labels)),labels,fontsize=9);fig.colorbar(im,ax=ax,label='Hệ số tương quan');ax.set_title('AP trong train sau biểu diễn RSSI và giá trị thiếu');save(12,'ma_tran_tuong_quan_ap_train')
    diagram=output/'diagram'
    if all((diagram/f'13_so_do_suy_luan.{ext}').exists() for ext in ('png','svg')):
        for ext in ('png','svg'):shutil.copyfile(diagram/f'13_so_do_suy_luan.{ext}',figures/f'13_so_do_suy_luan.{ext}')
        image_names.append('13_so_do_suy_luan')
    device_rows=[]
    if 'device_id' in test_raw.metadata:
        for seed in seeds:
            for variant in NAMES:
                errors=np.asarray([float(r['error_2d']) for r in predictions[(seed,variant,'test')]])
                for device in sorted(set(test_raw.metadata['device_id'])):
                    mask=test_raw.metadata['device_id']==device
                    device_rows.append({'seed':seed,'variant':variant,'device_id':int(device),'n':int(mask.sum()),
                        'mean_2d':float(errors[mask].mean()),'median_2d':float(np.median(errors[mask])),
                        'p90_2d':float(np.quantile(errors[mask],.9))})
        csv_save(tables/'test_per_device.csv',device_rows)
        devices=sorted(set(test_raw.metadata['device_id']));x=np.arange(len(devices))
        fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
        for j,variant in enumerate(NAMES):
            means=[];sds=[]
            for device in devices:
                values=[r['mean_2d'] for r in device_rows if r['device_id']==device and r['variant']==variant]
                means.append(np.mean(values));sds.append(np.std(values,ddof=1))
            bars=ax.bar(x+(j-.5)*.34,means,.34,yerr=sds,capsize=4,color=COLORS[variant],label=NAMES[variant])
            for bar,mean,sd in zip(bars,means,sds):ax.text(bar.get_x()+bar.get_width()/2,mean+sd+.04,f'{mean:.2f}',ha='center',va='bottom',fontsize=10)
        ax.set_xticks(x,[f'Thiết bị {int(d)}' for d in devices]);ax.set_ylabel(f'Mean 2D test ({unit})')
        ax.legend(loc='upper center',ncol=2);ax.set_ylim(0,ax.get_ylim()[1]*1.2);save(14,'test_theo_thiet_bi')
    source_train=read_supplied(name,'train')
    source_keys=set(zip(source_train.buildings.tolist(),source_train.floors.tolist(),source_train.xy[:,0],source_train.xy[:,1]))
    source_overlap=np.asarray([tuple([int(b),int(f),*xy]) in source_keys for b,f,xy in zip(test_raw.buildings,test_raw.floors,test_raw.xy)])
    result={'dataset':name,'seeds':seeds,'units':unit,'split_manifest':bundle.manifest,
        'per_seed':records,'aggregate':aggregate,'training_resources':resources,'figures':image_names,
        'test_known_train_position_rows':int(known.sum()),'test_unseen_train_position_rows':int((~known).sum()),
        'test_outside_same_floor_train_bbox_rows':int(outside.sum()),'nearest_train_position_distance_median':float(np.median(nearest)),
        'nearest_train_position_distance_max':float(nearest.max()),
        'test_all_undetected_rows':int((detected==0).sum()),'test_detected_AP_count_median':float(np.median(detected)),
        'paired_fraction_hybrid_better':float(np.mean([r['hybrid_minus_pure']<0 for r in paired])),
        'test_used_for_selection':False,'variants_share_neural_training':True,
        'prediction_average_ensemble':False,'std_ddof':1,'representative_confusion_seed':representative}
    result.update({'test_per_device':device_rows,'test_overlap_development_position_rows':int(source_overlap.sum()),
        'development_unique_positions':len(source_keys),'test_unique_positions':len(set(zip(test_raw.buildings.tolist(),test_raw.floors.tolist(),test_raw.xy[:,0],test_raw.xy[:,1]))),
        'train_unique_positions':len(train_keys),'validation_unique_positions':len({tuple([int(c),*xy]) for c,xy in zip(datasets['validation'].classes.numpy(),datasets['validation'].xy_original)})})
    save_json(report/'results.json',result)
    (report/'README.md').write_text(f'# Kết quả {name.upper()}\n\n12 hình dữ liệu thực nghiệm và sơ đồ suy luận tùy chọn, PNG/SVG, bảng CSV và results.json.\n\n'
        'Ba seed dùng cùng split. KAN tham chiếu và KAN lai chung trọng số neural; decoder/lai được đóng băng từ UJI. '
        'Train là chẩn đoán; catalogue lai có chứa chính truy vấn train. Không dùng test chọn checkpoint/cấu hình.\n',encoding='utf-8')
    print(f'EXPORTED {name}: {len(image_names)} figures, {len(records)} measured rows, {report}',flush=True)


def db_transform(features):
    return features[...,0]*(104/105)+features[...,1]/105


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--datasets',nargs='+',default=['dlulib','hcxy']);args=parser.parse_args()
    for name in args.datasets:generate(name,args.output_dir.resolve())


if __name__=='__main__':main()
