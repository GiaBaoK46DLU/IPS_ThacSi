"""Scientific plots only; no training, checkpoint selection or inference."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'runs/kan_v4_docx_20261026'
d=json.loads((OUT/'evidence.json').read_text(encoding='utf-8'))
font_manager.fontManager.addfont('C:/Windows/Fonts/arial.ttf')
plt.rcParams.update({'font.family':'Arial','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
colors=['#737b84','#1b5e84']; seeds=d['protocol']['seeds']
fig,ax=plt.subplots(figsize=(8.3,4.0))
x=np.arange(10)
for name,color in zip(('V3','V4'),colors):
 y=([d['V3_per_seed'][str(s)]['mean_2d'] for s in seeds] if name=='V3' else [r['mean_2d'] for r in d['V4']['per_seed']])
 ax.plot(x,y,'o-',label=name,color=color,lw=1.7,ms=5)
ax.axhline(7,color='#9f3a32',ls='--',lw=1.2,label='Mục tiêu 7 m');ax.set_xticks(x,[str(s) for s in seeds]);ax.set_xlabel('Seed cố định');ax.set_ylabel('Sai số trung bình trên Test (m)');ax.set_ylim(6.7,8.25);ax.grid(axis='y',alpha=.18);ax.legend(ncol=3);fig.tight_layout();fig.savefig(OUT/'figures/test_seeds.png');plt.close(fig)
errors=np.load(OUT/'errors.npz');fig,ax=plt.subplots(figsize=(8.3,3.7)); thresholds=np.linspace(0,30,301)
for name,color in zip(('V3','V4'),colors):
 rates=np.array([(errors[name]<=t).mean(axis=1) for t in thresholds]).mean(axis=1)*100
 ax.plot(thresholds,rates,label=name,color=color,lw=2)
ax.set_xlabel('Ngưỡng sai số 2D (m)');ax.set_ylabel('Tỷ lệ mẫu không vượt ngưỡng (%)');ax.set_ylim(0,100);ax.set_xlim(0,30);ax.grid(alpha=.18);ax.legend();fig.tight_layout();fig.savefig(OUT/'figures/test_cdf.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(8.3,3.5));x=np.arange(2);w=.33
ax.bar(x-w/2,[2/82*100,80/82*100],w,color=colors[0],label='V3 đều fingerprint')
ax.bar(x+w/2,[50,50],w,color=colors[1],label='V4 đều điểm tham chiếu')
ax.set_xticks(x,['Điểm A có 2 fingerprint','Điểm B có 80 fingerprint']);ax.set_ylabel('Xác suất lấy vào một lượt (%)');ax.set_ylim(0,112);ax.legend(ncol=2,loc='upper center');ax.grid(axis='y',alpha=.18);fig.tight_layout();fig.savefig(OUT/'figures/sampling_example.png');plt.close(fig)
fig,ax=plt.subplots(figsize=(8.3,3.5));x=np.arange(len(d['by_ap']));w=.34
for method,color,shift in [('V3',colors[0],-w/2),('V4',colors[1],w/2)]:ax.bar(x+shift,[r[method] for r in d['by_ap']],w,label=method,color=color)
ax.set_xticks(x,[r['label']+' AP\nN='+str(r['n']) for r in d['by_ap']]);ax.set_ylabel('Sai số trung bình trên Test (m)');ax.legend(ncol=2);ax.grid(axis='y',alpha=.18);fig.tight_layout();fig.savefig(OUT/'figures/test_ap_groups.png');plt.close(fig)
print('Created three empirical V3/V4 plots and one explicitly illustrative sampling example.')
