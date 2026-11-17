import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=json.load(open('/home/claude/answer-the-question-of-1029/results/summary.json'))
E=json.load(open('/home/claude/answer-the-question-of-1029/results/error_structure.json'))
CP=json.load(open('/home/claude/answer-the-question-of-1029/results/calib_per_epoch.json'))['per_fold']
F=[f"fold{i:02d}" for i in range(15)]; L=[f"p{i:02d}" for i in range(15)]
BL,GD='#2E4DA7','#D08C12'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#999','ytick.color':'#555','xtick.color':'#333'})
def save(fig,n): fig.tight_layout(); fig.savefig(__import__('os').path.join(__import__('os').path.dirname(__file__),'..','figures',n),dpi=200,transparent=True); plt.close(fig)
def grid(ax): ax.grid(axis='y',color='#ddd'); ax.set_axisbelow(True); ax.tick_params(axis='y',length=0)
# 1 B vs C
B=[R[f]['mae_B_last3'] for f in F]; C=[R[f]['mae_C_last3'] for f in F]
x=np.arange(16); fig,ax=plt.subplots(figsize=(13,5.2))
b1=ax.bar(x-0.2,B+[np.mean(B)],0.4,color=BL,label='B: original loss'); b2=ax.bar(x+0.2,C+[np.mean(C)],0.4,color=GD,label='C: + 3D angular loss')
ax.axvspan(14.5,15.5,color='#e9e9e9',zorder=0)
ax.text(b1[-1].get_x()+b1[-1].get_width(),np.mean(B)+0.55,f"{np.mean(B):.2f}°",ha='right',fontsize=12,fontweight='bold',color=BL)
ax.text(b2[-1].get_x(),np.mean(C)+0.15,f"{np.mean(C):.2f}°",ha='left',fontsize=12,fontweight='bold',color=GD)
ax.set_xticks(x,L+['Mean']); ax.set_ylabel('Mean angular error (°)'); ax.set_ylim(0,6.6); grid(ax)
ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(0.5,1.12)); save(fig,'c_bvc.png')
# 2 correlation
cb=[R[f]['B_corr_err_yaw_pitch'] for f in F]; cc=[R[f]['C_corr_err_yaw_pitch'] for f in F]; cg=[R[f]['corr_gt_yaw_pitch'] for f in F]
x=np.arange(15); fig,ax=plt.subplots(figsize=(13,5))
ax.bar(x-0.27,cb,0.27,color=BL,label='B'); ax.bar(x,cc,0.27,color=GD,label='C'); ax.bar(x+0.27,cg,0.27,color='#B5B5B5',label='Data itself (gt yaw vs gt pitch)')
ax.axhline(0,color='#555',lw=0.8); ax.set_ylim(-0.5,0.5); ax.set_xticks(x,L); ax.set_ylabel('Correlation r'); grid(ax)
ax.axhspan(-0.1,0.1,color='#2E4DA7',alpha=0.06,zorder=0)
ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.12)); save(fig,'c_corr.png')
# 3 pitch bias
pb=[E[f]['B']['pitch']['bias'] for f in F]
fig,ax=plt.subplots(figsize=(8.6,5.2)); bars=ax.bar(x,pb,0.62,color=[BL if v>=0 else GD for v in pb])
for b,v in zip(bars,pb): ax.text(b.get_x()+b.get_width()/2,v+(0.25 if v>=0 else -0.25),f"{v:+.1f}",ha='center',va='bottom' if v>=0 else 'top',fontsize=10.5)
ax.axhline(0,color='#555',lw=0.8); ax.set_xticks(x,L,fontsize=11,rotation=45); ax.set_ylabel('Mean pitch bias of B (°)'); ax.set_ylim(-7,6.5); grid(ax); save(fig,'c_bias.png')
# 4 calibration
nc=[CP[f]['B']['nocalib'] for f in F]; o9=[CP[f]['B']['off9'] for f in F]
x=np.arange(16); fig,ax=plt.subplots(figsize=(13,5.2))
b1=ax.bar(x-0.21,nc+[np.mean(nc)],0.4,color=BL,label='No calibration'); b2=ax.bar(x+0.21,o9+[np.mean(o9)],0.4,color=GD,label='9-image offset calibration')
ax.axvspan(14.5,15.5,color='#e9e9e9',zorder=0)
ax.text(b1[-1].get_x()+b1[-1].get_width()/2,np.mean(nc)+0.1,f"{np.mean(nc):.2f}°",ha='center',fontsize=12,fontweight='bold',color=BL)
ax.text(b2[-1].get_x()+b2[-1].get_width()/2,np.mean(o9)+0.1,f"{np.mean(o9):.2f}°",ha='center',fontsize=12,fontweight='bold',color=GD)
ax.set_xticks(x,L+['Mean']); ax.set_ylabel('Mean angular error (°)'); ax.set_ylim(0,6.6); grid(ax)
ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(0.5,1.12)); save(fig,'c_calib.png')
print(np.mean(nc),np.mean(o9),sum(a<b for a,b in zip(o9,nc)), min(cb),max(cb),min(cc),max(cc),min(cg),max(cg))
