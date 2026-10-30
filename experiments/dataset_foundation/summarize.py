"""Extra coverage, held-out-driver and rollout diagnostics; no model fitting."""
from pathlib import Path
import sys,os,json,csv
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from src.unified_driving_dataset import load_episode_records,UnifiedDrivingDataset

def main():
 r=load_episode_records(require_images=False);splits=json.loads((HERE/'results/splits.json').read_text())['episodes']
 ds=UnifiedDrivingDataset(r,action_offsets={'bench2drive':0,'person2drive':0});dur=[];split_rows=[]
 v_window,a_window,y_window,n_window=ds.dynamics_arrays();coverage=[]
 for part,keys in splits.items():
  for source in ['bench2drive','person2drive']:
   choose=np.isin(n_window,[key for key in keys if r[key]['source']==source]);state=np.c_[v_window[choose,-1],y_window[choose]];raw=a_window[choose]
   row=dict(source=source,split=part,eligible_windows=int(choose.sum()))
   for offset in [0,1]:
    plan=raw[:,3+offset:13+offset];partial=np.all((plan[:,:,2]>.01)&(plan[:,:,2]<.99),axis=1)&(state[:,0]>=2)
    row[f'coast_H{offset}']=int((np.all((plan[:,:,1]<=.01)&(plan[:,:,2]<=.01),axis=1)&np.all(state>=2,axis=1)).sum())
    row[f'partial_H{offset}']=int(partial.sum());row[f'partial_start_above5_H{offset}']=int((partial&(state[:,0]>5)).sum())
    row[f'multiple_transitions_H{offset}']=int((np.any(abs(np.diff(plan,axis=1))>.1,axis=2).sum(1)>=2).sum())
   coverage.append(row)
 pd.DataFrame(coverage).to_csv(HERE/'results/training_window_coverage.csv',index=False)
 for key,record in r.items():
  v,a,ids=record['speed'],record['actions'],record['frame_ids'];part=next(s for s,keys in splits.items() if key in keys)
  masks={'moving_coast':(v>=2)&(a[:,1]<=.01)&(a[:,2]<=.01),'partial_brake':(a[:,2]>.01)&(a[:,2]<.99),
         'strong_brake':a[:,2]>=.8,'low_throttle':(a[:,1]>.01)&(a[:,1]<.3)&(a[:,2]<=.01),
         'medium_throttle':(a[:,1]>=.3)&(a[:,1]<.7)&(a[:,2]<=.01),'high_throttle':(a[:,1]>=.7)&(a[:,2]<=.01),'turning':abs(a[:,0])>=.1}
  for regime,mask in masks.items():
   start=None
   for i in range(len(ids)+1):
    flag=i<len(ids) and mask[i]
    if start is not None and (not flag or ids[i]!=ids[i-1]+1):
     count=i-start;dur.append(dict(source=record['source'],episode=key,split=part,regime=regime,frames=count,duration_seconds=count*.1,span_seconds=(count-1)*.1));start=None
    if flag and start is None:start=i
  for name,mask in masks.items():split_rows.append(dict(source=record['source'],split=part,episode=key,regime=name,frames=int(mask.sum())))
 pd.DataFrame(dur).to_csv(HERE/'results/action_durations.csv',index=False)
 pd.DataFrame(split_rows).to_csv(HERE/'results/split_control_coverage.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(11,4))
 for source in ['bench2drive','person2drive']:
  for ax,regime in zip(axes,['moving_coast','partial_brake']):
   values=[d['duration_seconds'] for d in dur if d['source']==source and d['regime']==regime]
   if values:ax.hist(values,bins=np.arange(0,7,.2),histtype='step',label=source)
   ax.set(xlabel='Continuous run duration (frame count / 10 Hz)',ylabel='Runs',title=regime,yscale='log')
   if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(HERE/'results/action_durations.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(13,7))
 for row,source in enumerate(['bench2drive','person2drive']):
  speed=np.concatenate([rec['speed'] for rec in r.values() if rec['source']==source]);action=np.concatenate([rec['actions'] for rec in r.values() if rec['source']==source])
  for col,label in enumerate(['Steer','Throttle','Brake']):
   ax=axes[row,col];hist,xe,ye=np.histogram2d(speed,action[:,col],bins=[np.arange(0,max(32,2*np.ceil(speed.max()/2)+2),2),np.linspace(min(-1,action[:,0].min()),max(1,action[:,0].max()),31) if col==0 else np.linspace(0,1,21)])
   # Each speed column is a conditional distribution, not a joint frame count.
   hist=hist/np.maximum(hist.sum(1,keepdims=True),1)
   cmap=plt.get_cmap('magma').copy();cmap.set_bad('#dddddd')
   im=ax.pcolormesh(xe,ye,np.ma.masked_less_equal(hist.T,0),norm=LogNorm(vmin=1e-4,vmax=1),cmap=cmap);ax.set(xlabel='Speed (m/s)',ylabel=label,title=source);fig.colorbar(im,ax=ax,label='Fraction within speed bin')
 fig.tight_layout();fig.savefig(HERE/'results/speed_control_heatmaps.png',dpi=160);plt.close(fig)
 stats=[]
 for source in ['bench2drive','person2drive']:
  speed=np.concatenate([rec['speed'] for rec in r.values() if rec['source']==source]);action=np.concatenate([rec['actions'] for rec in r.values() if rec['source']==source])
  for field,values in [('speed',speed),('steer',action[:,0]),('throttle',action[:,1]),('brake',action[:,2])]:
   stats.append(dict(source=source,field=field,**{f'p{q}':float(np.percentile(values,q)) for q in [0,5,25,50,75,95,100]}))
 pd.DataFrame(stats).to_csv(HERE/'results/state_action_ranges.csv',index=False)
 metrics=pd.read_csv(HERE/'results/dynamics_metrics.csv');chosen=json.loads((HERE/'results/action_alignment.json').read_text())['action_offsets']
 episode=metrics[(metrics.episode!='ALL')&(metrics.subset=='all')&(metrics.step==10)&(metrics.split=='test')].copy()
 episode['driver']=episode.episode.map(lambda k:r[k]['driver']);episode.to_csv(HERE/'results/heldout_episode_metrics.csv',index=False)
 driver=[]
 for (source,name,model),g in episode.groupby(['source','driver','model']):
  if not name:continue
  driver.append(dict(source=source,driver=name,model=model,windows=int(g.n.sum()),mae=float(np.average(g.mae,weights=g.n)),rmse=float(np.sqrt(np.average(g.rmse**2,weights=g.n)))))
 pd.DataFrame(driver).to_csv(HERE/'results/heldout_driver_metrics.csv',index=False)
 fig,axes=plt.subplots(2,3,figsize=(14,7));quality=[]
 for row,source in enumerate(['bench2drive','person2drive']):
  data=np.load(HERE/'results'/f'{source}_test_rollouts.npz');target=data['target'];current=data['current'];offset=chosen[source]
  model=f'GRU_H{offset}_seed0';pred=data[model]
  for col,regime in enumerate(['coasting','partial_braking','strong_braking']):
   ax=axes[row,col];eligible=np.flatnonzero(data['mask_'+regime]);ax.set(title=f'{source}: {regime}',xlabel='Horizon (s)',ylabel='Speed (m/s)');ax.grid(alpha=.3)
   if not len(eligible):ax.text(.2,.5,'No held-out windows',transform=ax.transAxes);continue
   err=abs(pred[eligible,-1]-target[eligible,-1]);index=eligible[np.argsort(err)[len(err)//2]] # representative median-error example
   for key,label in [('target','Observed'),(model,'GRU'),('constant_speed','Constant speed'),('physics_H'+str(offset),'Physics')]:
    ax.plot(np.arange(11)*.1,np.r_[current[index],data[key][index]],label=label)
   ax.legend(fontsize=7)
  for model in [k for k in data.files if k.startswith('GRU')]:
   p=data[model];sequence=np.c_[current,p];acc=np.diff(sequence,axis=1)/.1
   quality.append(dict(source=source,model=model,finite=bool(np.isfinite(p).all()),minimum_speed=float(p.min()),maximum_speed=float(p.max()),
    absolute_accel_p995=float(np.quantile(abs(acc),.995)),observed_abs_accel_p995=float(np.quantile(abs(np.diff(np.c_[current,target],axis=1)/.1),.995)),
    predicted_accel_above15_fraction=float((abs(acc)>15).mean())))
 fig.tight_layout();fig.savefig(HERE/'results/rollout_examples.png',dpi=160);plt.close(fig)
 (HERE/'results/prediction_quality.json').write_text(json.dumps(quality,indent=2)+'\n')
 # Validation alignment consistency by episode, paired across the same held-out windows.
 val=metrics[(metrics.episode!='ALL')&(metrics.subset=='all')&(metrics.step==0)&(metrics.split=='validation')&(metrics.model.str.startswith('GRU'))].copy()
 val['alignment']=val.model.str.extract(r'(H[01])');paired=val.groupby(['source','episode','alignment']).rmse.mean().unstack('alignment');paired['winner']=np.where(paired.H0<paired.H1,'H0','H1');paired.to_csv(HERE/'results/alignment_by_episode.csv')
 print('Diagnostic summaries generated',flush=True)
if __name__=='__main__':main()
