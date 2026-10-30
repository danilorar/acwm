"""Probe fixed command ordering; these are synthetic inputs, not intervention validation."""
from pathlib import Path
import sys,json,csv
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from src.unified_driving_dataset import load_episode_records,UnifiedDrivingDataset
from experiments.dynamics_validation.run import Dynamics

def main():
 torch.set_num_threads(1);records=load_episode_records(require_images=False);splits=json.loads((HERE/'results/splits.json').read_text())['episodes'];config=json.loads((HERE/'results/dynamics_config.json').read_text())['source_config']
 ds=UnifiedDrivingDataset(records,action_offsets={'bench2drive':0,'person2drive':0});v,a,y,names=ds.dynamics_arrays();rows=[]
 for source in ['bench2drive','person2drive']:
  mask=np.isin(names,[k for k in splits['test'] if records[k]['source']==source]);rng=np.random.default_rng(42);groups=[]
  for low,high in [(2,5),(5,10),(10,15),(15,np.inf)]:
   eligible=np.flatnonzero(mask&(v[:,-1]>=low)&(v[:,-1]<high));groups.extend(rng.choice(eligible,min(250,len(eligible)),replace=False))
  idx=np.array(groups)
  mean=np.array(config[source]['mean'],np.float32);std=np.array(config[source]['std'],np.float32);mt,st=torch.tensor(mean),torch.tensor(std)
  for offset in [0,1]:
   hist=np.zeros((len(idx),4,5),np.float32);hist[:,:,0]=v[idx];hist[:,1:,1:4]=a[idx,offset:offset+3];hist[:,1:,4]=1;hist[:,:,:4]=(hist[:,:,:4]-mean)/std;hist[:,0,1:4]=0
   for seed in [0,1]:
    net=Dynamics();net.load_state_dict(torch.load(HERE/'results'/f'{source}_H{offset}_seed{seed}.pt',weights_only=True));net.eval();pred=[]
    with torch.no_grad():
     for command in [[0,0,1],[0,0,.3],[0,0,0],[0,1,0]]:
      plan=np.tile(command,(len(idx),10,1)).astype(np.float32);pred.append(net(torch.tensor(hist),torch.tensor((plan-mean[1:])/std[1:]),mt,st).numpy())
    inverted=np.logical_or.reduce([x>y+.05 for x,y in zip(pred,pred[1:])]);acc=np.concatenate([np.diff(np.c_[v[idx,-1],p],axis=1)/.1 for p in pred])
    rows.append(dict(source=source,offset=offset,seed=seed,cases=len(idx),minimum_observed_speed=float(v[idx,-1].min()),maximum_observed_speed=float(v[idx,-1].max()),any_horizon_order_failure_fraction=float(inverted.any(1).mean()),
     timestep_order_failure_fraction=float(inverted.mean()),candidate_abs_accel_p995=float(np.quantile(abs(acc),.995)),
     candidate_abs_accel_above15_fraction=float((abs(acc)>15).mean()),candidate_min_speed=float(min(p.min() for p in pred)),
     full_throttle_delta_mean=float((pred[3][:,-1]-v[idx,-1]).mean()),coast_delta_mean=float((pred[2][:,-1]-v[idx,-1]).mean()),
     partial_delta_mean=float((pred[1][:,-1]-v[idx,-1]).mean()),full_brake_delta_mean=float((pred[0][:,-1]-v[idx,-1]).mean())))
 with (HERE/'results/candidate_checks.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('All-horizon candidate checks completed',flush=True)
if __name__=='__main__':main()
