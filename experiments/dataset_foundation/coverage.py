"""Source/speed/action-duration coverage; no image encoder or model training."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import os
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT));os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.unified_driving_dataset import load_episode_records,nominal_controls


def write(name,rows):
    with (HERE/'results'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def longest(mask,ids):
    length=maximum=0
    for i,flag in enumerate(mask):
        length=(length+1 if i and ids[i]==ids[i-1]+1 else 1) if flag else 0
        maximum=max(maximum,length)
    return maximum


def inspect(records,verified):
    rows=[];conditioned=[];joint=defaultdict(int);qc=[]
    bins=[(-np.inf,2,'below2'),(2,5,'2to5'),(5,10,'5to10'),(10,15,'10to15'),(15,np.inf,'above15')]
    for key,r in records.items():
        ids,v,a=r['frame_ids'],r['speed'],r['actions'];n=len(ids)
        moving=v>=2;partial=(a[:,2]>.01)&(a[:,2]<.99);coast=moving&(a[:,1]<=.01)&(a[:,2]<=.01)
        count=dict(source=r['source'],episode=key,driver=r['driver'],route=r['route'],frames=n,out_of_nominal_control_frames=int((~nominal_controls(a)).sum()),
                   moving_frames=int(moving.sum()),coast_frames=int(coast.sum()),partial_frames=int(partial.sum()),
                   partial_above5=int((partial&(v>5)).sum()),simultaneous_pedal_frames=int(((a[:,1]>.01)&(a[:,2]>.01)).sum()),frames_above5=int((v>5).sum()),
                   max_coast_frames=longest(coast,ids),max_partial_frames=longest(partial,ids),
                   one_sec_coast_H0=0,one_sec_coast_H1=0,one_sec_partial_H0=0,one_sec_partial_H1=0,
                   multi_transition_windows_H0=0,multi_longitudinal_regime_windows_H0=0,valid_one_sec_windows=0)
        for i in range(n-10):
            if ids[i+10]-ids[i]!=10 or not nominal_controls(a[i:i+11]).all():continue
            count['valid_one_sec_windows']+=1
            block=a[i:i+10];speed_bin=next(label for low,high,label in bins if low<=v[i]<high)
            steer=float(abs(block[:,0]).max());steer_bin='straight' if steer<.05 else 'moderate_turn' if steer<.2 else 'strong_turn'
            regimes=[]
            for offset in [0,1]:
                plan=a[i+offset:i+offset+10]
                if np.all(v[i:i+11]>=2) and np.all((plan[:,1]<=.01)&(plan[:,2]<=.01)):
                    count[f'one_sec_coast_H{offset}']+=1
                    if offset==0:regimes.append('coast')
                if v[i]>=2 and np.all((plan[:,2]>.01)&(plan[:,2]<.99)):
                    count[f'one_sec_partial_H{offset}']+=1
                    if offset==0:regimes.append('partial_brake')
            if np.any(np.abs(np.diff(block,axis=0))>.1,axis=1).sum()>=2:
                count['multi_transition_windows_H0']+=1;regimes.append('multiple_transitions')
            pedal_mode=(block[:,1]>.01).astype(int)+2*(block[:,2]>.01).astype(int)
            count['multi_longitudinal_regime_windows_H0']+=int((np.diff(pedal_mode)!=0).sum()>=2)
            if np.all((block[:,1]>=.7)&(block[:,2]<=.01)):regimes.append('sustained_high_throttle')
            if np.all(block[:,2]>=.8) and v[i]>=2:regimes.append('sustained_strong_brake')
            for regime in regimes:joint[(r['source'],speed_bin,steer_bin,regime)]+=1
        rows.append(count)
        for low,high,label in bins:
            mask=(v>=low)&(v<high);part=a[mask]
            conditioned.append(dict(source=r['source'],episode=key,speed_bin=label,frames=int(mask.sum()),
                                    zero_pedal_frames=int(((part[:,1]<=.01)&(part[:,2]<=.01)).sum()),
                                    partial_frames=int(((part[:,2]>.01)&(part[:,2]<.99)).sum()),
                                    throttle_sum=float(part[:,1].sum()),brake_sum=float(part[:,2].sum()),abs_steer_sum=float(abs(part[:,0]).sum())))
        delta=np.diff(r['positions'][:,:2],axis=0);mean=(v[1:]+v[:-1])/2
        mask=(np.diff(ids)==1)&(mean>3);dt=np.linalg.norm(delta[mask],axis=1)/mean[mask]
        f=r['forward'][:-1,:2];projection=np.sum(delta*f,axis=1)/np.maximum(np.linalg.norm(delta,axis=1)*np.linalg.norm(f,axis=1),1e-8)
        qc.append(dict(source=r['source'],episode=key,dt_median=float(np.median(dt)) if len(dt) else None,
                       forward_cosine_median=float(np.median(projection[mask])) if mask.any() else None,
                       reverse_projection_while_speed_above3=int((mask&(projection<0)).sum()),
                       right_sign=r['right_sign']))
    summary=[]
    for source in ['bench2drive','person2drive','combined']:
        selected=[r for r in rows if source=='combined' or r['source']==source]
        row={'source':source,'episodes':len(selected)}
        for key in selected[0]:
            if key not in ['source','episode','driver','route']:
                row[key]=max(r[key] for r in selected) if key.startswith('max_') else sum(r[key] for r in selected)
        row['coast_fraction_moving']=row['coast_frames']/max(1,row['moving_frames'])
        row['partial_fraction_above5']=row['partial_above5']/max(1,row['frames_above5'])
        row['images_verified']=verified;summary.append(row)
    write('episode_coverage.csv',rows);write('coverage_summary.csv',summary);write('coordinate_checks.csv',qc)
    aggregate=[]
    for source in ['bench2drive','person2drive']:
        for _,_,label in bins:
            selected=[r for r in conditioned if r['source']==source and r['speed_bin']==label]
            aggregate.append(dict(source=source,speed_bin=label,**{k:sum(r[k] for r in selected) for k in ['frames','zero_pedal_frames','partial_frames','throttle_sum','brake_sum','abs_steer_sum']}))
    write('speed_conditioned.csv',aggregate)
    joints=[]
    for source in ['bench2drive','person2drive']:
        for _,_,speed_bin in bins:
            for steering in ['straight','moderate_turn','strong_turn']:
                for regime in ['coast','partial_brake','multiple_transitions','sustained_high_throttle','sustained_strong_brake']:
                    n=joint[(source,speed_bin,steering,regime)]
                    joints.append(dict(source=source,speed_bin=speed_bin,steering=steering,regime=regime,windows=n,low_support=n<20))
    write('joint_window_support.csv',joints)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for source in ['bench2drive','person2drive']:
        for ax,field,title in zip(axes,['zero_pedal_frames','partial_frames','abs_steer_sum'],['Zero pedals (%)','Partial brake (%)','Mean absolute steer']):
            subset=[r for r in aggregate if r['source']==source]
            scale=1 if field=='abs_steer_sum' else 100
            ax.plot([r['speed_bin'] for r in subset],[scale*r[field]/max(1,r['frames']) for r in subset],'o-',label=source)
            ax.set(xlabel='Speed bin (m/s)',ylabel=title);ax.legend(fontsize=8);ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(HERE/'results/speed_conditioned.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for source in ['bench2drive','person2drive']:
        actions=np.concatenate([r['actions'] for r in records.values() if r['source']==source])
        for i,ax in enumerate(axes):
            ax.hist(actions[:,i],bins=np.linspace(min(-1,actions[:,0].min()),max(1,actions[:,0].max()),61) if i==0 else np.linspace(0,1,41),histtype='step',density=True,label=source)
            ax.set_xlabel(['Steer','Throttle','Brake'][i]);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(HERE/'results/control_distributions.png',dpi=160);plt.close(fig)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-images',action='store_true');args=parser.parse_args()
    records=load_episode_records(require_images=args.verify_images)
    print(json.dumps(inspect(records,args.verify_images),indent=2))
