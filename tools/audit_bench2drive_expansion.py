"""Compare the extracted expansion with Mini and validate grouped PyTorch loading."""
from pathlib import Path
import csv
from collections import Counter, defaultdict
import gzip
import json
import os
import sys
import hashlib

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from src.bench2drive_dataset import Bench2DriveDataset, split_scenarios
from torch.utils.data import DataLoader
import torch
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(PROJECT/'data/bench2drive_train/manifests/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from download_bench2drive import save_json


def summarize(root):
    routes=[];controls=[];speeds=[]
    for anno in sorted(root.rglob('anno')):
        if not anno.is_dir():continue
        files=sorted(anno.glob('*.json.gz'))
        name=str(anno.parent.relative_to(root));moving=coast=overlap=intermediate=0
        runs=[];length=0;previous=None;count=0
        for file in files:
            with gzip.open(file,'rt') as f:row=json.load(f)
            frame=int(file.name.split('.')[0]);speed=float(row['speed'])
            action=[float(row[k]) for k in ['steer','throttle','brake']]
            ego=[box for box in row.get('bounding_boxes',[]) if box.get('class')=='ego_vehicle']
            if len(ego)!=1 or not np.isfinite([*ego[0]['location'],*ego[0]['rotation']]).all():
                raise ValueError(f'Invalid simulator ego pose: {file}')
            if not np.isfinite([speed,*action]).all():raise ValueError(f'Nonfinite annotation {file}')
            count+=1;controls.append(action);speeds.append(speed)
            moving+=speed>=2;coasting=speed>=2 and action[1]<=.01 and action[2]<=.01
            coast+=coasting;overlap+=action[1]>.01 and action[2]>.01
            intermediate+=.1<action[2]<.9
            if length and (not coasting or previous is None or frame!=previous+1):
                runs.append(length);length=0
            if coasting:length+=1
            previous=frame
        if length:runs.append(length)
        routes.append(dict(recording=name,frames=count,moving_frames=int(moving),coasting_frames=int(coast),
                           coasting_runs=len(runs),max_coasting_frames=max(runs,default=0),
                           sustained_coasting_runs_ge_10=sum(n>=10 for n in runs),
                           overlap_frames=int(overlap),intermediate_brake_frames=int(intermediate)))
    if not routes:raise ValueError(f'No extracted recordings: {root}')
    actions=np.array(controls);v=np.array(speeds)
    summary={key:sum(r[key] for r in routes) for key in ['frames','moving_frames','coasting_frames','coasting_runs','sustained_coasting_runs_ge_10','overlap_frames','intermediate_brake_frames']}
    summary.update(recordings=len(routes),max_coasting_frames=max(r['max_coasting_frames'] for r in routes),
                   coasting_fraction=summary['coasting_frames']/max(1,summary['moving_frames']),
                   brake_values=np.unique(actions[:,2]).tolist(),
                   intermediate_throttle_frames=int(((actions[:,1]>.1)&(actions[:,1]<.9)).sum()))
    summary['overlap_fraction']=summary['overlap_frames']/summary['frames']
    summary['intermediate_brake_fraction']=summary['intermediate_brake_frames']/summary['frames']
    summary['intermediate_throttle_fraction']=summary['intermediate_throttle_frames']/summary['frames']
    summary['max_coasting_sampled_seconds']=summary['max_coasting_frames']/10
    summary['max_coasting_endpoint_span_seconds']=max(0,summary['max_coasting_frames']-1)/10
    return summary,routes,actions


def main():
    base=PROJECT/'data/bench2drive_train';out=base/'manifests'
    expanded,routes,actions=summarize(base/'extracted')
    mini,_,mini_actions=summarize(PROJECT/'data/bench2drive_mini')
    manifest=json.loads((out/'selection.json').read_text())
    state=json.loads((out/'extracted.json').read_text())
    type_by_root={root:e['scenario_type'] for e in manifest['selected'] if (s:=state.get(e['path'])) for root in s['roots']}
    by_type=defaultdict(list)
    for row in routes:
        scenario_type=type_by_root[row['recording'].split('/')[0]]
        row['scenario_type']=scenario_type;by_type[scenario_type].append(row['recording'])
    # For the initial 3/type subset, explicitly hold out one route of every type.
    # The existing splitter validates coverage and rejects town/route crossovers.
    val=[]
    for scenario_type,names in sorted(by_type.items()):
        if len(names)<2:raise ValueError(f'Need at least 2 extracted routes/type for a diverse holdout: {scenario_type}')
        rank=lambda name:hashlib.sha256(f"{manifest['seed']}:validation:{name}".encode()).hexdigest()
        val.append(min(names,key=rank))
    train,val=split_scenarios([r['recording'] for r in routes],val_scenarios=val,seed=manifest['seed'])
    save_json(out/'splits.json',dict(seed=manifest['seed'],train=train,validation=val,
                                    policy='One held-out route per available scenario type; route grouping validated'))
    torch.set_num_threads(1)
    train_ds=Bench2DriveDataset(base/'extracted',scenarios=train,image_size=(32,18))
    val_ds=Bench2DriveDataset(base/'extracted',scenarios=val,image_size=(32,18))
    # Validate schema and pose source on one sample from every recording.
    for dataset in [train_ds,val_ds]:
        first={}
        for i,(name,_) in enumerate(dataset.samples):first.setdefault(name,i)
        for index in first.values():dataset[index]
    batch=next(iter(DataLoader(train_ds,batch_size=2,num_workers=0)))
    assert batch['future_actions'].shape==(2,10,3)
    assert batch['future_motion_targets'].shape==(2,10,5)
    sizes=lambda root:sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    report=dict(action_offset=1,action_channels=['steer','throttle','brake'],history=4,horizon=10,recording_hz=10,repository=manifest['repository'],revision=manifest['revision'],mini=mini,expanded=expanded,
                extracted_archives=len(state),selected_archives=len(manifest['selected']),scenario_distribution={k:len(v) for k,v in by_type.items()},
                archives_bytes=sizes(base/'archives'),extracted_bytes=sizes(base/'extracted'),
                training_windows=len(train_ds),validation_windows=len(val_ds),total_windows=len(train_ds)+len(val_ds),
                loader_batch_shapes={k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)},
                interpretation='Coverage must be assessed from measured coast/brake results; more clips alone do not establish counterfactual/MPC validity.')
    save_json(out/'coverage_report.json',report)
    with (out/'route_coverage.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(routes[0]));writer.writeheader();writer.writerows(routes)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for column,(ax,title) in enumerate(zip(axes,['Steer','Throttle','Brake'])):
        bins=np.linspace(-1,1,41) if column==0 else np.linspace(0,1,41)
        ax.hist(mini_actions[:,column],bins=bins,density=True,histtype='step',label='Mini')
        ax.hist(actions[:,column],bins=bins,density=True,histtype='step',label='0.0.4 subset')
        ax.set(title=title,xlabel='Recorded control',ylabel='Density');ax.legend()
    fig.tight_layout();fig.savefig(out/'action_coverage.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
