"""Comparable control audit; never changes the production loader or datasets."""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
import os
import re
import tarfile

import numpy as np
from PIL import Image
from safe_pickle import load as load_numeric_pickle

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
OUT=HERE/'results'
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def episode(dataset,name,fps,ids,speed,action,**metadata):
    ids=np.asarray(ids,int);speed=np.asarray(speed,float);action=np.asarray(action,float)
    order=np.argsort(ids)
    ids,speed,action=ids[order],speed[order],action[order]
    if len(ids)!=len(set(ids)):raise ValueError(f'Duplicate IDs: {name}')
    if not np.isfinite(np.r_[speed,action.ravel()]).all():raise ValueError(f'Nonfinite telemetry: {name}')
    if ((action[:,0]<-1.001)|(action[:,0]>1.001)|(action[:,1:]<-.001).any(1)|(action[:,1:]>1.001).any(1)).any():
        raise ValueError(f'Invalid raw control range: {name}')
    return dict(dataset=dataset,name=name,fps=fps,ids=ids,speed=speed,action=action,**metadata)


def load_episodes():
    episodes=[];qc={}
    tele=HERE/'samples/TeleopWM-Dataset'
    for path in sorted(tele.rglob('pd_dataframe.pkl')):
        df=load_numeric_pickle(path);a=np.stack(df.action)[:,[1,0,2]];v=np.stack(df.speed).ravel()
        ids=[int(re.findall(r'\d+',Path(s).stem)[-1]) for s in df.image_path]
        gnss=np.stack(df.gnss);imu=np.stack(df.imu)
        # Robust, long-span distance/speed timing check, not a replacement for timestamps.
        lag=30;dt=[]
        for i in range(len(v)-lag):
            mean=v[i:i+lag+1].mean()
            if mean>4 and ids[i+lag]-ids[i]==lag:
                lat=(gnss[i,0]+gnss[i+lag,0])/2
                displacement=(gnss[i+lag,:2]-gnss[i,:2])*111319.49*np.array([1,np.cos(np.deg2rad(lat))])
                dt.append(np.linalg.norm(displacement)/mean/lag)
        name=str(path.parent.relative_to(tele))
        episodes.append(episode('TeleopWM',name,15,ids,v,a))
        qc[name]=dict(frames=len(df),columns=list(df.columns),expert_flags={str(k):int(v) for k,v in df.is_expert.value_counts().items()},
                      long_span_gnss_dt_median=float(np.median(dt)) if dt else None,
                      first_image_path=str(df.image_path.iloc[0]),last_image_path=str(df.image_path.iloc[-1]),
                      route_ids=[str(x) for x in df.route_id.unique()],pose_source='GNSS + IMU compass; no precise pose in inspected table')
    person=HERE/'samples/Person2Drive'
    person_qc={}
    for path in sorted(person.glob('sensor_data/*/b2d_infos_train.pkl')):
        records=load_numeric_pickle(path);groups=defaultdict(list)
        for record in records:groups[record['folder']].append(record)
        for folder,rows in groups.items():
            rows.sort(key=lambda r:r['frame_idx']);name=f'{path.parent.name}/{folder}'
            ids=[r['frame_idx'] for r in rows];v=np.array([r['ego_vel'][0] for r in rows])
            a=[[r[k] for k in ['steer','throttle','brake']] for r in rows]
            matrices=np.stack([r['world2ego'] for r in rows]);poses=np.linalg.inv(matrices)
            if not np.isfinite(poses).all():raise ValueError('Invalid Person2Drive transform')
            pos=poses[:,:3,3];step=np.diff(ids);mean=(v[1:]+v[:-1])/2
            mask=(mean>3)&(step==1)
            est=np.linalg.norm(np.diff(pos[:,:2],axis=0),axis=1)[mask]/mean[mask]
            # Camera image +x points right. Its ego-space direction tests lateral sign.
            right_in_ego=np.asarray(rows[0]['sensors']['CAM_FRONT']['cam2ego'])[:3,0]
            episodes.append(episode('Person2Drive',name,10,ids,v,a))
            person_qc[name]=dict(frames=len(rows),gaps=int((step!=1).sum()),first_frame=int(ids[0]),last_frame=int(ids[-1]),
                                 position_speed_dt_median=float(np.median(est)) if len(est) else None,
                                 camera_right_in_ego=right_in_ego.tolist(),first_camera_path=rows[0]['sensors']['CAM_FRONT']['data_path'],
                                 world2ego_present=True,ego_translation_z_note='ego_translation z is flattened in inspected records; inverse world2ego retains height')
    pdd=HERE/'samples/PDD';pdd_qc={};native_rows=[]
    for path in sorted(pdd.glob('*/data/*/metric/metric_info.json')):
        state=json.loads(path.read_text());raw=json.loads(path.with_name('metrics.json').read_text())['records']
        controls={int(r['control']['step']):r['control'] for r in raw}
        if len(controls)!=len(raw):raise ValueError(f'Duplicate control steps: {path}')
        keys=sorted(map(int,state));shared=[k for k in keys if k in controls]
        if len(shared)!=len(keys):raise ValueError(f'Unmatched PDD state keys: {path}')
        if not all(k%5==0 for k in keys):raise ValueError('PDD cadence differs from fixed 20Hz→4Hz audit assumption')
        rows=[state[str(k)] for k in shared];a=[[controls[k][c] for c in ['steer','throttle','brake']] for k in shared]
        v=[r['speed'] for r in rows];name=str(path.parents[1].relative_to(pdd))
        episodes.append(episode('PDD',name,4,np.array(shared)//5,v,a))
        actual=np.array([[controls[k][c] for c in ['steer','throttle','brake']] for k in sorted(controls)])
        native_rows.append(dict(episode=name,control_steps=len(raw),intermediate_brake_fraction=float(((actual[:,2]>.01)&(actual[:,2]<.99)).mean()),
                                overlap_fraction=float(((actual[:,1]>.01)&(actual[:,2]>.01)).mean())))
        # Check cadence using available world positions and logged speed, not image ordinal guesses.
        pos=np.array([r['location'] for r in rows]);speed=np.array(v);mean=(speed[:-1]+speed[1:])/2
        mask=(mean>3)&(np.diff(shared)==5)
        dt=np.linalg.norm(np.diff(pos[:,:2],axis=0),axis=1)[mask]/mean[mask]
        pdd_qc[name]=dict(states=len(rows),controls=len(raw),first_state_keys=shared[:6],last_state_key=shared[-1],
                           control_step_min=min(controls),control_step_max=max(controls),
                           all_state_keys_have_exact_control_step=True,position_speed_dt_median=float(np.median(dt)) if len(dt) else None,
                           public_loader_control_mismatch_fraction=float(np.mean([int(raw[i]['control']['step'])!=k for i,k in enumerate(shared)])),
                           public_loader_action_value_disagreement_fraction=float(np.mean([max(abs(controls[k][c]-raw[i]['control'][c]) for c in ['steer','throttle','brake'])>1e-5 for i,k in enumerate(shared)])),
                           rotation_note='sample rotation fits [roll,pitch,yaw], despite card stating [pitch,roll,yaw]; yaw index=2 is consistent')
    root=PROJECT/'data/bench2drive_train/extracted'
    for folder in sorted(root.iterdir()):
        files=sorted((folder/'anno').glob('*.json.gz'));ids=[];speed=[];a=[]
        if not files:continue
        for path in files:
            with gzip.open(path,'rt') as f:r=json.load(f)
            ids.append(int(path.name.split('.')[0]));speed.append(r['speed']);a.append([r[k] for k in ['steer','throttle','brake']])
        episodes.append(episode('Bench2Drive',folder.name,10,ids,speed,a))
    (OUT/'quality_observations.json').write_text(json.dumps(dict(teleop=qc,person2drive=person_qc,pdd=pdd_qc),indent=2)+'\n')
    write('pdd_native_controls.csv',native_rows)
    return episodes


def max_run(mask,ids):
    longest=length=0
    for i,active in enumerate(mask):
        length=(length+1 if i and ids[i]==ids[i-1]+1 else 1) if active else 0
        longest=max(longest,length)
    return longest


def summarize(episodes):
    n=sum(len(e['ids']) for e in episodes);fps=episodes[0]['fps']
    speed=np.concatenate([e['speed'] for e in episodes]);a=np.concatenate([e['action'] for e in episodes])
    moving=speed>=2;coast=moving&(a[:,1]<=.01)&(a[:,2]<=.01);mid=(a[:,2]>.01)&(a[:,2]<.99)
    overlap=(a[:,1]>.01)&(a[:,2]>.01)
    sequences=Counter();full_windows=0
    for e in episodes:
        steps=int(e['fps']);ids=e['ids'];action=e['action']
        for i in range(len(ids)-steps):
            if not np.all(np.diff(ids[i:i+steps+1])==1):continue
            full_windows+=1
            # Common four control snapshots per second, using causal last-known command.
            idx=i+np.floor(np.arange(4)*e['fps']/4).astype(int)
            pattern=tuple(np.rint(action[idx]*10).astype(int).ravel())
            sequences[pattern]+=1
    frequencies=np.array(list(sequences.values()),float)/max(1,full_windows)
    longest_coast=max(max_run((e['speed']>=2)&(e['action'][:,1]<=.01)&(e['action'][:,2]<=.01),e['ids']) for e in episodes)
    longest_brake=max(max_run((e['action'][:,2]>.01)&(e['action'][:,2]<.99),e['ids']) for e in episodes)
    return dict(dataset=episodes[0]['dataset'],episodes=len(episodes),frames=n,fps=fps,
                valid_transitions=sum(int((np.diff(e['ids'])==1).sum()) for e in episodes),
                observed_endpoint_duration_seconds=sum(float(np.sum(np.diff(e['ids'])==1)/e['fps']) for e in episodes),
                moving_frames=int(moving.sum()),moving_coast_frames=int(coast.sum()),moving_coast_fraction=float(coast.sum()/max(1,moving.sum())),
                intermediate_brake_frames=int(mid.sum()),intermediate_brake_fraction=float(mid.mean()),
                moving_intermediate_brake_frames=int((mid&moving).sum()),moving_intermediate_brake_fraction=float((mid&moving).sum()/max(1,moving.sum())),
                overlap_frames=int(overlap.sum()),overlap_fraction=float(overlap.mean()),
                intermediate_throttle_fraction=float(((a[:,1]>.01)&(a[:,1]<.99)).mean()),
                max_coast_frames=longest_coast,max_coast_sampled_seconds=longest_coast/fps,
                max_coast_endpoint_span_seconds=max(0,longest_coast-1)/fps,
                max_intermediate_brake_frames=longest_brake,max_intermediate_brake_sampled_seconds=longest_brake/fps,
                one_second_windows=full_windows,unique_four_snapshot_patterns=len(sequences),
                pattern_entropy_bits=float(-np.sum(frequencies*np.log2(frequencies))) if len(frequencies) else 0.,
                dominant_pattern_fraction=float(max(frequencies)) if len(frequencies) else None)


def inspect_images():
    evidence={}
    for name,folder in [('TeleopWM','TeleopWM-Dataset'),('PDD','PDD')]:
        root=HERE/'samples'/folder;images=sorted(list(root.rglob('*.jpg'))+list(root.rglob('*.png')))
        sizes=Counter()
        for path in images:
            with Image.open(path) as im:im.load();sizes[str(im.size)]+=1
        evidence[name]=dict(downloaded_decoded_images=len(images),image_sizes=dict(sizes))
    import zstandard
    archive=HERE/'samples/Person2Drive/sensor_data/driver01/Town04_drive_1.tar.zst'
    selected=HERE/'samples/Person2Drive/front_sample';selected.mkdir(exist_ok=True)
    info=load_numeric_pickle(HERE/'samples/Person2Drive/sensor_data/driver01/b2d_infos_train.pkl')
    refs=sorted([r for r in info if r['folder']=='Town04_drive_1'],key=lambda r:r['frame_idx'])
    start=next(i for i in range(len(refs)-13) if np.all(np.diff([r['frame_idx'] for r in refs[i:i+14]])==1))
    window=refs[start:start+14];wanted={r['frame_idx'] for r in window}
    for old in selected.glob('*.jpg'):old.unlink()  # Only this audit's generated image excerpts.
    names=[];first_bytes=[];has_anno=False;camera_views=set()
    with archive.open('rb') as raw,zstandard.ZstdDecompressor().stream_reader(raw) as stream,tarfile.open(fileobj=stream,mode='r|') as tar:
        for member in tar:
            has_anno=has_anno or '/anno/' in member.name
            if '/camera/' in member.name:camera_views.add(member.name.split('/camera/')[1].split('/')[0])
            if member.isfile() and '/camera/rgb_front/' in member.name and member.name.endswith('.jpg'):
                frame=int(Path(member.name).stem);names.append(frame)
                if frame in wanted:
                    # Never use archive paths as extraction destinations.
                    payload=tar.extractfile(member).read();(selected/f'{frame:05}.jpg').write_bytes(payload);first_bytes.append(frame)
    for path in selected.glob('*.jpg'):
        with Image.open(path) as im:im.load()
    info=load_numeric_pickle(HERE/'samples/Person2Drive/sensor_data/driver01/b2d_infos_train.pkl')
    refs=[r for r in info if r['folder']=='Town04_drive_1'];ids={r['frame_idx'] for r in refs}
    evidence['Person2Drive']=dict(archive_front_frames=len(names),metadata_frames=len(refs),
                                    missing_image_references=sorted(ids-set(names)),
                                    decoded_front_frames=sorted(first_bytes),archive_has_anno=has_anno,camera_views=sorted(camera_views),
                                    metadata_first_frame=min(ids),raw_motion_records=len(json.loads((HERE/'samples/Person2Drive/driving_logs/driver01/Town04_drive_1/metric_info.json').read_text())))
    if set(first_bytes)!=wanted:raise ValueError('Missing image in contiguous Person2Drive 4+10 example')
    origin=window[3]['world2ego']
    target=[]
    for record in window[4:]:
        pose=np.linalg.inv(record['world2ego'])
        local=origin@pose[:,3]
        forward=origin[:3,:3]@pose[:3,0]
        # CAM_FRONT camera-right maps to ego -y in this right-handed metadata.
        dyaw=np.arctan2(-forward[1],forward[0])
        target.append([float(record['ego_vel'][0]),float(local[0]),float(-local[1]),float(np.sin(dyaw)),float(np.cos(dyaw))])
    if not np.isfinite(target).all():raise ValueError('Nonfinite motion-target preview')
    (OUT/'person2drive_target_preview.json').write_text(json.dumps(dict(
        driver='driver01',episode='Town04_drive_1',frame_ids=sorted(wanted),history=4,horizon=10,
        targets=target,columns=['speed','forward','right','sin_relative_right_yaw','cos_relative_right_yaw'],
        coordinate_evidence='CAM_FRONT cam2ego[:,0]=[0,-1,0]; negate local y for right-positive convention',
        pose_source='inverse world2ego; do not use flattened ego_translation z',
        action_alignment='No timing proof from this preview; retain configurable alignment'),indent=2)+'\n')
    (OUT/'image_compatibility.json').write_text(json.dumps(evidence,indent=2)+'\n')



def plot_pdd_indexing():
    folder=HERE/'samples/PDD/driver_01/data/Accident/metric'
    states=json.loads((folder/'metric_info.json').read_text())
    records=json.loads((folder/'metrics.json').read_text())['records']
    keys=sorted(map(int,states));controls={r['control']['step']:r['control'] for r in records}
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for ax,channel in zip(axes,['throttle','brake']):
        ax.plot(np.array(keys)/20,[controls[k][channel] for k in keys],'-o',markersize=3,label='Join on exact telemetry/control step')
        ax.plot(np.array(keys)/20,[records[i]['control'][channel] for i in range(len(keys))],'--',label='Public loader ordinal control index')
        ax.set_ylabel(channel.capitalize());ax.grid(alpha=.3);ax.legend(fontsize=8)
    axes[-1].set_xlabel('Simulation step / 20 (s; 4 Hz saved-state cadence)')
    fig.tight_layout();fig.savefig(OUT/'pdd_indexing.png',dpi=160);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True);episodes=load_episodes();groups=defaultdict(list)
    for e in episodes:groups[e['dataset']].append(e)
    rows=[summarize(group) for group in groups.values()];write('summary.csv',rows)
    write('episode_metrics.csv',[summarize([e])|{'episode':e['name']} for e in episodes])
    conditioned=[]
    for name,group in groups.items():
        v=np.concatenate([e['speed'] for e in group]);a=np.concatenate([e['action'] for e in group])
        for low,high,label in [(-np.inf,2,'below_2'),(2,5,'2_to_5'),(5,10,'5_to_10'),(10,np.inf,'above_10')]:
            mask=(v>=low)&(v<high);part=a[mask]
            conditioned.append(dict(dataset=name,speed_bin_mps=label,frames=int(mask.sum()),
                                    mean_steer=float(part[:,0].mean()) if len(part) else None,
                                    mean_throttle=float(part[:,1].mean()) if len(part) else None,
                                    mean_brake=float(part[:,2].mean()) if len(part) else None,
                                    coasting_fraction=float(((part[:,1]<=.01)&(part[:,2]<=.01)).mean()) if len(part) else None,
                                    intermediate_braking_fraction=float(((part[:,2]>.01)&(part[:,2]<.99)).mean()) if len(part) else None))
    write('speed_conditioned.csv',conditioned)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for name,group in groups.items():
        a=np.concatenate([e['action'] for e in group])
        for i,(ax,label) in enumerate(zip(axes,['Steering','Throttle','Brake'])):
            bins=np.linspace(-1,1,41) if i==0 else np.linspace(0,1,41)
            ax.hist(a[:,i],bins=bins,histtype='step',density=True,label=name)
            ax.set(title=label,xlabel='Raw normalized control',ylabel='Density')
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'control_distributions.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,field,label in zip(axes,['moving_coast_fraction','intermediate_brake_fraction','max_coast_sampled_seconds'],['Moving coasting (%)','Intermediate brake (%)','Longest sampled coast (s)']):
        values=[r[field]*(100 if 'fraction' in field else 1) for r in rows]
        ax.bar([r['dataset'] for r in rows],values);ax.set_ylabel(label);ax.tick_params(axis='x',rotation=25)
    fig.tight_layout();fig.savefig(OUT/'coverage_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in groups:
        part=[r for r in conditioned if r['dataset']==name]
        axes[0].plot([r['speed_bin_mps'] for r in part],[100*r['coasting_fraction'] if r['coasting_fraction'] is not None else np.nan for r in part],'o-',label=name)
        axes[1].plot([r['speed_bin_mps'] for r in part],[100*r['intermediate_braking_fraction'] if r['intermediate_braking_fraction'] is not None else np.nan for r in part],'o-',label=name)
    axes[0].set_ylabel('Coast (%) within speed bin');axes[1].set_ylabel('Intermediate brake (%) within speed bin')
    for ax in axes:ax.set_xlabel('Speed bin (m/s)');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'speed_conditioned.png',dpi=160);plt.close(fig)
    inspect_images()
    plot_pdd_indexing()
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
