"""Read-only Bench2Drive Mini audit; outputs only inside this experiment."""
from pathlib import Path
import csv
import json
import os
import sys
from collections import Counter

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, str(PROJECT))
from src.bench2drive_dataset import Bench2DriveDataset

OUT = HERE / 'results'
DT = .1


def write(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def runs(mask, ids):
    segments, start = [], None
    for i, active in enumerate(mask):
        if start is not None and (not active or ids[i] != ids[i - 1] + 1):
            segments.append((start, i - 1))
            start = None
        if active and start is None:
            start = i
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and np.std(a) > 1e-10 and np.std(b) > 1e-10 else None


def nearest(record):
    ego = record['bounding_boxes'][0]
    matrix = np.array(ego['world2ego'])
    choices = []
    for actor in record['bounding_boxes'][1:]:
        # Physical actor boxes only: signs/lights/road triggers do not prove contact.
        if actor.get('class') not in ['vehicle', 'walker', 'pedestrian']:
            continue
        points = np.array(actor.get('world_cord', []))
        if points.ndim != 2 or points.shape[1] != 3:
            continue
        local = (matrix @ np.c_[points, np.ones(len(points))].T).T[:, :3]
        lo, hi = local.min(0), local.max(0)
        ego_lo = np.array(ego['bbx_loc']) - np.array(ego['extent'])
        ego_hi = np.array(ego['bbx_loc']) + np.array(ego['extent'])
        sep = np.maximum(np.maximum(lo - ego_hi, ego_lo - hi), 0)
        # Lower-bound box gap, not exact oriented collision detection.
        gap = float(np.linalg.norm(sep))
        choices.append((gap, str(actor.get('id')), actor.get('type_id', actor['class'])))
    return min(choices) if choices else (None, None, None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scenarios = {}
    integrity = []
    raw = PROJECT / 'data/bench2drive_mini'
    for folder in sorted(raw.iterdir()):
        paths = sorted((folder / 'anno').glob('*.json.gz'))
        if not paths:
            continue
        ids = np.array([int(p.name.split('.')[0]) for p in paths])
        records = [Bench2DriveDataset._load_annotation(None, p) for p in paths]
        image_ids = {int(p.stem) for p in (folder / 'camera/rgb_front').iterdir()
                     if p.stem.isdigit() and p.suffix.lower() in ['.jpg', '.jpeg', '.png']}
        a = np.array([[r['steer'], r['throttle'], r['brake']] for r in records], dtype=float)
        v = np.array([r['speed'] for r in records])
        xyz = np.array([r['bounding_boxes'][0]['location'] for r in records])
        gps = np.array([[r['x'], r['y']] for r in records])
        yaw = np.deg2rad([r['bounding_boxes'][0]['rotation'][2] for r in records])
        theta = np.array([r['theta'] for r in records])
        imu = np.array([r['acceleration'] for r in records])
        scenarios[folder.name] = dict(ids=ids, records=records, a=a, v=v, xyz=xyz,
                                     gps=gps, yaw=yaw, theta=theta, imu=imu)
        valid = np.diff(ids) == 1
        from PIL import Image
        corrupt_images=[]
        for image in (folder/'camera/rgb_front').glob('*.jpg'):
            try:
                with Image.open(image) as img:img.verify()
            except Exception as exc:corrupt_images.append(str(image.name)+': '+str(exc))
        integrity.append(dict(corrupt_images=len(corrupt_images),scenario=folder.name, annotations=len(ids), images=len(image_ids),
                              missing_images=len(set(ids) - image_ids), missing_annotations=len(image_ids - set(ids)),
                              gaps=int((~valid).sum()), nonfinite=int(sum((~np.isfinite(x)).sum() for x in [a,v,xyz,gps,theta,imu])),
                              out_of_range_controls=int(((abs(a[:,0])>1)|(a[:,1]<0)|(a[:,1]>1)|(a[:,2]<0)|(a[:,2]>1)).sum())))
    assert len(scenarios) == 10
    write('integrity.csv', integrity)
    all_a = np.concatenate([d['a'] for d in scenarios.values()])
    all_v = np.concatenate([d['v'] for d in scenarios.values()])
    signed = all_a[:,1] - all_a[:,2]
    reconstructed_brake = (signed < 0).astype(float)
    representation = dict(brake_values=np.unique(all_a[:,2]).tolist(),
        throttle_when_braking_max=float(all_a[all_a[:,2] == 1,1].max()),
        max_brake_reconstruction_error=float(np.max(abs(reconstructed_brake-all_a[:,2]))),
        max_throttle_reconstruction_error=float(np.max(abs(signed+reconstructed_brake-all_a[:,1]))),
        conclusion='Signed encoding is empirically invertible on Mini; not generally invertible for arbitrary continuous controls.')
    (OUT/'action_representation.json').write_text(json.dumps(representation,indent=2)+'\n')
    overlap, coverage, sequences, coords, events, lags, triggered = [], [], [], [], [], [], []
    event_vectors = {'brake_rise': [], 'longitudinal_drop': []}
    threshold_grid = [.01, .05, .1]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    both = (all_a[:,1] > .01) & (all_a[:,2] > .01)
    axes[0].scatter(all_a[~both,1],all_a[~both,2],s=8,alpha=.15,label='Other')
    axes[0].scatter(all_a[both,1],all_a[both,2],s=14,alpha=.5,label='Overlap >0.01')
    axes[0].set(xlabel='Throttle',ylabel='Brake');axes[0].legend(fontsize=8)
    axes[1].hist([all_a[both,1],all_a[both,2]],bins=np.linspace(0,1,21),label=['Throttle','Brake'])
    axes[1].set(xlabel='Overlapping command magnitude',ylabel='Frames');axes[1].legend()
    axes[2].hist(np.minimum(all_a[both,1],all_a[both,2]),bins=20)
    axes[2].set(xlabel='min(throttle, brake): common component\n(recoverable from signed control on Mini)',ylabel='Frames')
    fig.tight_layout();fig.savefig(OUT/'control_overlap.png',dpi=160);plt.close(fig)
    for name,d in scenarios.items():
        ids,a,v=d['ids'],d['a'],d['v'];r=d['records']
        dv=np.diff(v);acc=dv/DT;valid=np.diff(ids)==1
        yaw,theta=d['yaw'],d['theta']
        delta_gps=np.diff(d['gps'],axis=0);delta_world=np.diff(d['xyz'][:,:2],axis=0)
        moving=v[:-1]>3
        transform_error = d['gps'] - d['xyz'][:,:2]
        mount_position = d['xyz'][:,:2] - 1.4*np.c_[np.cos(yaw),np.sin(yaw)]
        mount_residual = d['gps'] - mount_position
        c,s=np.cos(theta[:-1]),np.sin(theta[:-1])
        current_local=np.c_[c*delta_gps[:,0]+s*delta_gps[:,1],-s*delta_gps[:,0]+c*delta_gps[:,1]]
        c,s=np.cos(yaw[:-1]),np.sin(yaw[:-1])
        true_local=np.c_[c*delta_world[:,0]+s*delta_world[:,1],-s*delta_world[:,0]+c*delta_world[:,1]]
        coords.append(dict(scenario=name,compass_minus_yaw_median_deg=float(np.rad2deg(np.median(np.angle(np.exp(1j*(theta-yaw)))))),
                           gps_offset_x=float(np.mean(transform_error[:,0])),gps_offset_y=float(np.mean(transform_error[:,1])),
                           gps_residual_x_std=float(np.std(transform_error[:,0])),gps_residual_y_std=float(np.std(transform_error[:,1])),
                           mount_corrected_residual_x_std=float(np.std(mount_residual[:,0])),
                           mount_corrected_residual_y_std=float(np.std(mount_residual[:,1])),
                           current_local_forward_step_median=float(np.median(current_local[moving,0])),
                           current_local_lateral_step_median=float(np.median(current_local[moving,1])),
                           world_ego_forward_step_median=float(np.median(true_local[moving,0])),
                           world_ego_lateral_step_median=float(np.median(true_local[moving,1])),
                           negative_speed_frames=int((v<0).sum()),minimum_speed=float(v.min())))
        for threshold in threshold_grid:
            mask=(a[:,1]>threshold)&(a[:,2]>threshold)
            rr=runs(mask,ids)
            overlap.append(dict(scenario=name,threshold=threshold,frames=int(mask.sum()),fraction=float(mask.mean()),
                                moving_frames=int((mask&(v>2)).sum()),stopped_frames=int((mask&(abs(v)<.1)).sum()),
                                sequences=len(rr),max_frames=max([j-i+1 for i,j in rr],default=0),
                                throttle_median=float(np.median(a[mask,1])) if mask.any() else None,
                                brake_median=float(np.median(a[mask,2])) if mask.any() else None,
                                discarded_common_mean=float(np.minimum(a[mask,1],a[mask,2]).mean()) if mask.any() else None))
            for i,j in rr:
                sequences.append(dict(kind='overlap',scenario=name,threshold=threshold,speed_threshold='',
                                      start_frame=int(ids[i]),end_frame=int(ids[j]),frames=j-i+1,
                                      sampled_duration_seconds=(j-i+1)*DT,endpoint_span_seconds=(j-i)*DT,
                                      speed_min=float(v[i:j+1].min()),speed_max=float(v[i:j+1].max())))
            for speed_threshold in [.5,2.]:
                moving_mask=v>=speed_threshold
                coast=(a[:,1]<=threshold)&(a[:,2]<=threshold)&moving_mask
                rr=runs(coast,ids)
                coverage.append(dict(scenario=name,threshold=threshold,speed_threshold=speed_threshold,moving_frames=int(moving_mask.sum()),
                                     coasting_frames=int(coast.sum()),coasting_fraction=float(coast.sum()/moving_mask.sum()) if moving_mask.any() else None,
                                     sequences=len(rr),max_frames=max([j-i+1 for i,j in rr],default=0),
                                     coasting_speed_median=float(np.median(v[coast])) if coast.any() else None,
                                     intermediate_throttle=int(((a[:,1]>.1)&(a[:,1]<.9)).sum()),
                                     intermediate_brake=int(((a[:,2]>.1)&(a[:,2]<.9)).sum())))
                for i,j in rr:
                    sequences.append(dict(kind='coast',scenario=name,threshold=threshold,speed_threshold=speed_threshold,
                                          start_frame=int(ids[i]),end_frame=int(ids[j]),frames=j-i+1,
                                          sampled_duration_seconds=(j-i+1)*DT,endpoint_span_seconds=(j-i)*DT,
                                          speed_min=float(v[i:j+1].min()),speed_max=float(v[i:j+1].max())))
        for i in np.flatnonzero(valid & (acc <= -5)):
            near_before,near_after=nearest(r[i]),nearest(r[i+1])
            events.append(dict(scenario=name,frame_t=int(ids[i]),frame_next=int(ids[i+1]),index=int(i),
                               speed_t=float(v[i]),speed_next=float(v[i+1]),fd_acceleration=float(acc[i]),
                               steer_t=float(a[i,0]),throttle_t=float(a[i,1]),brake_t=float(a[i,2]),
                               steer_next=float(a[i+1,0]),throttle_next=float(a[i+1,1]),brake_next=float(a[i+1,2]),
                               imu_x_t=float(d['imu'][i,0]),imu_x_next=float(d['imu'][i+1,0]),
                               imu_y_next=float(d['imu'][i+1,1]),imu_z_next=float(d['imu'][i+1,2]),
                               ego_xyz_t=json.dumps(d['xyz'][i].tolist()),ego_xyz_next=json.dumps(d['xyz'][i+1].tolist()),
                               world_yaw_t=float(np.rad2deg(yaw[i])),world_yaw_next=float(np.rad2deg(yaw[i+1])),
                               compass_t=float(theta[i]),displacement=float(np.linalg.norm(delta_world[i])),
                               nearest_box_gap_lower_bound_t=near_before[0],nearest_actor_id_t=near_before[1],nearest_actor_type_t=near_before[2],
                               nearest_box_gap_lower_bound_next=near_after[0],nearest_actor_id_next=near_after[1],nearest_actor_type_next=near_after[2],
                               collision_metadata=','.join(k for k in r[i] if 'collis' in k.lower()) or 'absent',
                               interpretation='strong recorded deceleration; inspect IMU/control/context; collision unconfirmed'))
        u=a[:,1]-a[:,2]
        for lag in range(-5,6):
            # Correlate dv_i with recorded command at i+lag, without crossing scenarios.
            ii=np.arange(len(dv));jj=ii+lag
            mask=(jj>=0)&(jj<len(v))&valid&(v[:-1]>2)
            ii,jj=ii[mask],jj[mask]
            lags.append(dict(scenario=name,lag=lag,n=len(ii),signed_control_dv_correlation=corr(u[jj],dv[ii]),
                             brake_deceleration_correlation=corr(a[jj,2],-acc[ii]),
                             imu_x_fd_correlation=corr(d['imu'][jj,0],acc[ii])))
        for kind,indices in [('brake_rise',np.flatnonzero(np.diff(a[:,2])>.5)+1),
                             ('longitudinal_drop',np.flatnonzero(np.diff(u)<-1)+1)]:
            for j in indices:
                if j>=5 and j+5<len(dv) and v[j]>2 and np.all(np.diff(ids[j-5:j+7])==1):
                    event_vectors[kind].append(acc[j-5:j+6])
    events.sort(key=lambda e:e['fd_acceleration'])
    # Evidence labels are descriptive, not causal collision declarations.
    for event in events:
        d=scenarios[event['scenario']];i=event['index']
        near_imu=d['imu'][max(0,i-1):min(len(d['v']),i+3),0]
        near_brake=d['a'][max(0,i-2):min(len(d['v']),i+2),2]
        event['imu_corroboration_nearby']=bool(near_imu.min() <= min(-3,event['fd_acceleration']*.5))
        event['strong_brake_nearby']=bool(near_brake.max() >= .5)
        event['classification']=('ordinary_or_moderate_braking_candidate' if event['fd_acceleration'] >= -8 and event['strong_brake_nearby']
                                 else 'simulator_emergency_braking_candidate' if event['strong_brake_nearby'] and event['imu_corroboration_nearby']
                                 else 'unresolved_external_interaction_or_timing')
    write('overlap.csv',overlap);write('coverage.csv',coverage);write('sequences.csv',sequences)
    write('coordinates.csv',coords);write('extreme_events.csv',events);write('lag_correlations.csv',lags)
    for kind, vectors in event_vectors.items():
        if vectors:
            arr=np.array(vectors)
            for j,offset in enumerate(range(-5,6)):
                triggered.append(dict(trigger=kind,offset_steps=offset,n=len(arr),mean_fd_acceleration=float(arr[:,j].mean()),std=float(arr[:,j].std())))
    write('event_triggered.csv',triggered)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for col,title in [(1,'Throttle'),(2,'Brake')]:
        axes[0,0].hist(all_a[:,col],bins=np.linspace(0,1,21),histtype='step',label=title)
    axes[0,0].legend();axes[0,0].set(xlabel='Recorded command',ylabel='Frames')
    coast=(all_a[:,1]<=.01)&(all_a[:,2]<=.01)&(all_v>=2)
    axes[0,1].hist(all_v[coast],bins=12);axes[0,1].set(xlabel='Moving coasting speed (m/s)',ylabel='Frames')
    selected=[r for r in coverage if r['threshold']==.01 and r['speed_threshold']==2]
    axes[1,0].bar([r['scenario'].split('_Town')[0] for r in selected],[r['coasting_frames'] for r in selected])
    axes[1,0].tick_params(axis='x',rotation=60,labelsize=8);axes[1,0].set_ylabel('Moving coasting frames')
    seq=[r['frames']*DT for r in sequences if r['kind']=='coast' and r['threshold']==.01 and r['speed_threshold']==2]
    axes[1,1].hist(seq,bins=np.arange(.05,max(seq,default=.1)+.16,.1));axes[1,1].set(xlabel='Coasting sampled duration (s)',ylabel='Runs')
    fig.tight_layout();fig.savefig(OUT/'action_coverage.png',dpi=160);plt.close(fig)
    # One largest decrement from each of three different scenarios, no cherry-picking by model error.
    examples=[]
    for event in events:
        if event['scenario'] not in [e['scenario'] for e in examples]:examples.append(event)
        if len(examples)==3:break
    fig,axes=plt.subplots(3,3,figsize=(15,10))
    for col,e in enumerate(examples):
        d=scenarios[e['scenario']];i=e['index'];lo=max(0,i-10);hi=min(len(d['v']),i+12)
        x=(d['ids'][lo:hi]-d['ids'][i])*DT
        axes[0,col].plot(x,d['v'][lo:hi]);axes[0,col].set_title(e['scenario'].split('_Town')[0]);axes[0,col].set_ylabel('Speed (m/s)')
        axes[1,col].plot(x[:-1],np.diff(d['v'][lo:hi])/DT,label='FD: interval starting at x')
        axes[1,col].plot(x,d['imu'][lo:hi,0],label='Recorded IMU x',alpha=.8)
        axes[1,col].set_ylabel('Acceleration (m/s²)');axes[1,col].legend(fontsize=7)
        for j,label in [(0,'Steer'),(1,'Throttle'),(2,'Brake')]:axes[2,col].step(x,d['a'][lo:hi,j],where='post',label=label)
        axes[2,col].legend(fontsize=7);axes[2,col].set(xlabel='Seconds relative to selected transition start',ylabel='Control')
        for ax in axes[:,col]:ax.axvspan(0,.1,alpha=.15,color='red');ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/'extreme_deceleration.png',dpi=160);plt.close(fig)
    # Front RGB context at the next frame, where each selected decrement ends.
    from PIL import Image
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,e in zip(axes,examples):
        path=raw/e['scenario']/'camera/rgb_front'/f"{e['frame_next']:05}.jpg"
        with Image.open(path) as img:ax.imshow(img)
        ax.set_title(f"{e['scenario'].split('_Town')[0]} frame {e['frame_next']}");ax.axis('off')
    fig.tight_layout();fig.savefig(OUT/'event_context.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name in scenarios:
        rr=[r for r in lags if r['scenario']==name]
        axes[0].plot([r['lag'] for r in rr],[r['signed_control_dv_correlation'] for r in rr],alpha=.6,label=name.split('_Town')[0])
    axes[0].axvline(0,color='k',ls='--',label='H0');axes[0].axvline(1,color='k',ls=':',label='H1')
    axes[0].set(xlabel='Command index lag relative to transition start (steps)',ylabel='corr(throttle − brake, speed change)')
    for kind in event_vectors:
        rr=[r for r in triggered if r['trigger']==kind]
        if rr:axes[1].plot([r['offset_steps']*DT for r in rr],[r['mean_fd_acceleration'] for r in rr],label=f"{kind} (n={rr[0]['n']})")
    axes[1].axvline(0,color='k',ls='--');axes[1].set(xlabel='Transition-start time relative to command switch (s)',ylabel='Mean finite-difference acceleration (m/s²)');axes[1].legend(fontsize=8)
    axes[0].legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(OUT/'alignment_analysis.png',dpi=160);plt.close(fig)
    summary=[]
    def add(section,metric,value):summary.append(dict(section=section,metric=metric,value=value))
    add('integrity','scenarios',len(scenarios));add('integrity','annotations',len(all_v))
    for threshold in threshold_grid:
        rr=[r for r in overlap if r['threshold']==threshold]
        add('overlap',f'frames_threshold_{threshold}',sum(r['frames'] for r in rr))
        add('overlap',f'max_run_frames_threshold_{threshold}',max(r['max_frames'] for r in rr))
        rr=[r for r in coverage if r['threshold']==threshold and r['speed_threshold']==2]
        add('coast',f'frames_threshold_{threshold}_speed_ge_2',sum(r['coasting_frames'] for r in rr))
        add('coast',f'max_run_frames_threshold_{threshold}_speed_ge_2',max(r['max_frames'] for r in rr))
    add('coast','moving_frames_speed_ge_2',int((all_v>=2).sum()))
    add('coast','speed_median_coast_0.01_ge_2',float(np.median(all_v[(all_v>=2)&(all_a[:,1]<=.01)&(all_a[:,2]<=.01)])))
    add('actions','intermediate_throttle_0.1_0.9',int(((all_a[:,1]>.1)&(all_a[:,1]<.9)).sum()))
    add('actions','intermediate_brake_0.1_0.9',int(((all_a[:,2]>.1)&(all_a[:,2]<.9)).sum()))
    add('overlap','moving_above_2_frames',int((both&(all_v>2)).sum()));add('overlap','stopped_abs_speed_below_0.1_frames',int((both&(abs(all_v)<.1)).sum()))
    add('events','deceleration_ge_5_mps2',len(events));add('events','maximum_deceleration_magnitude',-events[0]['fd_acceleration'])
    write('summary.csv',summary)
    # Compact local-doc evidence: no additional dataset downloads.
    docs=PROJECT/'external/Bench2Drive'
    evidence={}
    for rel in ['README.md','docs/v004_update.md','docs/anno.md']:
        text=(docs/rel).read_text()
        evidence[rel]=[dict(line=i+1,text=line) for i,line in enumerate(text.splitlines()) if any(s in line.lower() for s in ['1,100','1100','13638','13,638','44 scenarios','coordinate','carla api'])]
    metadata=[]
    for path in raw.rglob('*'):
        if path.is_file() and any(s in path.name.lower() for s in ['collision','result','measurement','metadata']):metadata.append(str(path.relative_to(raw)))
    (OUT/'metadata_evidence.json').write_text(json.dumps(dict(local_docs=evidence,metadata_files=metadata,annotation_keys=sorted(scenarios[next(iter(scenarios))]['records'][0]),actor_classes=dict(Counter(a['class'] for d in scenarios.values() for r in d['records'] for a in r['bounding_boxes']))),indent=2)+'\n')
    previous=[]
    for experiment in ['action_alignment','dynamics_validation']:
        with (PROJECT/'experiments'/experiment/'results/summary.csv').open() as f:
            for row in csv.DictReader(f):
                if row['scenario']=='ALL' and (experiment=='action_alignment' or (row['subset']=='all' and row['step'] in ['0','1','5','10'])):
                    previous.append(dict(experiment=experiment,model=row['model'],step=row.get('step','one_step'),
                                         mae_mean=row['mae_mean'],rmse_mean=row['rmse_mean'],
                                         mae_std=row['mae_std'],rmse_std=row['rmse_std']))
    write('previous_results.csv',previous)
    # Cross-check independent aggregations before declaring successful completion.
    assert sum(r['annotations'] for r in integrity)==len(all_v)
    for threshold in threshold_grid:
        mask=(all_a[:,1]>threshold)&(all_a[:,2]>threshold)
        assert mask.sum()==sum(r['frames'] for r in overlap if r['threshold']==threshold)
        assert mask.sum()==sum(r['frames'] for r in sequences if r['kind']=='overlap' and r['threshold']==threshold)
    assert all(e['frame_next']==e['frame_t']+1 and e['fd_acceleration']<=-5 for e in events)
    import hashlib
    protected=[PROJECT/'src/bench2drive_dataset.py',PROJECT/'experiments/action_alignment/run.py',PROJECT/'experiments/dynamics_validation/run.py']
    (OUT/'source_manifest.json').write_text(json.dumps({str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected},indent=2)+'\n')
    print(json.dumps(summary,indent=2));print('Top events:')
    for e in events[:10]:print(e['scenario'],e['frame_t'],round(e['fd_acceleration'],2),'IMU next',round(e['imu_x_next'],2),'box gap',e['nearest_box_gap_lower_bound_next'])
    print('Results:',OUT)


if __name__=='__main__':main()
