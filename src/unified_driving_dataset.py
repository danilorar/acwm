"""Source-aware 10 Hz RGB/ego training dataset for Bench2Drive and Person2Drive."""
from pathlib import Path
import json
import gzip
import hashlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT=Path(__file__).resolve().parents[1]


def load_episode_records(project=PROJECT, *, require_images=True):
    """Return numerical episode records; expensive JSON parsing happens once, not per sample.

    Matrices come from simulator poses, only used for supervision. Person2Drive's
    camera calibration defines ego +y left; Bench2Drive's actor +y is right.
    """
    project=Path(project);records={}
    bench=project/'data/bench2drive/extracted'
    for folder in sorted(bench.iterdir()):
        paths=sorted((folder/'anno').glob('*.json.gz'))
        if not paths:continue
        rows=[]
        for path in paths:
            with gzip.open(path,'rt') as f:rows.append(json.load(f))
        ids=np.array([int(p.name.split('.')[0]) for p in paths])
        matrices=[]
        for row in rows:
            ego=[b for b in row['bounding_boxes'] if b.get('class')=='ego_vehicle']
            if len(ego)!=1:raise ValueError('Expected unique simulator ego pose')
            matrices.append(ego[0]['world2ego'])
        records['bench2drive/'+folder.name]=dict(source='bench2drive',episode=folder.name,driver='',
            route=folder.name.split('_Weather')[0].split('_',1)[-1],
            frame_ids=ids,speed=np.array([r['speed'] for r in rows],np.float32),
            actions=np.array([[r[c] for c in ['steer','throttle','brake']] for r in rows],np.float32),
            world2body=np.array(matrices,np.float64),right_sign=1,
            image_paths=[folder/'camera/rgb_front'/f'{i:05}.jpg' for i in ids])
    person=project/'data/person2drive'
    selection=json.loads((project/'data/config/person_selection.json').read_text())
    for entry in selection['selected']:
        with np.load(project/entry['index'],allow_pickle=False) as data:
            arrays={k:data[k].copy() for k in data.files}
        driver,episode=entry['driver'],entry['episode']
        if len(arrays['image_paths'])!=len(arrays['frame_ids']):raise ValueError('Mismatched image reference count')
        paths=[]
        for frame_id,value in zip(arrays['frame_ids'],arrays['image_paths']):
            relative=Path(str(value))
            if relative.is_absolute() or relative.parts!=(episode,'camera','rgb_front',relative.name) or relative.suffix!='.jpg' or not relative.stem.isdigit() or int(relative.stem)!=int(frame_id):
                raise ValueError(f'Unsafe or mismatched image/frame reference: {driver}/{value}')
            paths.append(person/'extracted'/driver/relative)
        records[f'person2drive/{driver}/{episode}']=dict(source='person2drive',episode=episode,driver=driver,
            route=entry['route'],frame_ids=arrays['frame_ids'],speed=arrays['speed'],actions=arrays['actions'],
            world2body=arrays['world2ego'],right_sign=-1,image_paths=paths)
    for key,r in records.items():
        ids=r['frame_ids'];n=len(ids)
        if r['speed'].shape!=(n,) or r['actions'].shape!=(n,3) or r['world2body'].shape!=(n,4,4) or len(r['image_paths'])!=n:raise ValueError(f'Invalid episode array shapes: {key}')
        if len(ids)!=len(set(ids)) or not np.all(np.diff(ids)>0):raise ValueError(f'Invalid frame IDs: {key}')
        if not all(np.isfinite(r[c]).all() for c in ['speed','actions','world2body']):raise ValueError(f'Nonfinite data: {key}')
        if require_images:
            missing=[str(p) for p in r['image_paths'] if not p.is_file()]
            if missing:raise FileNotFoundError(f'{key}: {len(missing)} missing referenced images; first {missing[0]}')
        poses=np.linalg.inv(r['world2body'])
        r['positions']=poses[:,:3,3]
        r['forward']=poses[:,:3,0]
        r['fps']=10.
    return records


class UnifiedDrivingDataset(Dataset):
    """Planar anchor-yaw targets with source-calibrated right-positive axes.

    Explicit action_offsets is mandatory: neither source's recording-time
    convention is proved. Offsets 0/1 are supervised label hypotheses; deployment
    must provide its own known command history and planned outgoing commands.
    """
    def __init__(self,records,*,action_offsets,episode_ids=None,history=4,horizon=10,image_size=(224,224),exclude_uncertain_controls=True):
        if any(isinstance(v,bool) or not isinstance(v,int) or v<1 for v in [history,horizon]):raise ValueError('Positive integer history/horizon required')
        if len(image_size)!=2 or any(not isinstance(v,int) or v<1 for v in image_size):raise ValueError('image_size=(width,height) must be positive')
        selected=list(records) if episode_ids is None else list(episode_ids)
        if len(selected)!=len(set(selected)) or not set(selected)<=set(records):raise ValueError('Unknown/duplicate episode selection')
        self.records={k:records[k] for k in selected}
        self.action_offsets=dict(action_offsets)
        for record in self.records.values():
            source=record['source'];offset=self.action_offsets.get(source)
            if isinstance(offset,bool) or not isinstance(offset,int) or offset not in (0,1):raise ValueError(f'Explicit H0/H1 offset required for {source}')
        self.history,self.horizon,self.image_size=history,horizon,tuple(image_size)
        self.exclude_uncertain_controls=bool(exclude_uncertain_controls)
        self.samples=[]
        for key,record in self.records.items():
            ids=record['frame_ids'];width=history+horizon
            nominal=nominal_controls(record['actions'])
            for i in range(len(ids)-width+1):
                if ids[i+width-1]-ids[i]==width-1 and (not self.exclude_uncertain_controls or nominal[i:i+width].all()):self.samples.append((key,i))
        if not self.samples:raise ValueError('No contiguous matched windows')

    def __len__(self):return len(self.samples)

    def numeric_sample(self,index):
        key,start=self.samples[index];r=self.records[key]
        k,h=self.history,self.horizon;end=start+k+h
        offset=self.action_offsets[r['source']]
        speed=r['speed'][start:end];actions=r['actions'][start:end]
        anchor=start+k-1;future=np.arange(start+k,end)
        heading=r['forward'][anchor,:2];heading=heading/np.linalg.norm(heading)
        right=r['right_sign']*np.array([-heading[1],heading[0]])
        delta=r['positions'][future,:2]-r['positions'][anchor,:2]
        future_heading=r['forward'][future,:2]
        angle=np.arctan2(future_heading@right,future_heading@heading)
        targets=np.column_stack([speed[k:],delta@heading,delta@right,np.sin(angle),np.cos(angle)]).astype(np.float32)
        result=dict(past_ego_observations=torch.tensor(speed[:k,None]),
                    past_actions=torch.tensor(actions[offset:offset+k-1]),
                    future_actions=torch.tensor(actions[k-1+offset:k-1+offset+h]),
                    future_motion_targets=torch.from_numpy(targets),
                    source=r['source'],scenario=key,driver=r['driver'],route=r['route'],
                    start_frame=int(r['frame_ids'][start]),frame_ids=torch.tensor(r['frame_ids'][start:end]),
                    action_offset=offset,action_in_nominal_range=torch.tensor(nominal_controls(actions)),motion_target_source='simulator_world2body',
                    motion_frame='anchor_yaw_forward_right')
        if not all(torch.isfinite(v).all() for v in result.values() if isinstance(v,torch.Tensor)):raise ValueError(f'Nonfinite sample {key}')
        return result

    def __getitem__(self,index):
        result=self.numeric_sample(index);key,start=self.samples[index]
        images=[]
        for path in self.records[key]['image_paths'][start:start+self.history+self.horizon]:
            with Image.open(path) as image:
                a=np.asarray(image.convert('RGB').resize(self.image_size),dtype=np.float32)/255
                images.append(torch.from_numpy(a.transpose(2,0,1).copy()))
        images=torch.stack(images)
        result['past_images']=images[:self.history];result['future_images']=images[self.history:]
        return result

    def dynamics_arrays(self):
        """Image-free arrays for the same valid windows, for lightweight diagnostics."""
        past=[];actions=[];future=[];names=[]
        for key,start in self.samples:
            r=self.records[key];k,h=self.history,self.horizon;offset=self.action_offsets[r['source']]
            past.append(r['speed'][start:start+k]);future.append(r['speed'][start+k:start+k+h])
            # Include all 14 raw actions, permitting both alignments on identical targets.
            actions.append(r['actions'][start:start+k+h]);names.append(key)
        return np.array(past,np.float32),np.array(actions,np.float32),np.array(future,np.float32),np.array(names)


def numeric_record_fingerprint(records, splits):
    """Hash numeric supervision and episode assignments, independent of disk budgeting."""
    digest=hashlib.sha256(b'quality_policy:nominal_controls_all_14_v1')
    digest.update(json.dumps({k:sorted(v) for k,v in splits.items()},sort_keys=True).encode())
    for key in sorted(records):
        r=records[key];digest.update(key.encode());digest.update(str(r['right_sign']).encode())
        for field,dtype in [('frame_ids',np.int64),('speed',np.float32),('actions',np.float32),('world2body',np.float64)]:
            digest.update(np.ascontiguousarray(r[field],dtype=dtype).tobytes())
    return digest.hexdigest()


def nominal_controls(actions):
    """Nominal CARLA range, a conservative quality flag rather than an applied-control claim."""
    return np.isfinite(actions).all(axis=-1)&(np.abs(actions[...,0])<=1)&(actions[...,1:]>=0).all(axis=-1)&(actions[...,1:]<=1).all(axis=-1)


def load_training_datasets(project=PROJECT, *, require_images=True, image_size=(224,224)):
    """Load the frozen, quality-filtered train/validation/test views once."""
    project=Path(project)
    config=project/'data/config'
    splits=json.loads((config/'splits.json').read_text())['episodes']
    offsets=json.loads((config/'action_alignment.json').read_text())['action_offsets']
    records=load_episode_records(project,require_images=require_images)
    return {part:UnifiedDrivingDataset(records,episode_ids=keys,action_offsets=offsets,image_size=image_size)
            for part,keys in splits.items()}
