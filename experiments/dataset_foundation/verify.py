"""Verify extracted RGB, all split identities, source transforms and real batches."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys,json,hashlib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader,Subset
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from src.unified_driving_dataset import load_episode_records,UnifiedDrivingDataset,numeric_record_fingerprint,nominal_controls
from src.bench2drive_dataset import route_group

def main():
 records=load_episode_records(require_images=True)
 negative=[]
 for key,r in records.items():
  v=r['speed']
  if (v<0).any():negative.append(dict(episode=key,source=r['source'],minimum=float(v.min()),negative_frames=int((v<0).sum()),below_minus001=int((v<-.01).sum()),below_minus01=int((v<-.1).sum()),below_minus1=int((v<-1).sum())))
 (HERE/'results/negative_speed_audit.json').write_text(json.dumps(negative,indent=2)+'\n')
 split=json.loads((HERE/'results/splits.json').read_text())['episodes']
 config=json.loads((HERE/'results/dynamics_config.json').read_text())
 fingerprint=numeric_record_fingerprint(records,split)
 if config.get('data_fingerprint')!=fingerprint:raise ValueError('Stale dynamics results: numeric data or split changed; rerun dynamics.py')
 offsets=json.loads((HERE/'results/action_alignment.json').read_text())['action_offsets']
 flat=sum(split.values(),[]);assert len(flat)==len(set(flat)) and set(flat)==set(records)
 identities={};drivers={};routes={};counts={};speed_ranges={};sample_indices=[];jpeg_paths=[];path_split={}
 ds=UnifiedDrivingDataset(records,action_offsets=offsets)
 first={}
 for i,(key,start) in enumerate(ds.samples):first.setdefault(key,i)
 for part,keys in split.items():
  identities[part]={(key,int(frame)) for key in keys for frame in records[key]['frame_ids']}
  drivers[part]={records[k]['driver'] for k in keys if records[k]['source']=='person2drive'}
  routes[part]={route_group(records[k]['episode']) for k in keys if records[k]['source']=='bench2drive'}
  counts[part]={}
  for source in ['bench2drive','person2drive']:
   chosen=[k for k in keys if records[k]['source']==source];used=0;windows=0;gaps=0
   for key in chosen:
    r=records[key];ids=r['frame_ids'];cover=np.zeros(len(ids),bool)
    for start in range(len(ids)-13):
     if ids[start+13]-ids[start]==13 and nominal_controls(r['actions'][start:start+14]).all():cover[start:start+14]=True;windows+=1
    used+=int(cover.sum());gaps+=int((np.diff(ids)!=1).sum());sample_indices.append(first[key]);jpeg_paths.extend(r['image_paths']);path_split.update({str(p):part for p in r['image_paths']})
    assert np.all((r['actions'][:,1:]>=0)&(r['actions'][:,1:]<=1))
    # Out-of-range raw steering is retained; default windows exclude its frames.
    # Inverse-pose rotations remain orthonormal (both coordinate handednesses).
    rot=r['world2body'][:,:3,:3]
    assert np.allclose(rot@rot.transpose(0,2,1),np.eye(3),atol=1e-4)
   values=np.concatenate([records[k]['speed'] for k in chosen])
   speed_ranges.setdefault(part,{})[source]=dict(minimum=float(values.min()),maximum=float(values.max()),negative_frames=int((values<0).sum()),below_minus01_frames=int((values<-.1).sum()))
   counts[part][source]=dict(out_of_nominal_control_frames=int(sum((~nominal_controls(records[k]['actions'])).sum() for k in chosen)),episodes=len(chosen),matched_frames=sum(len(records[k]['frame_ids']) for k in chosen),
      frames_participating_in_windows=used,windows=windows,gap_boundaries=gaps)
 for a,b in [('train','validation'),('train','test'),('validation','test')]:
  assert not identities[a]&identities[b];assert not drivers[a]&drivers[b];assert not routes[a]&routes[b]
 # PIL verification scans every referenced compressed JPEG; representative full decodes
 # exercise RGB conversion and resizing at each episode's first/middle/last valid window.
 def check(path):
  with Image.open(path) as image:
   if image.format!='JPEG':raise ValueError(f'Not JPEG: {path}')
   image.verify()
  return hashlib.sha256(path.read_bytes()).hexdigest(),str(path)
 hashes={}
 with ThreadPoolExecutor(max_workers=8) as pool:
  for digest,path in pool.map(check,jpeg_paths):hashes.setdefault(digest,[]).append(path)
 duplicates=[paths for paths in hashes.values() if len({path_split[p] for p in paths})>1]
 (HERE/'results/cross_split_duplicate_images.json').write_text(json.dumps(duplicates,indent=2)+'\n')
 if duplicates:raise ValueError(f'{len(duplicates)} identical JPEG hashes cross splits; resolve before training')
 torch.set_num_threads(1)
 for key,index in first.items():
  members=[i for i,(name,start) in enumerate(ds.samples) if name==key]
  for i in [members[0],members[len(members)//2],members[-1]]:
   sample=ds[i]
   assert sample['past_images'].shape==(4,3,224,224)
   assert sample['future_images'].shape==(10,3,224,224)
   assert sample['past_actions'].shape==(3,3)
   assert sample['future_actions'].shape==(10,3)
   assert sample['past_ego_observations'].shape==(4,1)
   assert sample['future_motion_targets'].shape==(10,5)
 # Mixed-source batching includes all six contractual fields.
 indices=[next(i for i,(key,_) in enumerate(ds.samples) if records[key]['source']==s) for s in ['bench2drive','person2drive']]
 batch=next(iter(DataLoader(Subset(ds,indices),batch_size=2,num_workers=0)))
 shapes={k:list(v.shape) for k,v in batch.items() if hasattr(v,'shape')}
 result=dict(data_fingerprint=fingerprint,counts=counts,raw_speed_ranges=speed_ranges,jpeg_headers_verified=len(jpeg_paths),representative_full_decode_windows=len(first)*3,
             no_shared_frames=True,cross_split_identical_jpegs=len(duplicates),bench_route_disjoint=True,person_driver_disjoint=True,mixed_batch_shapes=shapes,
             action_offsets=offsets,limitation='JPEG verify is structural; full decoding sampled at three windows per episode, not every image')
 (HERE/'results/verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
