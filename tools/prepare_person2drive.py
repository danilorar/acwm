"""Pinned selective Person2Drive preparation; explicit metadata/plan/download/extract stages."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import argparse
import json
import os
import shutil
import sys
import tarfile
import time

import numpy as np
from huggingface_hub import hf_hub_download

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from src.numeric_metadata import load
from download_bench2drive import save_json, verified

BASE=PROJECT/'data/person2drive_train'


def download(entry,destination,revision):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if verified(destination,entry):return
    reuse=PROJECT/'experiments/dataset_selection/samples/Person2Drive'/entry['path']
    if verified(reuse,entry):
        if destination.exists():raise ValueError('Conflicting local file; not overwriting')
        os.link(reuse,destination)
        return
    for attempt in range(3):
        try:
            # local_dir reproduces the repository-relative path.
            relative=Path(entry['path']);local_root=destination
            for _ in relative.parts:local_root=local_root.parent
            hf_hub_download('dongxr7/Person2Drive',entry['path'],repo_type='dataset',revision=revision,
                            local_dir=local_root,force_download=destination.exists())
            if not verified(destination,entry):raise ValueError('Size/SHA256 mismatch')
            return
        except Exception as exc:
            with (BASE/'manifests/failures.jsonl').open('a') as f:f.write(json.dumps(dict(path=entry['path'],attempt=attempt+1,error=str(exc)))+'\n')
            if attempt==2:raise
            time.sleep(3)


def descriptor(ids,v,a):
    coast=(v>=2)&(a[:,1]<=.01)&(a[:,2]<=.01)
    partial=(a[:,2]>.01)&(a[:,2]<.99)
    def windows(mask):
        if len(ids)<11:return 0
        return int(((np.convolve(mask.astype(int),np.ones(10,dtype=int),'valid')[:-1]==10)&(ids[10:]-ids[:-10]==10)).sum())
    return dict(frames=len(ids),valid_transitions=int((np.diff(ids)==1).sum()),
                coasting_windows=windows(coast),partial_windows=windows(partial),
                fast_partial=int((partial&(v>5)).sum()),steering_turns=int((abs(a[:,0])>.1).sum()),
                fast_frames=int((v>10).sum()),low_frames=int(((v>=2)&(v<5)).sum()),
                moderate_throttle=int(((a[:,1]>.1)&(a[:,1]<.9)).sum()))


def plan(catalog,count):
    entries={f['path']:f for f in catalog['files']}
    options=[]
    for driver in catalog['drivers']:
        records=load(BASE/'metadata'/f'sensor_data/{driver}/b2d_infos_train.pkl')
        grouped=defaultdict(list)
        for row in records:grouped[row['folder']].append(row)
        for folder,rows in sorted(grouped.items()):
            rows.sort(key=lambda r:r['frame_idx'])
            ids=np.array([r['frame_idx'] for r in rows],np.int64)
            if len(set(ids))!=len(ids):raise ValueError('Duplicate metadata IDs')
            speed=np.array([r['ego_vel'][0] for r in rows],np.float32)
            action=np.array([[r[c] for c in ['steer','throttle','brake']] for r in rows],np.float32)
            matrices=np.stack([r['world2ego'] for r in rows]).astype(np.float64)
            if not all(np.isfinite(x).all() for x in [speed,action,matrices]):raise ValueError('Nonfinite metadata')
            path=BASE/'manifests/indexes'/driver/(folder+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,frame_ids=ids,speed=speed,actions=action,world2ego=matrices,
                                image_paths=np.array([r['sensors']['CAM_FRONT']['data_path'] for r in rows]))
            archive=f'sensor_data/{driver}/{folder}.tar.zst'
            if archive not in entries:continue
            options.append(dict(**entries[archive],driver=driver,episode=folder,route='108' if folder.startswith('Town04') else '103',
                                index=str(path.relative_to(PROJECT)),**descriptor(ids,speed,action)))
        print('Indexed',driver,flush=True)
    chosen=[]
    # Broad human coverage first. For each driver and route, alternate the main coverage objective,
    # then maximize remaining complementary controls; length is only a tie breaker.
    for driver in catalog['drivers']:
        for route in ['103','108']:
            candidates=[x for x in options if x['driver']==driver and x['route']==route]
            for objective in ['coasting_windows','fast_partial','steering_turns'][:count]:
                selected=max(candidates,key=lambda e:(e[objective],e['partial_windows'],e['moderate_throttle'],e['frames'],e['path']))
                chosen.append(selected);candidates.remove(selected)
    if count>3:raise ValueError('Initial study supports up to 3 selected episodes per route')
    compressed=sum(e['size'] for e in chosen)
    # Metadata and front-only RGB are much smaller, but retain a conservative allowance.
    free=shutil.disk_usage(BASE).free
    required=compressed*2+20*1024**3
    result=dict(repository=catalog['repo'],revision=catalog['revision'],seed=catalog['seed'],drivers=catalog['drivers'],
                selected=chosen,all_inspected_episodes=options,
                expected_person_frames=sum(e['frames'] for e in chosen),expected_combined_frames=35510+sum(e['frames'] for e in chosen),
                archive_bytes=compressed,free_bytes=free,conservative_additional_bytes=required,fits=required<free)
    save_json(BASE/'manifests/selection.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['selected','all_inspected_episodes']},indent=2),flush=True)
    if not result['fits']:raise OSError('Planned subset exceeds available storage allowance')
    return result


def extract(entry,selection):
    import zstandard
    archive=BASE/'archives'/entry['path']
    if not verified(archive,entry):raise ValueError('Archive not verified')
    final=BASE/'extracted'/entry['driver']/entry['episode']
    marker=BASE/'manifests/completed'/entry['driver']/(entry['episode']+'.json')
    if marker.exists():
        record=json.loads(marker.read_text())
        if record['sha256']==entry['sha256'] and all((final/'camera/rgb_front'/name).is_file() and (final/'camera/rgb_front'/name).stat().st_size==size for name,size in record['images'].items()):return
        raise ValueError('Incomplete/conflicting extraction; not overwriting')
    if final.exists():raise ValueError('Unmarked extraction already exists')
    stage=BASE/'.staging'/entry['driver']/entry['episode'];stage.mkdir(parents=True,exist_ok=True)
    images={}
    with archive.open('rb') as raw,zstandard.ZstdDecompressor().stream_reader(raw) as stream,tarfile.open(fileobj=stream,mode='r|') as tar:
        for member in tar:
            parts=Path(member.name).parts
            if member.name.startswith('/') or '..' in parts or '\\' in member.name:raise ValueError('Unsafe tar path')
            if not(member.isfile() or member.isdir()):raise ValueError('Tar links/special members refused')
            if member.isfile() and '/camera/rgb_front/' in member.name and member.name.endswith('.jpg'):
                name=Path(member.name).name
                if not Path(name).stem.isdigit() or name in images:raise ValueError('Invalid/duplicate image')
                if shutil.disk_usage(BASE).free<20*1024**3+member.size:raise OSError('Disk reserve reached')
                out=stage/'camera/rgb_front'/name;out.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as inp,out.open('wb') as target:shutil.copyfileobj(inp,target,4*1024**2)
                if out.stat().st_size!=member.size:raise ValueError('Truncated member')
                images[name]=member.size
    with np.load(PROJECT/entry['index']) as data:
        missing=[str(p) for p in data['image_paths'] if Path(str(p)).name not in images]
    if missing:raise ValueError(f'Missing referenced images: {missing[:3]}')
    final.parent.mkdir(parents=True,exist_ok=True);stage.rename(final)
    save_json(marker,dict(sha256=entry['sha256'],images=images,retained_bytes=sum(images.values())))
    print('Extracted',entry['driver'],entry['episode'],len(images),'front RGB files',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',action='store_true');p.add_argument('--plan',action='store_true')
    p.add_argument('--download',action='store_true');p.add_argument('--extract',action='store_true')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--episodes-per-route',type=int,default=3)
    args=p.parse_args()
    # A fresh checkout restores the small pinned catalog/selection, never raw data.
    for name in ['catalog.json','selection.json']:
        destination=BASE/'manifests'/name
        if not destination.exists():
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(PROJECT/'experiments/dataset_foundation/manifests'/name,destination)
    catalog=json.loads((BASE/'manifests/catalog.json').read_text())
    if args.metadata:
        def fetch(entry):
            download(entry,BASE/'metadata'/entry['path'],catalog['revision']);print('Metadata verified:',entry['path'],flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(fetch,catalog['metadata']))
    selection=plan(catalog,args.episodes_per_route) if args.plan else None
    if args.download or args.extract:
        selection=selection or json.loads((BASE/'manifests/selection.json').read_text())
        if not selection['fits']:raise ValueError('Non-fitting plan')
        if args.download:
            def transfer(entry):
                download(entry,BASE/'archives'/entry['path'],selection['revision']);print('Archive verified:',entry['path'],flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(transfer,selection['selected']))
        if args.extract:
            for entry in selection['selected']:extract(entry,selection)


if __name__=='__main__':main()
