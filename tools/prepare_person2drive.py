"""Restore the frozen Person2Drive training cohort; explicit metadata/index/download/extract stages."""
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

BASE=PROJECT/'data/person2drive'


def download(entry,destination,revision):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if verified(destination,entry):return
    reuse=PROJECT/'data/archive/dataset_selection_samples/Person2Drive'/entry['path']
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


def build_indexes(selection):
    """Index only frozen selected episodes; never reselect the training cohort."""
    for driver in sorted({e['driver'] for e in selection['selected']}):
        grouped=defaultdict(list)
        for row in load(BASE/'metadata'/f'sensor_data/{driver}/b2d_infos_train.pkl'):
            grouped[row['folder']].append(row)
        for entry in selection['selected']:
            if entry['driver']!=driver:continue
            rows=sorted(grouped[entry['episode']],key=lambda r:r['frame_idx'])
            path=PROJECT/entry['index'];path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,frame_ids=np.array([r['frame_idx'] for r in rows],np.int64),
                speed=np.array([r['ego_vel'][0] for r in rows],np.float32),
                actions=np.array([[r[c] for c in ['steer','throttle','brake']] for r in rows],np.float32),
                world2ego=np.stack([r['world2ego'] for r in rows]).astype(np.float64),
                image_paths=np.array([r['sensors']['CAM_FRONT']['data_path'] for r in rows]))
        print('Indexed',driver,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ['metadata','index','download','extract']:parser.add_argument('--'+flag,action='store_true')
    parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    config=PROJECT/'data/config';catalog=json.loads((config/'person_catalog.json').read_text())
    selection=json.loads((config/'person_selection.json').read_text())
    (BASE/'manifests').mkdir(parents=True,exist_ok=True)
    save_json(BASE/'manifests/selection.json',selection)
    if args.metadata:
        drivers=set(selection['drivers'])
        entries=[e for e in catalog['files'] if e['path'].endswith('/b2d_infos_train.pkl') and e['path'].split('/')[1] in drivers]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda e:download(e,BASE/'metadata'/e['path'],selection['revision']),entries))
    if args.index:build_indexes(selection)
    if args.download:
        required=sum(e['size'] for e in selection['selected'] if not (BASE/'archives'/e['path']).exists())
        if shutil.disk_usage(BASE).free<required+20*1024**3:raise OSError('Download would breach disk reserve')
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(lambda e:download(e,BASE/'archives'/e['path'],selection['revision']),selection['selected']))
    if args.extract:
        for entry in selection['selected']:extract(entry,selection)
    print('Frozen selection:',len(selection['selected']),'episodes; no cohort reselection performed.')


if __name__=='__main__':main()
