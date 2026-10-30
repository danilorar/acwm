"""Safely extract verified selected archives, retaining annotations and front RGB."""
from pathlib import Path, PurePosixPath
import argparse
import gzip
import json
import shutil
import tarfile

from download_bench2drive import PROJECT, save_json, verified


def safe_member(member):
    path=PurePosixPath(member.name)
    if path.is_absolute() or '..' in path.parts or '\\' in member.name or (not path.parts and not member.isdir()):
        raise ValueError(f'Unsafe archive path: {member.name}')
    if not(member.isfile() or member.isdir()):
        raise ValueError(f'Links/special archive members are forbidden: {member.name}')
    return path


def keep_member(path, all_rgb=False):
    parts=path.parts
    annotation=len(parts)>=3 and parts[-2]=='anno' and parts[-1].endswith('.json.gz')
    rgb=len(parts)>=4 and parts[-3]=='camera' and (parts[-2]=='rgb_front' or (all_rgb and parts[-2].startswith('rgb_'))) and path.suffix.lower() in ['.jpg','.jpeg','.png']
    return annotation or rgb


def extract_one(archive, entry, base, state, *, all_rgb=False, reserve_bytes=20*1024**3):
    if not verified(archive,entry):raise ValueError(f'Unverified archive: {archive.name}')
    mode='all_rgb' if all_rgb else 'rgb_front'
    prior=state.get(entry['path'])
    if prior:
        if prior['sha256'] != entry['sha256'] or prior['mode'] != mode:
            raise ValueError('Existing extraction has different hash/modalities; use a separate destination to avoid overwriting')
        files=[base/'extracted'/p for p in prior['files']]
        if not all(p.is_file() and p.stat().st_size==size for p,size in zip(files,prior['sizes'])):
            raise ValueError('Existing extraction is incomplete; not overwriting it silently')
        return prior,False
    staging=base/'.extract_staging'/entry['path'].removesuffix('.tar.gz')
    if staging.exists():shutil.rmtree(staging)  # Only this utility's unfinished transaction.
    staging.mkdir(parents=True)
    try:
        with tarfile.open(archive,'r:gz') as tar:
            members=tar.getmembers()
            paths=[safe_member(m) for m in members]  # Validate all paths, even unwanted modalities.
            selected=[(m,p) for m,p in zip(members,paths) if m.isfile() and keep_member(p,all_rgb)]
            if not selected:raise ValueError('No requested annotation/RGB members in archive')
            if len({str(p) for _,p in selected}) != len(selected):raise ValueError('Duplicate archive member paths')
            roots=sorted({p.parts[0] for _,p in selected})
            if len(roots)!=1:raise ValueError('Expected a single preserved top-level recording directory')
            if (base/'extracted'/roots[0]).exists():raise ValueError('Extracted recording already exists without a matching completion record')
            if not any(p.parts[-2]=='anno' for _,p in selected) or not any(p.parts[-2]=='rgb_front' for _,p in selected):
                raise ValueError('Archive lacks required anno or camera/rgb_front modality')
            retained=sum(m.size for m,_ in selected)
            if retained+reserve_bytes>shutil.disk_usage(base).free:raise OSError('Retained member sizes exceed free disk plus reserve')
            for member,path in selected:
                destination=staging/str(path)
                destination.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source,destination.open('xb') as target:
                    shutil.copyfileobj(source,target,8*1024**2)
                if destination.stat().st_size!=member.size:raise ValueError('Truncated tar member')
        # Read through gzip trailer to verify CRC/length; tar end markers alone do not do this.
        with gzip.open(archive,'rb') as stream:
            for _ in iter(lambda:stream.read(8*1024**2),b''):pass
        folder=staging/roots[0]
        annos=list(folder.rglob('anno/*.json.gz'))
        if not annos:raise ValueError('No annotation files after extraction')
        # Preflight expected pose/schema and numeric-ID pairing before commit.
        for anno_dir in {p.parent for p in annos}:
            image_dir=anno_dir.parent/'camera/rgb_front'
            ids={p.name.split('.')[0] for p in anno_dir.glob('*.json.gz')}
            images={p.stem for p in image_dir.iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png']}
            if ids!=images:raise ValueError(f'Image/annotation mismatch: {anno_dir}')
            first=next(anno_dir.glob('*.json.gz'))
            with gzip.open(first,'rt') as f:row=json.load(f)
            for key in ['speed','steer','throttle','brake']:float(row[key])
            ego=[box for box in row['bounding_boxes'] if box.get('class')=='ego_vehicle']
            if len(ego)!=1 or len(ego[0]['location'])!=3 or len(ego[0]['rotation'])!=3:
                raise ValueError('No compatible simulator ego pose')
        folder.rename(base/'extracted'/roots[0])
        record=dict(sha256=entry['sha256'],mode=mode,roots=roots,
                    files=[str(p) for _,p in selected],sizes=[m.size for m,_ in selected],retained_bytes=retained)
        return record,True
    finally:
        if staging.exists():shutil.rmtree(staging)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=PROJECT/'data/bench2drive_train')
    parser.add_argument('--only',nargs='+',help='Exact selected archive filenames')
    parser.add_argument('--all-rgb',action='store_true',help='Keep all RGB cameras, not only front')
    parser.add_argument('--reserve-gib',type=float,default=20.)
    args=parser.parse_args()
    base=args.destination.resolve()
    manifest=json.loads((base/'manifests/selection.json').read_text())
    state_path=base/'manifests/extracted.json'
    state=json.loads(state_path.read_text()) if state_path.exists() else {}
    (base/'extracted').mkdir(parents=True,exist_ok=True)
    selected=manifest['selected']
    if args.only:
        if set(args.only)-{e['path'] for e in selected}:raise ValueError('--only contains unselected archive')
        selected=[e for e in selected if e['path'] in args.only]
    failed=[]
    for entry in selected:
        path=base/'archives'/entry['path']
        if not path.exists():
            print('Not downloaded:',entry['path'],flush=True)
            continue
        try:
            record,changed=extract_one(path,entry,base,state,all_rgb=args.all_rgb,reserve_bytes=int(args.reserve_gib*1024**3))
            state[entry['path']]=record
            save_json(state_path,state)
            print('Extracted:' if changed else 'Verified existing extraction:',entry['path'],flush=True)
        except Exception as exc:
            failed.append(dict(path=entry['path'],error=str(exc)))
            with (base/'manifests/extraction_failures.jsonl').open('a') as f:f.write(json.dumps(failed[-1])+'\n')
    if failed:raise RuntimeError(f'{len(failed)} extractions failed; see manifests/extraction_failures.jsonl')
    print(f'Completed extractions recorded: {len(state)}. Archives retained.',flush=True)


if __name__=='__main__':main()
