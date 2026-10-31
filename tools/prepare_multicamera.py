"""Audit selected archives and optionally add only the six surrounding RGB views.

Existing files are compared to archive bytes, never overwritten. Per-episode
markers make interruption/resumption safe. No metadata or cohort changes.
"""
from pathlib import Path,PurePosixPath
from collections import defaultdict
from contextlib import contextmanager
import argparse,hashlib,io,json,os,shutil,sys,tarfile,time
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.download_bench2drive import verified,save_json
CAMERAS=('rgb_front','rgb_front_left','rgb_front_right','rgb_back','rgb_back_left','rgb_back_right')
REPORT=ROOT/'docs/multicamera'

@contextmanager
def open_tar(path):
    if path.name.endswith('.zst'):
        import zstandard
        with path.open('rb') as raw,zstandard.ZstdDecompressor().stream_reader(raw) as stream,tarfile.open(fileobj=stream,mode='r|') as tar:yield tar
    else:
        with tarfile.open(path,mode='r|gz') as tar:yield tar

def audit_episode(source,entry,extract=False):
    import numpy as np
    base=ROOT/'data'/source;archive=base/'archives'/entry['path']
    if not verified(archive,entry):raise ValueError('Missing or unverified archive: '+str(archive))
    if source=='person2drive':
        episode=entry['episode'];key=f"person2drive/{entry['driver']}/{episode}";folder=base/'extracted'/entry['driver']/episode
        with np.load(ROOT/entry['index'],allow_pickle=False) as index:expected=set(map(int,index['frame_ids']))
    else:
        episode=entry['path'].removesuffix('.tar.gz');key='bench2drive/'+episode;folder=base/'extracted'/episode
        expected={int(p.name.split('.')[0]) for p in (folder/'anno').glob('*.json.gz')}
    if not expected:raise ValueError('No state/action frame IDs; restore annotations/indexes before RGB extraction')
    marker=base/'manifests/multicamera'/key.removeprefix(source+'/')
    marker=marker.with_name(marker.name+'.json');stage=base/'.staging_multicamera'/key.removeprefix(source+'/')
    if marker.exists():
        old=json.loads(marker.read_text())
        if old['archive_sha256']==entry['sha256'] and (not extract or old['extracted']):
            if not extract or all((folder/'camera'/c/f'{i:05}.jpg').is_file() for c in CAMERAS for i in old['cameras'][c]['ids']):return old
    stats={c:dict(ids=[],bytes=0,resolutions=set(),formats=set(),exif_timestamp_images=0,existing_matches=0) for c in CAMERAS}
    directories=defaultdict(int);new_bytes=0;seen=set()
    for member in iter_members(archive):
        # iter_members yields RGB payload only, while validating every tar member.
        camera,name,size,data=member
        directories[camera]+=1
        if camera not in CAMERAS:continue
        if (camera,name) in seen:raise ValueError('Duplicate RGB member')
        seen.add((camera,name));frame=int(Path(name).stem);s=stats[camera];s['ids'].append(frame);s['bytes']+=size
        with Image.open(io.BytesIO(data)) as image:
            s['resolutions'].add(image.size);s['formats'].add(image.format)
            exif=image.getexif();s['exif_timestamp_images']+=int(any(k in exif for k in [306,36867,36868]));image.verify()
        target=folder/'camera'/camera/name
        if target.exists():
            if target.stat().st_size!=size or hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(data).digest():raise ValueError('Conflicting existing RGB: '+str(target))
            s['existing_matches']+=1
        else:
            new_bytes+=size
            if extract:
                if shutil.disk_usage(base).free<size+20*1024**3:raise OSError('20 GiB reserve reached')
                out=stage/camera/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
    front=set(stats['rgb_front']['ids']);missing={}
    for camera,s in stats.items():
        s['ids'].sort();s['resolutions']=sorted(s['resolutions']);s['formats']=sorted(s['formats']);ids=set(s['ids'])
        s['frames']=len(ids);s['missing_metadata_ids']=sorted(expected-ids);s['extra_vs_metadata_ids']=sorted(ids-expected)
        s['missing_vs_front_ids']=sorted(front-ids);s['extra_vs_front_ids']=sorted(ids-front)
        s['gaps_inside_range']=sorted(set(range(min(ids),max(ids)+1))-ids) if ids else []
        missing[camera]=len(expected-ids)
    all_views=all(stats[c]['frames'] for c in CAMERAS)
    # Publish only complete six-camera episodes. Incomplete episodes remain reported,
    # never acquire an extraction-complete marker or silently substitute imagery.
    if extract and all_views and not any(missing.values()):
        for camera in CAMERAS:
            staged=stage/camera
            if not staged.exists():continue
            final=folder/'camera'/camera
            if not final.exists():staged.rename(final)
            else:
                for p in staged.iterdir():
                    target=final/p.name
                    if target.exists():raise ValueError('Concurrent destination appeared')
                    p.rename(target)
                staged.rmdir()
        if stage.exists():shutil.rmtree(stage)
    result=dict(episode=key,archive=str(archive.relative_to(ROOT)),archive_sha256=entry['sha256'],available_camera_directories=dict(directories),
        metadata_frames=len(expected),cameras=stats,all_six_available=all_views,all_metadata_frames_covered=not any(missing.values()),
        additional_rgb_bytes=new_bytes,extracted=bool(extract and all_views and not any(missing.values())),
        synchronization='Frame-ID and metadata correspondence only; archive JPEGs have no verified capture-time clock. See synchronization.md.')
    save_json(marker,result);return result

def iter_members(archive):
    with open_tar(archive) as tar:
        for member in tar:
            path=PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or '\\' in member.name or not (member.isfile() or member.isdir()):raise ValueError('Unsafe archive member '+member.name)
            parts=path.parts
            if not member.isfile() or len(parts)<3 or parts[-3]!='camera' or not parts[-2].startswith('rgb_'):continue
            name=parts[-1]
            if Path(name).suffix.lower() not in ['.jpg','.jpeg','.png'] or not Path(name).stem.isdigit():raise ValueError('Unrecognized RGB member')
            # Do not even materialize unrequested extra RGB (e.g. top-down) payloads.
            data=tar.extractfile(member).read() if parts[-2] in CAMERAS else None
            yield parts[-2],name,member.size,data

def compact_report(row):
    """Keep exact missing IDs; full present-ID lists stay in ignored local markers."""
    result={k:v for k,v in row.items() if k!='cameras'};result['cameras']={}
    for c,stats in row['cameras'].items():
        present=stats.get('ids',[])
        item={k:v for k,v in stats.items() if k not in ['ids','extra_vs_metadata_ids','extra_vs_front_ids']}
        item['id_range']=[min(present),max(present)] if present else stats.get('id_range')
        item['extra_vs_metadata_count']=len(stats['extra_vs_metadata_ids']) if 'extra_vs_metadata_ids' in stats else stats.get('extra_vs_metadata_count',0)
        item['extra_vs_front_count']=len(stats['extra_vs_front_ids']) if 'extra_vs_front_ids' in stats else stats.get('extra_vs_front_count',0)
        result['cameras'][c]=item
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',choices=['bench2drive','person2drive','all'],default='all');parser.add_argument('--extract',action='store_true');parser.add_argument('--wait-for-archives',action='store_true');parser.add_argument('--wait-for-audit',action='store_true');args=parser.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True)
    sources=['bench2drive','person2drive'] if args.source=='all' else [args.source];rows=[]
    for source in sources:
        prefix='bench' if source=='bench2drive' else 'person';selection=json.loads((ROOT/'data/config'/f'{prefix}_selection.json').read_text())
        for n,entry in enumerate(selection['selected']):
            archive=ROOT/'data'/source/'archives'/entry['path']
            while args.wait_for_archives and (not archive.is_file() or archive.stat().st_size!=entry['size']):time.sleep(5)
            audit_key=Path(entry['driver'])/entry['episode'] if source=='person2drive' else Path(entry['path'].removesuffix('.tar.gz'))
            audit_marker=(ROOT/'data'/source/'manifests/multicamera'/audit_key).with_suffix('.json')
            while args.wait_for_audit and not audit_marker.exists():time.sleep(5)
            row=audit_episode(source,entry,args.extract);rows.append(row)
            print(source,n+1,'/',len(selection['selected']),row['episode'],'missing',sum(len(s['missing_metadata_ids']) for s in row['cameras'].values()),flush=True)
        save_json(REPORT/f'{source}_availability.json',dict(episodes=[compact_report(r) for r in rows if r['episode'].startswith(source+'/')]))
    print('RGB archive audit/extraction complete',flush=True)
if __name__=='__main__':main()
