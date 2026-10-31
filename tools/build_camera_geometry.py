"""Preserve source calibration and export explicit optical-camera to shared-ego geometry."""
from pathlib import Path
from collections import defaultdict
import argparse,gzip,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.numeric_metadata import load
from tools.download_bench2drive import save_json
CAMERAS=('rgb_front','rgb_front_left','rgb_front_right','rgb_back','rgb_back_left','rgb_back_right')
SENSORS=tuple('CAM_'+c.removeprefix('rgb_').upper() for c in CAMERAS)
# Native CARLA camera axes (forward,right,up) -> optical (right,down,forward).
ACTOR_TO_OPTICAL=np.array([[0,1,0,0],[0,0,-1,0],[1,0,0,0],[0,0,0,1]],float)
LEFT_TO_RIGHT=np.diag([1.,-1,1,1]);REPORT=ROOT/'docs/multicamera'

def export(source,key,rows,ids,path):
    n=len(rows);K=np.full((n,6,3,3),np.nan);native=np.full((n,6,4,4),np.nan);raw=np.full_like(native,np.nan);canonical=np.full_like(native,np.nan)
    sizes=np.full((n,6,2),-1,dtype=np.int64);
    if source=='person2drive':
        marker=ROOT/'data/person2drive/manifests/multicamera'/(key.removeprefix('person2drive/')+'.json')
        if marker.exists():
            audit=json.loads(marker.read_text())
            for j,c in enumerate(CAMERAS):
                resolutions=audit['cameras'][c]['resolutions']
                if len(resolutions)==1:sizes[:,j]=resolutions[0]
    valid=np.zeros((n,6),bool);errors=[];reference_mismatches=[];timestamps=0
    for i,row in enumerate(rows):
        if source=='bench2drive':
            ego=next(b for b in row['bounding_boxes'] if b.get('class')=='ego_vehicle');world2ego=np.asarray(ego['world2ego'],float)
        else:world2ego=np.asarray(row['world2ego'],float)
        for j,(name,sensor) in enumerate(zip(CAMERAS,SENSORS)):
            s=row['sensors'].get(sensor)
            if s is None:continue
            K[i,j]=s['intrinsic'];raw[i,j]=s['world2cam'];derived=world2ego@np.linalg.inv(raw[i,j])
            if source=='bench2drive':
                native[i,j]=s.get('cam2ego',derived);canonical[i,j]=derived@np.linalg.inv(ACTOR_TO_OPTICAL)
                sizes[i,j]=[s['image_size_x'],s['image_size_y']]
                if 'cam2ego' in s:errors.append(float(abs(derived-native[i,j]).max()))
            else:
                native[i,j]=s['cam2ego'];canonical[i,j]=LEFT_TO_RIGHT@native[i,j]
                errors.append(float(abs(derived-native[i,j]).max()))
                p=Path(s['data_path'])
                if p.parent.name!=name or not p.stem.isdigit() or int(p.stem)!=int(ids[i]):reference_mismatches.append([int(ids[i]),name,s['data_path']])
            timestamps+=sum(k in s for k in ['timestamp','capture_timestamp','sensor_timestamp'])
            valid[i,j]=np.isfinite(K[i,j]).all() and np.isfinite(canonical[i,j]).all()
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,frame_ids=np.asarray(ids,np.int64),intrinsics=K,camera_to_ego=canonical,native_camera_to_ego=native,
        world_to_camera_raw=raw,image_size=sizes,valid=valid,camera_names=np.array(CAMERAS))
    # Compare each mounting transform with its episode median; preserve variations.
    median=np.nanmedian(canonical,axis=0);deviation=np.nanmax(abs(canonical-median),axis=(0,2,3))
    result=dict(episode=key,frames=n,valid_camera_calibrations=int(valid.sum()),missing_camera_calibrations=int((~valid).sum()),
        metadata_path_id_mismatches=reference_mismatches,timestamp_fields_found=timestamps,
        native_world_to_camera_axes='CARLA actor forward/right/up' if source=='bench2drive' else 'optical right/down/forward',
        canonical_frame='camera optical right/down/forward to ego forward/right/up, at the same metadata observation',
        native_ego_y='right' if source=='bench2drive' else 'left',
        reported_extrinsic_comparison_count=len(errors),reported_vs_pose_derived_max_abs=float(max(errors)) if errors else None,mounting_max_abs_deviation_by_camera=deviation.tolist(),
        median_intrinsics=np.nanmedian(K,axis=0).tolist(),median_camera_to_ego=median.tolist(),image_sizes=np.unique(sizes.reshape(-1,2),axis=0).tolist(),
        sidecar=str(path.relative_to(ROOT)))
    print('Geometry',key,'pose discrepancy',result['reported_vs_pose_derived_max_abs'],flush=True);return result

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',choices=['bench2drive','person2drive','all'],default='all');args=parser.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True);report=[]
    previous=REPORT/'calibration.json'
    if args.source!='all' and previous.exists():report=[r for r in json.loads(previous.read_text())['episodes'] if not r['episode'].startswith(args.source+'/')]
    for folder in (sorted((ROOT/'data/bench2drive/extracted').iterdir()) if args.source!='person2drive' else []):
        files=sorted((folder/'anno').glob('*.json.gz'));rows=[]
        for p in files:
            with gzip.open(p,'rt') as f:rows.append(json.load(f))
        if rows:report.append(export('bench2drive','bench2drive/'+folder.name,rows,[int(p.name.split('.')[0]) for p in files],ROOT/'data/bench2drive/manifests/cameras'/(folder.name+'.npz')))
    selection=json.loads((ROOT/'data/config/person_selection.json').read_text())
    for driver in (selection['drivers'] if args.source!='bench2drive' else []):
        grouped=defaultdict(list)
        for row in load(ROOT/f'data/person2drive/metadata/sensor_data/{driver}/b2d_infos_train.pkl'):grouped[row['folder']].append(row)
        for entry in selection['selected']:
            if entry['driver']!=driver:continue
            rows=sorted(grouped[entry['episode']],key=lambda r:r['frame_idx'])
            report.append(export('person2drive',f"person2drive/{driver}/{entry['episode']}",rows,[r['frame_idx'] for r in rows],ROOT/'data/person2drive/manifests/cameras'/driver/(entry['episode']+'.npz')))
    save_json(REPORT/'calibration.json',dict(camera_order=CAMERAS,episodes=report))
if __name__=='__main__':main()
