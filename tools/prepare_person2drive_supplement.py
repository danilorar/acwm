"""Add the rare-control episodes identified by the complete release audit."""
from collections import defaultdict
import json,sys,shutil
import numpy as np
from prepare_person2drive import BASE,PROJECT,download,extract,descriptor
from src.numeric_metadata import load
from download_bench2drive import save_json

PAIRS=[('driver10','Town05_drive_1'),('driver10','Town05_drive_5'),('driver03','Town05_drive_1'),('driver03','Town05_drive_2'),
       ('driver05','Town05_drive_5'),('driver05','Town05_drive_6'),('driver43','Town05_drive_7'),('driver43','Town05_drive_4')]

def main():
 catalog=json.loads((BASE/'manifests/catalog.json').read_text());selection=json.loads((BASE/'manifests/selection.json').read_text());files={e['path']:e for e in catalog['files']};entries=[]
 for driver in sorted({d for d,e in PAIRS}):
  records=load(BASE/'metadata'/f'sensor_data/{driver}/b2d_infos_train.pkl');groups=defaultdict(list)
  for r in records:groups[r['folder']].append(r)
  for d,ep in PAIRS:
   if driver!=d:continue
   group=sorted(groups[ep],key=lambda r:r['frame_idx']);ids=np.array([r['frame_idx'] for r in group],np.int64);v=np.array([r['ego_vel'][0] for r in group],np.float32);a=np.array([[r[c] for c in ['steer','throttle','brake']] for r in group],np.float32)
   path=BASE/'manifests/indexes'/driver/(ep+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
   np.savez_compressed(path,frame_ids=ids,speed=v,actions=a,world2ego=np.stack([r['world2ego'] for r in group]).astype(np.float64),image_paths=np.array([r['sensors']['CAM_FRONT']['data_path'] for r in group]))
   entries.append(dict(**files[f'sensor_data/{driver}/{ep}.tar.zst'],driver=driver,episode=ep,route='103',index=str(path.relative_to(PROJECT)),**descriptor(ids,v,a)))
 additional=sum(e['size'] for e in entries if not (BASE/'archives'/e['path']).exists())
 if shutil.disk_usage(BASE).free<additional*2+20*1024**3:raise OSError('Supplement does not fit reserve')
 known={e['path'] for e in selection['selected']}
 selection['selected'] += [e for e in entries if e['path'] not in known]
 selection['complete_metadata_inspection']='data/person2drive_train/manifests/full_release_inspection.json'
 selection['inspected_sensor_episode_count']=800
 selection['drivers']=sorted({e['driver'] for e in selection['selected']})
 selection['expected_person_frames']=sum(e['frames'] for e in selection['selected']);selection['expected_combined_frames']=35510+selection['expected_person_frames'];selection['archive_bytes']=sum(e['size'] for e in selection['selected'])
 selection['conservative_additional_bytes']=selection['archive_bytes']*2+20*1024**3
 selection['supplement']=dict(free_bytes_before_download=shutil.disk_usage(BASE).free,additional_storage_estimate_bytes=additional*2,
    episodes=[e['path'] for e in entries],reason='Full 800-episode audit: all 14 H0 low-speed sustained partial-braking windows; all 21 turning-partial windows; additional fast turning coast',
    extra_driver_splits={'driver10':'train','driver43':'train','driver03':'validation'},selection_uses_model_errors=False)
 save_json(BASE/'manifests/selection.json',selection)
 snapshot=PROJECT/'experiments/dataset_foundation/manifests/selection.json';save_json(snapshot,selection)
 print('Final selected episodes',len(selection['selected']),'combined frames',selection['expected_combined_frames'],flush=True)
 # Only the supplementary entries, so this can safely run beside the primary transfer.
 for e in entries:
  download(e,BASE/'archives'/e['path'],selection['revision']);extract(e,selection)
 print('Rare-control supplement complete',flush=True)
if __name__=='__main__':main()
