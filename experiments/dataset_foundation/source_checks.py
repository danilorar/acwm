"""Record source-specific calibration evidence and unresolved collection provenance."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from src.numeric_metadata import load
from src.unified_driving_dataset import load_episode_records,nominal_controls

def main():
 selection=json.loads((ROOT/'data/person2drive_train/manifests/selection.json').read_text());rows=[]
 for driver in selection['drivers']:
  selected={x['episode'] for x in selection['selected'] if x['driver']==driver};found={}
  for row in load(ROOT/f'data/person2drive_train/metadata/sensor_data/{driver}/b2d_infos_train.pkl'):
   if row['folder'] not in selected or row['folder'] in found:continue
   matrix=np.asarray(row['sensors']['CAM_FRONT']['cam2ego']);right=matrix[:3,0]
   if not np.allclose(right,[0,-1,0],atol=1e-5):raise ValueError('Unexpected camera-right convention')
   found[row['folder']]=True;rows.append(dict(driver=driver,episode=row['folder'],camera_right_in_ego=right.tolist(),cam2ego=matrix.tolist()))
  assert set(found)==selected
 steering_checks={};action_ranges={}
 records=load_episode_records(require_images=False)
 for source in ['bench2drive','person2drive']:
  actions=np.concatenate([r['actions'] for r in records.values() if r['source']==source]);action_ranges[source]=dict(minimum=actions.min(0).tolist(),maximum=actions.max(0).tolist(),out_of_nominal_frames=int((~nominal_controls(actions)).sum()),applied_steering_saturation_or_conversion='Unverified; preserve raw labels and conservatively exclude ambiguous windows')
  turns=[];steers=[[],[]]
  for r in records.values():
   if r['source']!=source:continue
   f=r['forward'][:,:2];angle=np.arctan2(f[:-1,0]*f[1:,1]-f[:-1,1]*f[1:,0],np.sum(f[:-1]*f[1:],axis=1))*r['right_sign']
   valid=(np.diff(r['frame_ids'])==1)&(r['speed'][:-1]>3)
   turns.extend(angle[valid]);steers[0].extend(r['actions'][:-1,0][valid]);steers[1].extend(r['actions'][1:,0][valid])
  angle=np.array(turns);steering_checks[source]={}
  for offset in [0,1]:
   steer=np.array(steers[offset]);mask=(abs(steer)>.05)&(abs(angle)>1e-4)
   steering_checks[source]['H'+str(offset)]=dict(transitions=int(mask.sum()),correlation=float(np.corrcoef(steer[mask],angle[mask])[0,1]),positive_steer_right_yaw_sign_agreement=float((steer[mask]*angle[mask]>0).mean()))
 sources=['external/Bench2Drive/tools/data_collect.py','external/Bench2Drive/leaderboard/leaderboard/scenarios/scenario_manager.py',
          'experiments/dataset_selection/sources/Person2Drive_DATASET_DETAILS.md','experiments/dataset_selection/sources/Person2Drive_RELEASE_MANIFEST.md']
 result=dict(raw_action_ranges=action_ranges,steering_right_yaw_diagnostics=steering_checks,person_calibration_episode_samples=rows,source_file_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
  bench2drive=dict(native_axes='CARLA actor x forward, y right, z up; simulator inverse world2ego',
   collection_evidence='Env_Manager.tick reads sensor packet, speed and vehicle.get_control; save serializes shared tick_data with shared frame count',
   missing='Expert collection caller linking Env_Manager.tick/save to world ticks, control generation/application and count increments is absent. Evaluation scenario_manager order is not proof for these archives.'),
  person2drive=dict(native_axes='Processed metadata x forward, y left, z up; camera-right column [0,-1,0]',
   image_join='Use explicit CAM_FRONT.data_path and frame_idx in b2d_infos_train.pkl; never zip 20-Hz motion logs to 10-Hz RGB',
   missing='Original human recording/rendering/metadata-export caller linking applied command tick to rendered RGB and frame_idx is unavailable in the inspected release. No authoritative applied-control timestamps.'),
  frequency='Both treated as 10 Hz; selected-episode pose distance/speed median interval 0.1s independently supports this.',
  convention='Targets project world displacement and heading into planar anchor-forward and calibrated anchor-right axes; raw steer/throttle/brake are never negated or merged.',
  alignment='Explicit per-source offsets, selected using validation predictive RMSE only; actual collection-time convention unproved.')
 (HERE/'results/source_checks.json').write_text(json.dumps(result,indent=2)+'\n');print('Source-specific calibration checked for',len(rows),'Person episodes',flush=True)
if __name__=='__main__':main()
