"""Inspect rare sustained control/speed/steering combinations in all 800 episodes."""
from pathlib import Path
import sys,json,csv
from collections import defaultdict
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from src.numeric_metadata import load

def main():
 catalog=json.loads((ROOT/'data/person2drive_train/manifests/catalog.json').read_text())
 selected=json.loads((ROOT/'data/person2drive_train/manifests/selection.json').read_text());chosen={(e['driver'],e['episode']) for e in selected['selected']};rows=[]
 for entry in catalog['files']:
  if not entry['path'].endswith('/b2d_infos_train.pkl'):continue
  driver=entry['path'].split('/')[1];groups=defaultdict(list)
  for r in load(ROOT/'data/person2drive_train/metadata'/entry['path']):groups[r['folder']].append(r)
  for ep,group in groups.items():
   group.sort(key=lambda x:x['frame_idx']);ids=np.array([r['frame_idx'] for r in group]);v=np.array([r['ego_vel'][0] for r in group]);a=np.array([[r[c] for c in ['steer','throttle','brake']] for r in group]);valid=ids[10:]-ids[:-10]==10
   partial=np.convolve(((a[:,2]>.01)&(a[:,2]<.99)).astype(int),np.ones(10,dtype=int),'valid')[:-1]==10
   coast=np.convolve(((a[:,1]<=.01)&(a[:,2]<=.01)).astype(int),np.ones(10,dtype=int),'valid')[:-1]==10
   moving=np.lib.stride_tricks.sliding_window_view(v,11).min(1)>=2
   turns=np.lib.stride_tricks.sliding_window_view(abs(a[:,0]),10).max(1)[:-1]>=.2
   row=dict(driver=driver,episode=ep,selected=(driver,ep) in chosen,frames=len(ids),
    low_speed_partial=int((valid&partial&(v[:-10]>=2)&(v[:-10]<5)).sum()),
    middle_speed_partial=int((valid&partial&(v[:-10]>=5)&(v[:-10]<10)).sum()),
    turning_partial=int((valid&partial&turns&(v[:-10]>=2)).sum()),
    fast_turning_coast=int((valid&coast&moving&turns&(v[:-10]>=10)).sum()))
   rows.append(row)
  print('Rare coverage inspected',driver,flush=True)
 with (HERE/'results/full_release_rare_coverage.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print({k:sum(r[k] for r in rows) for k in ['low_speed_partial','middle_speed_partial','turning_partial','fast_turning_coast']},flush=True)
if __name__=='__main__':main()
