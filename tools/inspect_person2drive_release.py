"""Inspect all 50 drivers' numeric metadata, without downloading their sensor archives."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import json
import numpy as np
from prepare_person2drive import BASE,download,descriptor
from src.numeric_metadata import load
from download_bench2drive import save_json


def main():
    catalog=json.loads((BASE/'manifests/catalog.json').read_text())
    entries=[f for f in catalog['files'] if f['path'].endswith('/b2d_infos_train.pkl')]
    def fetch(e):
        download(e,BASE/'metadata'/e['path'],catalog['revision'])
        print('Metadata verified',e['path'],flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(fetch,entries))
    rows=[]
    for entry in entries:
        driver=entry['path'].split('/')[1];data=load(BASE/'metadata'/entry['path']);groups=defaultdict(list)
        for row in data:groups[row['folder']].append(row)
        for folder,group in groups.items():
            group.sort(key=lambda r:r['frame_idx'])
            ids=np.array([r['frame_idx'] for r in group]);v=np.array([r['ego_vel'][0] for r in group]);a=np.array([[r[c] for c in ['steer','throttle','brake']] for r in group])
            rows.append(dict(driver=driver,episode=folder,**descriptor(ids,v,a)))
        print('Inspected',driver,len(groups),'episodes',flush=True)
    save_json(BASE/'manifests/full_release_inspection.json',dict(revision=catalog['revision'],drivers=len(entries),episodes=len(rows),
        referenced_frames=sum(r['frames'] for r in rows),records=rows,note='Telemetry metadata only; RGB existence verified only for downloaded recordings'))


if __name__=='__main__':main()
