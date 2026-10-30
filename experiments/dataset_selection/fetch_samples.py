"""Fetch only the fixed, small audit selection; never snapshot a whole dataset."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import urllib.request

from huggingface_hub import HfApi, hf_hub_download

HERE=Path(__file__).resolve().parent
REPOS={'TeleopWM-Dataset':'bimilab/TeleopWM-Dataset','Person2Drive':'dongxr7/Person2Drive','PDD':'tasl-lab/PDD'}


def main():
    api=HfApi();manifest=[]
    selections={name:[] for name in REPOS}
    episodes=['train/Town01/0000','train/Town01/0001','train/Town03/0000','train/Town03/0001',
              'train/Town04/0000','train/Town04/0001','val/Town02/0000','test/Town05/0000']
    selections['TeleopWM-Dataset']=[p+'/pd_dataframe.pkl' for p in episodes]
    for driver in ['driver01','driver50']:
        selections['Person2Drive'].append(f'sensor_data/{driver}/b2d_infos_train.pkl')
        for town in ['Town04','Town05']:
            for drive in [1,5]:
                for file in ['metric_info.json','result.json','route.json']:
                    selections['Person2Drive'].append(f'driving_logs/{driver}/{town}_drive_{drive}/{file}')
    selections['Person2Drive'].append('sensor_data/driver01/Town04_drive_1.tar.zst')
    for driver in ['driver_01','driver_10','driver_30']:
        for scenario in ['Accident','ControlLoss','VanillaNonSignalizedTurn','NonSignalizedJunctionLeftTurn']:
            prefix=f'{driver}/data/{scenario}'
            selections['PDD'] += [prefix+'/metric/metric_info.json',prefix+'/metric/metrics.json']
            selections['PDD'] += [prefix+f'/images/{i}.jpg' for i in range(16)]
    selections['TeleopWM-Dataset'] += [f'train/Town01/0000/image/image_{i:09}.png' for i in range(100,119)]
    for name,paths in selections.items():
        source=json.loads((HERE/'sources/dataset_revisions.json').read_text())[name]
        revision=source['revision']
        for start in range(0,len(paths),80):
            infos=api.get_paths_info(REPOS[name],paths[start:start+80],repo_type='dataset',revision=revision)
            by_path={p.path:p for p in infos}
            for path in paths[start:start+80]:
                if path not in by_path:raise ValueError(f'Missing sample {name}/{path}')
                info=by_path[path]
                manifest.append(dict(dataset=name,repo=REPOS[name],revision=revision,path=path,size=info.size,
                                     expected_sha256=info.lfs.sha256 if info.lfs else None,git_blob_id=info.blob_id))
    total=sum(e['size'] for e in manifest)
    print(f'Fixed sample plan: {len(manifest)} files, {total/2**20:.1f} MiB',flush=True)
    if total>1200*1024**2:raise ValueError('Audit sample exceeds fixed 1,200 MiB budget; review before increasing')
    (HERE/'sample_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def fetch(entry):
        path=Path(hf_hub_download(entry['repo'],entry['path'],repo_type='dataset',revision=entry['revision'],
                                  local_dir=HERE/'samples'/entry['dataset']))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size!=entry['size'] or (entry['expected_sha256'] and digest!=entry['expected_sha256']):
            raise ValueError(f'Bad sample checksum: {path}')
        entry['local_sha256']=digest
        return entry
    with ThreadPoolExecutor(max_workers=6) as pool:
        manifest=list(pool.map(fetch,manifest))
    (HERE/'sample_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Verified audit samples downloaded.',flush=True)


if __name__=='__main__':main()
