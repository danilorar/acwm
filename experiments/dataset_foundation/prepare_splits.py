"""Create new three-way splits without changing historical assignments."""
from pathlib import Path
import json
import sys
import hashlib
import numpy as np

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT))
from src.unified_driving_dataset import load_episode_records,nominal_controls
from src.bench2drive_dataset import route_group


def make_splits(records,seed=42):
    old_path=PROJECT/'data/bench2drive_train/manifests/splits.json'
    old=json.loads(old_path.read_text());held=sorted(old['validation'])
    order=np.random.default_rng(seed).permutation(held).tolist()
    result={'train':['bench2drive/'+n for n in old['train']],
            'validation':['bench2drive/'+n for n in order[:len(order)//2]],
            'test':['bench2drive/'+n for n in order[len(order)//2:]]}
    catalog=json.loads((PROJECT/'data/person2drive_train/manifests/catalog.json').read_text())
    drivers=sorted(catalog['drivers'])
    order=np.random.default_rng(seed).permutation(drivers).tolist()
    assignments={'train':order[:-4],'validation':order[-4:-2],'test':order[-2:]}
    selection=json.loads((PROJECT/'data/person2drive_train/manifests/selection.json').read_text())
    for driver,part in selection.get('supplement',{}).get('extra_driver_splits',{}).items():
        assignments[part].append(driver)
    for split in result:
        result[split]+=sorted(k for k,r in records.items() if r['source']=='person2drive' and r['driver'] in assignments[split])
    flat=sum(result.values(),[])
    if len(flat)!=len(set(flat)) or set(flat)!=set(records):raise ValueError('Incomplete/overlapping episode splits')
    bench_routes={s:{route_group(records[k]['episode']) for k in keys if records[k]['source']=='bench2drive'} for s,keys in result.items()}
    for i,a in enumerate(result):
        for b in list(result)[i+1:]:
            if bench_routes[a]&bench_routes[b] or set(assignments[a])&set(assignments[b]):raise ValueError('Route/driver leakage')
    counts={}
    for split,keys in result.items():
        counts[split]={}
        for source in ['bench2drive','person2drive']:
            chosen=[records[k] for k in keys if records[k]['source']==source]
            count=sum(sum(r['frame_ids'][i+13]-r['frame_ids'][i]==13 and nominal_controls(r['actions'][i:i+14]).all() for i in range(len(r['frame_ids'])-13)) for r in chosen)
            counts[split][source]=dict(episodes=len(chosen),frames=sum(len(r['frame_ids']) for r in chosen),windows=int(count))
    return dict(seed=seed,episodes=result,person_driver_assignments=assignments,counts=counts,
                bench_original_split_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
                policy='Bench original train retained; old holdout divided into validation/test. Person base drivers 8/2/2 preserved; rare supplement adds 2 train and 1 validation driver, giving 10/3/2; routes repeat across splits.')


if __name__=='__main__':
    records=load_episode_records(require_images=False)
    result=make_splits(records);(HERE/'results').mkdir(exist_ok=True)
    (HERE/'results/splits.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(drivers=result['person_driver_assignments'],counts=result['counts']),indent=2))
