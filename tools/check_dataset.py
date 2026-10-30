"""Offline check that the organized corpus matches the frozen training foundation."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch.utils.data import DataLoader,Subset
from src.unified_driving_dataset import load_training_datasets,numeric_record_fingerprint

def main():
    torch.set_num_threads(1)
    datasets=load_training_datasets()
    config=ROOT/'data/config'
    expected=json.loads((config/'verification.json').read_text())
    splits=json.loads((config/'splits.json').read_text())['episodes']
    records={key:r for ds in datasets.values() for key,r in ds.records.items()}
    assert numeric_record_fingerprint(records,splits)==expected['data_fingerprint'],'Numeric corpus or splits changed'
    for part,ds in datasets.items():
        count=sum(row['windows'] for row in expected['counts'][part].values())
        assert len(ds)==count,(part,len(ds),count)
        samples=[next(i for i,(key,_) in enumerate(ds.samples) if ds.records[key]['source']==source) for source in ['bench2drive','person2drive']]
        batch=next(iter(DataLoader(Subset(ds,samples),batch_size=2,num_workers=0)))
        assert batch['past_images'].shape==(2,4,3,224,224)
        assert batch['future_images'].shape==(2,10,3,224,224)
        assert batch['past_actions'].shape==(2,3,3)
        assert batch['future_actions'].shape==(2,10,3)
        assert batch['future_motion_targets'].shape==(2,10,5)
        print(part,len(ds),'windows; mixed-source RGB batch passed',flush=True)
    print('Frozen corpus fingerprint, image paths and all three split views passed.')

if __name__=='__main__':main()
