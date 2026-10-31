"""Offline six-camera integrity, contact sheets, CPU decoding and memory benchmarks."""
from pathlib import Path
from collections import defaultdict
import argparse,csv,gc,hashlib,json,multiprocessing as mp,os,resource,sys,threading,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from torch.utils.data import DataLoader,Subset
from src.unified_driving_dataset import CAMERA_ORDER,UnifiedDrivingDataset,load_episode_records,numeric_record_fingerprint
OUT=ROOT/'docs/multicamera'

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
def csvfile(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])*1024

def bench_child(records,keys,offsets,cameras,batch_size,connection):
    try:
        torch.set_num_threads(1);dataset=UnifiedDrivingDataset(records,episode_ids=keys,action_offsets=offsets,cameras=cameras,require_calibration=True)
        indices=np.random.default_rng(42).choice(len(dataset),batch_size,replace=False).tolist()
        gc.collect();baseline=rss();peak=[baseline];stop=threading.Event()
        def monitor():
            while not stop.wait(.005):peak[0]=max(peak[0],rss())
        monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start();start=time.perf_counter()
        batch=next(iter(DataLoader(Subset(dataset,indices),batch_size=batch_size,num_workers=0)))
        seconds=time.perf_counter()-start;peak[0]=max(peak[0],rss());stop.set();monitor_thread.join()
        tensor_bytes=sum(v.numel()*v.element_size() for v in batch.values() if isinstance(v,torch.Tensor))
        connection.send(dict(batch_size=batch_size,cameras=len(cameras),num_workers=0,seconds=seconds,seconds_per_window=seconds/batch_size,
            baseline_rss_bytes=baseline,peak_rss_bytes=peak[0],incremental_peak_rss_bytes=peak[0]-baseline,batch_tensor_bytes=tensor_bytes,
            past_shape=list(batch['past_images'].shape),future_shape=list(batch['future_images'].shape)))
    except Exception as exc:connection.send(dict(error=repr(exc)))
    finally:connection.close()

def sheet(key,r,index):
    ids=r['frame_ids'];frame=int(ids[index]);canvas=Image.new('RGB',(1440,630),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    draw.text((10,5),key,fill='black',font=font);draw.text((10,29),f'Recorded frame {frame}; order 1..6, capture timestamp unavailable',fill='black',font=font)
    for j,camera in enumerate(CAMERA_ORDER):
        path=r['camera_paths'][camera][index]
        with Image.open(path) as image:thumbnail=image.convert('RGB').resize((480,270))
        x=(j%3)*480;y=60+(j//3)*285;canvas.paste(thumbnail,(x,y));draw.rectangle((x,y,x+479,y+25),fill='black');draw.text((x+6,y+2),f'{j+1}: {camera}',fill='white',font=font)
    filename=r['source']+'_contact_sheet.jpg';canvas.save(OUT/filename,quality=92);return dict(episode=key,frame=frame,file=filename)

def verify_person_path_references():
    from src.numeric_metadata import load
    selection=json.loads((ROOT/'data/config/person_selection.json').read_text())
    selected={(e['driver'],e['episode']) for e in selection['selected']};checked=0;mismatches=[]
    for driver in selection['drivers']:
        for row in load(ROOT/f'data/person2drive/metadata/sensor_data/{driver}/b2d_infos_train.pkl'):
            if (driver,row['folder']) not in selected:continue
            front=Path(row['sensors']['CAM_FRONT']['data_path'])
            for camera in CAMERA_ORDER:
                sensor='CAM_'+camera.removeprefix('rgb_').upper();path=Path(row['sensors'][sensor]['data_path']);checked+=1
                expected=Path(row['folder'])/'camera'/camera/front.name
                if path!=expected or not path.stem.isdigit() or int(path.stem)!=int(row['frame_idx']):mismatches.append([driver,str(path),str(expected),int(row['frame_idx'])])
    result=dict(references_checked=checked,mismatches=mismatches)
    save('person_path_references.json',result)
    if mismatches:raise ValueError('Explicit per-camera metadata references disagree; do not infer substitute paths')
    print('Verified explicit Person camera path references',checked,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--paths-only',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    verify_person_path_references()
    if args.paths_only:return
    # Read raw source records only once; no visual features or model weights.
    records=load_episode_records();splits=json.loads((ROOT/'data/config/splits.json').read_text())['episodes'];offsets=json.loads((ROOT/'data/config/action_alignment.json').read_text())['action_offsets']
    expected=json.loads((ROOT/'data/config/verification.json').read_text())
    assert numeric_record_fingerprint(records,splits)==expected['data_fingerprint'],'Numerical corpus or assignments changed'
    identities={p:{(key,int(i)) for key in keys for i in records[key]['frame_ids']} for p,keys in splits.items()}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not identities[a]&identities[b]
    drivers={p:{records[k]['driver'] for k in keys if records[k]['source']=='person2drive'} for p,keys in splits.items()}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not drivers[a]&drivers[b]
    # Route identity is town/route, independent of weather/scenario.
    import re
    routes={p:{re.search(r'(Town\d+(?:HD)?)_Route(\d+)',records[k]['episode']).groups() for k in keys if records[k]['source']=='bench2drive'} for p,keys in splits.items()}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:assert not routes[a]&routes[b]
    counts=[];batches=[]
    for part,keys in splits.items():
        front=UnifiedDrivingDataset(records,episode_ids=keys,action_offsets=offsets,cameras=('rgb_front',),require_calibration=True)
        six=UnifiedDrivingDataset(records,episode_ids=keys,action_offsets=offsets,require_calibration=True)
        expected_windows=sum(x['windows'] for x in expected['counts'][part].values());assert len(front)==expected_windows
        counts.append(dict(split=part,front_windows=len(front),six_camera_windows=len(six),missing_view_windows=six.rejected_windows_missing_cameras,missing_calibration_windows=six.rejected_windows_missing_calibration))
        sample_indices=[next(i for i,(key,_) in enumerate(six.samples) if records[key]['source']==source) for source in ['bench2drive','person2drive']]
        batch=next(iter(DataLoader(Subset(six,sample_indices),batch_size=2,num_workers=0)))
        assert batch['past_images'].shape==(2,4,6,3,224,224);assert batch['future_images'].shape==(2,10,6,3,224,224)
        assert batch['camera_geometry_valid'].all();assert torch.isfinite(batch['camera_to_ego']).all()
        for i in sample_indices:
            a=six.numeric_sample(i);key,start=six.samples[i];j=front.samples.index((key,start));b=front.numeric_sample(j)
            for field in ['past_actions','future_actions','past_ego_observations','future_motion_targets']:torch.testing.assert_close(a[field],b[field])
        batches.append(dict(split=part,shapes={k:list(v.shape) for k,v in batch.items() if isinstance(v,torch.Tensor)}));del batch
        print('Validated split',part,len(six),'six-camera windows',flush=True)
    csvfile('windows.csv',counts);save('batches.json',batches)
    contacts=[];decode=[];cadence=[]
    for source in ['bench2drive','person2drive']:
        key=next(k for k in splits['train'] if records[k]['source']==source);r=records[key];moving=np.flatnonzero(r['speed']>5);index=int(moving[len(moving)//2]) if len(moving) else len(r['frame_ids'])//2
        contacts.append(sheet(key,r,index))
        for n in range(5):
            frame=min(index+n,len(r['frame_ids'])-1);start=time.perf_counter()
            for c in CAMERA_ORDER:
                with Image.open(r['camera_paths'][c][frame]) as image:image.convert('RGB').load()
            seconds=time.perf_counter()-start;decode.append(dict(source=source,frame=int(r['frame_ids'][frame]),camera_count=6,decode_seconds=seconds))
        intervals=[]
        for rr in records.values():
            if rr['source']!=source:continue
            speed=(rr['speed'][1:]+rr['speed'][:-1])/2;mask=(np.diff(rr['frame_ids'])==1)&(speed>3)
            intervals.extend((np.linalg.norm(np.diff(rr['positions'][:,:2],axis=0)[mask],axis=1)/speed[mask]).tolist())
        cadence.append(dict(source=source,median_displacement_over_speed_seconds=float(np.median(intervals)),note='Kinematic cadence check, not capture timestamps or camera skew'))
    save('contact_sheets.json',contacts);csvfile('decode_times.csv',decode);save('cadence.json',cadence)
    memory=[];context=mp.get_context('fork')
    for source in ['bench2drive','person2drive']:
        keys=[k for k in splits['train'] if records[k]['source']==source]
        for cameras in [('rgb_front',),CAMERA_ORDER]:
            for batch_size in [1,2,4,8]:
                receive,send=context.Pipe(False);process=context.Process(target=bench_child,args=(records,keys,offsets,cameras,batch_size,send));process.start();send.close()
                result=receive.recv();process.join();receive.close()
                if process.exitcode or 'error' in result:raise RuntimeError(result)
                result['source']=source;memory.append(result);print('Measured',source,len(cameras),'cameras batch',batch_size,flush=True)
    save('loader_benchmarks.json',memory)
    save('validation.json',dict(numeric_fingerprint=expected['data_fingerprint'],split_integrity=True,person_driver_disjoint=True,bench_route_disjoint=True,
        camera_order=CAMERA_ORDER,counts=counts,action_offsets=offsets,no_training_or_features=True))
    print('MULTICAMERA VALIDATION COMPLETE',flush=True)
if __name__=='__main__':main()
