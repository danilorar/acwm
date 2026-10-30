"""Small source-independent alignment/rollout check, with locked held-out tests."""
from pathlib import Path
import os,sys,json,csv,hashlib
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT));os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.matplotlib'))
from src.unified_driving_dataset import load_episode_records,UnifiedDrivingDataset,numeric_record_fingerprint,nominal_controls
# Reuse the historical compact architecture and sign-constrained physics fit,
# without changing or running the historical experiment's entry point.
from experiments.dynamics_validation.run import Dynamics,fit_physics,physics_rollout
import matplotlib.pyplot as plt


def write(name,rows):
    with (HERE/'results'/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    records=load_episode_records(require_images=False)
    split=json.loads((HERE/'results/splits.json').read_text())['episodes']
    ds=UnifiedDrivingDataset(records,action_offsets={'bench2drive':0,'person2drive':0})
    v,a,y,names=ds.dynamics_arrays()
    all_metrics=[];ordering=[];predictions={};config={};training=[]
    sources=np.array([records[n]['source'] for n in names])
    for source in ['bench2drive','person2drive']:
        masks={s:(sources==source)&np.isin(names,keys) for s,keys in split.items()}
        # Same capped, episode-balanced training targets for both offsets/seeds.
        groups=sorted(set(names[masks['train']]))
        rng=np.random.default_rng(42);quota=max(1,12000//len(groups));indices=[]
        for name in groups:
            candidates=np.flatnonzero(names==name)
            indices.extend(rng.choice(candidates,min(quota,len(candidates)),replace=False))
        train_idx=np.array(sorted(indices));pool=np.concatenate([a[train_idx,:13].reshape(-1,3),a[train_idx,1:].reshape(-1,3)])
        mean=np.r_[v[train_idx].mean(),pool.mean(0)].astype(np.float32)
        std=np.r_[v[train_idx].std(),pool.std(0)].astype(np.float32);std[std<1e-6]=1
        mt,st=torch.tensor(mean),torch.tensor(std)
        unique=[]
        for name in split['train']:
            if records[name]['source']!=source:continue
            r=records[name];valid=(np.diff(r['frame_ids'])==1)&nominal_controls(r['actions'])[:-1]&nominal_controls(r['actions'])[1:]
            unique.extend(np.c_[r['speed'][:-1],r['speed'][1:],r['actions'][:-1,1:],r['actions'][1:,1:]][valid])
        unique=np.array(unique)
        acceleration=np.diff(np.c_[v[train_idx,-1],y[train_idx]],axis=1)/.1
        config[source]=dict(training_windows_available=int(masks['train'].sum()),training_windows_used=len(train_idx),mean=mean.tolist(),std=std.tolist(),
                            training_speed_max=float(np.max(y[train_idx])),training_abs_accel_995=float(np.quantile(abs(acceleration),.995)))
        source_pred={s:{} for s in ['validation','test']}
        for offset in [0,1]:
            hist=np.zeros((len(v),4,5),np.float32);hist[:,:,0]=v
            hist[:,1:,1:4]=a[:,offset:offset+3];hist[:,1:,4]=1
            hist[:,:,:4]=(hist[:,:,:4]-mean)/std;hist[:,0,1:4]=0
            act=(a[:,3+offset:13+offset]-mean[1:])/std[1:]
            ht,at=torch.from_numpy(hist),torch.from_numpy(act)
            coef=fit_physics(unique,'H'+str(offset));config[source]['physics_H'+str(offset)]=coef.tolist()
            for seed in [0,1]:
                torch.manual_seed(seed);net=Dynamics();optimizer=torch.optim.Adam(net.parameters(),lr=.003)
                for epoch in range(60):
                    optimizer.zero_grad();pred=net(ht[train_idx],at[train_idx],mt,st)
                    loss=torch.nn.functional.mse_loss(pred,torch.tensor(y[train_idx]))
                    loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step()
                training.append(dict(source=source,offset=offset,seed=seed,last_train_mse=float(loss.detach())))
                net.eval();torch.save(net.state_dict(),HERE/'results'/f'{source}_H{offset}_seed{seed}.pt')
                for part in ['validation','test']:
                    idx=np.flatnonzero(masks[part]);out=[]
                    with torch.no_grad():
                        for begin in range(0,len(idx),2048):
                            batch=idx[begin:begin+2048];out.append(net(ht[batch],at[batch],mt,st).numpy())
                    source_pred[part][f'GRU_H{offset}_seed{seed}']=np.concatenate(out)
                idx=np.flatnonzero(masks['test']&(v[:,-1]>=2)&(v[:,-1]<=15))
                idx=np.random.default_rng(42).choice(idx,min(1000,len(idx)),replace=False)
                candidates={}
                with torch.no_grad():
                    for label,command in [('throttle',[0,1,0]),('coast',[0,0,0]),('partial',[0,0,.3]),('brake',[0,0,1])]:
                        raw=np.tile(command,(len(idx),10,1)).astype(np.float32)
                        candidates[label]=net(ht[idx],torch.tensor((raw-mean[1:])/std[1:]),mt,st).numpy()
                final=[candidates[k][:,-1] for k in ['brake','partial','coast','throttle']]
                failure=np.logical_or.reduce([first>second+.05 for first,second in zip(final,final[1:])])
                ordering.append(dict(source=source,offset=offset,seed=seed,cases=len(idx),ordering_failures=int(failure.sum()),failure_fraction=float(failure.mean()),
                                     mean_throttle_delta=float((candidates['throttle'][:,-1]-v[idx,-1]).mean()),
                                     mean_coast_delta=float((candidates['coast'][:,-1]-v[idx,-1]).mean()),
                                     mean_partial_delta=float((candidates['partial'][:,-1]-v[idx,-1]).mean()),
                                     mean_brake_delta=float((candidates['brake'][:,-1]-v[idx,-1]).mean())))
                print(source,'H'+str(offset),'seed',seed,'trained and evaluated',flush=True)
            for part in ['validation','test']:
                idx=np.flatnonzero(masks[part]);source_pred[part]['physics_H'+str(offset)]=physics_rollout(v[idx,-1],a[idx,3+offset:13+offset],coef)
        for part,outs in source_pred.items():
            idx=np.flatnonzero(masks[part]);target=y[idx];current=v[idx,-1]
            outs['constant_speed']=np.repeat(current[:,None],10,axis=1)
            outs['constant_accel']=np.maximum(0,current[:,None]+(v[idx,-1]-v[idx,-2])[:,None]*np.arange(1,11))
            raw=a[idx,3:14];state=np.c_[current,target]
            subsets={'all':np.ones(len(idx),bool),
                     'coasting':np.all((raw[:,:,1]<=.01)&(raw[:,:,2]<=.01),axis=1)&np.all(state>=2,axis=1),
                     'partial_braking':np.all((raw[:,:,2]>.01)&(raw[:,:,2]<.99),axis=1)&(current>=2),
                     'strong_braking':np.any(raw[:,:,2]>=.8,axis=1)&(current>=2),
                     'strong_deceleration':np.any(np.diff(state,axis=1)<=-.5,axis=1)&(current>=2)}
            for model,pred in outs.items():
                if not np.isfinite(pred).all():raise ValueError('Nonfinite rollout')
                for scope in ['ALL',*sorted(set(names[idx]))]:
                    for subset,mask in subsets.items():
                        selected=mask&((names[idx]==scope) if scope!='ALL' else True)
                        if not selected.any():continue
                        for step in [0,1,5,10]:
                            error=(pred-target)[selected]
                            if step:error=error[:,step-1]
                            all_metrics.append(dict(source=source,split=part,model=model,episode=scope,subset=subset,step=step,n=int(selected.sum()),
                                                    mae=float(abs(error).mean()),rmse=float(np.sqrt((error**2).mean())),
                                                    above_train_max_plus5_fraction=float((pred[selected]>config[source]['training_speed_max']+5).mean())))
            np.savez_compressed(HERE/'results'/f'{source}_{part}_rollouts.npz',target=target,current=current,episode=names[idx],
                                **outs,**{f'mask_{k}':val for k,val in subsets.items()})
        predictions[source]=source_pred
    write('dynamics_metrics.csv',all_metrics);write('action_ordering.csv',ordering);write('training.csv',training)
    offsets={}
    for source in ['bench2drive','person2drive']:
        losses={o:np.mean([r['rmse'] for r in all_metrics if r['source']==source and r['split']=='validation' and r['episode']=='ALL' and r['subset']=='all' and r['step']==0 and r['model'].startswith(f'GRU_H{o}_')]) for o in [0,1]}
        offsets[source]=min(losses,key=losses.get)
        config[source]['validation_alignment_rmse']={str(k):float(v) for k,v in losses.items()}
    (HERE/'results/action_alignment.json').write_text(json.dumps(dict(action_offsets=offsets,
        selection_rule='Mean validation full-horizon RMSE across two seeds; test never selects offsets',
        uncertainty='Predictive choice only; actual collection-time control timing remains unverified'),indent=2)+'\n')
    (HERE/'results/dynamics_config.json').write_text(json.dumps(dict(quality_policy='Exclude all 14-frame windows touching out-of-nominal controls; raw labels preserved',data_fingerprint=numeric_record_fingerprint(records,split),epochs=60,seeds=[0,1],lr=.003,train_cap_per_source=12000,device='cpu',
        source_config=config,model_source='experiments/dynamics_validation/run.py:Dynamics (unchanged)',
        evaluation_subsets='Coast/partial require all 11 control labels to agree, so H0/H1 share exactly the same target subsets'),indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,source in zip(axes,['bench2drive','person2drive']):
        data=np.load(HERE/'results'/f'{source}_test_rollouts.npz')
        for model in ['constant_speed','constant_accel','physics_H'+str(offsets[source]),'GRU_H0','GRU_H1']:
            keys=[model+'_seed0',model+'_seed1'] if model.startswith('GRU') else [model]
            errors=[np.sqrt(((data[key]-data['target'])**2).mean(0)) for key in keys]
            ax.plot(np.arange(1,11)*.1,np.mean(errors,axis=0),label=model)
        ax.set(title=source,xlabel='Horizon (s)',ylabel='Held-out speed RMSE (m/s)');ax.legend(fontsize=7);ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(HERE/'results/dynamics_horizons.png',dpi=160);plt.close(fig)
    print('Validation-chosen offsets:',offsets,flush=True)


if __name__=='__main__':main()
