"""Write the final report only after image verification has completed."""
from pathlib import Path
import json,hashlib,subprocess,shutil
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'

def table(headers,rows):
 return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(map(str,row))+' |' for row in rows]])
def main():
 verification=json.loads((OUT/'verification.json').read_text());counts=verification['counts'];cov=pd.read_csv(OUT/'coverage_summary.csv');metrics=pd.read_csv(OUT/'dynamics_metrics.csv');offsets=json.loads((OUT/'action_alignment.json').read_text())['action_offsets'];config=json.loads((OUT/'dynamics_config.json').read_text())
 storage={}
 for name,path in [('bench2drive',ROOT/'data/bench2drive_train'),('person_archives',ROOT/'data/person2drive_train/archives'),('person_metadata',ROOT/'data/person2drive_train/metadata'),('person_front_rgb',ROOT/'data/person2drive_train/extracted'),('person_manifests',ROOT/'data/person2drive_train/manifests')]:
  files=[p for p in path.rglob('*') if p.is_file()];storage[name]=dict(logical_bytes=sum(p.stat().st_size for p in files),allocated_bytes=int(subprocess.check_output(['du','-sB1',str(path)]).split()[0]),files=len(files))
 storage['free_bytes']=shutil.disk_usage(ROOT).free
 cache=Path.home()/'.cache/huggingface'
 if cache.exists():storage['global_huggingface_cache_allocated_bytes']=int(subprocess.check_output(['du','-sB1',str(cache)]).split()[0])
 storage['note']='Per-tree storage includes retained source archives and all 50 metadata files. The global Hugging Face cache may include pre-existing unrelated files; it is not counted as dataset storage. Historical samples may share hardlinks with this tree.'
 (OUT/'storage.json').write_text(json.dumps(storage,indent=2)+'\n')
 scripts=[ROOT/'src/unified_driving_dataset.py',ROOT/'src/numeric_metadata.py',ROOT/'experiments/dynamics_validation/run.py',*HERE.glob('*.py'),*HERE.glob('manifests/*.json'),OUT/'splits.json']
 (OUT/'provenance.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},indent=2)+'\n')
 lines=['# Dataset foundation: final assessment','',
  'The fixed cohort is ready for DINOv3 extraction and a bounded **observational world-model prototype**. It is not sufficient evidence for a large final model intended to rank counterfactual plans or drive MPC. Resolve those requirements with intervention tests and targeted rare-control collection before committing to that use. No visual feature extraction or visual-model training was started in this experiment.','',
  '## Inventory and integrity','']
 rows=[]
 for source in ['bench2drive','person2drive']:
  rows.append([source,*[f"{sum(c[source][key] for c in counts.values()):,}" for key in ['episodes','matched_frames','frames_participating_in_windows','windows']]])
 lines += [table(['Source','Episodes','Matched frames','Frames in valid K4/H10 windows','Windows'],rows),'',
  f"Verified {verification['jpeg_headers_verified']:,} indexed JPEG structures and {verification['representative_full_decode_windows']:,} representative decoded windows. No source-frame identities, overlapping windows, or exact JPEG hashes cross splits. Every indexed image exists; all numeric values are finite and pose rotations are orthonormal. Default samples exclude windows touching 61 Person steering labels outside the nominal [-1,1] range (observed raw range -1.1987 to +1.9646). Those raw rows are preserved and flagged; applied clipping/scaling is unknown, so no conversion is guessed. Full JPEG decoding was sampled at three windows per episode, rather than every image. Bench has 3,154 negative near-zero signed speeds (minimum −0.06124 m/s; none below −0.1 m/s). Raw values are preserved and included in losses; small rollback versus simulator jitter is not resolved. The baseline zero clamp cannot reproduce those small negatives.",'',
  '## Splits','']
 rows=[]
 for part,c in counts.items():rows.append([part,*[f"{sum(r[key] for r in c.values()):,}" for key in ['episodes','matched_frames','frames_participating_in_windows','windows']]])
 lines += [table(['Split','Episodes','Matched frames','Frames in windows','Windows'],rows),'',
  'Bench2Drive: 88/22/22 route-disjoint episodes, retaining all original training routes. Person2Drive: 52/14/14 episodes from 10/3/2 disjoint drivers. Test drivers remain driver05 and driver39. Both physical Person2Drive routes repeat across splits, so this evaluates new drivers on familiar environments. Bench test routes were part of a historical validation set; this is not an untouched prospective benchmark. Coverage-curated held-out episodes are not a random population sample.','',
  '## Storage','']
 rows=[[name,f"{value['allocated_bytes']/1024**3:.2f}",f"{value['files']:,}"] for name,value in storage.items() if isinstance(value,dict)]
 lines += [table(['Tree','Allocated GiB','Files'],rows),'',f"Dataset trees above total {sum(v['allocated_bytes'] for v in storage.values() if isinstance(v,dict))/1024**3:.2f} GiB; free disk: {storage['free_bytes']/1024**3:.2f} GiB. Source archives are retained for resumption. No LiDAR or other camera views were extracted. Hugging Face cache accounting is separate in storage.json; prior data are preserved.",'',
  '## Coverage','']
 rows=[]
 for _,r in cov.iterrows():rows.append([r.source,f'{100*r.coast_fraction_moving:.2f}%',f'{100*r.partial_frames/r.frames:.2f}%',int(r.partial_above5),f'{100*r.partial_fraction_above5:.2f}%',f'{int(r.one_sec_coast_H0):,} / {int(r.one_sec_coast_H1):,}',f'{int(r.one_sec_partial_H0):,} / {int(r.one_sec_partial_H1):,}',f'{int(r.multi_transition_windows_H0):,}'])
 lines += [table(['Source','Coast / moving frames','Partial / all frames','Partial brake frames >5 m/s','Partial / frames >5 m/s','1s coast H0/H1','1s partial H0/H1','Multi-transition H0 windows'],rows),'',
  'Moving means speed >=2 m/s; pedals <=0.01 define coasting; 0.01<brake<0.99 defines partial braking. Intermediate-braking counts above 5 m/s are frame-level, not independent events. Longest selected moving coast is 65 frames (6.5 seconds by frame count; 6.4-second observation span); partial-braking runs reach 39 frames (3.9/3.8 seconds).','',
  'Person moving-coasting and partial-braking percentages are lower than in the earlier two-driver sample; broadening the cohort changes its proportions even while increasing absolute support. The old sample was not a population estimate. Explicit maneuver weighting is still needed. All 50 drivers and 800 sensor episodes were inspected through their image-associated numeric metadata (1,175,820 referenced frames). Sensor archives were downloaded only for the selected 80 episodes. The final 15-driver subset combines a balanced 12-driver primary cohort and eight rare-control supplements. Person has no matching curated hazard-type annotation in the consumed metadata, so its scenario analysis uses routes and control-event proxies. Bench contributes 44 scenario types, 12 towns and 23 weather settings; Person contributes human controls on routes 103/108, not 80 distinct physical routes.','',
  'The complete release contains only 14 H0 one-second partial-braking windows starting at 2–5 m/s and 21 partial-braking windows with strong steering; all are present in the selected raw metadata. Some raw windows may be excluded by the conservative action-range gate; consult training_window_coverage.csv for exact eligible training sequences. They overlap heavily and come from very few episodes/drivers. No low-speed sustained partial-braking test windows are available: the two supporting drivers are in train/validation. Only 20 of 75 fast-coasting/strong-turning windows are selected. Bench has no partial braking and no one-second moving-coasting windows. High-speed strong steering, long combined turning/braking, varied intervention sequences and new Person routes remain poorly covered. More random frames from these same routes will not resolve those structural gaps. The 117,798 selected Person metadata rows have minimum speed 0.2001 m/s: exact standstill/terminal-stop states are absent from this metadata subset. The reason for that cutoff is not established. Bench supplies stopped states, but does not establish human partial-braking-to-stop dynamics. Additional front images without image-linked metadata are retained but excluded; raw 20-Hz logs cannot safely fill them by ordinal joining.','',
  'Plots: [speed-conditioned summaries](results/speed_conditioned.png), [conditional control heatmaps](results/speed_control_heatmaps.png), [raw controls](results/control_distributions.png), [continuous durations](results/action_durations.png). Counts by speed/steering/regime and episode are in the coverage CSVs; zero cells are retained.','',
  '## Dynamics sanity check','',
  'All compared models use the same conservative control-quality gate, and no raw control is clipped. Separate source-specific GRUs use identical H0/H1 samples/initializations and training-only normalization. Two seeds, 60 updates, and at most 12,000 episode-balanced training windows per source keep this diagnostic small. The first predicted step measures one-step accuracy from the same model trained for free-running ten-step prediction. No future ground-truth speed is used as decoder input. Models were not pooled across different source physics.','']
 rows=[]
 test=metrics[(metrics.split=='test')&(metrics.episode=='ALL')&(metrics.subset=='all')]
 for source in ['bench2drive','person2drive']:
  for model in ['constant_speed','constant_accel','physics_H'+str(offsets[source]),'GRU_H'+str(offsets[source])]:
   group=test[(test.source==source)&test.model.str.startswith(model)];vals=[]
   for step,metric in [(1,'rmse'),(5,'rmse'),(10,'mae'),(10,'rmse'),(0,'rmse')]:
    x=group[group.step==step][metric];vals.append(f'{x.mean():.3f}'+(f' ± {x.std(ddof=1):.3f}' if len(x)>1 else ''))
   rows.append([source,model,*vals])
 lines += [table(['Source','Model','0.1s RMSE','0.5s RMSE','1s MAE','1s RMSE','Pooled horizon RMSE'],rows),'',
  'All errors are speed m/s. ± denotes sample standard deviation across two seeds, not a confidence interval across episodes/drivers. Constant acceleration extrapolates the last observed speed increment. The simple physics model fits acceleration = bias + positive throttle gain × throttle − positive brake gain × brake − positive drag × speed, on unique training transitions only. Both controls hypotheses for physics are retained in the metrics.','',
  'The GRU beats the baselines over one second on both sources, but error grows with horizon. Person one-step constant-acceleration error is competitive; this is not a uniform GRU win at every metric.','',
  '### Held-out episode and driver consistency','']
 episode=metrics[(metrics.split=='test')&(metrics.episode!='ALL')&(metrics.subset=='all')&(metrics.step==10)]
 for source in ['bench2drive','person2drive']:
  g=episode[(episode.source==source)&episode.model.str.startswith('GRU_H'+str(offsets[source]))].groupby('episode').rmse.mean()
  b=episode[(episode.source==source)&(episode.model=='constant_speed')].set_index('episode').rmse
  wins=int((g<b.loc[g.index]).sum());lines += [f'- {source}: GRU improves 1s RMSE over constant speed on {wins}/{len(g)} test episodes; episode RMSE ranges {g.min():.3f}–{g.max():.3f} m/s (median {g.median():.3f}).']
 drivers=pd.read_csv(OUT/'heldout_driver_metrics.csv');driver_rows=[]
 for driver in sorted(drivers.driver.unique()):
  subset=drivers[drivers.driver==driver];g=subset[subset.model.str.startswith('GRU_H'+str(offsets['person2drive']))];b=subset[subset.model=='constant_speed']
  driver_rows.append([driver,int(g.windows.iloc[0]),f'{g.rmse.mean():.3f} ± {g.rmse.std(ddof=1):.3f}',f'{b.rmse.mean():.3f}'])
 lines += ['',table(['Person held-out driver','Windows','GRU 1s RMSE','Constant 1s RMSE'],driver_rows),'','### Braking/coasting subsets at one second','']
 rows=[]
 for source in ['bench2drive','person2drive']:
  for regime in ['coasting','partial_braking','strong_braking','strong_deceleration']:
   group=metrics[(metrics.source==source)&(metrics.split=='test')&(metrics.episode=='ALL')&(metrics.subset==regime)&(metrics.step==10)]
   gru=group[group.model.str.startswith('GRU_H'+str(offsets[source]))];base=group[group.model=='constant_speed']
   rows.append([source,regime,int(gru.n.iloc[0]) if len(gru) else 0,f'{gru.mae.mean():.3f}' if len(gru) else 'N/A',f'{gru.rmse.mean():.3f}' if len(gru) else 'N/A',f'{base.rmse.mean():.3f}' if len(base) else 'N/A'])
 lines += [table(['Source','Subset','Windows','GRU MAE','GRU RMSE','Constant RMSE'],rows),'',
  'Dynamics coast/partial masks require all 11 control labels to agree so both hypotheses share the same target windows. Strong braking means any brake>=0.8 command while starting moving; strong deceleration means a recorded single-step speed drop>=0.5 m/s. Counts are overlapping windows; scarce cases are not independent replications. H1 is slightly better than H0 on Person partial braking, so the preferred aggregate hypothesis is not uniformly best across maneuvers.','',
  '### Alignment and action response','']
 for source in ['bench2drive','person2drive']:
  loss=config['source_config'][source]['validation_alignment_rmse'];lines += [f"- {source}: validation pooled-horizon RMSE H0={loss['0']:.4f}, H1={loss['1']:.4f}; provisional choice H{offsets[source]}."]
 cand=pd.read_csv(OUT/'candidate_checks.csv')
 for source in ['bench2drive','person2drive']:
  c=cand[(cand.source==source)&(cand.offset==offsets[source])];lines += [f"- {source}: candidate ordering fails in {100*c.any_horizon_order_failure_fraction.min():.1f}–{100*c.any_horizon_order_failure_fraction.max():.1f}% of speed-stratified held-out histories across seeds (any horizon, 0.05 m/s tolerance). Mean 1s coast change ranges {c.coast_delta_mean.min():.2f} to {c.coast_delta_mean.max():.2f} m/s."]
 lines += ['',
  'The synthetic command probe covers speed bins 2–5, 5–10, 10–15 and >=15 m/s, up to 250 histories/bin. The expected speed order is full brake <= partial brake <= coast <= full throttle. Finite/nonnegative outputs and broad acceleration thresholds do not establish physical realism: nonnegative speed is imposed by the model, and long Bench coasting is unsupported. Observed Bench 99.5th-percentile absolute acceleration is about 22 m/s², indicating abrupt/unmodeled events; the GRU smooths some of these rather than explaining them. Candidate-plan changes have no intervention ground truth.','',
  'Bench steering/right-yaw association favors H0, whereas aggregate speed prediction favors H1. Different channels, sampling/actuator delays, hidden variables and losses can produce conflicting predictive preferences. Neither source has the authoritative collection/export caller needed to certify controls at a saved RGB tick. `source_checks.json` names the missing callers; the adapter requires explicit per-source offsets and preserves raw command indices. Coordinate calibration is independently checked, not treated as timing proof. Ten Hz describes saved observations, not a verified zero-order hold of each command for the whole 0.1-second interval; within-interval control changes are not established by the available caller provenance.','',
  'See [error accumulation](results/dynamics_horizons.png), [representative rollouts](results/rollout_examples.png), per-episode/driver CSVs, `prediction_quality.json`, and `candidate_checks.csv`.','',
  '## Recommendation and remaining uncertainty','',
  'Proceed with a fixed-dataset visual prototype and DINOv3 features when requested. Keep source identity, source-specific normalization/physics sensitivity, explicit action offsets and the saved splits. Use explicit source/maneuver sampling weights rather than letting the larger Person window count dominate. Keep an honest rare-event test set and report per-source results.','',
  'Do not treat this as certification for counterfactual planning or launch a large final planning model based only on these scores. Required evidence for that use includes known applied-command timestamps, vehicle/gear/slope/physics context, intervention-response tests, and better low-speed/turning partial-braking coverage. The full-release audit shows the latter is intrinsically rare in the current source; human full-stop supervision also needs investigation because the selected metadata omit exact standstill. Two Person routes and two test drivers cannot establish broad environment or human-driver generalization. More seeds, grouped resampling and a prospective test split would strengthen uncertainty estimates.','',
  'This diagnostic evaluates longitudinal speed only, not learned lateral displacement/yaw dynamics. Coordinate transforms for those targets are checked, but lateral rollout accuracy is not established. It did not test visual predictability or a shared mixed-source world model. Existing data, original dataset code and historical experiments were retained. No DINOv3 cache, visual encoder, Transformer or RSSM training was run.','']
 (HERE/'REPORT.md').write_text('\n'.join(lines));print('Final report and storage/provenance written',flush=True)
if __name__=='__main__':main()
