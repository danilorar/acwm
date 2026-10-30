# Dataset selection for action-conditioned visual dynamics

**Recommendation: add a selective Person2Drive subset to the existing Bench2Drive
collection. Do not replace Bench2Drive outright and do not download a full
candidate collection yet.** Person2Drive is the strongest measured addition for
our current 10 Hz, one-second RGB-plus-ego-motion task: it has substantial sustained
moving coasting, nonbinary braking, explicitly referenced RGB, and simulator pose
transforms. TeleopWM is useful for continuous-braking/visual prediction but does
not solve coasting and lacks clean pose labels in the inspected tables. PDD is a
promising small human-driving source after correcting its loader indexing and
handling its 4 Hz visuals / faster control stream.

This is a data suitability study, **not proof of causal control dynamics or MPC
reliability**. All alternative statistics below refer to the stated samples.
No production dataset or existing recording was modified; no models were trained.

## Reproduce

From the project root, using the existing environment:

```bash
.venv/bin/python -m pip install -r experiments/dataset_selection/requirements.txt
.venv/bin/python experiments/dataset_selection/fetch_samples.py
.venv/bin/python experiments/dataset_selection/run_audit.py
.venv/bin/python -m unittest discover -s experiments/dataset_selection -p test_audit.py -v
```

The downloader is fixed to **270 sample files, 788.2 MiB total**, capped at
1,200 MiB. It downloads telemetry/metadata primarily, one Person2Drive sensor
archive, 19 TeleopWM RGB frames and 192 PDD RGB frames. It never snapshots a whole
repository. sample_manifest.json records exact paths, sizes, revisions and hashes.
Existing matching Hub downloads are reused. Source revisions and small documentary
snapshots are under sources/. Large sample payloads and initial broad file listings
are ignored by this experiment's local .gitignore. The audit also reads the
existing 132-route Bench2Drive subset in data/bench2drive_train/extracted/.

Pickle files are inspected through a restricted numeric/pandas unpickler; remote
loading scripts are not executed. JSON and RGB files are read directly. Only
requested front-image excerpts are copied from the Person2Drive tar.zst; source
archive paths are never used as output paths. Five focused tests pass, covering
threshold definitions, gap boundaries, one-second windows, duration units and
blocking arbitrary pickle globals.

## Sampling and comparable definitions

- **Bench2Drive:** all 132 existing V0.0.4 recordings; 35,510 frames.
- **TeleopWM:** first two training episodes in each of Town01/03/04 plus episode
  0000 in validation Town02 and test Town05. Eight metadata tables, 35,263 frames.
  These are inspected for coverage only; no training on validation/test is done.
- **Person2Drive:** complete driver01 and driver50 training metadata, each containing
  eight drives in each of Town04 and Town05: 32 drives and 39,322 selected frames.
  Separate motion/result/route JSONs for eight representative drives were also
  inspected. One driver01/Town04_drive_1 sensor archive provides RGB validation.
- **PDD:** driver_01, driver_10, driver_30 × Accident, ControlLoss,
  NonSignalizedJunctionLeftTurn, VanillaNonSignalizedTurn: twelve sessions, 878
  matched saved states. The same sessions contain 4,360 native control steps.

Selection is deterministic and explicitly listed; it is not a population-random
sample. Coverage is not extrapolated to all drivers/routes. Person2Drive driver01
and driver50 respectively have 16.34% and 18.09% moving coasting, and 3.37% and
5.00% intermediate braking over all their sampled metadata frames. This supports
the finding in both inspected drivers, without establishing it for all fifty.

All controls are reordered to [steer, throttle, brake], retained as original
normalized values (steer [-1,1], pedals [0,1]). No signed compression or clipping.
Speed uses the released m/s-equivalent field; Person2Drive uses ego_vel[0], not
an assumed world-coordinate vector norm. Timing/units are checked against pose
motion where available. Moving means released speed>=2 m/s, as in our Bench2Drive
audit. Coasting requires throttle<=0.01 AND brake<=0.01 while moving. Intermediate
brake is strictly **0.01<brake<0.99**, as requested (different from the earlier
expansion report's 0.1–0.9 definition). Overlap means both pedals>0.01.
Intermediate throttle uses the analogous strict 0.01–0.99 interval.

Native-rate paired observations are compared: 10 Hz Bench2Drive/Person2Drive,
15 Hz TeleopWM, 4 Hz PDD. Durations use n/fps sampled duration; first-to-last
observed span is (n-1)/fps. Missing frame IDs break runs and windows; episode
boundaries are never bridged. Future-action sampling and control-to-transition
causality are separate from matching an observed frame's label.

## Measured control coverage

| Dataset / inspected scope | Frames / valid transitions | Moving coasting | Intermediate brake (all / moving) | Longest coast | Longest intermediate brake |
|---|---:|---:|---:|---:|---:|
| Bench2Drive: 132 routes | 35,510 / 35,378 | 522/19,003 = **2.75%** | 0.00% / 0.00% | 6 frames / 0.60s | 0.00s |
| TeleopWM: 8 episodes, 5 towns | 35,263 / 35,255 | 82/20,369 = **0.40%** | 25.97% / 4.54% | 2 frames / 0.13s | 50.20s |
| Person2Drive: 32 drives, 2 drivers, 2 routes | 39,322 / 39,069 | 5,704/33,165 = **17.20%** | 4.12% / 4.44% | 50 frames / 5.00s | 2.50s |
| PDD: 12 sessions, 3 drivers, 4 scenarios | 878 / 866 | 102/726 = **14.05%** | 7.63% / 7.02% | 6 frames / 1.50s | 1.75s |

Longest-coast endpoint spans are 0.50s Bench2Drive, 0.067s TeleopWM, **4.90s
Person2Drive**, and 1.25s PDD. Thus the Person2Drive advantage is not just a larger
number of short coasting samples. The longest reported intermediate-brake run
includes low-speed/stationary samples; it is not necessarily a long moving stop.

**TeleopWM's headline 25.97% intermediate-brake rate is misleading if interpreted
as moving-action coverage:** 8,233 of its 9,157 intermediate-brake frames occur
below 2 m/s. Moving intermediate braking is 4.54%, close to Person2Drive's 4.44%.
PDD's moving intermediate braking is 7.02%, though based on a smaller sample.
PDD's complete native control stream has 7.61% intermediate braking versus 7.63%
on the 4 Hz paired subset. No intermediate braking is present in Bench2Drive.

Simultaneous throttle/brake rates are 0.37% Bench2Drive, 0% TeleopWM,
0.17% Person2Drive and 1.48% PDD. Intermediate-throttle rates are respectively
38.55%, 53.08%, 73.92% and 68.45%. These are observational command distributions,
not a calibration that normalized pedal magnitudes produce identical forces.

Speed-conditioned rates and mean steer/throttle/brake are saved in
results/speed_conditioned.csv, with bins <2, 2–5, 5–10 and >=10 m/s.
Person2Drive retains substantial coasting across all moving bins (about
19.0%, 15.9%, 17.3%), whereas TeleopWM has 1.04%, 0.20%, and no sampled frames
in the >=10 m/s bin. Different speed/context distributions still confound a
claim about universal dataset superiority.

## One-second control-sequence diversity

For every contiguous window covering an actual one-second frame interval, sample
four controls at offsets 0,0.25,0.5,0.75s using last-known native controls. Quantize
each channel to nearest 0.1 and count the resulting 12-value patterns. No invented
camera interpolation is used. This compares coarse temporal signatures, not full
native-rate trajectory coverage.

| Dataset | Valid one-second windows | Unique coarse patterns | Entropy (bits) | Dominant pattern |
|---|---:|---:|---:|---:|
| Bench2Drive | 34,190 | 4,553 | 8.417 | 15.46% |
| TeleopWM | 35,143 | 9,107 | 9.065 | 15.39% |
| Person2Drive | 36,915 | 6,682 | 8.798 | 8.22% |
| PDD | 830 | 622 | 8.793 | 4.82% |

Pattern count and entropy depend on sample size, speed mix, temporal correlations,
quantization and sampling rate. In particular, PDD has only 830 windows here.
Do not interpret these as a statistically balanced ranking or a count of
independent experiments. TeleopWM's higher pattern entropy does not compensate
for its lack of sustained moving coasting or clean pose targets.

## Structure and compatibility comparison

| Dataset | Front RGB / controls | Ego-motion supervision | Rate and four-past/one-second horizon | Split implications | Verdict |
|---|---|---|---|---|---|
| Bench2Drive V0.0.4 | camera/rgb_front + anno JSON.gz; three raw channels | Simulator ego location/rotation/world2ego; current loader verified | 10 Hz; K=4, H=10 | Existing route-held-out split preserved | Keep for broad visual/scenario coverage; poor coast/brake support |
| TeleopWM | image PNG + pd_dataframe.pkl with explicit image_path, action=[throttle,steer,brake], speed | GNSS and IMU compass/acceleration available; no precise actor pose in 34 inspected columns | Nominal 15 Hz; K=4, H=15 for one second | Official town-disjoint train/val/test layout | Auxiliary continuous-control visual/speed source; not the coasting solution |
| Person2Drive | Six-camera JPEG tar.zst; metadata explicitly references CAM_FRONT data_path and frame_idx | world2ego, ego_yaw, ego_translation, ego_vel, ego_accel; use transform inverse for precise position | 10 Hz; K=4, H=10; break missing IDs | Only two pre-rendered routes; episode/driver splits possible, strict three-way route holdout unavailable without more rendered routes | **Best addition for this project's current targets** |
| PDD | images JPEG; metric_info.json plus metrics.json control steps | Simulator location, rotation, forward/right vectors, speed, velocity and acceleration | Saved RGB/state 4 Hz, faster control stream consistent with 20 Hz; K=4, H=4 for one second | Driver/scenario sessions; verified route IDs not supplied in inspected files | Useful secondary human source after indexing/multirate fixes |

The full releases are described as 27 TeleopWM episodes (verified directory count:
21 train, 3 val, 3 test), 50 Person2Drive drivers with 800 sensor-equipped drives
and 200 additional sensorless driving records, and 630 PDD sessions / 70,087 images
(card figures). Full-release total durations were not independently enumerated.
The inspected contiguous observed-transition durations are 58.96 minutes
Bench2Drive, 39.17 minutes TeleopWM, 65.12 minutes Person2Drive and 3.61 minutes PDD.

### TeleopWM findings

The [release card](https://huggingface.co/datasets/bimilab/TeleopWM-Dataset) and
[official loader](https://github.com/bimilab/paper-TeleopWM/blob/main/src/datasets/carla_rollout_dataset.py)
specify the raw action order and metadata-based image pairing. All eight tables
are flagged is_expert=True; those flags, action_mu/action_sigma and policy features
should not be mistaken for verified human teleoperation traces. We audit the raw
action column, not action_mu or a predicted action head.

The [official model README](https://github.com/bimilab/paper-TeleopWM) uses a 15 FPS
reference. GNSS displacement over 30-frame spans divided by speed yields median
per-frame intervals 0.06625–0.06738s across these episodes, supporting that rate.
Single-step GNSS differences are too noisy to infer timing. There are no explicit
frame timestamps in the inspected 34 columns. Exact CARLA build/physics provenance
was not established from the release/setup docs.

The image sample decoded at 960x600. GPS/compass can provide noisy motion labels
with a documented world-coordinate projection and covariance treatment, but this
is not equivalent to the simulator pose labels used by our current loader.
Do not silently supervise precision trajectory prediction with GPS noise, or feed
privileged expert features into deployment observations.

The upstream [MILE collector](https://github.com/wayveai/mile/blob/main/data_collect.py)
generates control from obs_t, steps the environment, then writes obs_t with the
corresponding supervision. This supports an outgoing-control interpretation for
that upstream implementation, **not proof of the exact TeleopWM collection fork**.
The release's collection-plan generator is not an execution trace. Its public
loader's future action slicing likewise does not establish actuator timestamps.

### Person2Drive findings

The [release documentation](https://huggingface.co/datasets/dongxr7/Person2Drive)
allows per-file/per-drive download. The [paper](https://arxiv.org/html/2602.18757)
states 10 Hz sampling; pose displacement divided by stored speed supports median
0.10025s across our episode checks. The exact simulator version was not verified.
Offline image/metadata loading works without running CARLA; matching physics to
our later evaluation simulator is a separate check.

The sampled archive contains **900 front RGB images, six RGB views, no anno/**.
Its 866 selected metadata entries reference existing front images with zero
missing references. A contiguous frame 2–15 example decoded and produced finite
4-history/10-future motion targets. The separate motion JSON has 1,844 records;
indexing it by image frame number would not be a valid join. Official
[dataset details](https://huggingface.co/datasets/dongxr7/Person2Drive/blob/main/DATASET_DETAILS.md)
also warn about that discrepancy. Use the explicitly image-associated training
metadata, not an ordinal merge with those raw logs.

The 32 metadata episodes contain **221 frame-gap boundaries**. They reduce valid
one-step transitions to 39,069; the audit does not bridge them. The cause of every
omission is not established. Metadata selection and fixed-route driving may bias
coverage, and only two of fifty drivers were inspected quantitatively.

Coordinate handling is important: CAM_FRONT cam2ego maps camera-right to ego
[0,-1,0], so these processed metadata use **left-positive ego y**. For our target's
right-positive displacement, transform positions with the anchor world2ego then
negate local y. Derive relative right-positive heading from transformed forward
vectors, avoiding an assumed yaw sign. ego_translation z is flattened in inspected
entries, while inverse world2ego retains actor height; prefer the latter as the
pose source. The target preview and calibration evidence are saved, rather than
inventing a new ground-truth source.

Controls in the metadata are the published human-driving control labels; the
rendering-to-control call order is not published. We cannot certify whether they
are outgoing/incoming or whether every label reflects the final applied server
command. Keep configurable offsets and validate causal rollout response before
MPC. Calibration/format consistency alone is not temporal alignment proof.

**Splits:** Town04 means route 108 and Town05 route 103, repeated across drivers.
Never treat drive_1 and drive_5 as different physical routes for route splitting.
Supply explicit route group IDs to our splitting utility. A three-way
route-disjoint RGB train/validation/test split is impossible using only these two
rendered routes. Driver/episode-disjoint splits evaluate a different question.
Routes 100/101 supply recorder/log files but no released camera frames; the
additional rendering pipeline is not yet public.

### PDD findings and loader issue

The [card](https://huggingface.co/datasets/tasl-lab/PDD) documents CARLA 0.9.15 and
4 FPS RGB. The sample's world displacement/speed estimates are about 0.250–0.251s
per saved state. State keys are 0,5,10,..., while control.step runs at finer
increments. For Accident driver_01 there are 70 states and 350 controls.

We join **metric_info key k to control.step k**, never zip the two sequences.
For RGB, the provided loader associates sorted images with sorted state keys;
the card's 4 FPS and sampled counts support that intended mapping, but explicit
camera timestamps are absent and capture phase remains unproven.

The [public loader](https://huggingface.co/datasets/tasl-lab/PDD/blob/main/load_pdd.py)
selects state using metric_keys[frame_idx] but control using records[frame_idx].
Once state steps advance by five, those refer to different simulator steps.
Step IDs disagree in 97.2–99.3% of sampled rows; control values differ by >1e-5
in 92.6–98.7%. This is an integration/preprocessing issue, not evidence that the
underlying raw human controls are invalid. results/pdd_indexing.png visualizes it.
The audit does not execute or modify the published loader.

Use human controls from metrics.json; **expert_control_* are reference actions**,
not the human commands. The sensor data also contain simulator world poses.
The source get_metric_info helper emits rotation=[roll,pitch,yaw], despite the
card stating [pitch,roll,yaw]; yaw index 2 remains usable. This was checked against
forward vectors. Our 192 sampled RGB images decoded at 1024x512.

One second of visuals is four future frames, with approximately five raw control
steps per saved interval. Do not upsample these into alleged 10 Hz ground-truth
images or discard four of five commands while assuming constant control. A
multirate/native-4-Hz adapter is needed, with action blocks or explicit timestamps.
Group repeated scenario routes across drivers; route identifiers are not verified
from session names alone. The card's license is CC-BY-NC-4.0, unlike the licenses
listed for the other releases.

## Required adapter changes (not implemented)

1. **Person2Drive:** add a source-specific metadata reader grouped by (driver,folder),
   index images by data_path/frame_idx, preserve gaps, read raw pedals, use full
   world2ego pose for supervision, and convert lateral/yaw signs via calibration.
   Keep only speed and known command history as ego inputs. Retain the same output
   interface and H=10/K=4 at 10 Hz. Map route groups to Town04/108 and Town05/103.
2. **TeleopWM:** read image_path/action/speed, reorder channels, retain native 15 Hz
   timing (H=15) or use a timestamp-aware approach. Keep GNSS labels explicitly
   noisy or omit position supervision. Group by actual town+route_id, not folder
   episode number. Preserve the published town-disjoint splits.
3. **PDD:** join state/control by saved step keys, validate image correspondence,
   use simulator pose/yaw with documented ordering, and retain native 4 Hz visuals
   plus faster action blocks. Maintain explicit route/driver grouping. Do not use
   the public loader's ordinal control join unchanged.

All three can provide four historical frames and future control sequences, but
those four frames span different physical histories (0.2s TeleopWM, 0.3s at 10 Hz,
0.75s PDD). Include time in the model contract; variable frame rates are not merely
image-format differences. No changes were made to src/bench2drive_dataset.py.

## Selection decision and remaining uncertainty

**Strongest source to add: Person2Drive.** It directly improves the two observed
action gaps, matches our desired 10 Hz prediction grid, and supports the complete
motion target through image-linked metadata. **Strongest training strategy:
Bench2Drive + selective Person2Drive**, with episode/route-aware evaluation and
source-aware sampling. Bench2Drive retains 44 scenario types and broader existing
route diversity; replacing it with two pre-rendered routes would trade away that
coverage. PDD is a secondary human-control option; TeleopWM is an auxiliary
continuous-braking/visual source, not a sustained-coasting replacement.

Next step is a small separate adapter and matched dynamics check on these samples,
then a selectively approved RGB expansion across more Person2Drive drivers and
both available routes. Do not download 734 GB simply because the metadata is
promising. No further dataset transfer or model training is started here.

Remaining limitations for counterfactual MPC:

- Marginal pedal diversity is not conditional coverage of speed, steering, grade,
  traffic and arbitrary one-second action sequences. All are observational traces.
- Applied-control timing, actuator delay and postprocessing are not conclusively
  traced for any alternative. Person2Drive's unpublished renderer and PDD's missing
  explicit image timestamps require care; do not transfer Bench2Drive's H1 as fact.
- Human feedback correlates actions with hazards and unseen state. A good recorded
  trajectory fit does not identify intervention response or prove calibrated risk.
- CARLA build, vehicle type/physics settings and camera calibration may differ.
  Offline format compatibility does not guarantee matching actuation dynamics.
  Check them before treating normalized command magnitudes as interchangeable.
- Dataset mixing should preserve source identity/timing, fit normalization on
  training only, and hold out complete episodes/routes/drivers as appropriate.
  Uncertainty and support constraints need controlled simulator validation later.

The measured gains justify **selective supplementation**, not a claim that the
dataset problem is solved or that a CEM planner can safely explore any command.

## Evidence, sizes and artifacts

Source revisions are pinned in sources/dataset_revisions.json. Initial broad HF
file lists for TeleopWM/PDD stopped at 100,000 entries; their summed sizes are
**not full-dataset size estimates**. Targeted episode listings establish the 27
TeleopWM directories. Dataset card sizes are approximate (~90 GB TeleopWM,
~734 GB Person2Drive, ~13 GB PDD); Hub storage accounting differs and is preserved
separately, not substituted for current-commit payload totals. Only sample file
sizes/hashes were summed to enforce the 788.2 MiB download plan.

- dataset_comparison.md: this report and recommendation.
- fetch_samples.py, sample_manifest.json: fixed selection and verified downloads.
- safe_pickle.py: restricted numeric metadata reader.
- run_audit.py, test_audit.py: reproducible metrics and correctness tests.
- results/summary.csv, episode_metrics.csv: pooled and per-episode statistics.
- results/speed_conditioned.csv, pdd_native_controls.csv: speed bins and native controls.
- results/control_distributions.png, coverage_comparison.png, speed_conditioned.png:
  comparable visual summaries.
- results/pdd_indexing.png: public-loader indexing discrepancy.
- results/quality_observations.json, image_compatibility.json: gaps, cadence, schema,
  image references and decoding checks.
- results/person2drive_target_preview.json: contiguous real RGB/pose example.
- sources/: pinned code-tree metadata, small official documentary/source snapshots.

No additional candidate was promoted without measured evidence. The investigation
focuses on the requested three alternatives rather than adding untested datasets
to a long recommendation list. Existing Bench2Drive data, dataset implementation,
experimental results and train/validation manifests remain unchanged.
