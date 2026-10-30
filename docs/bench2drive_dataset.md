# Bench2Drive training dataset

The dataset returns deployment observations, future action conditioning, and
training supervision as separate fields. Defaults: history K=4, horizon H=10,
10 Hz recordings, image_size=(224,224), action_offset=1, pose_source='simulator'.
Four observations span 0.3 seconds; predictions cover 0.1–1.0 seconds after the
last observation. Values are float32; images are RGB in [0,1]. No control values,
large decelerations, overlapping commands, or small negative speeds are clipped.

## Interface

| Field | Shape (one sample) | Contents / role |
|---|---|---|
| past_images | (K,3,height,width) | Observed front RGB, input |
| past_ego_observations | (K,1) | Recorded sensor speed (m/s), input |
| past_actions | (K-1,3) | Recorded controls assigned to the observed transitions |
| future_actions | (H,3) | Candidate/recorded control conditioning |
| future_images | (H,3,height,width) | Future RGB supervision only |
| future_motion_targets | (H,5) | [speed, forward_m, right_m, sin(relative_yaw), cos(relative_yaw)], supervision only |
| scenario | string | Relative recording directory (flat or nested) |
| start_frame | integer | First observed frame |
| frame_ids | (K+H,) int64 | Matched consecutive recorded IDs |
| action_offset | integer | Explicit recorded alignment hypothesis |
| motion_target_source | string | simulator or gps_compass |

A DataLoader adds a batch dimension to tensors, batches numeric metadata into
tensors and string metadata into lists. With batch_size=8, future_actions is
(8,10,3), past_actions (8,3,3), future_motion_targets (8,10,5).
For K=1, past_actions is an empty (0,3) tensor: there is no observed transition.
Old `past_states`/`future_states` keys have intentionally been removed; do not
replace input observations with the privileged motion targets.

Controls are always `[steer, throttle, brake]`. Signed throttle-minus-brake can be
derived externally, but is not used as the primary representation. Pixel resize
uses PIL's (width,height), preserving the former direct-resize behavior. Square
resizing distorts aspect ratio; use the same policy at deployment and decide
encoder-specific preprocessing later. No pretrained-encoder normalization is
applied here.

## Pose provenance and coordinate convention

The collector `external/Bench2Drive/tools/data_collect.py:get_bounding_boxes`
records the ego actor's get_transform().location and rotation=[pitch,roll,yaw]
in the bounding box entry with class='ego_vehicle'. This is a verified simulator
pose source, not the bounding-box center or the GNSS sensor location. The default
supervision reads this entry by class (never assumes its list index). It raises
an error when no unique ego entry is available; there is no silent fallback.

Motion targets use actor-origin world positions and world yaw in radians. For
the last observed actor pose (x0,y0,yaw0):

```text
forward =  cos(yaw0)*(x-x0) + sin(yaw0)*(y-y0)
right   = -sin(yaw0)*(x-x0) + cos(yaw0)*(y-y0)
relative heading = yaw - yaw0, represented by sin and cos
```

Positive forward follows the vehicle's heading; positive right follows CARLA's
right-positive local y. These are planar XY targets in the cutoff's yaw-aligned
frame; they intentionally omit elevation, pitch and roll. Sloped-road 3D targets
would require a full transform and a separate schema change. Relative sin/cos
heading is continuous across ±pi.

The collector's x/y are projected GPS measurements and theta is IMU compass.
The audit verifies compass=world_yaw+pi/2 modulo 2pi in all ten scenarios. Default
simulator supervision therefore needs **no arbitrary compass rotation**. The
explicit pose_source='gps_compass' option converts yaw=theta-pi/2 and uses x/y as
noisy sensor-position labels (including mount effects). It is never mixed per
frame with precise simulator targets and the source is included in each sample.

No GPS/compass/actor pose is fed into past_ego_observations: speed is the only
initial ego observation. The last observed precise pose is a supervision reference,
not an input. The trainer must explicitly select input keys and use future fields
only as losses. In deployment, use measured/estimated speed and the agent's own
actually applied command history, not simulator ground truth or future recordings.

## Action indexing and its uncertainty

Let window indices be 0..K+H-1; the observation cutoff is t=K-1.

| Hypothesis | past_actions (K-1 observed transitions) | future_actions (H future transitions) |
|---|---|---|
| H0, action_offset=0 | a[0:K-1] | a[K-1:K-1+H] |
| H1, action_offset=1 | a[1:K] | a[K:K+H] |

No past slice uses a recorded frame after the cutoff. Future recorded controls
are supervised conditioning labels, not observations known in advance. H1 is the
provisional default because it performed slightly better in the multi-step GRU
diagnostic, **not because collection-time action timing has been verified**.
The expert collection caller remains unavailable. A label at the current or next
recorded observation might reflect feedback; indexing alone does not establish
causal actuator timing. Recorded past actions are aligned training labels under
the chosen hypothesis, not a guarantee of physical actuation timestamps.

At deployment, the first planned outgoing command conditions the first future
transition, regardless of which recorded offset was used during training. Never
wait for a future observation to construct input action history. Keep H0 as a
sensitivity check and record the offset with checkpoints/configuration.

## Grouped train/validation split

```python
from src.bench2drive_dataset import Bench2DriveDataset, split_scenarios

root = 'data/bench2drive_mini'
index = Bench2DriveDataset(root)
train_names, val_names = split_scenarios(index.scenarios, seed=2026, val_fraction=.2)
train = Bench2DriveDataset(root, scenarios=train_names, action_offset=1)
validation = Bench2DriveDataset(root, scenarios=val_names, action_offset=1)
```

Automatic splitting deterministically shuffles sorted town/route groups, selects
ceil(group_count*val_fraction) validation groups (at least one, leaving one for
training), and keeps every recording/weather variant of a route together. This
is a split of complete recording directories, not a holdout of an entire scenario
*type* across all routes in a larger collection. If that stronger generalization
protocol is wanted, provide explicit assignments that also hold types together.
The validation window fraction need not equal val_fraction.

Pass train_scenarios and/or val_scenarios to split_scenarios for explicit
assignments. An omitted side is the complement. Assignments must cover the named
recordings, have no overlap, and share no town/route across sets. This can reuse
any of the historical five-fold validation lists unchanged; the automatic helper
is a new train/validation convenience, not a replacement for those saved folds.

Both flat `Scenario_Town03_Route156_Weather0` and nested `Scenario/Town03_156`
recordings are supported. Frame discovery recursively finds anno directories
and camera/rgb_front siblings. route_group recognizes town/route names; use
route_ids={recording: unique_route_id} for other naming/provenance. Unknown routes
fail closed during splitting rather than silently treating possibly shared routes
as distinct. Use globally consistent route IDs when combining collections.
Never randomly split overlapping frame windows. The dataset itself does not
force a split: apply the helper or validated explicit assignments before training.

## Validation and limitations

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python test_dataset.py
```

Synthetic tests check exact action slices, future-history isolation, input/target
separation, forward/right motion at five yaw angles, heading wrapping, missing
frames, duplicate IDs, explicit noisy fallback, invalid configurations and
batching. Real integration checks cover all ten clips' heading/forward conventions,
all 2,165 matched windows, old fold assignments, split isolation and finite outputs.
Real integration tests skip when Mini is unavailable; synthetic tests still run.
A one-time preservation check uses the /tmp pre-refactor manifest when present.

The dataset warns and reports unmatched files in matching_report, never bridges
frame gaps, rejects duplicate numeric IDs, and raises on empty window selection
or nonfinite returned tensors. Matching filenames do not prove original sensor
capture/control application timing. Numeric/pose validation occurs as samples are
loaded; image preprocessing is lazy. No normalization statistics are fitted here;
any later state scaling must use training data only.

Mini contains only ten clips, binary braking, very little moving coasting, rare
sustained candidate plans, and extreme decelerations. It supports exploratory
recorded-plan prediction, not demonstrated arbitrary-action MPC reliability.
Speed-only observations omit grade, reverse/gear, contact and latent actuator
state. Visual context and uncertainty may help but are not guaranteed solutions.
No historical experiment source/results were changed or rerun to replace saved
metrics; their documented old interface/assumptions remain historical records.
