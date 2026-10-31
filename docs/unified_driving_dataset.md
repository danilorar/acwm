# Joint driving dataset contract

`src/unified_driving_dataset.py` is a separate adapter. It supports both sources; the superseded Bench-only adapter and historical
experiments are preserved in tag dataset-foundation-v1.

```python
import json
from pathlib import Path
from torch.utils.data import DataLoader
from src.unified_driving_dataset import load_episode_records, UnifiedDrivingDataset

results = Path('data/config')
splits = json.loads((results / 'splits.json').read_text())['episodes']
offsets = json.loads((results / 'action_alignment.json').read_text())['action_offsets']
records = load_episode_records()  # verifies all indexed image paths
train = UnifiedDrivingDataset(records, episode_ids=splits['train'],
                              action_offsets=offsets, image_size=(224, 224))
batch = next(iter(DataLoader(train, batch_size=8, shuffle=True, num_workers=0)))
```

At 10 Hz, an anchor at recorded observation `t` has observations `t-3..t`
and targets `t+1..t+10`. By default, a sample requires 14 frames with controls in the nominal CARLA
range (steer [-1,1], throttle/brake [0,1]) and strictly consecutive matched IDs
within one episode. Missing metadata frames break windows; no interpolation or
ordinal joins are allowed. Images without metadata are not training frames. In the selected Person2Drive
metadata, minimum speed is about 0.2001 m/s and no exact standstill is present;
the cause of that selection cutoff is unverified. Do not infer human full-stop
coverage from unindexed RGB or fill missing labels from ordinal raw-log indices.

| Output | Sample shape | Meaning |
|---|---|---|
| `past_images` | `(4,6,3,224,224)` | Six-view RGB float32 in [0,1], oldest first |
| `past_ego_observations` | `(4,1)` | Recorded forward speed in m/s; no future/privileged poses |
| `past_actions` | `(3,3)` | Controls for the three observed transitions |
| `future_actions` | `(10,3)` | Candidate controls for the ten predicted transitions |
| `future_images` | `(10,6,3,224,224)` | Future RGB supervision |
| `future_motion_targets` | `(10,5)` | Speed, forward meters, right meters, sin(right yaw), cos(right yaw) |

Batching prepends the batch dimension. `image_size` is PIL `(width,height)`.
Bench recorded speed can be slightly negative near stops (observed minimum
−0.06124 m/s); the adapter preserves these signed values. Person selected
metadata contain only positive forward speeds.

Raw action columns are **steer, throttle, brake**. Pedals range from 0 to 1,
but applied-force calibration is unverified. Person2Drive has 61 steering labels
outside the nominal [-1,1] range, from -1.1987 to +1.9646; their conversion or
saturation into applied steering is not established. These labels are preserved,
not clipped or rescaled. Default training windows conservatively exclude any
14-frame block touching them; `exclude_uncertain_controls=False` exposes them
for audit. `action_in_nominal_range` returns the 14-frame quality flags.
Source steering labels are not certified interchangeable physical angles or
applied commands. No pedal subtraction, steering
sign conversion, or implied cross-source actuator calibration is applied.
Metadata includes `source`, episode-qualified `scenario`, `driver`, physical
`route`, `frame_ids`, `action_offset`, and the motion-label frame/provenance.
The targets use simulator poses, not GPS approximations. Pose information is
supervision only. Lateral displacement and heading are projected into the
anchor's planar yaw frame, rather than its pitched/rolled full body frame.

## Independent source conventions

Bench2Drive uses native CARLA actor x forward / y right. Person2Drive's processed
metadata use x forward / y left: front-camera calibration maps camera-right to
`[0,-1,0]`. The adapter uses inverse `world2ego` positions and forward vectors,
then projects into each source's calibrated right direction. It derives
relative heading from forward-vector projections, avoiding assumed yaw signs.
It does not use Person2Drive's flattened `ego_translation.z`. Selected-episode
camera calibration checks and displacement/speed interval checks are saved in
the diagnostic results. Synthetic tests cover different yaw angles and both
handednesses. These coordinate checks do **not** certify action timing.

Alignment must be supplied explicitly for each source:

| Hypothesis | Past controls | Future controls |
|---|---|---|
| H0 / offset 0 | `a[t-3:t]` (last index t-1) | a[t] through a[t+9] |
| H1 / offset 1 | `a[t-2:t+1]` (last index t) | a[t+1] through a[t+10] |

Neither past sequence reads controls after observation t. H1's future labels
are the hypothesized incoming controls; at deployment this means supplying
known previous commands and the planned outgoing sequence, **not waiting for
future observations**. Future ground-truth speeds are never fed into the GRU
rollout. The diagnostic's per-source validation choice is provisional, and
`action_alignment.json` explicitly preserves that uncertainty.

The expert Bench2Drive collector caller connecting `tick/save` to world ticks,
control generation/application, and count increments is missing. Person2Drive's
original human recording/rendering/metadata-export caller is also unavailable.
Evaluation-agent execution order and predictive error cannot substitute for
these missing collection-time callers. Ten Hz is the observation frequency;
it does not establish that each recorded command was held for the entire next
0.1-second transition. Candidate plans use a 10-Hz discretization assumption. See docs/dataset_foundation.md and the archived
experiments/dataset_foundation/results/source_checks.json at dataset-foundation-v1.

## Splits and normalization

Use only the saved episode lists. Bench2Drive preserves its old training routes
and divides the old holdout into validation/test; physical town/route groups
are disjoint. Person2Drive holds out complete drivers and episodes, with 10/3/2
drivers (the original 8/2/2 assignments are preserved; supplementary drivers
add two training and one validation driver). Both physical routes necessarily repeat across splits: this evaluates
new drivers on familiar routes, not generalization to unseen environments.
The Bench test partition comes from a historical validation set, so it is not
a pristine prospective benchmark despite being held out of this experiment.
No windows or source frame identities cross splits. The final verifier also
checks exact JPEG hashes across splits.

The adapter does not normalize states or commands. Fit scalers on training
records only. The diagnostic fits separate source scalers and separate dynamics
models; it does not assert identical physical responses across the two sources.
Person2Drive has more windows, so a future mixed-source training sampler should
make its source weighting explicit. Speed-only observations intentionally omit
gear, slope, tire/vehicle parameters and other possible hidden variables.

## Six-camera RGB and geometry

Six views are the default, in fixed order: `rgb_front`, `rgb_front_left`,
`rgb_front_right`, `rgb_back`, `rgb_back_left`, `rgb_back_right`. Use `cameras=`
to select a subset, which retains canonical order. Front-only output keeps the
camera axis: `(4,1,3,H,W)` and `(10,1,3,H,W)`. Every required view must exist for
all 14 observations; no frame/camera substitution is performed. Missing views
reject windows; counters expose losses. The frozen cohort retains all 143,431
windows after requiring six views. Numerical targets/actions are unchanged.

```python
from src.unified_driving_dataset import load_training_datasets
six = load_training_datasets()  # requires calibration; all six views
front = load_training_datasets(cameras=('rgb_front',))
# Optional explicit crop, applied identically to every view, with updated K:
cropped = load_training_datasets(image_crop=(160,90,1440,810))
```

For `L=history+horizon=14` and `C` selected views, samples also return:

| Field | Shape | Meaning |
| --- | --- | --- |
| `camera_ids`, `camera_indices` | C | Directory identities and canonical 0..5 IDs |
| `camera_intrinsics_original` | `(L,C,3,3)` | Original pinhole K |
| `camera_intrinsics` | `(L,C,3,3)` | K after crop/resize |
| `image_affine` | `(L,C,3,3)` | Original-pixel to processed-pixel mapping |
| `original_image_size` | `(L,C,2)` | Actual decoded width,height |
| `camera_to_ego` | `(L,C,4,4)` | Optical camera to same-observation forward/right/up ego |
| `camera_native_to_ego` | `(L,C,4,4)` | Original source convention, labelled by camera_native_axes |
| `camera_geometry_valid` | `(L,C)` | Calibration availability/validity |
| `camera_capture_timing_verified` | scalar bool | False: capture clocks are not saved |

Pixel coordinates use zero-based pixel centers. For crop offset left/top and
resize scale sx/sy, `u' = sx*(u-left+0.5)-0.5`, likewise for v. The loader uses
`K_processed = image_affine @ K_original`. Default preprocessing is bicubic
resize only; explicit crop bounds are validated before PIL can pad. Original
matrices and pixels are retained. No undistortion is performed.

Bench raw world2cam is actor-camera forward/right/up; Person raw camera matrices
are optical right/down/forward, and its native ego y is left-positive. They are
converted independently to a common optical-camera to forward/right/up ego.
That common ego convention is left-handed, so these coordinate transforms can
have determinant -1; do not convert them to SO(3) quaternions without reconciling
handedness. The six cameras do not share one K: rear FOV is wider. Camera mounts
are relative to the body at each observation, not the anchored planar motion
label frame. Raw world2cam is preserved on disk but is not returned as privileged
future global pose input.

Lossless calibration sidecars are in `data/<source>/manifests/cameras/`, derived
from the original retained JSON/PKL calibration. `load_training_datasets()`
requires valid geometry by default. Explicit `require_calibration=False` is an
audit option: unavailable geometry returns invalid masks/NaNs, not invented
identity matrices. Direct `UnifiedDrivingDataset` users must choose whether to
require calibration.

All views align to available metadata frame IDs, but native capture timestamps
are unavailable. The reference Bench interface requests a common simulator
frame; the original production/export caller and empirical camera timing skew
are not certified. Do not interpret the timing-verification flag as true merely
because camera_geometry_valid is true. See [the audit](multicamera/README.md)
for per-episode frame statistics, calibration diagnostics, contact sheets,
loading memory/time and remaining synchronization limits.
