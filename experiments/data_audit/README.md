# Bench2Drive Mini data audit

**Decision: READY WITH CHANGES.** No fundamental corruption or matching failure
was found. Correct the ego-position representation and explicitly configure action
indexing before using the present dataset class for action-conditioned visual
training. Preserve all real transitions, including hard stops and overlapping
controls. Limited control coverage constrains subsequent planning claims, not an
initial observational world-model prototype.

The audit only writes under `experiments/data_audit/`. It does not modify the
main dataset, previous experiments, raw recordings, or train any models. No new
dataset is downloaded. No visual encoder, Transformer, RSSM or MPC is implemented.

## Reproduction and scope

From `/home/danilo/Desktop/world_model`:

```bash
.venv/bin/python experiments/data_audit/run.py
```

Uses existing NumPy, PyTorch/Pillow dependencies of `Bench2DriveDataset`, and
Matplotlib. Reuses its gzip JSON reader without loading its training windows.
All ten folders are audited, not a random sample. Outputs include CSVs, five
focused figures, raw-source hashes, and snapshots of local documentation and
prior experiment summaries. Existing outputs in this experiment are overwritten.
No training or downloads occur. The script reads prior experiment summary CSVs
and local collection documentation, so run it within this project checkout.

Integrity: **2,295 annotations, 2,295 matching front RGB images, zero missing
counterparts, zero discontinuous frame IDs, zero nonfinite audited numeric
values, zero out-of-range controls, and zero JPEG header/structure verification
failures**. JPEG `verify()` is a structural check, not an exhaustive pixel decode.
All gzip annotations parse successfully. Numeric checks cover speed, controls,
positions, heading and acceleration, not every nested actor field.
Recording interval is 0.1 seconds. Valid sequence counts remain 2,165 for 4+10
frames. Filename matching establishes dataset pairing, not original simulation
capture/application timestamps.

## A. Throttle and brake overlap

| Threshold (both strictly above) | Frames | Percent of all 2,295 |
|---|---:|---:|
| 0.01 | 187 | 8.15% |
| 0.05 | 186 | 8.10% |
| 0.10 | 154 | 6.71% |

All overlapping brake values are exactly 1. Overlapping throttle ranges from
0.02092 to 0.25649, median 0.12749. At the 0.01 threshold, overlaps occur only
in ControlLoss (129), VehicleTurningRoute (30), ParkedObstacle (14), and
OppositeVehicleTakingPriority (14). There are 50 runs, mostly short; the longest
is ControlLoss frames 177–295 inclusive: **119 frames**, 11.9 sampled seconds
(endpoint span 11.8s). Most of that run is a prolonged stop. Of all 187 frames,
143 have |speed|<0.1 m/s, 38 have speed>2 m/s, and the remaining six are in the
intermediate speed range. Some overlap occurs during strong deceleration and
ordinary moving segments too. It is not exclusively a collision or stop artifact.
No collision cause is established for overlapping commands.

**Important correction to earlier interpretation:** signed control does *not*
lose raw action information on these actual Mini recordings. Brake is strictly
binary {0,1}, and throttle when brake=1 never exceeds 0.25649. Thus the observed
signed values for brake-on and brake-off are disjoint:

```python
u = throttle - brake
recovered_brake = float(u < 0)
recovered_throttle = u + recovered_brake
```

This reconstructs every original throttle/brake value with **zero error**.
Simultaneous commands alone do not prove information loss on a restricted data
support. Our earlier report overstated this issue. The mapping is nevertheless
many-to-one over arbitrary continuous controls: (0.2,0.2) and (0,0) both give zero.
`u` is also not a calibrated physical force or acceleration; no claim that
throttle and brake effects subtract linearly is justified.

**Recommendation:** retain `[steer, throttle, brake]` for the final world model.
It preserves raw actuator semantics and extends safely to other collections or
candidate plans with fractional brakes. This is a moderate design improvement,
not evidence that the old Mini experiments lost control values. Keep overlap
frames; do not force either control to zero or classify all overlap as erroneous.

## B. Coasting and action coverage

Define moving coasting as speed>=2 m/s, throttle<=threshold and brake<=threshold.
At thresholds 0.01, 0.05 and 0.1 the result is unchanged: **20/1,527 moving frames
(1.31%)**, in **11 runs**, maximum **three consecutive frames** (0.3 sampled
seconds; 0.2s between endpoints). Median coasting speed is 6.586 m/s. The CSV
also evaluates moving threshold 0.5 m/s for sensitivity. Run and speed ranges
are saved individually in `sequences.csv`.

| Scenario | Frames | Overlap >0.01 | Moving ≥2 | Coast ≤0.01 | Max coast run |
|---|---:|---:|---:|---:|---:|
| AccidentTwoWays | 349 | 0 | 176 | 0 | 0 frames |
| Accident | 207 | 0 | 179 | 3 | 3 frames |
| ConstructionObstacle | 207 | 0 | 142 | 3 | 2 frames |
| ControlLoss | 350 | 129 | 195 | 3 | 3 frames |
| DynamicObjectCrossing | 214 | 0 | 171 | 0 | 0 frames |
| HardBreakRoute | 293 | 0 | 177 | 0 | 0 frames |
| OppositeVehicleTakingPriority | 134 | 14 | 92 | 0 | 0 frames |
| ParkedObstacle | 164 | 14 | 130 | 0 | 0 frames |
| VehicleTurningRoute | 228 | 30 | 149 | 7 | 3 frames |
| YieldToEmergencyVehicle | 149 | 0 | 116 | 4 | 2 frames |

Intermediate throttle is common: **1,074/2,295 (46.8%)** lie strictly between
0.1 and 0.9. Intermediate brake is absent: **0 frames** strictly between 0.1 and
0.9; the complete brake alphabet is {0,1}. Thus this is not a generally continuous
three-actuator coverage dataset. Coasting appears in only five scenario clips,
and no full-second moving coasting sequence exists.

The previous candidate-control experiment's severe coasting deceleration is
therefore an extrapolation failure hypothesis supported by poor coverage; it
is not proof that the recorded coasting frames are wrong. Even constant full
throttle/full brake one-second plans are rare. Observationally correlated
controls, road context and expert feedback make arbitrary control interventions
hard to identify from this Mini set.

**Larger collection:** verified local README lines 39/67 and
`external/Bench2Drive/docs/v004_update.md:21` describe a newer 0.0.4 training set
with **44 scenarios ×25 routes =1,100 routes**, uniform across scenarios. The
README also describes the older Full collection of 13,638 clips. These are
different releases and must not be conflated. Mini recordings were obtained
from the original `rethinklab/Bench2Drive` download list; a 0.0.4 code checkout
does not establish that their expert collection logic is from the new release.

More clips provide more scenario/route diversity and may increase rare-action
counts. **Nothing in the inspected documentation verifies more coasting,
fractional brake commands, or adequate MPC action coverage.** No control
histograms for the larger collection were available locally. Audit a small
representative sample or obtain collection metadata before making that claim;
do not assume downloading more expert demonstrations fixes support gaps.

## C. Extreme deceleration

There are **106 consecutive transitions** with finite-difference acceleration
<=-5 m/s², including low-speed transitions. The previous dynamics audit's 79
count additionally required current speed>=2 m/s; these counts are compatible.
Maximum magnitude is **26.65 m/s²**, about 2.7g, which is unusually large for
ordinary road braking and should be treated as simulator-domain behavior until
vehicle physics/collection settings are known.

All values below use speed in m/s, acceleration in m/s²:

| Scenario / transition | Speed before → after | FD acceleration | IMU x at t / t+1 | Control (T,B) at t / t+1 |
|---|---:|---:|---:|---|
| OppositeVehicleTakingPriority 42→43 | 3.50→0.84 | -26.65 | -19.45 / -27.04 | (0.26,1) / (0.26,1) |
| HardBreakRoute 33→34 | 5.31→2.65 | -26.64 | -4.69 / -20.69 | (0.00,1) / (0.00,1) |
| ConstructionObstacle 49→50 | 5.61→2.98 | -26.26 | -14.63 / -23.31 | (0.50,0) / (0.00,1) |
| Accident 66→67 | 2.98→0.37 | -26.16 | -18.53 / -26.71 | (0.00,1) / (0.00,1) |
| Accident 65→66 | 5.60→2.98 | -26.14 | -3.84 / -18.53 | (0.00,1) / (0.00,1) |

Recorded IMU x becomes strongly negative around the major drops. At
OppositeVehicleTakingPriority 42→43, FD=-26.65 and next IMU x=-27.04. At
HardBreakRoute 33→34, FD=-26.64, IMU x progresses -4.69→-20.69 and then -26.37
in the following frame. Several other scenarios have similar corroboration.
This argues against treating large speed drops as isolated bad speed numbers.
It does not establish realistic real-world friction or a unique causal mechanism.

The RGB context shows an open intersection in the first selected case, a lead
vehicle braking in HardBreakRoute, and a construction obstruction in the third.
Physical vehicle/walker box gaps at the next frame for these cases are positive
(15.40m, 4.61m, 1.93m lower bounds respectively). These are **not confirmed ego
collisions**. The construction prop may not be represented among those actor
classes. No collision flags/logs were found in annotation keys or extracted
metadata filenames. Scenario names like Accident describe the scenario, not
proof that ego collided.

A descriptive evidence rule labels the 106 events as 40 ordinary/moderate
braking candidates, 45 simulator emergency-braking candidates (nearby strong
brake plus negative IMU), and 21 unresolved interaction/timing candidates. These
are heuristic triage labels, not ground-truth causal classifications. Five events
have overlapping ego-frame axis-aligned bounding envelopes of a recorded physical
actor; that is only a possible-contact cue. Oriented shape, z, collision response,
missing static props and actor annotation reliability still matter. No event is
labeled a confirmed collision.

Each event in `extreme_events.csv` includes before/after speed and controls,
three-axis next IMU, ego positions, yaw/compass, displacement, nearest recorded
physical actor ID/type and approximate box gaps. The box gap is a lower bound
from actor vertices enclosed by an ego-frame AABB, not an exact contact test.
`extreme_deceleration.png` covers ±about one second around the three most
negative events from distinct scenarios. Red bands mark the selected interval.

**Acceleration conventions:** annotation `acceleration` comes from IMU sensor
components (`tools/data_collect.py:76`), mounted with zero yaw/roll/pitch
(`:624` onward). It is not a world-x velocity derivative. IMU x is approximately
longitudinal; sensor-frame rotation, road attitude, possible gravity effects,
instantaneous sampling and finite-interval averaging matter. The FD signal is
plotted at interval start, while IMU samples are observations at their timestamps.
Across scenarios, FD acceleration correlation with IMU x at t+1 is 0.87–0.96,
versus 0.21–0.72 at t for moving transitions. This supports the time-label
convention for acceleration, not the timing of saved controls. A full calibrated
3D specific-force-to-world-acceleration comparison is outside this audit.

**Action:** preserve events with flags; report event-stratified losses, optionally
use a robust/heteroscedastic training objective, and retain context in the visual
model. Do not clip targets to ordinary-car braking limits or automatically delete
extreme samples. Physics parameter provenance and collision logs would clarify
causes, but their absence does not establish data corruption.

## D. Action alignment

Lag analysis uses transitions v_(i+1)-v_i, matching only consecutive IDs within
one scenario and current speed>2 m/s. Lag L means recorded command at i+L.
L=0 is H0, L=1 is H1. Correlations are scenario-specific, with counts in the CSV.
Signed-command/delta-speed correlation favors lag 0 over lag 1 in 8/10 scenarios;
OppositeVehicleTakingPriority and ParkedObstacle favor lag 1. Correlation maxima
across the wider lag range need not be at either hypothesis.

Event-triggered plots include 213 sharp brake rises (>0.5) and 208 signed-command
drops (<-1), with full ±five-transition context and current speed>2 m/s. The
strongest mean negative acceleration occurs around the command-change index.
Events can overlap and involve controller feedback; they are not independent
impulse-response experiments. Physical response delays, incoming observations,
expert reactions to emerging hazards and high autocorrelation can produce
similar timing signatures under different collection conventions.

Previously saved results are reviewed, not retrained:

| Experiment | H0 aggregate RMSE | H1 aggregate RMSE |
|---|---:|---:|
| One-step MLP, delta speed | 0.3972 | 0.4111 |
| Multi-step GRU, full speed horizon | 0.8852 | 0.8601 |
| Multi-step GRU, 1-second speed | 1.1642 | 1.1419 |

These metrics have different targets/windows/model classes and should not be
compared as absolute loss values across experiments. The previous MLP also used
signed controls, while the GRU used separate controls and temporal history;
this is not a clean ablation isolating action timing or history.

The 0.0.4 public collector `Env_Manager.tick()` reads current `get_control()` and
sensor data; `save()` copies that tick dictionary into shared-index RGB and
annotation files. The evaluation loop advances the world, gets sensor data,
runs the agent, and applies its returned control. However, **the expert collection
caller linking those methods to control application, world ticks, save cadence
and count increments is missing**. That gap is unchanged. Evaluation order alone
cannot prove training recording timing; no new log timestamps resolve it here.

**Provisional strategy:** make alignment an explicit dataset/config option. For
the upcoming multi-step recorded-plan prototype, provisionally use H1, matching
the direct temporal diagnostic, while retaining H0 as an evaluation sensitivity
check. H1's recorded frame t+1 control is treated as the hypothesized control
of t→t+1, not as an instruction to wait for a future sensor frame at deployment.
At deployment the model receives the candidate plan from its first outgoing
transition onward. Preserve raw recording indices and the hypothesis in metadata.
Do not claim this establishes causal intervention semantics or permanently shift
raw files. Recorded future actions may react to future states, unlike an open-loop
candidate plan; this remains a planning-identification limitation.

## E. Dataset implementation review and exact locations

All line numbers refer to the unchanged `src/bench2drive_dataset.py`.

| Lines | Finding | Proposed action |
|---|---|---|
| 40–53 | Annotation/image numeric matching is correct on all ten clips. Non-numeric annotation basenames would raise an error; none occur. | Keep matching; report missing pairs instead of silently masking future dataset issues. |
| 65–75 | Consecutive windows are correctly constrained to a scenario. | Keep; validate history/horizon are positive integers and dataset nonempty. |
| 87–96 | RGB conversion, tensor order and pixel scaling are correct. Square resizing distorts 16:9 images; PIL size is width,height. | Document a consistent deployment resize/crop policy and chosen encoder normalization. Not a verified data bug. |
| 116–137 | Compass used as world yaw rotates positions about 90 degrees from intended ego-forward/right axes. | Use explicit compass→yaw conversion or precise world2ego transform; define handedness and verify straight-line motion. |
| 120–133 | Raw x/y are projected GPS sensor positions, not clean actor-center ground truth. | Separate noisy input odometry from precise pose supervision and account for sensor lever arm. |
| 139–147 | Relative sin/cos heading representation correctly handles angle wrapping; constant compass offset cancels in dtheta. | Keep representation; document angle convention. |
| 152–159 | Two-dimensional signed control is empirically invertible on Mini but not generally on arbitrary controls. | Prefer raw three-control actions; do not alter/drop overlap records. |
| 167–175 | Returns future targets and past observations separately; no future target enters past slicing. | Preserve separation; trainer must never feed future_images/states into rollout inputs. |
| 171–172 | Slice is correct **conditional on H0**, not a verified universal convention. | Add explicit H0/H1 selection; H0 indices K-1:K-1+H, H1 K:K+H. |
| 167–179 | Past controls are omitted, although history-aware dynamics benefit from them. | Add past transition controls: H0 0:K-1, H1 1:K, for K observed frames. Neither extends beyond cutoff. |
| 25–75 | No train/validation grouping is enforced by the class. | Split by scenario before sampling windows; random_split on overlapping windows would leak. Previous experiment folds are already valid. |

**Confirmed coordinate issue:** every scenario has median
`wrap(compass - actor_world_yaw)` approximately **+90 degrees**. For example,
HardBreakRoute moving transitions have median true ego-forward displacement
+0.776 m and near-zero lateral displacement, while current preprocessing gives
+0.081 m in its nominal forward coordinate and -0.728 m in its nominal lateral
coordinate. This is a consistent rotated frame, not random bad positions, but it
contradicts the stated ego-aligned convention and would misalign trajectory
labels with vehicle controls/geometry.

For these recordings, a planar correction is:

```python
yaw0 = theta0 - np.pi / 2
forward = np.cos(yaw0) * dx + np.sin(yaw0) * dy
right = -np.sin(yaw0) * dx + np.cos(yaw0) * dy
```

This assumes CARLA world x/y and a right-positive lateral convention. For precise
training trajectory targets, prefer the recorded ego pose/world2ego and actor
locations; use a consistent 3D transform when slope matters. Do not pass privileged
future actor poses or ground-truth odometry to a deployment-time input encoder.

**Position residuals:** raw projected x/y differ from actor location by sensor
mount offsets and variability, not merely a constant coordinate translation.
The collector's GPS mount is -1.4m along ego x. After approximate yaw-based mount
correction, residual standard deviations are **0.39–0.53m in x** and
**0.54–0.57m in y** across clips. These are observed residuals, which may include
GNSS noise, projection/calibration and ignored pitch effects; they are not a
calibrated pure sensor-noise estimate. Differencing GPS positions at 0.1s would
create large noisy velocity targets. Keep the recorded speed signal and distinguish
sensor observations from precise supervised position targets.

Other state limitations: speed, local position and relative heading do not encode
reverse/gear, previous actuation, pitch/road grade, wheel state, collision/contact
or surrounding actors. Some are observable from permitted sensors/history/images;
some are hidden variables for a stochastic model. There are small negative recorded
speeds down to -0.0277 m/s; distinguish numerical near-zero noise from real reverse
motion using reverse/gear rather than blindly clamping every negative speed.

## Findings, severity and whether training is blocked

| Problem / status | Evidence | Severity | Recommended action | Blocks training? |
|---|---|---|---|---|
| Confirmed ego-axis convention mismatch | Compass minus world yaw≈90°, wrong forward/lateral displacement | Critical for current pose-conditioned pipeline | Correct and document axes; test forward motion and turns | Yes, when using current position states/targets |
| Confirmed noisy GPS/mount-derived position labels | 0.39–0.57m mount-corrected coordinate residual SD | Moderate | Separate estimated observations from precise ground-truth pose supervision | Before using clean trajectory loss; not speed/image-only training |
| Confirmed implicit H0 indexing, uncertain true timing | Dataset line 172; conflicting diagnostics; missing caller | Moderate | Explicit configurable alignment; provisional H1 plus H0 sensitivity | Blocks an undocumented assumption, not exploratory training |
| Confirmed lack of fractional brake/coasting support | Brake binary; 20 moving coast frames; max run 3 | Critical for unrestricted counterfactual planning | Restrict support claims; audit larger data/collect targeted controls before MPC | No for visual prototype; yes for claiming arbitrary-plan validation |
| Confirmed overlap, but no observed signed information loss | Exact reconstruction on all Mini actions; 187 overlaps | Moderate design concern | Retain original three commands for generality | No by itself |
| Confirmed abrupt simulation deceleration; causal origin uncertain | FD/IMU corroboration, no collision logs | Moderate | Preserve/flag; event-stratified metrics and uncertainty; investigate failures | No; do not delete automatically |
| Potential overlapping-window split leakage | Class emits overlapping windows but does not split | Critical if random window split is used | Enforce scenario-grouped train/validation | Yes until split policy is enforced; prior folds already safe |
| Potential deployment privilege mismatch | x/y-derived past pose and future arrays available in dictionary | Critical if unavailable inputs/targets are fed to model | Explicit input/target schema and sensor availability contract | Before training pipeline is wired; no current leakage demonstrated |
| Confirmed missing past controls in returned schema | No past_actions key | Moderate | Add three observed-transition controls for history-aware training | For planned action-history model, yes |
| Unvalidated resize/state dimensionality settings | Fixed square resize, no positive K/H validation | Minor | Document resize/crop; validate arguments | No for current valid settings |

## Answers and minimal pre-training change list

A. **Suitable for learning recorded-plan dynamics:** yes, with history and context.
These are valid simulator observations; complete causal actuator identification
and arbitrary-plan MPC reliability are not established.

B. **Use three controls:** yes for semantics and future data, while correcting the
previous claim that Mini signed controls necessarily lost information.

C. **Coasting limitation:** significant for MPC and calibration of hypothetical
plans; not a reason to stop initial visual representation/dynamics training.

D. **Extreme deceleration:** corroborated in other signals and plausibly simulator
emergency braking for many events. No proof that all are realistic braking,
collisions, or annotation corruption. Preserve them and keep causal labels tentative.

E. **Provisional alignment:** configurable H1 for multi-step training, H0 sensitivity;
record exact indexing. No conclusive timing proof.

F. **Dataset changes needed:** correct pose axes and label provenance, expose
alignment and past controls, enforce grouped splits in the training setup, and
keep original action channels. The class has correct frame/window matching and
relative-angle encoding. There is no demonstrated future-state leakage in its slices.

G. **Before visual-model training:** agree and implement the above dataset/schema
corrections in a separate approved edit; add concise regression checks for forward
motion, turn handedness, both action slices, history cutoff and group split; define
which ego quantities are permitted sensor inputs versus supervision; retain the
existing held-out scenario protocol and braking/coverage diagnostics. Future target
fields must be consumed only as losses. No additional large dataset download or
model architecture work is needed to finish these preprocessing changes.

Once these concrete changes are applied, proceed with the prototype. Additional
coasting/partial-brake coverage and event provenance are requirements for later
planning-validation claims, not open-ended prerequisites for starting research.

## Artifacts and limits

Required figures: `control_overlap.png`, `action_coverage.png`,
`extreme_deceleration.png`, `alignment_analysis.png`. One extra focused figure,
`event_context.png`, shows selected RGB evidence. CSVs preserve scenario/threshold
counts, runs, coordinate residuals, event details, correlations, event-triggered
means, and prior-model results. `metadata_evidence.json` captures local documentation
lines and missing collision metadata, `action_representation.json` the reconstruction
check, and `source_manifest.json` identifies unchanged source versions used.

Counts are descriptive for ten clips. Overlapping windows, repeated stop frames,
correlated switches and expert feedback are not independent experimental samples.
Box envelopes are approximate and actor coverage incomplete. No IMU calibration,
collection-driver reconstruction, collision-log recovery, new causal experiment or
new model fit is claimed. Proposed source corrections are documented only; main
code and prior results remain untouched.
