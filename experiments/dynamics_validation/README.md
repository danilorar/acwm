# Multi-step ego-speed dynamics diagnostic

This experiment tests speed prediction over 1 second from four ego observations
at 10 Hz and a supplied 10-step candidate control sequence. No visual inputs,
encoder or Transformer are implemented. The production dataset and previous
experiment are unchanged.

## Reproduce

From `/home/danilo/Desktop/world_model`:

```bash
.venv/bin/python experiments/dynamics_validation/run.py
.venv/bin/python experiments/dynamics_validation/audit.py
```

Uses the project's installed NumPy, PyTorch, Pillow and Matplotlib. CPU only,
one thread; the default 30 model fits took about 54 seconds before plotting on
this machine. Flags: `--root`, `--output`, `--epochs`, `--seeds`, `--lr`.
The audit uses the default results path. Results are replaced on rerun.
All new scripts, results and Matplotlib cache are inside this experiment folder.
The fixed folds are read from the previous experiment's `results/config.json`;
the audit optionally compares against its existing `predictions.csv` (required
for that comparison). No dependency on downloaded model weights.

## Data and exact indexing

Reuse `Bench2DriveDataset(history=4, horizon=10)` for paired image/annotation
frame IDs, consecutive windows and gzip annotation reading. Images are never
decoded. There are 2,165 windows across ten scenarios. Windows do not cross
scenario boundaries or missing frames. Cutoff t is the fourth observation.
Four observations span 0.3 seconds; ten future transitions span 1.0 second.

| Input or target | H0 outgoing hypothesis | H1 incoming hypothesis |
|---|---|---|
| Observed speeds | v_(t-3), ..., v_t | Same |
| Three observed transition controls | a_(t-3), a_(t-2), a_(t-1) | a_(t-2), a_(t-1), a_t |
| Future candidate controls | a_t, ..., a_(t+9) | a_(t+1), ..., a_(t+10) |
| Targets | v_(t+1), ..., v_(t+10) | Same |

Here a = [steer, throttle, brake], kept separate. History token i contains
observed speed i and the control assigned to its incoming observed transition.
The first token has no preceding observed transition: control features and a
validity mask are zero. Other tokens have mask one. Neither history uses any
control whose recorded frame is later than t. In H0, a_t is provided as the first
candidate, not duplicated in history. In H1, a_t is history and a_(t+1) is the
first candidate. The complete future plan is an explicit conditional input,
not an observation available at the cutoff.

`audit.py` verifies every slice, perturbs future controls to check that observed
history is unaffected, checks paired initialization, and independently recomputes
all metrics from saved rollouts. No ground-truth future speed enters the model's
forward method or recurrent feedback, during training or validation.

## Model and fair evaluation

- Encoder: one GRU layer, input width 5 (speed, 3 controls, validity), hidden 32.
- Decoder: GRUCell, input width 4 (own previous speed, current candidate control),
  hidden 32; linear speed-increment head; add increment and clamp speed at zero.
  Decoder state begins at the encoder's final hidden state. Each of the ten steps
  uses the model's own previous predicted speed, never teacher forcing.
- Train with full-horizon speed MSE, Adam lr=0.003, 80 full-batch updates,
  gradient norm clipped to 5. Fixed last epoch, no validation model selection.
- Same architecture and initialization for each H0/H1 pair, seeds 0, 1, 2.
  Deterministic PyTorch operations, one CPU thread.
- Reuse all five previous scenario folds exactly: eight training and two held-out
  scenarios per fold. Each window has one out-of-fold rollout per seed/model.
  Overlapping windows remain entirely within their scenario partition.
- Shared input scaler per fold uses training history speeds and pooled training
  H0/H1 history/future controls only. No validation information enters scaling.
  Future speed targets are not normalized. Scaler statistics and folds are saved.

Baselines, evaluated on exactly the same windows:

1. Constant speed: repeat v_t ten times.
2. Longitudinal physics, separately fitted for H0 and H1:
   acceleration = bias + throttle_gain*throttle - brake_gain*brake - drag*speed;
   v_next = max(0, v + 0.1*acceleration).
   The three gains are nonnegative; bias is unrestricted. Fit one-step acceleration
   by sign-constrained least squares on **unique training-scenario transitions**,
   solving all eight active sets. Roll forward using own predicted speed.
   No validation data fits the parameters. This intentionally simple baseline
   lacks control lag, collisions and road/vehicle effects.

## Aggregate results

Errors are speed errors in m/s. Full-horizon errors pool all ten steps and all
held-out windows before computing each seed's score. Means and sample SDs are
across seeds, not confidence intervals. Long scenarios receive more weight.

| Model | Full MAE | Full RMSE | 0.1s RMSE | 0.5s RMSE | 1.0s RMSE |
|---|---:|---:|---:|---:|---:|
| GRU_H0 | 0.4778 ± 0.0049 | 0.8852 ± 0.0020 | 0.3370 | 0.8674 | 1.1642 |
| GRU_H1 | 0.4637 ± 0.0042 | 0.8601 ± 0.0026 | 0.3198 | 0.8374 | 1.1419 |
| physics_H0 | 0.7008 ± 0.0000 | 1.2729 ± 0.0000 | 0.3766 | 1.2063 | 1.7639 |
| physics_H1 | 0.7758 ± 0.0000 | 1.3505 ± 0.0000 | 0.3930 | 1.2803 | 1.8726 |
| constant | 0.9956 ± 0.0000 | 1.6740 ± 0.0000 | 0.4236 | 1.5363 | 2.4077 |

GRU H1 lowers full RMSE by 2.8% relative to GRU H0 and about 32% relative to the
better physics baseline. Its 1-second RMSE is about 53% below constant speed.
Both GRUs beat both baselines on full-horizon RMSE in every scenario.
H1 has lower aggregate MAE and RMSE at every evaluated horizon (0.1–1.0s).
Thus H0 is not preferred in this temporal setup. H1 wins scenario full RMSE
on 9/10 clips (Accident is the exception), and 1-second RMSE on 9/10
(OppositeVehicleTakingPriority is the exception). The small aggregate margin
must not be confused with proof of the recording convention.

Per-scenario **full-horizon RMSE**, seed means:

| Scenario | GRU H0 | GRU H1 | Physics H0 | Constant |
|---|---:|---:|---:|---:|
| AccidentTwoWays | 0.4526 | 0.4405 | 0.8797 | 1.2789 |
| Accident | 0.5368 | 0.5396 | 1.1717 | 1.6815 |
| ConstructionObstacle | 0.8491 | 0.8270 | 1.4476 | 1.9236 |
| ControlLoss | 0.9888 | 0.9410 | 1.3429 | 1.6666 |
| DynamicObjectCrossing | 0.8525 | 0.8364 | 1.2616 | 1.6831 |
| HardBreakRoute | 0.5400 | 0.5359 | 1.0428 | 1.4709 |
| OppositeVehicleTakingPriority | 1.5053 | 1.4963 | 1.7865 | 2.2582 |
| ParkedObstacle | 1.0617 | 1.0286 | 1.5459 | 1.9830 |
| VehicleTurningRoute | 1.1984 | 1.1575 | 1.4231 | 1.7306 |
| YieldToEmergencyVehicle | 0.9435 | 0.8954 | 1.0860 | 1.4484 |

All individual scenario, horizon, MAE and RMSE results, including both physics
alignments and seed variation, are available in metrics.csv and summary.csv.

## Braking and abrupt changes

Strong-braking subset: initial speed >=2 m/s and at least one observed future
speed decrement <=-0.5 m/s in one 0.1s step (deceleration >=5 m/s²). There are
390 overlapping windows. This defines **strong observed deceleration**, which
may include collision effects; it does not establish commanded braking.

On this subset, GRU H1 has full MAE **1.041 m/s**, full RMSE **1.675 m/s**, and
1-second MAE/RMSE **1.382/2.176 m/s**. GRU H0's full RMSE is 1.713; physics H0's
is 2.489; constant speed's is 2.786. The models improve on baselines but do not
reliably reproduce abrupt stops. Example rollouts visibly smooth or delay sharp
speed drops, especially in ControlLoss and VehicleTurningRoute.

The immediate-braking subset selects a strong decrement in the *first* future
step, not necessarily the onset of a braking episode (79 windows). GRU H1's
0.1s MAE/RMSE is **1.010/1.235 m/s**, versus constant **1.333/1.531**.
Across the unique annotation stream, only 38/79 abrupt moving-speed decrements
have brake>=0.5 at t, and 37/79 at t+1. Neither alignment makes every abrupt
deceleration directly attributable to a strong recorded brake command.

The previous MLP's saved predictions were also evaluated on exactly the same
first target transitions: overall H0/H1 RMSE **0.403/0.418**, versus GRU
**0.337/0.320**. However, on immediate strong deceleration the previous H1 MLP's
RMSE was **1.195**, slightly better than GRU H1's **1.235**. Temporal modeling
helps ordinary prediction but does not uniformly solve the abrupt-change issue.
This is a comparison of different models/objectives/control representations,
not a controlled ablation isolating the benefit of history alone.

## Error accumulation

GRU H1 RMSE grows from **0.320 at 0.1s** to **0.837 at 0.5s** and **1.142 at
1.0s**, about 3.6 times the first-step value. Its MAE grows from 0.177 to 0.648.
The growth is substantial, though much smaller than constant speed's 2.408 m/s
1-second RMSE. These are free-running errors, not teacher-forced scores.

## Candidate controls and coverage

For identical held-out histories with initial speed >=2 m/s, replace every
future command with full throttle, coasting, or full brake (steering zero).
Mean predicted 1-second speed changes:

| Model | Full throttle | Coast | Full brake |
|---|---:|---:|---:|
| GRU H0 | +5.590 | -5.227 | -7.153 |
| GRU H1 | +5.035 | -4.676 | -7.157 |

Both models preserve the expected brake <= coast <= throttle ordering in all
4,281 tested history/seed pairs per alignment (0.05 m/s tolerance). Magnitudes
are not independently validated; the large coasting loss is a warning sign.
Only **20/1,527 moving annotations (1.31%)** have both throttle and brake <=0.01.
There are **zero full-second coasting windows** from moving starts; sustained
full throttle/brake windows are also very rare (1–4 per alignment/command,
ignoring steering). These hypothetical plans are outside observed sequence
coverage even though individual command values occur in the data.

There are **187/2,295 annotations** with both throttle and brake >0.01. Separate
commands preserve information that throttle-minus-brake alone would discard.
This overlap is an observation, not proof that the controls are inconsistent.

## Assessment and uncertainty

1. **Predictive feasibility:** yes, the temporal model materially improves
   recorded-plan speed prediction over both baselines over one second.
2. **Alignment:** H1 has a small but more consistent temporal advantage. This
   reverses the prior aggregate one-step preference. Do not change the main
   dataset alignment automatically or claim collection-time timing is resolved.
3. **Braking:** improved relative to simple baselines but still poor on abrupt
   deceleration; selected hard stops are badly mistimed or underestimated.
4. **Rollout reliability:** error grows with horizon; 1-second RMSE around 1.14
   m/s overall and 2.18 m/s during strong deceleration is not precise dynamics.
5. **Observed coverage concerns:** coasting and sustained candidate plans are
   poorly represented. Large speed changes do not always coincide with strong
   recorded braking. Missing collisions, slope, gear/reverse, controller/actuator
   delay and additional state are plausible explanations, not established causes.
6. **Next development decision:** there is enough signal to proceed with an
   exploratory visual world-model prototype and retain these diagnostic baselines.
   There is not enough evidence to call this a reliable counterfactual dynamics
   model or use it for trusted MPC. Targeted event/control-coverage investigation
   remains necessary before making planning claims. No visual model work is done
   by this experiment.

H1's future recorded actions may themselves respond to future observed states.
Conditional predictive success on recorded plans can therefore reflect expert
feedback/endogeneity rather than causal control effectiveness. Neither this
experiment nor the monotonicity check proves behavior under interventions.
The missing expert collection caller still prevents conclusive timing tracing.

Limitations: only ten clips, one scenario-fold partition, three optimizer seeds,
correlated overlapping windows, a small fixed training budget, and speed-only
validation. Seed SD excludes uncertainty over new scenarios. Four scalar speed
observations cannot describe full vehicle/environment dynamics. No matched
no-history ablation or larger-data generalization study is performed. Nonnegative
prediction clipping ignores tiny negative recorded speed noise.

## Files

- run.py: train paired models, fit baselines, evaluate, plot.
- audit.py: indexing/leakage/metrics checks, previous MLP comparison and coverage.
- results/metrics.csv: per-seed/model/scenario/subset/horizon scores;
  step 0 means full horizon, steps 1–10 indicate forecast endpoint.
- results/summary.csv: seed means and sample SDs.
- results/config.json: fold membership, parameters, normalizers, runtime.
- results/rollouts.npz: held-out arrays for independently reproducing scores.
- results/training.csv: first/final training losses (fixed budget, no tuning).
- results/control_responses.csv and control_response_summary.csv: candidate tests.
- results/audit.json and previous_mlp_reference.csv: audit and matched reference.
- results/horizon_errors.png: MAE/RMSE versus horizon, seed SD bands.
- results/rollout_examples.png: four stress cases selected by largest observed
  1-second speed drop within each named scenario; target-based selection is for
  visualization only, never training/checkpoint selection.
- results/braking_analysis.png: strong-deceleration and immediate-braking scores.
- results/control_response.png: hypothetical candidate-control response.
