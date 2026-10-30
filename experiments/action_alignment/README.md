# Bench2Drive action alignment diagnostic

Compare H0 (`[v_t, steer_t, throttle_t - brake_t]`) with H1
(`[v_t, steer_(t+1), throttle_(t+1) - brake_(t+1)]`) to predict the same
`delta_v = v_(t+1) - v_t`. Recording interval: 0.1 s (10 Hz).
This experiment does not change the production dataset or training pipeline.

## Run

From the project root:

```bash
.venv/bin/python experiments/action_alignment/run.py
```

Dependencies are the project's existing NumPy, PyTorch, Pillow and Matplotlib.
Defaults: CPU, one PyTorch thread, seeds 0/1/2, 200 full-batch Adam steps,
learning rate 0.003. Flags: `--root`, `--output`, `--seeds`, `--split-seed`,
`--epochs`, `--lr`. Paths resolve relative to the script, so execution from
another working directory works too. Results in the output directory are replaced
on rerun. No checkpoints or image tensors are needed.

## Protocol

- Reuse `Bench2DriveDataset(history=1, horizon=1)` for matching RGB/annotation
  frame IDs and indexing consecutive frame pairs, and its gzip annotation reader.
  Only annotation values are loaded; images are not decoded.
- All 10 scenarios contribute 2,285 one-step transitions. Missing frame IDs are
  never bridged, and pairs never cross scenario boundaries.
- Five fixed scenario folds, shuffled with split seed 2026: train on eight
  scenarios and validate on two. Each transition receives exactly one held-out
  prediction per model and training seed. Fold membership is in `results/config.json`.
- Three independent training seeds per fold; reset the seed before constructing
  each model so H0 and H1 have identical initial parameters.
- MLP: 3 inputs, Linear(3,64), ReLU, Linear(64,64), ReLU, Linear(64,1).
  MSE loss; identical optimizer and training settings. Fixed final epoch;
  no validation-based checkpoint selection or hyperparameter tuning.
- Both models use the same input normalization: mean/std of the pooled H0 and H1
  training inputs in that fold. No held-out data enters normalization.
  Targets stay in physical units. Full-batch updates avoid batch-order differences.
- Zero baseline predicts delta_v=0 (equivalently next speed equals current speed).
- Aggregate metrics pool all held-out transitions, weighting scenarios by their
  transition counts. Per-scenario metrics are also reported. Seed SD is sample
  standard deviation, not a confidence interval or independent-sample uncertainty.
- Next-speed MAE/RMSE are identical to delta-v MAE/RMSE when current speed is
  treated as observed exactly, because adding v_t cancels in the prediction error.

## Results from the default run

Values are mean ± sample SD across three training seeds, in m/s per recorded
transition (speed change, not acceleration).

| Model | Held-out MAE | Held-out RMSE |
|---|---:|---:|
| H0: action at t | 0.236469 ± 0.000506 | 0.397163 ± 0.003827 |
| H1: action at t+1 | 0.249941 ± 0.003418 | 0.411149 ± 0.010044 |
| Zero change | 0.248219 ± 0 | 0.416122 ± 0 |

H0 lowers aggregate RMSE about 3.4% relative to H1 and 4.6% relative to the
baseline. It lowers MAE about 5.4% relative to H1 and 4.7% relative to baseline.
H1 slightly improves baseline RMSE but has worse baseline MAE.

Per-scenario seed-mean RMSE:

| Scenario | H0 | H1 | Zero |
|---|---:|---:|---:|
| AccidentTwoWays | 0.2360 | 0.2523 | 0.3139 |
| Accident | 0.3006 | 0.3188 | 0.4070 |
| ConstructionObstacle | 0.3683 | 0.3625 | 0.4748 |
| ControlLoss | 0.4917 | 0.5850 | 0.4113 |
| DynamicObjectCrossing | 0.3440 | 0.3200 | 0.3930 |
| HardBreakRoute | 0.2832 | 0.2991 | 0.3731 |
| OppositeVehicleTakingPriority | 0.4906 | 0.4404 | 0.5286 |
| ParkedObstacle | 0.4121 | 0.4302 | 0.4595 |
| VehicleTurningRoute | 0.5526 | 0.5354 | 0.4657 |
| YieldToEmergencyVehicle | 0.4657 | 0.4201 | 0.4302 |

H0 wins MAE on 8/10 scenarios but RMSE on only 5/10. Both models are worse
than the zero baseline on ControlLoss and VehicleTurningRoute. H0 also loses
baseline RMSE on YieldToEmergencyVehicle. ParkedObstacle is sensitive to seed.
These results support H0 as the better aggregate predictive convention for this
small diagnostic, with substantial scenario dependence.

## Artifacts

- `results/metrics.csv`: aggregate and per-scenario metrics for each seed/model.
- `results/summary.csv`: seed means and standard deviations.
- `results/predictions.csv`: paired held-out predictions with scenario/frame IDs.
- `results/config.json`: training settings, validation folds and normalizers.
- `results/comparison.png`: actual-vs-predicted scatter plots (seed 0),
  per-scenario RMSE with seed SD, and a held-out HardBreakRoute trace (seed 0).

## Interpretation and limitations

This is evidence about predictive usefulness, **not proof of collection-time
control alignment**. The missing expert collection caller remains unresolved.
H1 also uses an action recorded at the target observation, which could be a
response to that observation; predictive success would not establish causality.

There are only ten clips, one fixed fold partition, three seeds and a small MLP.
Adjacent transitions within a scenario are correlated. Seed variation measures
optimization variation, not uncertainty across new scenarios or recording runs.
The state omits road slope, gear/reverse, collisions, control history and actuator
lag; these can dominate speed changes. Signed throttle-minus-brake may discard
information if both controls overlap. Abrupt switching and long stationary
segments make average error an incomplete diagnostic. This evaluates only
one-step speed dynamics, not steering/yaw dynamics or multi-step rollout quality.

Keep H0 as the provisional convention for returning to world-model development;
do not change the main dataset alignment based on this experiment alone.
