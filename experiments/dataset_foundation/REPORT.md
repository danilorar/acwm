# Dataset foundation: final assessment

The fixed cohort is ready for DINOv3 extraction and a bounded **observational world-model prototype**. It is not sufficient evidence for a large final model intended to rank counterfactual plans or drive MPC. Resolve those requirements with intervention tests and targeted rare-control collection before committing to that use. No visual feature extraction or visual-model training was started in this experiment.

## Inventory and integrity

| Source | Episodes | Matched frames | Frames in valid K4/H10 windows | Windows |
| --- | --- | --- | --- | --- |
| bench2drive | 132 | 35,510 | 35,510 | 33,794 |
| person2drive | 80 | 117,798 | 117,359 | 109,637 |

Verified 153,308 indexed JPEG structures and 636 representative decoded windows. No source-frame identities, overlapping windows, or exact JPEG hashes cross splits. Every indexed image exists; all numeric values are finite and pose rotations are orthonormal. Default samples exclude windows touching 61 Person steering labels outside the nominal [-1,1] range (observed raw range -1.1987 to +1.9646). Those raw rows are preserved and flagged; applied clipping/scaling is unknown, so no conversion is guessed. Full JPEG decoding was sampled at three windows per episode, rather than every image. Bench has 3,154 negative near-zero signed speeds (minimum −0.06124 m/s; none below −0.1 m/s). Raw values are preserved and included in losses; small rollback versus simulator jitter is not resolved. The baseline zero clamp cannot reproduce those small negatives.

## Splits

| Split | Episodes | Matched frames | Frames in windows | Windows |
| --- | --- | --- | --- | --- |
| train | 140 | 102,673 | 102,375 | 95,927 |
| validation | 36 | 27,454 | 27,357 | 25,810 |
| test | 36 | 23,181 | 23,137 | 21,694 |

Bench2Drive: 88/22/22 route-disjoint episodes, retaining all original training routes. Person2Drive: 52/14/14 episodes from 10/3/2 disjoint drivers. Test drivers remain driver05 and driver39. Both physical Person2Drive routes repeat across splits, so this evaluates new drivers on familiar environments. Bench test routes were part of a historical validation set; this is not an untouched prospective benchmark. Coverage-curated held-out episodes are not a random population sample.

## Storage

| Tree | Allocated GiB | Files |
| --- | --- | --- |
| bench2drive | 2.27 | 71,030 |
| person_archives | 67.77 | 242 |
| person_metadata | 10.37 | 148 |
| person_front_rgb | 9.41 | 171,726 |
| person_manifests | 0.02 | 281 |

Dataset trees above total 89.85 GiB; free disk: 271.24 GiB. Source archives are retained for resumption. No LiDAR or other camera views were extracted. Hugging Face cache accounting is separate in storage.json; prior data are preserved.

## Coverage

| Source | Coast / moving frames | Partial / all frames | Partial brake frames >5 m/s | Partial / frames >5 m/s | 1s coast H0/H1 | 1s partial H0/H1 | Multi-transition H0 windows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bench2drive | 2.75% | 0.00% | 0 | 0.00% | 0 / 0 | 0 / 0 | 24,183 |
| person2drive | 13.87% | 2.81% | 2287 | 3.20% | 2,744 / 2,860 | 537 / 586 | 26,120 |
| combined | 12.07% | 2.16% | 2287 | 2.62% | 2,744 / 2,860 | 537 / 586 | 50,303 |

Moving means speed >=2 m/s; pedals <=0.01 define coasting; 0.01<brake<0.99 defines partial braking. Intermediate-braking counts above 5 m/s are frame-level, not independent events. Longest selected moving coast is 65 frames (6.5 seconds by frame count; 6.4-second observation span); partial-braking runs reach 39 frames (3.9/3.8 seconds).

Person moving-coasting and partial-braking percentages are lower than in the earlier two-driver sample; broadening the cohort changes its proportions even while increasing absolute support. The old sample was not a population estimate. Explicit maneuver weighting is still needed. All 50 drivers and 800 sensor episodes were inspected through their image-associated numeric metadata (1,175,820 referenced frames). Sensor archives were downloaded only for the selected 80 episodes. The final 15-driver subset combines a balanced 12-driver primary cohort and eight rare-control supplements. Person has no matching curated hazard-type annotation in the consumed metadata, so its scenario analysis uses routes and control-event proxies. Bench contributes 44 scenario types, 12 towns and 23 weather settings; Person contributes human controls on routes 103/108, not 80 distinct physical routes.

The complete release contains only 14 H0 one-second partial-braking windows starting at 2–5 m/s and 21 partial-braking windows with strong steering; all are present in the selected raw metadata. Some raw windows may be excluded by the conservative action-range gate; consult training_window_coverage.csv for exact eligible training sequences. They overlap heavily and come from very few episodes/drivers. No low-speed sustained partial-braking test windows are available: the two supporting drivers are in train/validation. Only 20 of 75 fast-coasting/strong-turning windows are selected. Bench has no partial braking and no one-second moving-coasting windows. High-speed strong steering, long combined turning/braking, varied intervention sequences and new Person routes remain poorly covered. More random frames from these same routes will not resolve those structural gaps. The 117,798 selected Person metadata rows have minimum speed 0.2001 m/s: exact standstill/terminal-stop states are absent from this metadata subset. The reason for that cutoff is not established. Bench supplies stopped states, but does not establish human partial-braking-to-stop dynamics. Additional front images without image-linked metadata are retained but excluded; raw 20-Hz logs cannot safely fill them by ordinal joining.

Plots: [speed-conditioned summaries](results/speed_conditioned.png), [conditional control heatmaps](results/speed_control_heatmaps.png), [raw controls](results/control_distributions.png), [continuous durations](results/action_durations.png). Counts by speed/steering/regime and episode are in the coverage CSVs; zero cells are retained.

## Dynamics sanity check

All compared models use the same conservative control-quality gate, and no raw control is clipped. Separate source-specific GRUs use identical H0/H1 samples/initializations and training-only normalization. Two seeds, 60 updates, and at most 12,000 episode-balanced training windows per source keep this diagnostic small. The first predicted step measures one-step accuracy from the same model trained for free-running ten-step prediction. No future ground-truth speed is used as decoder input. Models were not pooled across different source physics.

| Source | Model | 0.1s RMSE | 0.5s RMSE | 1s MAE | 1s RMSE | Pooled horizon RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| bench2drive | constant_speed | 0.386 | 1.373 | 1.302 | 2.100 | 1.479 |
| bench2drive | constant_accel | 0.297 | 1.480 | 1.839 | 2.779 | 1.766 |
| bench2drive | physics_H1 | 0.355 | 1.131 | 1.018 | 1.631 | 1.186 |
| bench2drive | GRU_H1 | 0.279 ± 0.002 | 0.665 ± 0.004 | 0.471 ± 0.002 | 0.838 ± 0.003 | 0.660 ± 0.001 |
| person2drive | constant_speed | 0.286 | 1.293 | 1.795 | 2.359 | 1.524 |
| person2drive | constant_accel | 0.139 | 0.888 | 1.558 | 2.206 | 1.265 |
| person2drive | physics_H0 | 0.157 | 0.549 | 0.631 | 0.877 | 0.604 |
| person2drive | GRU_H0 | 0.137 ± 0.000 | 0.408 ± 0.009 | 0.434 ± 0.020 | 0.580 ± 0.022 | 0.422 ± 0.012 |

All errors are speed m/s. ± denotes sample standard deviation across two seeds, not a confidence interval across episodes/drivers. Constant acceleration extrapolates the last observed speed increment. The simple physics model fits acceleration = bias + positive throttle gain × throttle − positive brake gain × brake − positive drag × speed, on unique training transitions only. Both controls hypotheses for physics are retained in the metrics.

The GRU beats the baselines over one second on both sources, but error grows with horizon. Person one-step constant-acceleration error is competitive; this is not a uniform GRU win at every metric.

### Held-out episode and driver consistency

- bench2drive: GRU improves 1s RMSE over constant speed on 22/22 test episodes; episode RMSE ranges 0.450–1.536 m/s (median 0.747).
- person2drive: GRU improves 1s RMSE over constant speed on 14/14 test episodes; episode RMSE ranges 0.487–0.651 m/s (median 0.583).

| Person held-out driver | Windows | GRU 1s RMSE | Constant 1s RMSE |
| --- | --- | --- | --- |
| driver05 | 8139 | 0.590 ± 0.030 | 2.594 |
| driver39 | 8799 | 0.569 ± 0.014 | 2.118 |

### Braking/coasting subsets at one second

| Source | Subset | Windows | GRU MAE | GRU RMSE | Constant RMSE |
| --- | --- | --- | --- | --- | --- |
| bench2drive | coasting | 0 | N/A | N/A | N/A |
| bench2drive | partial_braking | 0 | N/A | N/A | N/A |
| bench2drive | strong_braking | 2019 | 0.738 | 1.126 | 2.473 |
| bench2drive | strong_deceleration | 822 | 1.156 | 1.691 | 3.398 |
| person2drive | coasting | 343 | 0.702 | 0.813 | 3.818 |
| person2drive | partial_braking | 134 | 0.853 | 1.123 | 6.698 |
| person2drive | strong_braking | 117 | 0.789 | 1.107 | 7.099 |
| person2drive | strong_deceleration | 2561 | 0.604 | 0.814 | 3.958 |

Dynamics coast/partial masks require all 11 control labels to agree so both hypotheses share the same target windows. Strong braking means any brake>=0.8 command while starting moving; strong deceleration means a recorded single-step speed drop>=0.5 m/s. Counts are overlapping windows; scarce cases are not independent replications. H1 is slightly better than H0 on Person partial braking, so the preferred aggregate hypothesis is not uniformly best across maneuvers.

### Alignment and action response

- bench2drive: validation pooled-horizon RMSE H0=0.6301, H1=0.6131; provisional choice H1.
- person2drive: validation pooled-horizon RMSE H0=0.3527, H1=0.3576; provisional choice H0.
- bench2drive: candidate ordering fails in 0.0–0.0% of speed-stratified held-out histories across seeds (any horizon, 0.05 m/s tolerance). Mean 1s coast change ranges -4.63 to -4.62 m/s.
- person2drive: candidate ordering fails in 0.0–3.0% of speed-stratified held-out histories across seeds (any horizon, 0.05 m/s tolerance). Mean 1s coast change ranges -3.34 to -3.33 m/s.

The synthetic command probe covers speed bins 2–5, 5–10, 10–15 and >=15 m/s, up to 250 histories/bin. The expected speed order is full brake <= partial brake <= coast <= full throttle. Finite/nonnegative outputs and broad acceleration thresholds do not establish physical realism: nonnegative speed is imposed by the model, and long Bench coasting is unsupported. Observed Bench 99.5th-percentile absolute acceleration is about 22 m/s², indicating abrupt/unmodeled events; the GRU smooths some of these rather than explaining them. Candidate-plan changes have no intervention ground truth.

Bench steering/right-yaw association favors H0, whereas aggregate speed prediction favors H1. Different channels, sampling/actuator delays, hidden variables and losses can produce conflicting predictive preferences. Neither source has the authoritative collection/export caller needed to certify controls at a saved RGB tick. `source_checks.json` names the missing callers; the adapter requires explicit per-source offsets and preserves raw command indices. Coordinate calibration is independently checked, not treated as timing proof. Ten Hz describes saved observations, not a verified zero-order hold of each command for the whole 0.1-second interval; within-interval control changes are not established by the available caller provenance.

See [error accumulation](results/dynamics_horizons.png), [representative rollouts](results/rollout_examples.png), per-episode/driver CSVs, `prediction_quality.json`, and `candidate_checks.csv`.

## Recommendation and remaining uncertainty

Proceed with a fixed-dataset visual prototype and DINOv3 features when requested. Keep source identity, source-specific normalization/physics sensitivity, explicit action offsets and the saved splits. Use explicit source/maneuver sampling weights rather than letting the larger Person window count dominate. Keep an honest rare-event test set and report per-source results.

Do not treat this as certification for counterfactual planning or launch a large final planning model based only on these scores. Required evidence for that use includes known applied-command timestamps, vehicle/gear/slope/physics context, intervention-response tests, and better low-speed/turning partial-braking coverage. The full-release audit shows the latter is intrinsically rare in the current source; human full-stop supervision also needs investigation because the selected metadata omit exact standstill. Two Person routes and two test drivers cannot establish broad environment or human-driver generalization. More seeds, grouped resampling and a prospective test split would strengthen uncertainty estimates.

This diagnostic evaluates longitudinal speed only, not learned lateral displacement/yaw dynamics. Coordinate transforms for those targets are checked, but lateral rollout accuracy is not established. It did not test visual predictability or a shared mixed-source world model. Existing data, original dataset code and historical experiments were retained. No DINOv3 cache, visual encoder, Transformer or RSSM training was run.
