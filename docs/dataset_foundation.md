# Dataset foundation

Completed analysis: tag `dataset-foundation-v1`, branch `archive/dataset-analysis`
(commit `17b4cf3`). Full reports, metrics, plots, source evidence and experimental
code remain there, rather than in main's working tree.

## Training inventory

| Source | Episodes | Matched frames | Usable frames | Windows |
| --- | ---: | ---: | ---: | ---: |
| Bench2Drive V0.0.4 | 132 | 35,510 | 35,510 | 33,794 |
| Person2Drive | 80 / 15 drivers | 117,798 | 117,359 | 109,637 |
| Total | 212 | 153,308 | 152,869 | 143,431 |

| Split | Episodes | Usable frames | Windows |
| --- | ---: | ---: | ---: |
| Train | 140 | 102,375 | 95,927 |
| Validation | 36 | 27,357 | 25,810 |
| Test | 36 | 23,137 | 21,694 |

Data paths: data/bench2drive/extracted and data/person2drive/extracted. Person
annotations are compact NPZ indexes under data/person2drive/manifests/indexes,
backed by the original driver PKLs in metadata/. Do not keep the images alone.
Canonical splits, source selections and action offsets are in data/config/.
The training accessor and shape contract are in unified_driving_dataset.md.

Bench has 44 scenario types, 12 towns and 23 weather settings. Its original
training routes are retained, with route-ID-disjoint validation/test. Person
has 10/3/2 disjoint train/validation/test drivers. Both physical Person routes
repeat across splits: this is driver generalization on familiar environments.
Bench test was historical validation, not a pristine prospective benchmark.

## Observed findings

- All 50 Person drivers / 800 sensor episodes were inspected through metadata;
  80 sensor episodes were selected, including eight rare-control supplements.
- Person moving coasting: 13.87%; partial braking: 2.81% of all frames, 3.20%
  above 5 m/s. H0 one-second windows: 2,744 moving coast and 537 partial brake.
  These overlap and are not independent events. Bench has neither type of
  one-second window. The complete Person release has only 14 low-speed sustained
  partial-braking windows and 21 turning-partial windows; all are present in the
  selected raw metadata. No low-speed sustained partial-braking test windows.
- Selected Person metadata contain no exact stops (minimum 0.2001 m/s), despite
  additional unindexed RGB. Do not merge 20-Hz raw logs by ordinal image index.
- 61 Person steering labels fall outside nominal [-1,1], ranging -1.1987 to
  +1.9646. Applied scaling/clipping is unknown. Preserve raw fields; default
  windows exclude blocks touching them. The permissive audit view flags them.
- Bench has 3,154 near-zero negative signed speeds, minimum -0.06124 m/s; cause
  (rollback vs jitter) is unverified. Keep them, rather than silently clipping.
- Person native ego y points left; Bench y points right. Simulator poses are
  converted to shared planar anchor-forward/right targets. Steering sign was
  checked empirically; physical magnitude equivalence is not established.

## Lightweight dynamics result

Two-seed, source-specific autoregressive GRUs; train-only normalization and
identical H0/H1 target windows/settings. Held-out speed errors in m/s:

| Source / provisional alignment | GRU 0.1s RMSE | GRU 1s RMSE | Constant 1s RMSE | Physics, same alignment, 1s RMSE |
| --- | ---: | ---: | ---: | ---: |
| Bench / H1 | 0.279 | 0.838 | 2.100 | 1.631 |
| Person / H0 | 0.137 | 0.580 | 2.359 | 0.877 |

GRU improves over constant speed on all 36 held-out episodes. Strong-braking
RMSE is about 1.1 m/s, with limited independent human events. Error grows with
horizon; some synthetic candidate-control orderings fail. Only longitudinal
speed prediction was trained; lateral transforms were checked numerically.
The GRU zero clamp cannot reproduce Bench's small negative target speeds.

H1 Bench / H0 Person were selected by validation error, not test labels.
Collection/export callers linking saved RGB, controls, simulation ticks and
application are missing. A 10-Hz recording rate does not prove commands were
held for the full 0.1-second transition. Predictive usefulness is not proof of
collection-time alignment or counterfactual/intervention validity.

## Readiness and storage

Proceed with DINOv3 features and an observational visual-model prototype when
requested. A final planning/MPC model needs intervention tests, authoritative
applied-control timing/semantics, and stronger rare-control/full-stop evidence.
Keep source conditioning and explicit source/maneuver sampling weights.

Prepared corpus trees occupy about 89.85 GiB, including 67.77 GiB retained Person
archives, 10.37 GiB full-release metadata, 9.41 GiB front RGB and 2.27 GiB Bench.
Only front RGB and required annotations were extracted. Archives are optional
for loading but useful for recovery; no deletion was performed in this cleanup.

Mini, exploratory samples, former outputs and the upstream checkout now live in
ignored data/archive/. They are NOT training inputs and are NOT stored by the
Git tag. Historical scripts/results are stored by Git. No visual encoder,
DINOv3 cache, Transformer or RSSM training has started.
