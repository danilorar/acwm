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

## Six-camera milestone and encoder readiness

Complete six-camera analysis: tag `dataset-six-camera-v1`, branch
`archive/dataset-analysis`. The earlier front-only experiments remain at
`dataset-foundation-v1`. Read reports with git show or in a separate worktree;
main contains only the runtime adapter, focused tests and a smoke checker.

All 132 Bench routes and 80 Person episodes provide the six requested cameras:
front, front-left, front-right, rear, rear-left, rear-right. The 1,243,416 stored
RGB images are 1600×900 JPEGs, with zero missing state-linked camera IDs and
matching per-view raw frame sets. There are 919,848 metadata-linked images;
additional unindexed Person RGB remains available but is not in the frozen
supervised cohort. All 143,431 windows survive six-camera requirements.

Intrinsics/extrinsics are retained in lossless calibration sidecars; returned
geometry reflects resize/crop. Bench native camera transforms are actor axes,
Person native camera transforms are optical axes with left-positive ego y.
After explicit conversion, measured median intrinsics match and normalized
mount matrices agree within about 6e-6. The common forward/right/up body frame
is left-handed. The rear camera has a wider FOV than the other five.
Frame references/calibration are consistent, but native capture timestamps are
absent, so actual cross-camera skew is unknown. Do not equate geometry validity
or matching filenames with a certified capture clock.

Six-view CPU decode/resize/collation took about 1 second per K4/H10 window.
Batch eight took 7.5–7.7 seconds with ~386 MiB tensor payload and ~2.2 GiB process
RSS (including ~1.4 GiB of loaded records). Full measurements, contact sheets,
per-episode inventories and calibration diagnostics are in the archived report:

```bash
git show dataset-six-camera-v1:docs/multicamera/README.md
```

Proceed with DINOv3 features and an observational visual-model prototype when
requested. A final planning/MPC model still needs intervention tests,
authoritative applied-command semantics/timing, and better rare-control and
human full-stop evidence. Preserve source conditioning and explicit sampling
weights. No visual encoder, feature cache, Transformer or RSSM has started.

## Completed cleanup

The six-camera checkpoint was pushed before deleting raw archives, Mini,
exploratory samples, old outputs/upstream checkout, metadata for 35 unselected
Person drivers, unused episode indexes and preparation-completion bookkeeping.
All extracted RGB (including unindexed frames), all Bench annotations, 80 Person
state indexes, 212 calibration sidecars and original metadata for the 15 selected
Person drivers were preserved. Runtime numerical data, splits and action-offset
choices are unchanged.

Cleanup reclaimed approximately **128.38 GiB**, leaving approximately
**287.80 GiB free** on this machine. Prepared data now occupy about
**77.27 GiB**. Removed archives could only be recovered by
re-downloading the archived pinned sources; they are not stored in Git. The
analysis branch preserves a deletion ledger in docs/cleanup.json. Main removes
preparation scripts/heavy reports/catalogs and keeps this summary plus the
runtime field contract. Minimal runtime dependencies are numpy, Pillow and torch.
