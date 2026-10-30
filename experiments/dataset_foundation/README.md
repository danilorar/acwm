# Joint Bench2Drive / Person2Drive foundation

This is a dataset preparation and diagnostic experiment. It performs no visual
feature caching, DINOv3, Transformer, or RSSM training. Historical datasets,
adapters and experiments are preserved. Detailed findings are in `REPORT.md`.
The common field contract and indexing examples are documented in
`../../docs/unified_driving_dataset.md`.

## Reproduce

Run commands from the project root using the existing virtual environment.
Dependencies are listed in the root `requirements.txt`; measured package
versions are in `results/environment.json`.

```bash
# Restore the pinned catalog, fetch the selected drivers' metadata, rebuild indexes.
.venv/bin/python tools/prepare_person2drive.py --metadata --plan
# SHA256-verified/resumable downloads; archives contain six cameras and LiDAR,
# but extraction writes only front RGB. Archives are deliberately retained.
.venv/bin/python tools/prepare_person2drive.py --download --extract --workers 4
# Inspect all 50 drivers / 800 episodes and fetch the supplement metadata.
.venv/bin/python tools/inspect_person2drive_release.py
.venv/bin/python experiments/dataset_foundation/release_coverage.py
# Final eight rare-control episodes require the all-driver metadata above.
.venv/bin/python tools/prepare_person2drive_supplement.py
# Diagnostic: fixed splits, two seeds per source/alignment, final verification.
.venv/bin/python experiments/dataset_foundation/run.py
.venv/bin/python experiments/dataset_foundation/source_checks.py
.venv/bin/python -m unittest tests.test_unified_driving_dataset
```

The full metadata audit needs about 10.37 GiB. The primary subset's archives total 58.88 GiB; the eight
rare-control supplements add 8.89 GiB (67.77 GiB total). Before transfer, 356 GiB was free and the conservative
allowance was approximately 138 GiB plus 7.94 GiB for the remaining metadata.
Actual storage is recorded after extraction in `results/storage.json`.
Archive files are indivisible: downloading a sensor archive also transfers its
unused cameras/LiDAR, but none of those extra modalities are extracted.

`manifests/` preserves the small pinned catalog, selected paths/SHA256 values,
and the all-driver inspection. The final selection contains 80 Person2Drive
episodes across 15 drivers, with 117,798 indexed frames; adding Bench2Drive
gives 212 episodes and 153,308 indexed frames. Downloads reside in ignored
`data/person2drive_train/archives/`; numeric metadata in `metadata/`; compact
indexes and completion markers in `manifests/`; front images in `extracted/`.
The Hugging Face client resumes incomplete transfers. Extraction streams through
Zstandard/tar into a staging folder, rejects unsafe members and links, requires
every indexed image, and atomically renames each completed episode. A 20 GiB
free-space reserve is checked while extracting. Existing final folders are not
overwritten. The restricted metadata unpickler allows numeric/pandas types only.

## Method

Default K4/H10 windows exclude every 14-frame block touching a raw control
outside nominal CARLA ranges. Person has 61 out-of-nominal steering labels
(-1.1987 to +1.9646); they are preserved and flagged, never clipped or rescaled.
Applied-control saturation/conversion is unknown. Set
`exclude_uncertain_controls=False` only for explicit audits. Coverage plots show
raw marginals; one-second coverage windows require nominal controls, and
`training_window_coverage.csv` gives exact eligible K4/H10 sequence counts.

Twelve saved drivers include the two previously audited drivers and ten additional
drivers. For each driver and each physical route, three distinct episodes
maximize complementary coasting duration, higher-speed partial braking, and
steering coverage. Raw frame count is only a tie breaker. This deliberately
curated sample is not an unbiased estimate of the complete release. Inspecting
all 800 episode metadata identifies what the selected subset still misses.

The all-driver rare-coverage scan then adds eight episodes: all 14 released
one-second low-speed partial-braking windows, all 21 turning-partial-braking
windows (including those already selected), and more fast turning coast. Those
14/21 windows overlap heavily and remain too scarce for strong claims. The
supplement is selected using coverage only, not model errors. Driver10 and
driver43 enter training; driver03 enters validation. Two extra driver05 episodes
remain in the existing test-driver partition. Final Person driver counts are
10/3/2; both physical routes still repeat across splits.

The baseline is a small GRU history encoder / autoregressive GRUCell decoder,
reusing the existing diagnostic architecture without modifying it. It receives
four observed speeds, three aligned observed controls, and ten candidate
controls. Its own predicted speed drives each subsequent rollout step. It uses
raw separate steering/throttle/brake channels, source-specific training-only
normalization, at most 12,000 episode-balanced training windows per source,
60 Adam updates at 0.003, and seeds 0/1. H0/H1 use identical target windows,
training samples, initialization and optimization. Source-specific models avoid
assuming identical vehicle physics. Constant speed, constant acceleration, and
a training-fitted sign-constrained longitudinal model are evaluated on identical
windows. A single split and two seeds limit statistical claims.

Alignment is chosen separately by source using mean validation full-horizon
RMSE; test labels never select the hypothesis. Scores use all held-out windows.
`step=0` in `dynamics_metrics.csv` pools all ten future speeds; steps 1,5,10 mean
0.1,0.5,1.0 seconds. `episode=ALL` is window-weighted, not episode-weighted.
Per-episode and held-out-driver metrics are provided separately. Variability
across seeds measures optimization variability, not uncertainty across drivers.

Coverage moving speed is >=2 m/s. Coasting means throttle/brake <=0.01;
partial braking means 0.01<brake<0.99; strong braking means brake>=0.8.
One-second coverage uses 11 contiguous states and ten commands under each
offset. Partial-braking windows start moving; coast windows remain moving.
The dynamics coast/partial subsets require **all 11 raw labels** to agree so
H0 and H1 are evaluated on exactly the same targets. Strong-braking dynamics
windows contain at least one strong command and start moving. Control
transitions are channel changes >0.1, at least twice within ten commands.
Counts of overlapping windows are not counts of independent events.
The speed-conditioned zero-pedal curve includes stopped states in the below-2
m/s bin; only the separately defined moving-coasting counts imply motion.
Continuous duration uses run frames / 10 Hz; the timestamp span is one tick
shorter. CSVs retain both definitions.

## Results files

- `splits.json`, `verification.json`, `source_checks.json`: split identities,
  image/shape/pose checks, source calibration and unresolved timing provenance.
- `coverage_summary.csv`, `episode_coverage.csv`, `joint_window_support.csv`,
  `split_control_coverage.csv`, `action_durations.csv`: marginal and joint support.
- `speed_conditioned.png`, `speed_control_heatmaps.png`,
  `control_distributions.png`, `action_durations.png`: coverage plots.
- `dynamics_metrics.csv`, `heldout_episode_metrics.csv`,
  `heldout_driver_metrics.csv`, `alignment_by_episode.csv`: paired errors.
- `dynamics_horizons.png`, `rollout_examples.png`: error accumulation and
  median-error held-out examples (selection rule fixed, not cherry-picked).
- `scenario_inventory.csv`, `scenario_coverage.png`,
  `split_sequence_coverage.csv`: hazard types, repeated routes, drivers and
  per-split support. `state_action_ranges.csv` retains numerical quantiles.
- `action_ordering.csv`, `prediction_quality.json`: candidate-command response
  and numerical/acceleration checks. `candidate_checks.csv` checks ordering at every horizon. Synthetic candidate
  tests have no observed
  intervention outcomes and cannot demonstrate counterfactual correctness.
- `action_alignment.json`, `dynamics_config.json`, `training.csv`, `.pt` weights,
  rollout `.npz` arrays: reproducibility and sensitivity checks.
