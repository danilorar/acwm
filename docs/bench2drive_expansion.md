# Bench2Drive 0.0.4 expansion

## Current status: complete

The selected source is **rethinklab/Bench2Drive-V0.0.4**, not the older Base
collection. The official Hub file listing contains 1,100 route archives and
44 scenario types. Selection is pinned to revision
`d84ae18cf20a88b9220df1bc15c073d8824b7ac1`.

Prepared selection: **132 unique routes, three per scenario type**, across
12 towns and all 23 represented weather labels. All types receive equal weight,
including ordinary/Vanilla driving, intersections, emergency braking, following,
cut-ins, pedestrians, turns and lane changes. Within each type, selection prefers
less-represented towns/weather, with deterministic SHA256(seed:filename) tie
breaking. Seed: 42. Existing selection prefixes are preserved when expanding.

All **132 selected archives** were downloaded, checked against authoritative
SHA256 hashes, and safely extracted. The retained data are raw annotations and
front RGB; all original archives are still present. Mini, the refactored dataset
class, and all historical experiment files/results are unchanged. No visual
encoder or Transformer training was started.

## Measured result

| Item | Result |
|---|---:|
| Routes / scenario types | 132 / 44 (exactly three routes per type) |
| Towns / weather labels | 12 / 23 |
| Frames | 35,510 |
| Available 4-history + 10-future windows | 33,794 |
| Training routes / windows | 88 / 22,274 |
| Validation routes / windows | 44 / 11,520 |
| Retained raw archives including small HF metadata | 49.25 GiB |
| Extracted annotations and front RGB | 2.12 GiB |
| Archives plus extracted data | 51.37 GiB (55.16 GB logical file sizes) |
| Available disk after completion | about 310.26 GiB |

Every type is represented in both train and validation. Scenario distribution
and exact filenames, SHA256 hashes, source revision and split assignments are
saved in manifests. No selected town/route ID appears on both sides. No selected
ID matched the official benchmark evaluation XML route IDs.

## Measured action-coverage comparison

Same definitions in both collections: moving speed>=2 m/s; coasting throttle and
brake<=0.01; intermediate controls strictly between 0.1 and 0.9; overlap both>0.01.

| Metric | Mini | Selected 0.0.4 subset |
|---|---:|---:|
| Total frames | 2,295 | 35,510 |
| Moving frames | 1,527 | 19,003 |
| Moving-coasting frames | 20 | 522 |
| Moving-coasting fraction | 1.31% | 2.75% |
| Moving-coasting runs | 11 | 383 |
| Longest coasting run | 3 frames | 6 frames |
| Coasting runs with >=10 recorded frames | 0 | 0 |
| Intermediate throttle | 1,074 (46.80%) | 13,637 (38.40%) |
| Intermediate brake | 0 | 0 |
| Brake alphabet | {0,1} | {0,1} |
| Throttle/brake overlap | 187 (8.15%) | 132 (0.37%) |

The largest coasting run is 0.6 sampled seconds, or 0.5 seconds between first and
last observed samples. Coasting frame count is 26 times Mini's and the moving
coasting rate roughly doubles, but the runs are still very short. **This is a
modest improvement in transient coasting, not a resolution of sustained-coasting
or fractional-braking coverage.** Increased route/scenario/visual diversity is
substantial; unrestricted candidate-control coverage for MPC is still inadequate.
No causal/generalization improvement is inferred solely from more frames.

The steering, throttle and brake distribution plot is
`data/bench2drive_train/manifests/action_coverage.png`, using common bins for both
collections. Detailed metrics are in coverage_report.json and route_coverage.csv.

## Compatibility and readiness

The unmodified Bench2DriveDataset loaded the new archive layout successfully.
All 35,510 annotations were checked for finite speed/control values and a unique
finite simulator ego pose. Extraction checked image/annotation ID pairing in
every recording. Integration loaded a sample from each of the 132 recordings and
batched two samples, using small 32x18 image resizing for the diagnostic:

```text
past_images:           (2,4,3,18,32)
past_ego_observations: (2,4,1)
past_actions:          (2,3,3)
future_actions:        (2,10,3)
future_motion_targets:(2,10,5)
```

At the dataset's normal 224x224 default, image spatial dimensions become 224x224;
actions and motion dimensions are unchanged. Controls remain three raw channels,
and action_offset=1 remains provisional. Simulator poses are supervision only.
The full suite's 25 tests passed, and the downloader/extractor tests were also
rerun after the transfer-worker change. Completion checks confirmed every selected
archive and extraction, all 88/44 split assignments, and unchanged historical file
hashes. JPEG decoding was checked on integration samples, not exhaustively across
all images; archive hashes and gzip CRCs were verified for all recordings.

**Ready for the next development stage:** the larger subset can now be used for
visual feature extraction and recorded-plan world-model training. This task does
not start that work. Additional targeted coasting/fractional-brake data and
counterfactual validation remain necessary for later MPC claims. Exact
collection-time action timing is still unresolved; no dataset repair is inferred
from these coverage measurements.

## Reviewed storage plan

At planning time:

| Item | Size |
|---|---:|
| Actual selected archive bytes, from authoritative Hub metadata | 52,879,234,513 bytes = 52.88 GB = 49.25 GiB |
| Available disk space | about 361.94 GiB |
| Conservative additional-space budget | 218.82 GiB |
| Safety reserve included in budget | 20 GiB |

The original reviewed budget included archives, three times their compressed size
as a retained extraction allowance, one maximum-archive cache/partial margin and
the reserve. The downloader now defaults to four workers and budgets one maximum
archive per worker (about 224.3 GiB from an empty destination). Both estimates
comfortably fit; actual retained storage after completion is only 51.37 GiB.
**The multiplier is an allowance, not a measured expansion ratio.** Hugging Face
metadata publishes compressed sizes/hashes, not tar member expanded sizes.
Before writing a recording, extraction reads its tar headers, sums the exact
retained annotation/front-RGB sizes, and checks available space plus reserve.
The plan fits without reducing the initial 132-route count. No full-collection
transfer is initiated. Archives are retained until explicit deletion approval.

Destination:

```text
data/bench2drive_train/
  archives/       # Verified raw tar.gz downloads, HF resume metadata
  extracted/      # Only completed recordings, original directory names
  manifests/
    catalog.json
    selection.json
    storage_plan.json
    download_status.json       # Created during transfer
    extracted.json             # Created during extraction
    download_failures.jsonl    # Only if failures occur
    extraction_failures.jsonl  # Only if failures occur
```

Staging is a sibling `.extract_staging/`, outside `extracted/`, so interrupted
partial recordings cannot accidentally enter the recursive dataset loader.

## Tools and execution

Use the project virtual environment; huggingface_hub was added to requirements.

Plan only (default; no archive transfer):

```bash
.venv/bin/python tools/download_bench2drive.py --num-routes 132 --seed 42
```

Commands used after approval (safe to rerun; verified files are reused):

```bash
.venv/bin/python tools/download_bench2drive.py --num-routes 132 --seed 42 --download
.venv/bin/python tools/extract_bench2drive.py
.venv/bin/python tools/audit_bench2drive_expansion.py
```

For an initial approved compatibility check, both download/extract commands accept
`--only EXACT_SELECTED_ARCHIVE.tar.gz`. The full selection remains unchanged.
Inspect the first extraction and a batch before proceeding to the rest if the new
archive format differs from Mini. Explain any necessary dataset changes before
making them. The extractor preflights core annotation fields and a unique
simulator ego pose and refuses to silently substitute another pose source.

Downloader defaults to four concurrent transfers (`--workers 4`), with serialized
status writes. It uses official huggingface_hub local-directory downloads (resumable
partial files). It checks authoritative byte size and SHA256 before skipping an
existing file and after transfer. Failed attempts are logged and retried; rerun
the same command to continue interrupted work. No global duplicate snapshot is
requested. The manifest pins the repository revision; it is not refreshed silently.

Extractor independently verifies every source archive's SHA256, rejects traversal,
absolute paths, links, duplicate selected paths and special members, validates
required modality pairing and schema, and checks the gzip CRC through its trailer.
It stages, then promotes complete recordings. Existing extraction markers are
checked against expected file sizes; incomplete or conflicting data are reported
rather than overwritten. Raw archive files are never deleted.

Default retained modalities are **all raw annotation JSON.gz files and original
front RGB image bytes**, which are the current world model's required modalities.
`--all-rgb` additionally retains every RGB camera. Unselected modalities remain
recoverable from retained raw archives. Changing modalities in an existing
completed destination is rejected to prevent duplicate/conflicting extraction;
choose the intended policy before extracting. No resizing/recompression or
annotation rewriting is performed.

## Evaluation exclusion and train/validation policy

Catalog candidates are checked against town/route IDs in
`external/Bench2Drive/leaderboard/data/bench2drive_0.0.4_val.xml` before selection.
No archive IDs in the training repository matched that evaluation list (zero
excluded matches). Only the specified official training repository is queried.
This is a town/route-ID check, not a claim that roads or geographic regions never
overlap between training and evaluation.

Once extracted, the audit chooses one held-out route of each represented type
using a fixed seeded filename hash. It passes those explicit assignments through
existing `split_scenarios`, which rejects shared town/route groups. For the full
initial subset this yields 88 training recordings and 44 validation recordings,
with all 44 scenario types represented on both sides. Actual names and counts
are written to manifests/splits.json after data compatibility checks.
These are recording/route-held-out splits, not unseen-scenario-type holdouts.
No overlapping windows from one route enter both sets. Existing five-fold
experimental splits are not changed.

## Coverage and compatibility report after extraction

`tools/audit_bench2drive_expansion.py` streams original annotations, compares
against Mini using identical definitions, and writes:

- manifests/coverage_report.json: completed archive count, raw/extracted sizes,
  frame/window counts, scenario distribution, control metrics, batch shapes.
- manifests/route_coverage.csv: per-recording coverage statistics.
- manifests/action_coverage.png: steering/throttle/brake distributions vs Mini.
- manifests/splits.json: exact reproducible recording assignments.

It checks schema/pose and finite tensors on one loaded sample from every
recording and batches two samples using the existing dataset. Counts include
moving-coasting frames (speed>=2 m/s, throttle/brake<=0.01), their run durations,
coasting runs of at least ten recorded frames, intermediate throttle/brake
(strictly 0.1–0.9), and overlapping throttle/brake (>0.01). Gaps terminate runs.
At least two recordings per type are required for the diverse holdout, so run
this complete report after the full selected subset is extracted.

Known Mini baseline: 2,295 frames; 20/1,527 moving-coasting frames (1.31%);
longest moving-coast run three frames; 0 intermediate-brake frames; 187 overlapping
throttle/brake frames. **No expanded-data improvement is assumed or claimed in
advance.** More scenario diversity is guaranteed by the selection, but more
coasting/fractional braking remains an empirical question. Recorded-plan
predictive success does not establish causal counterfactual or MPC validity.

## Expand later

```bash
.venv/bin/python tools/download_bench2drive.py --num-routes 176 --seed 42
# Review the new incremental space plan and obtain transfer approval, then:
.venv/bin/python tools/download_bench2drive.py --num-routes 176 --seed 42 --download
.venv/bin/python tools/extract_bench2drive.py
.venv/bin/python tools/audit_bench2drive_expansion.py
```

The original 132 selections remain a prefix. Existing verified archives and
completed matching extractions are reused. Re-running the coverage utility
recomputes a documented seeded split over the expanded collection; freeze/copy
splits.json for experiments requiring unchanged validation membership. The tool
refuses full-collection selection and selection shrink/seed changes in an existing
destination. Use a different destination for a different seed/release/modality
policy. Check storage again before expansion.

## Validation performed so far

All **25 tests pass**, comprising the 22 dataset tests and three new offline
selection/extraction tests. New tests cover balanced prefix-preserving selection,
unsafe path/link rejection, authoritative checksum verification, selective
extraction, completed-extraction reuse and corruption rejection. The completed real integration pass validated loading samples from every
recording and DataLoader batching. The final coverage pass validated pose fields
throughout the annotation collection. No DINOv2/Transformer feature extraction or training is started.
