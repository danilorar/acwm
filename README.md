# ACWM

Clean starting point for the action-conditioned visual world model's **visual
encoder stage**. Six-camera data preparation is complete; no encoder, visual
feature cache, Transformer or RSSM training has been implemented.

```text
data/
  bench2drive/          # six-camera RGB, annotations and calibration
  person2drive/        # six-camera RGB, selected indexes/calibration + source metadata
  config/              # tracked frozen splits, offsets, cohort and expected counts
src/
  unified_driving_dataset.py
tools/
  check_dataset.py     # offline integrity/batching check
tests/
  test_unified_driving_dataset.py
docs/
  dataset_foundation.md
  unified_driving_dataset.md
```

## Use the prepared data

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/check_dataset.py
```

```python
from torch.utils.data import DataLoader
from src.unified_driving_dataset import load_training_datasets

datasets = load_training_datasets()
loader = DataLoader(datasets['train'], batch_size=2, shuffle=True, num_workers=0)
batch = next(iter(loader))
```

Images have shapes `(B,4,6,3,224,224)` and `(B,10,6,3,224,224)`. For front-only,
use `load_training_datasets(cameras=('rgb_front',))`, retaining camera axis C=1.
Intrinsics, extrinsics, camera IDs and validity masks accompany the images.
The training view has **95,927 windows**, validation **25,810**, test **21,694**.
The loader reads only metadata-linked images; extra unindexed RGB is retained
but is not automatically part of the training cohort.

Read [the dataset contract](docs/unified_driving_dataset.md) and
[the foundation/context summary](docs/dataset_foundation.md) before implementing
the encoder. Keep source identity, frozen splits, raw controls and camera
geometry conventions. File-ID alignment does not certify capture-time sync or
applied-control timing.

## Analysis and data recovery

Preparation/download/audit/benchmark scripts, detailed reports, plots and source
manifests are on **`archive/dataset-analysis`**, with immutable six-camera tag
**`dataset-six-camera-v1`**. Earlier front-only experiments are preserved at
`dataset-foundation-v1`. Main deliberately has no dependency on their code.

```bash
# Read the complete camera report without leaving main:
git show dataset-six-camera-v1:docs/multicamera/README.md
# Open the full analysis/preparation workspace separately:
git worktree add ../acwm-dataset-analysis archive/dataset-analysis
```

Raw compressed archives, Mini/exploratory backups and unselected driver metadata
were removed after the analysis checkpoint was pushed. Prepared RGB, all Bench
annotations, selected Person indexes, all calibration sidecars, and original
metadata for the 15 selected Person drivers remain. To recover deleted raw
sources, use the archived pinned manifests/tools; substantial re-downloads would
be required. Git tags contain code/reports/manifests, not the large datasets.
