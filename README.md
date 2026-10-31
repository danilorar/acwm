# Think Before Act

Action-conditioned visual world model. Dataset preparation and diagnostics are
complete; visual feature extraction and world-model training have not started.

```text
data/                   # local datasets; only config/ is tracked
  bench2drive/          # active Bench2Drive corpus
  person2drive/         # active Person2Drive corpus
  config/               # frozen splits, action hypotheses, source manifests
  archive/              # ignored legacy data and backups; never used for training
src/                    # unified dataset and restricted metadata reader
tools/                  # dataset restoration and real-data smoke check
tests/                  # focused adapter/extraction tests
docs/                   # dataset contract and analysis summary
```

## Setup and use

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

Default image shapes are `(B,4,6,3,224,224)` and `(B,10,6,3,224,224)`.
Use `load_training_datasets(cameras=('rgb_front',))` for front-only data (camera
axis size one). Intrinsics, extrinsics, camera IDs and validity masks accompany
the images. See [the six-camera preparation report](docs/multicamera/README.md).

Both sources are needed. Train/validation/test select disjoint episodes in the
same physical folders; a folder name does not define the split.
See [the dataset contract](docs/unified_driving_dataset.md) and
[the foundation summary](docs/dataset_foundation.md).

## Restore missing data

The following commands use frozen manifests, never choose a different cohort.
They are unnecessary on this machine, where the dataset is already prepared.
Downloading requires substantial disk space; archive transfers happen only with
explicit download flags. Sensor archives contain extra modalities, but only
annotations/front RGB first; the multi-camera step below then extracts exactly
the six surrounding RGB views, preserving front files. No LiDAR/depth is extracted.

```bash
python tools/download_bench2drive.py             # show frozen selection
python tools/download_bench2drive.py --download
python tools/extract_bench2drive.py
python tools/prepare_person2drive.py --metadata --index
python tools/prepare_person2drive.py --download --extract
python tools/prepare_multicamera.py              # audit and measure missing RGB
python tools/prepare_multicamera.py --extract
python tools/build_camera_geometry.py
python tools/check_dataset.py
```

## Archived investigation

The complete analysis is retained in branch `archive/dataset-analysis` and tag
`dataset-foundation-v1` (commit `17b4cf3`). The clean main branch does not depend
on experimental scripts, the old Bench-only adapter, Mini, or the upstream
Bench2Drive checkout. Therefore `.gitmodules` and the old patch/CSV-specific
`.gitattributes` are unnecessary on main.

Read a historical report without changing branches:

```bash
git show dataset-foundation-v1:experiments/dataset_foundation/REPORT.md
```

For old experiments, use a separate worktree and follow its historical setup:

```bash
git worktree add ../world_model-analysis archive/dataset-analysis
```

Ignored data are not stored in Git. The historical code expects the old names
`data/bench2drive_train`, `data/person2drive_train`, and sometimes Mini; point
those names in the analysis worktree to the corresponding current datasets.
Historical Person manifests/index paths may also need the old-name symlink.
Keep those aliases in the analysis worktree, not the clean main project.
