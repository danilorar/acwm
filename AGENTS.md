# Project context

- Read README.md, docs/dataset_foundation.md and docs/unified_driving_dataset.md.
- Runtime data: data/bench2drive/extracted and data/person2drive/extracted.
  Person labels/indexes are in data/person2drive/manifests/indexes.
- Canonical cohort/splits/hypotheses live in data/config; do not silently reselect,
  randomly split overlapping windows, clip raw controls, or bridge missing IDs.
- Use src.unified_driving_dataset.load_training_datasets for the frozen views.
- Keep source identity and conservative control-quality filtering. Person native
  ego y is left-positive; shared targets are right-positive. Raw controls are
  preserved. H1 Bench / H0 Person are predictive hypotheses, not timing proof.
- Main has no dependency on external/ or experiments/. Complete analysis is in
  archive/dataset-analysis and tag dataset-foundation-v1. Do not reconstruct it
  on main unless the current task needs it.
- Tests: python -m unittest discover -s tests -v. Real corpus check:
  python tools/check_dataset.py. No network download is needed for either.
- Do not delete local data/archive merely because Git ignores it: its Mini,
  exploratory samples, outputs and upstream worktree are not backed up by a tag.
- Current milestone ends at dataset preparation. No DINOv3/Transformer/RSSM
  training has been performed. Follow the user's next requested scope.
