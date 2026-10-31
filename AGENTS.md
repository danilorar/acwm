# ACWM: encoder-stage context

- Read README.md, docs/dataset_foundation.md and docs/unified_driving_dataset.md.
- Only runtime dataset code and focused tests belong on main now. Analysis,
  source restoration, calibration generation and benchmark tools are preserved
  on archive/dataset-analysis; immutable checkpoint dataset-six-camera-v1.
  Older experiments: dataset-foundation-v1. Do not restore them onto main unless
  the current task requires them. Use git show or a separate worktree to inspect.
- Input roots: data/bench2drive/extracted and data/person2drive/extracted.
  Person states/actions: data/person2drive/manifests/indexes; calibration for
  both sources: data/<source>/manifests/cameras. Keep these together.
- Frozen splits, source selections and provisional action offsets are in
  data/config. Do not reselect/split overlapping windows, bridge missing IDs,
  silently clip controls, or delete extracted data without a user request.
- Use src.unified_driving_dataset.load_training_datasets. Default camera axis
  C=6; explicit front-only keeps C=1. Geometry accompanies all 14 observations.
  Native source axes differ; shared camera geometry is optical-to-ego
  forward/right/up (left-handed). Reconcile handedness before assuming SO(3).
  Resize/crop must keep pixel affines and intrinsics consistent.
- Keep source identity and control-quality filtering. H1 Bench / H0 Person are
  predictive hypotheses, not collection timing proof. Capture timestamps/skew
  and applied steering conversion remain unverified. No counterfactual claim.
- Main no longer includes raw archives, Mini/exploratory data or unused driver
  metadata. All extracted RGB and selected original metadata are retained.
- Tests: python -m unittest discover -s tests -v. Real corpus check:
  python tools/check_dataset.py. Neither requires network access.
- Six-view raw RGB costs ~48 MiB per window before collation; batch eight tensors
  ~386 MiB and process RSS ~2.2 GiB in the audit. Start conservatively.
- Current milestone: prepared data, not an implemented model. Do not begin
  visual feature caching/training except within the user's next requested task.
