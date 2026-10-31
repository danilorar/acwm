# Frozen runtime data configuration

- splits.json: episode-disjoint train/validation/test assignments.
- action_alignment.json: provisional source offsets (Bench H1, Person H0).
- person_selection.json: exactly 80 selected episodes and runtime index paths;
  also retains source revision/checksums and the 15 selected driver identities.
- bench_selection.json: pinned 132-route source selection/checksums.
- verification.json: expected numerical fingerprint, counts, cameras and timing.

Data roots are ../bench2drive and ../person2drive; index paths are project-relative.
Large source catalogs and selection-search bookkeeping are archived at
dataset-six-camera-v1, not needed by the runtime loader. Splits and offsets
were not changed by cleanup. Raw data are ignored by Git.
