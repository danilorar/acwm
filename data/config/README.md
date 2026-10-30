# Frozen training configuration

splits.json: episode-disjoint train/validation/test assignments.
action_alignment.json: provisional per-source offsets (Bench H1, Person H0).
bench_catalog.json / bench_selection.json: pinned Bench V0.0.4 source/checksums.
person_catalog.json / person_selection.json: pinned Person source/checksums and
final 80 selected episodes; index paths use the new data/person2drive name.
verification.json: expected numeric fingerprint, counts and prior image checks.

Raw data are ignored. Current source roots are ../bench2drive and ../person2drive;
index paths are project-relative. Older source manifests remain beside the data
for resumability; files in this directory are authoritative for main.
