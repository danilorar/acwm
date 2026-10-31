# Six-camera RGB preparation

Complete for all 132 Bench2Drive V0.0.4 routes and 80 Person2Drive episodes. Existing front RGB and state/action metadata were preserved. No visual encoder, feature cache, model training or feature-fusion architecture was introduced.

## Availability, extraction and synchronization

| Source | Episodes | Frames per requested camera | Stored six-view images | Missing state-linked IDs |
| --- | --- | --- | --- | --- |
| bench2drive | 132 | 35,510 | 213,060 | 0 |
| person2drive | 80 | 171,726 | 1,030,356 | 0 |

All six views are 1600×900 JPEGs. Total stored requested RGB: **1,243,416 images**. All views have the same raw frame-ID set as their episode's front camera, with no internal raw-ID gaps. Missing state/action-linked camera IDs: **0**. Each per-episode available directory, count, ID range, missing-ID list, format and resolution appears in the two availability JSON files. Full raw ID inventories remain in data/<source>/manifests/multicamera/.

Person has 171,726 raw frames per view but only 117,798 metadata-linked observations; the additional 53,928 per-view frames remain unlabelled and are excluded from supervised windows. There are 919,848 metadata-linked RGB images across both sources (153,308 observations × six). Bench also publishes top-down RGB; it was reported as available but not extracted. No LiDAR, depth, semantic or extra camera files were extracted.

Bench source archives were genuinely missing (0/132 at the canonical location). The frozen 132-archive manifest was recovered once, with all SHA256 values verified, transferring about 49.25 GiB. All 80 Person archives were retained and reused. Existing front images were compared byte-for-byte by SHA256 with archive members during audit/extraction and were never overwritten. New camera directories were staged and published only after all six views covered the episode metadata.

The preliminary extra-RGB allowance was 60 GiB, followed by an exact Person audit of 51.34 GiB before its extraction. Bench was audited episode by episode before each episode's extraction; its final additional RGB payload is 9.15 GiB. Total additional RGB payload: 60.49 GiB. Available space remains 159.43 GiB, above the 20 GiB extraction reserve.

**Synchronization status: frame-index aligned; timestamp-level synchronization is not certified.** Matching filenames alone were not treated as proof. Person's six explicit sensor data_path references match every metadata frame_idx, and camera/world/body transforms are geometrically consistent. Bench's reference SensorInterface.get_data(frame) rejects other native simulator frames, and Env_Manager.save() writes the six tick_data views and annotations with one shared count. However, the exact production collection/export caller is not certified, and saved files do not retain the native simulator frame IDs or per-camera capture clock.

No timestamp/capture_timestamp/sensor_timestamp camera fields were found in the consumed metadata, and no JPEG EXIF date tags were found in the scanned RGB payloads. Archive/filesystem modification times are not sensor capture times. Therefore cross-camera timing skew cannot be measured or bounded from these files; it is **unknown, not zero**. Pose-distance/speed cadence checks are in cadence.json and support the 10-Hz observation scale, not capture-time synchronization. Public reference code intent and calibration consistency do not close that gap.

## Geometry

Lossless float64 calibration sidecars live under data/<source>/manifests/cameras/. They retain frame IDs, camera identity, original intrinsics, native camera-to-ego matrices, raw world-to-camera matrices and explicitly converted optical-camera-to-shared-ego transforms. The original JSON/PKL calibration is retained. Bench has no directly reported cam2ego in these annotations; its mount is derived from inverse world2cam and the same-row ego pose. Person reports cam2ego directly.

Bench raw world2cam is CARLA actor-camera coordinates (forward/right/up), not a pinhole optical transform. Person raw world2cam and cam2ego use optical camera coordinates (right/down/forward), with ego y left-positive. Both are converted explicitly to optical-camera → body ego (forward/right/up). This shared ego convention is left-handed: the change-of-basis can have determinant −1. Do not interpret the converted matrices as SO(3) quaternions without reconciling handedness. Raw matrices remain available for alternative conventions.

Across episode-median calibrations, maximum cross-source intrinsic difference is 0 pixels and maximum normalized mount-matrix difference is 6.10352e-06. This agreement was measured, not inferred from camera names. Person reported-versus-pose-derived mount maximum absolute residual is 0.000163795; Bench has no direct reported comparison. Bench per-frame derived mounting variation reaches 0.00305356 in matrix elements, with original variations preserved. These diagnostics do not identify whether small differences are rounding, pose sampling or another collection detail.

The rear view has a wider FOV: fx≈560.166 at 1600 pixels width, versus fx≈1142.518 for the other five views. Do not replace its intrinsics with front-camera K. Consumed calibration provides the pinhole K/transforms; no undistortion is performed.

Images are decoded as RGB float32 in [0,1]. Resizing and optional explicit crop update K with the same affine transform as pixels. Using zero-based pixel centers and PIL resize, u' = sx·(u−left+0.5)−0.5 (and similarly for v). Thus K_processed = image_affine @ K_original. Crop bounds are checked before PIL can pad; absent views are never substituted. No crop is applied by default.

## Unified dataset contract

Fixed order: rgb_front, rgb_front_left, rgb_front_right, rgb_back, rgb_back_left, rgb_back_right. Subsets retain this order. Default tensors: past_images=(4,6,3,224,224), future_images=(10,6,3,224,224). Front-only is cameras=('rgb_front',), retaining a camera dimension of size one. Ego observations, past/future actions and motion targets are unchanged.

For all 14 history/future observations the sample exposes camera_intrinsics, camera_intrinsics_original, camera_to_ego, camera_native_to_ego, camera_geometry_valid, original_image_size and image_affine. camera_ids/camera_indices identify the view axis. camera_to_ego is the mount at that observation, not a future world/anchor pose. Privileged raw world2cam is retained on disk rather than fed as a future model input. camera_capture_timing_verified=False explicitly preserves the timing uncertainty.

load_training_datasets() requires valid calibration by default. Windows touching a missing selected camera or required calibration are rejected, with rejection counters. Explicit permissive geometry auditing returns invalid masks/NaNs, not invented matrices. File disappearance after indexing raises a decoding error rather than selecting another time/view. The existing 61 ambiguous control labels and all gap/episode quality gates remain unchanged.

## Validation

| Split | Front-only windows | Six-camera windows | Missing-view rejection | Missing-calibration rejection |
| --- | --- | --- | --- | --- |
| train | 95,927 | 95,927 | 0 | 0 |
| validation | 25,810 | 25,810 | 0 | 0 |
| test | 21,694 | 21,694 | 0 | 0 |

Total remains **143,431 windows** and **152,869 participating observations**. The numerical fingerprint matches the original foundation. Split files and H1 Bench/H0 Person action-offset files are byte-identical to Git HEAD. Route/driver/episode/frame separation was checked, and complete mixed-source batches loaded from all three splits. Contact sheets show one genuine six-camera frame from each source in the fixed order:

- [Bench contact sheet](bench2drive_contact_sheet.jpg)
- [Person contact sheet](person2drive_contact_sheet.jpg)

## CPU decode and DataLoader memory

Measurements use this Linux machine, CPU torch threads=1, warm OS file cache, num_workers=0, and no model/GPU. Independent forked benchmark processes inherit loaded records; RSS includes shared parent pages. A 5-ms sampler measures process peak RSS. The reported incremental peak excludes initial dataset construction. Loader time includes JPEG decode, crop/resize, tensor construction and collation. No claim is made about disk-cold or GPU/worker-prefetch performance.

| Source | Batch | Seconds | Base RSS MiB | Peak RSS MiB | Batch tensor MiB |
| --- | --- | --- | --- | --- | --- |
| bench2drive | 1 | 0.989 | 1433 | 1521 | 48.26 |
| bench2drive | 8 | 7.511 | 1433 | 2203 | 386.04 |
| person2drive | 1 | 1.013 | 1438 | 1523 | 48.26 |
| person2drive | 8 | 7.683 | 1438 | 2214 | 386.04 |

One six-camera K4/H10 image window alone contains 48.23 MiB of float32 pixels; batch eight needs 385.88 MiB before geometry, temporary samples and collation. Actual batch tensors are about 386.04 MiB. Start raw-RGB experiments with batch 1–2 and num_workers=0; worker prefetch can multiply queued memory and was not benchmarked. Full front/six-camera batch sizes 1/2/4/8 are in loader_benchmarks.json; native-resolution six-image decode-only timings are in decode_times.csv.

## Storage and reproduction

| Tree | Allocated GiB |
| --- | --- |
| bench2drive/extracted | 11.76 |
| bench2drive/archives | 49.25 |
| bench2drive/manifests | 0.05 |
| person2drive/extracted | 62.42 |
| person2drive/archives | 67.77 |
| person2drive/metadata | 10.37 |
| person2drive/manifests | 0.07 |

Total active source trees: **201.69 GiB**, including retained source archives. This excludes legacy data/archive/. No original data or source archives were deleted.

```bash
# Restore missing frozen sources only when necessary:
python tools/download_bench2drive.py --download
python tools/extract_bench2drive.py
python tools/prepare_person2drive.py --metadata --index --download --extract
# Audit first; exact RGB payload per episode is persisted:
python tools/prepare_multicamera.py
python tools/prepare_multicamera.py --extract
python tools/build_camera_geometry.py
python -m unittest discover -s tests -v
python tools/check_dataset.py
python tools/validate_multicamera.py
python tools/report_multicamera.py
```

Existing valid archives are reused; completed episode markers support resumption. Audits validate archive SHA256, safe tar members, six-view frame sets and original-front byte consistency. Additional views are staged; incomplete episodes are not published as complete. The tools extract only the six requested RGB directories. Source manifests/cohort/splits are never reselected.

Remaining uncertainty is capture-clock synchronization and the original applied-control timing/conversion already documented in the foundation, not missing six-camera coverage. Future spatial fusion should use the explicitly labelled matrices and masks, rather than assuming native source coordinate conventions are identical. No DINOv3 inference, visual feature caching, Transformer/RSSM training or fusion architecture was run.
