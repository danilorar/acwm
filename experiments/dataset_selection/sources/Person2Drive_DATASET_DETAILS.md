# Person2Drive Dataset Details

Last updated: **2026-10-14**. For current coverage and release history, see [RELEASE_MANIFEST.md](RELEASE_MANIFEST.md).

## Routes and Released Data

| Route | CARLA map | Published data |
|---|---|---|
| 103 | Town05 | Six-view RGB, LiDAR, training metadata, and original driving records |
| 108 | Town04 | Six-view RGB, LiDAR, training metadata, and original driving records |
| 100 | Town06 | Original driving records for unseen-route generalization evaluation |
| 101 | Town06 | Original driving records for unseen-route generalization evaluation |

The benchmark has 50 anonymized drivers. Route 100/101 records are published for all 50 drivers, with two recorded drives per route per driver; see the release manifest for component counts.

## Public Directory Structure

For the complete repository structure, see [README.md](README.md).

`driverXX` identifies the driver. Within each driver's directory, `Town04_drive_N` contains records for route 108, and `Town05_drive_N` contains records for route 103. Routes 100 and 101 both use Town06, so their directory names include `route100` or `route101` to distinguish them. `N` is the drive index within each driver and route.

## Inside a Sensor Archive

Example: `sensor_data/driver01/Town04_drive_1.tar.zst` extracts to:

```text
Town04_drive_1/
├── camera/
│   ├── rgb_front/
│   ├── rgb_front_left/
│   ├── rgb_front_right/
│   ├── rgb_back/
│   ├── rgb_back_left/
│   └── rgb_back_right/
└── lidar/
```

The six camera directories contain RGB images in JPEG (`.jpg`) format. The `lidar/` directory contains compressed LAS point clouds (`.laz`). Images can be read with Pillow; LAZ files can be read with laspy and the lazrs decompression backend.

```bash
pip install pillow "laspy[lazrs]"
```

Retained frame names and numbers are unchanged, and frame counts vary by drive.

The published sensor archives contain the six-view RGB images and LiDAR point clouds used in our paper. They exclude depth images, top-down RGB images, semantic top-down images, `anno/`, videos, CSV files, driving-result files, and local symbolic links. Five-type driving records are provided separately in the logging directories. Additional outputs can be generated through the rendering pipeline described below.

## Training and Map Metadata

| File | Location | Purpose |
|---|---|---|
| `b2d_infos_train.pkl` | `sensor_data/driverXX/` | Driver-specific training metadata, including frame references and model input/target information |
| `b2d_map_infos.pkl` | Repository root | Shared map-level metadata used by the benchmark pipeline |

Camera `data_path` values in the training metadata are relative to the corresponding `sensor_data/driverXX/` directory. For example, in the inspected driver50 metadata, an entry with `folder="Town04_drive_1"` and `frame_idx=2` references the front-camera image at `Town04_drive_1/camera/rgb_front/00002.jpg`. After extraction, its full path is `sensor_data/driver50/Town04_drive_1/camera/rgb_front/00002.jpg`.

Use the recorded paths and frame identifiers to locate files. The position of an entry in the metadata list is not its sensor frame number.

Extract the sensor archives inside their corresponding driver directory so drive folders and `b2d_infos_train.pkl` are located together. Training metadata may select a subset of sensor frames, so metadata entry counts need not equal the total number of image frames.

## Original Driving Record Files

These five file types are published directly, without compression:

| File | Content |
|---|---|
| `metric_info.json` | Original indexed vehicle motion/state records, including position, rotation, acceleration, angular velocity, and orientation vectors |
| `route.json` | Recorded route coordinates |
| `result.json` | Route evaluation results, scores, run metadata, and recorded infractions |
| `live_results.txt` | Text summary of the run and recorded events |
| `RouteScenario_*.log` | CARLA binary recorder data for simulation replay; the filename varies with the route |

For routes 103/108, files are stored under `driving_logs/driverXX/TownXX_drive_N/`. For routes 100/101, files are stored under `driving_logs_without_sensor_data/driverXX/Town06_route100_drive_N/` or `Town06_route101_drive_N/`.

For routes 100/101, drive directories are numbered chronologically within each driver and route. No camera images, LiDAR point clouds, or annotation directories are published for those records. Original file contents, timestamps, and indices within the files are retained.

Motion/state logging and sensor capture can have different record counts. The numeric keys in `metric_info.json` index the original motion-record sequence and must not be assumed to match sensor frame numbers. For example, key `"2"` does not necessarily correspond to image `00002.jpg`. A general mapping between these indices has not yet been verified. Original records are preserved without truncation or renumbering.

## Sensor Data Generation

The `RouteScenario_*.log` files support replay of the recorded driving sessions in CARLA. Our rendering pipeline uses these recordings to generate sensor data and annotations in the same directory structure and file formats as Bench2Drive.

- **Routes 100 and 101:** Pre-rendered sensor data are not included in this release. Camera images, LiDAR point clouds, and annotations can be generated from the published recorder files. The directory name `driving_logs_without_sensor_data` describes the published files; it does not mean sensor data cannot be generated.
- **Routes 103 and 108:** The published archives contain the six-view RGB images and LiDAR point clouds used in our paper. The accompanying recorder files also support generation of additional sensor modalities and annotations through the same pipeline.

Rendering scripts are not yet publicly available. Generated sensor data depend on the rendering configuration; format compatibility does not imply byte-for-byte identity with the published sensor files.

## Availability and Feedback

The dataset is actively maintained. [RELEASE_MANIFEST.md](RELEASE_MANIFEST.md) lists published components and release history, while [Files and versions](https://huggingface.co/datasets/dongxr7/Person2Drive/tree/main) shows the latest uploaded files.

If a file is missing or a download, extraction, or data-loading step fails, please [open a Community discussion](https://huggingface.co/datasets/dongxr7/Person2Drive/discussions) with the public driver ID, affected file path, command, and error message. Questions about the documentation are also welcome.
