---
license: apache-2.0
viewer: false
tags:
- autonomous-driving
- carla
- benchmark
- end-to-end-driving
- personalized-driving
---

# Person2Drive

Person2Drive is a dataset and benchmark for closed-loop personalized end-to-end autonomous driving. This repository hosts its dataset release.

### Driving Like Yourself: A Benchmark for Closed-Loop Personalized End-to-End Autonomous Driving

[**ECCV 2026**](https://link.springer.com/chapter/10.1007/978-3-032-37435-6_18) · [Paper on arXiv](https://arxiv.org/abs/2602.18757) · [PDF](https://arxiv.org/pdf/2602.18757)

## Updates & Feedback

**Actively maintained · Last updated: 2026-10-14**

We actively maintain the Person2Drive dataset and its documentation. For current file availability and release history, see [RELEASE_MANIFEST.md](RELEASE_MANIFEST.md).

Questions and reports of problems are welcome. If you encounter missing files, download or extraction issues, or problems loading the data, please [open a discussion in the Hugging Face Community](https://huggingface.co/datasets/dongxr7/Person2Drive/discussions). Include the affected driver/file path, the command you ran, and any error messages so we can investigate. Documentation questions are welcome too.

## Overview

Person2Drive is a CARLA-based benchmark for studying personalized end-to-end autonomous driving. It contains human driving records from anonymized drivers and supports research on driving style modeling, driver-level personalization, route generalization, and closed-loop evaluation.

The full benchmark covers **50 drivers and four routes**. Routes 103 and 108 provide driving records and selected sensor data. Routes 100 and 101 are used for generalization evaluation on unseen routes; their published driving records do not include sensor data.

## Dataset Contents

| Data component | Routes | Description |
|---|---|---|
| Six-camera RGB and LiDAR archives in `sensor_data/` | 103 (Town05), 108 (Town04) | 50 complete drivers; 800 archives available |
| Driver-level `b2d_infos_train.pkl` | 103, 108 | 50 files available |
| Original driving records in `driving_logs/` | 103, 108 | 50 drivers; 800 records / 4000 files |
| Driving records in `driving_logs_without_sensor_data/` | 100, 101 (Town06) | 50 drivers; 200 records / 1000 files for unseen-route evaluation, without sensor data |
| Shared map metadata | Benchmark maps | `b2d_map_infos.pkl` available |

## Repository Structure

```text
Person2Drive/
├── sensor_data/driverXX/
│   ├── Town04_drive_1.tar.zst ... Town04_drive_8.tar.zst
│   ├── Town05_drive_1.tar.zst ... Town05_drive_8.tar.zst
│   └── b2d_infos_train.pkl
├── driving_logs/
│   └── driverXX/
│       ├── Town04_drive_N/
│       └── Town05_drive_N/
├── driving_logs_without_sensor_data/
│   └── driverXX/
│       ├── Town06_route100_drive_N/
│       └── Town06_route101_drive_N/
├── b2d_map_infos.pkl
├── RELEASE_MANIFEST.md
├── DATASET_DETAILS.md
└── README.md
```

`driverXX` is an anonymized public identifier. Each complete sensor package contains eight Town04 drives, eight Town05 drives, and one driver-level metadata file.

**Naming convention.** `TownXX` identifies the CARLA map, while route IDs identify individual driving routes. In this dataset, `Town05_drive_N` corresponds to route 103, and `Town04_drive_N` corresponds to route 108. Routes 100 and 101 both use Town06, so their directory names explicitly include `route100` or `route101` to distinguish them. `N` is the drive index within each driver and route.

## Sensor Archives and Metadata

The published sensor archives for routes 103 and 108 contain the **six-view RGB images and LiDAR point clouds used in our paper**. Each `.tar.zst` archive contains one drive directory. The camera views are `rgb_front`, `rgb_front_left`, `rgb_front_right`, `rgb_back`, `rgb_back_left`, and `rgb_back_right`.

Frame counts vary by drive; retained sensor files keep their original names and frame numbers. Additional sensor modalities can be generated from the accompanying CARLA recorder files using our rendering pipeline.

`b2d_infos_train.pkl` provides driver-specific training metadata and frame references. The shared `b2d_map_infos.pkl` provides map-level metadata. See [DATASET_DETAILS.md](DATASET_DETAILS.md) for the internal structure and file descriptions.

## Original Driving Records

`driving_logs/` stores original driving records, route information, and run results for routes 103 and 108. Corresponding camera and LiDAR data remain in `sensor_data/`.

```text
driving_logs/driver01/Town04_drive_1/
├── metric_info.json
├── route.json
├── result.json
├── live_results.txt
└── RouteScenario_108_rep0.log
```

`driving_logs_without_sensor_data/` provides the same five file types for routes 100 and 101 in Town06, organized as `driverXX/Town06_route100_drive_N/` or `driverXX/Town06_route101_drive_N/`. Each of the 50 drivers has two recorded drives per route. Pre-rendered camera images, LiDAR point clouds, and annotation files are not included in this part of the release. These outputs can be generated by replaying the CARLA recorder files with our rendering pipeline.

The `RouteScenario_*.log` name varies with the route. These are **CARLA binary recorder files**, rather than plain-text console logs. JSON and text files are uploaded directly, without compression. Original file contents, timestamps, and indices within the files are preserved; motion record counts can differ from camera frame counts.

**Rendering pipeline.** Generated data follow the same directory structure and file formats as Bench2Drive. Rendering scripts are not yet publicly available. See [Sensor Data Generation](DATASET_DETAILS.md#sensor-data-generation) for details.

## Download and Extraction

Install the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
pip install -U huggingface_hub
```

### Try one driving record first

Download the sensor archive and the five original driving-record files for `driver01/Town04_drive_1`. The download is approximately **325 MiB**, excluding training and map metadata. All sizes below refer to downloaded files; allow additional disk space for extraction.

```bash
hf download dongxr7/Person2Drive --repo-type dataset \
  --include "sensor_data/driver01/Town04_drive_1.tar.zst" \
            "driving_logs/driver01/Town04_drive_1/*" \
  --local-dir Person2Drive
```

On Linux, install `tar` and `zstd`, then extract the archive:

```bash
tar -I zstd -xf \
  Person2Drive/sensor_data/driver01/Town04_drive_1.tar.zst \
  -C Person2Drive/sensor_data/driver01
```

This extracts the six RGB camera directories and the LiDAR directory under `Person2Drive/sensor_data/driver01/Town04_drive_1/`. The five driving-record files remain under `Person2Drive/driving_logs/driver01/Town04_drive_1/`.

On Windows, use an archive tool that supports Zstandard and TAR, preserving the directory structure.

### Download training metadata when needed

To inspect the training metadata for the same driver, download it separately:

```bash
hf download dongxr7/Person2Drive \
  sensor_data/driver01/b2d_infos_train.pkl \
  --repo-type dataset --local-dir Person2Drive
```

The training metadata covers multiple drives. Downloading one archive is sufficient for inspecting that drive, but using all metadata entries requires the corresponding sensor archives. Extract those archives into the same driver directory so the drive folders and `b2d_infos_train.pkl` are located together.

### Download a complete driver package

Download all sensor archives, training metadata, and route 103/108 driving records for `driver01`. This example downloads approximately **11.8 GiB**, excluding route 100/101 records and shared map metadata.

```bash
hf download dongxr7/Person2Drive --repo-type dataset \
  --include "sensor_data/driver01/*" "driving_logs/driver01/*" \
  --local-dir Person2Drive
```

Replace `driver01` in the paths to select another driver. Download sizes vary by driver.

### Download route 100/101 driving records

Download all four route 100/101 driving records for the same driver.

```bash
hf download dongxr7/Person2Drive --repo-type dataset \
  --include "driving_logs_without_sensor_data/driver01/*" \
  --local-dir Person2Drive
```

### Download shared map metadata

The root-level `b2d_map_infos.pkl` is shared across drivers and can be downloaded separately:

```bash
hf download dongxr7/Person2Drive b2d_map_infos.pkl \
  --repo-type dataset --local-dir Person2Drive
```

For reproducible experiments, record the Hugging Face commit revision used for your download. The CLI supports `--revision <commit_hash>`.

## Privacy and Anonymization

Public driver identifiers use `driverXX`. The private name-to-ID mapping is not published. Original driving records retain run metadata, including recording timestamps.

## Citation

If you use Person2Drive in your research, please cite our paper:

```bibtex
@inproceedings{dong2026person2drive,
  title={Driving Like Yourself: A Benchmark for Closed-Loop Personalized End-to-End Autonomous Driving},
  author={Dong, Xiaoru and Li, Ruiqin and Han, Xiao and Wu, Zhenxuan and Wang, Jiamin and Chen, Jian and Jiang, Qi and Yiu, Siu Ming and Zhu, Xinge and Ma, Yuexin},
  booktitle={Computer Vision -- ECCV 2026},
  series={Lecture Notes in Computer Science},
  volume={17039},
  pages={326--344},
  year={2026},
  publisher={Springer},
  address={Cham},
  doi={10.1007/978-3-032-37435-6_18},
  url={https://doi.org/10.1007/978-3-032-37435-6_18}
}
```

## Contact

For direct inquiries, please contact Xiaoru Dong at [xrdong@cs.hku.hk](mailto:xrdong@cs.hku.hk).
