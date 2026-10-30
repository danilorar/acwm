---
license: cc-by-nc-4.0
task_categories:
  - robotics
  - image-to-text
tags:
  - autonomous-driving
  - personalized-driving
  - CARLA
  - human-driving-data
  - vision-language
  - driving-behavior
pretty_name: "PDD: Personalized Driving Dataset"
size_categories:
  - 10K<n<100K
---

# PDD: Personalized Driving Dataset

## Dataset Description

PDD (Personalized Driving Dataset) is a multi-driver, multi-scenario driving dataset collected in CARLA 0.9.15. It captures real human driving behavior from **30 individual drivers**, each performing **21 challenging driving scenarios**. The dataset is designed for research on personalized autonomous driving, where models learn to mimic individual driving styles.

Each driver has a detailed profile capturing demographics, driving experience, habits, and self-reported driving style. The driving data includes front-camera RGB images, 3D bounding boxes for surrounding objects, and per-frame vehicle telemetry (speed, acceleration, steering, throttle, brake, etc.).

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Drivers | 30 |
| Scenarios per driver | 21 |
| Total scenario instances | 630 |
| Total image frames | 70,087 |
| Total bounding box files | 70,087 |
| Dataset size | ~13 GB |
| Simulator | CARLA 0.9.15 |
| Frame rate (saved) | 4 FPS |

## Dataset Structure

```
PDD/
├── driver_01/
│   └── data/
│       ├── Accident/
│       │   ├── images/          # Front-camera RGB images (JPEG)
│       │   │   ├── 0.jpg
│       │   │   ├── 1.jpg
│       │   │   └── ...
│       │   ├── boxes/           # 3D bounding boxes (compressed JSON)
│       │   │   ├── 0.json.gz
│       │   │   ├── 1.json.gz
│       │   │   └── ...
│       │   └── metric/
│       │       ├── metrics.json       # Per-step control inputs
│       │       └── metric_info.json   # Per-frame telemetry
│       ├── BlockedIntersection/
│       │   └── ...
│       └── ... (21 scenarios)
├── driver_02/
│   └── ...
├── ... (30 drivers)
└── user_profiles/
    ├── driver_01.json
    ├── driver_02.json
    └── ... (30 profiles)
```

## Data Fields

### Images (`images/*.jpg`)
Front-forward RGB camera images captured at 4 FPS during driving.

### Bounding Boxes (`boxes/*.json.gz`)
Gzip-compressed JSON files, one per frame. Each contains a list of detected objects:
- `class`: Object type (`ego_car`, `car`, `walker`, `static`)
- `position`: [x, y, z] relative to ego vehicle
- `extent`: [length, width, height] of bounding box
- `yaw`: Heading angle
- `speed`: Object speed
- `id`: Unique object identifier
- `distance`: Distance from ego vehicle

### Telemetry (`metric/metric_info.json`)
Per-frame driving telemetry indexed by frame number:
- `location`: [x, y, z] world coordinates
- `rotation`: [pitch, roll, yaw]
- `speed`: Current speed (m/s)
- `speed_limit`: Road speed limit (m/s)
- `acceleration`: [x, y, z] acceleration vector
- `velocity`: [x, y, z] velocity vector
- `angular_velocity`: [x, y, z]
- `distance_to_front_vehicle`: Distance to lead vehicle (m)
- `lane_change_count`: Number of lane changes
- `lane_info`: Current lane information
- `target_point`, `target_point_next`: Navigation waypoints
- `expert_target_speed`: Expert reference speed
- `expert_control_steer/throttle/brake`: Expert reference controls
- `other_vehicles`: Nearby vehicle information
- `walkers`: Nearby pedestrian information

### Control Inputs (`metric/metrics.json`)
Sequential list of control commands applied at each simulation step:
- `steer`: Steering angle [-1, 1]
- `throttle`: Throttle input [0, 1]
- `brake`: Brake input [0, 1]
- `gear`, `hand_brake`, `reverse`: Additional vehicle state

### Driver Profiles (`user_profiles/driver_XX.json`)
- `basic_information`: Age, gender, occupation
- `driving_experience`: Years of experience
- `driving_frequency_per_week`: Typical weekly driving hours
- `driving_purposes`: Common driving use cases
- `driving_habits_preferences`: Self-reported driving habits
- `health_and_driving_records`: Health conditions, accident history
- `driving_style`: Self-classified style (Aggressive / Assertive / Balanced / Calm / Cautious)
- `international_driving_experience`: Driving experience in other regions

## Usage

```python
from huggingface_hub import snapshot_download

# Download the full dataset
snapshot_download(repo_id="tasl-lab/PDD", repo_type="dataset", local_dir="./PDD")

# Download a specific driver only
snapshot_download(repo_id="tasl-lab/PDD", repo_type="dataset", local_dir="./PDD",
                  allow_patterns=["driver_01/**", "user_profiles/**"])
```

Or use the provided loading script (`load_pdd.py`) for a structured PyTorch-compatible loader:

```python
# Copy load_pdd.py to your project, then:
from datasets import load_dataset

dataset = load_dataset("./load_pdd.py", name="driver_01", trust_remote_code=True)
sample = dataset["train"][0]
print(sample["driver_id"])        # "driver_01"
print(sample["scenario"])         # "Accident"
print(sample["speed"])            # 0.001
print(sample["image"])            # PIL Image
print(sample["driver_profile"])   # {...}
```

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{wang2026drivewaypreferencealignment,
      title={Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving},
      author={Zehao Wang and Huaide Jiang and Shuaiwu Dong and Yuping Wang and Hang Qiu and Jiachen Li},
      year={2026},
      eprint={2603.25740},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.25740},
}
```
