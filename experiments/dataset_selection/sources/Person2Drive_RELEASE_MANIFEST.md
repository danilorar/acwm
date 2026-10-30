# Person2Drive Release Manifest

Last updated: **2026-10-14**. This manifest describes the currently published files. The release covers all 50 drivers on routes 103, 108, 100, and 101, with sensor data provided for routes 103 and 108.

## Current Availability

| Data component | Routes | Current availability |
|---|---|---|
| Six-camera RGB and LiDAR archives in `sensor_data/` | 103 (Town05), 108 (Town04) | 50 complete drivers; 800 archives available |
| Driver-level `b2d_infos_train.pkl` | 103, 108 | 50 files available |
| Original driving records in `driving_logs/` | 103, 108 | 50 drivers; 800 records / 4000 files |
| Driving records in `driving_logs_without_sensor_data/` | 100, 101 (Town06) | 50 drivers; 200 records / 1000 files |
| Shared map metadata | Benchmark maps | `b2d_map_infos.pkl` available |

## Sensor Package Coverage

- Complete sensor packages: **`driver01`–`driver50`**.
- Published sensor archives: **800 `.tar.zst` files** covering routes 103 and 108.
- Driver-level training metadata: **50 `b2d_infos_train.pkl` files**, one per driver.
- Shared map metadata: **one `b2d_map_infos.pkl`**.

Each complete driver package under `sensor_data/driverXX/` contains 16 `.tar.zst` archives: eight for Town04 (route 108) and eight for Town05 (route 103), plus `b2d_infos_train.pkl`.

## Original Driving Record Coverage

| Published directory | Drivers | Records | Files |
|---|---|---:|---:|
| `driving_logs/` — routes 103 and 108 | `driver01`–`driver50` | 800 | 4000 |
| `driving_logs_without_sensor_data/` — routes 100 and 101 | `driver01`–`driver50` | 200 | 1000 |

Routes 100/101 driving records are published for **all 50 drivers**, with two recorded drives per route per driver.

Every published driving record contains `metric_info.json`, `route.json`, `result.json`, `live_results.txt`, and one corresponding `RouteScenario_*.log`. Sensor data are not included in the route 100/101 release.

For the repository structure and download instructions, see [README.md](README.md).

## Recent Updates

### 2026-10-14

- Completed the release of driving logs for routes 100 and 101 across all 50 drivers.

### 2026-10-13

- Completed the release of RGB images, LiDAR point clouds, and training metadata for routes 103 and 108 across all 50 drivers.

### 2026-10-12

- Renamed `drivers/` to `sensor_data/`; earlier download and loading scripts should use the new path.
- Added `driving_logs/` and `driving_logs_without_sensor_data/`.
