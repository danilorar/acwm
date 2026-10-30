"""PDD: Personalized Driving Dataset - HuggingFace loading script."""

import gzip
import json
import os
from pathlib import Path

import datasets

_DESCRIPTION = """\
PDD (Personalized Driving Dataset) is a multi-driver, multi-scenario driving dataset
collected in CARLA 0.9.15. It captures real human driving behavior from 30 individual
drivers, each performing 21 challenging driving scenarios.
"""

_HOMEPAGE = ""
_LICENSE = "cc-by-nc-4.0"

_SCENARIOS = [
    "Accident",
    "BlockedIntersection",
    "ConstructionObstacle",
    "ControlLoss",
    "CrossingBicycleFlow",
    "DynamicObjectCrossing",
    "EnterActorFlow",
    "HazardAtSideLane",
    "HighwayExit",
    "InterurbanActorFlow",
    "InvadingTurn",
    "MergerIntoSlowTraffic",
    "NonSignalizedJunctionLeftTurn",
    "NonSignalizedJunctionRightTurn",
    "ParkedObstacle",
    "ParkingCutIn",
    "SignalizedJunctionLeftTurn",
    "SignalizedJunctionRightTurn",
    "StaticCutIn",
    "VanillaNonSignalizedTurn",
    "VehicleOpensDoorTwoWays",
]

_DRIVERS = [f"driver_{i:02d}" for i in range(1, 31)]


class PDDConfig(datasets.BuilderConfig):
    """BuilderConfig for PDD."""

    def __init__(self, driver_ids=None, scenarios=None, **kwargs):
        """
        Args:
            driver_ids: List of driver IDs to load (e.g. ["driver_01", "driver_02"]).
                        If None, loads all 30 drivers.
            scenarios: List of scenario names to load. If None, loads all 21 scenarios.
            **kwargs: Passed to super.
        """
        super().__init__(**kwargs)
        self.driver_ids = driver_ids or _DRIVERS
        self.scenarios = scenarios or _SCENARIOS


# Build one config per driver + an "all" config
_CONFIGS = [
    PDDConfig(
        name="all",
        version=datasets.Version("1.0.0"),
        description="All 30 drivers, all 21 scenarios",
        driver_ids=_DRIVERS,
        scenarios=_SCENARIOS,
    ),
] + [
    PDDConfig(
        name=driver_id,
        version=datasets.Version("1.0.0"),
        description=f"Data for {driver_id}",
        driver_ids=[driver_id],
        scenarios=_SCENARIOS,
    )
    for driver_id in _DRIVERS
]


class PDD(datasets.GeneratorBasedBuilder):
    """PDD: Personalized Driving Dataset."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = _CONFIGS
    DEFAULT_CONFIG_NAME = "all"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "driver_id": datasets.Value("string"),
                    "scenario": datasets.Value("string"),
                    "frame_index": datasets.Value("int32"),
                    "image": datasets.Image(),
                    "boxes": datasets.Sequence(
                        {
                            "class": datasets.Value("string"),
                            "position": datasets.Sequence(datasets.Value("float64"), length=3),
                            "extent": datasets.Sequence(datasets.Value("float64"), length=3),
                            "yaw": datasets.Value("float64"),
                            "speed": datasets.Value("float64"),
                            "id": datasets.Value("int64"),
                            "distance": datasets.Value("float64"),
                        }
                    ),
                    # Telemetry
                    "speed": datasets.Value("float64"),
                    "speed_limit": datasets.Value("float64"),
                    "location": datasets.Sequence(datasets.Value("float64"), length=3),
                    "rotation": datasets.Sequence(datasets.Value("float64"), length=3),
                    "acceleration": datasets.Sequence(datasets.Value("float64"), length=3),
                    "velocity": datasets.Sequence(datasets.Value("float64"), length=3),
                    "steer": datasets.Value("float64"),
                    "throttle": datasets.Value("float64"),
                    "brake": datasets.Value("float64"),
                    "distance_to_front_vehicle": datasets.Value("float64"),
                    "lane_change_count": datasets.Value("int32"),
                    "expert_target_speed": datasets.Value("float64"),
                    "expert_control_steer": datasets.Value("float64"),
                    "expert_control_throttle": datasets.Value("float64"),
                    "expert_control_brake": datasets.Value("float64"),
                    "target_point": datasets.Sequence(datasets.Value("float64"), length=2),
                    "target_point_next": datasets.Sequence(datasets.Value("float64"), length=2),
                    # Driver profile
                    "driver_profile": datasets.Value("string"),  # JSON string
                    "driving_style": datasets.Value("string"),
                }
            ),
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):
        data_dir = os.path.dirname(os.path.abspath(__file__))
        if dl_manager.is_streaming:
            data_dir = dl_manager.download_config.download_dir or data_dir
        # For HuggingFace Hub, data_dir will be set by the downloader
        # For local loading, use the script directory
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": data_dir},
            ),
        ]

    def _generate_examples(self, data_dir):
        config = self.config
        driver_ids = config.driver_ids
        scenarios = config.scenarios

        # Load driver profiles
        profiles = {}
        profiles_dir = os.path.join(data_dir, "user_profiles")
        for driver_id in driver_ids:
            profile_path = os.path.join(profiles_dir, f"{driver_id}.json")
            if os.path.exists(profile_path):
                with open(profile_path, "r") as f:
                    profiles[driver_id] = json.load(f)

        idx = 0
        for driver_id in sorted(driver_ids):
            profile = profiles.get(driver_id, {})
            driving_style = profile.get("driving_style", "")
            profile_json = json.dumps(profile, ensure_ascii=False)

            for scenario in sorted(scenarios):
                scenario_dir = os.path.join(data_dir, driver_id, "data", scenario)
                images_dir = os.path.join(scenario_dir, "images")
                boxes_dir = os.path.join(scenario_dir, "boxes")
                metric_info_path = os.path.join(scenario_dir, "metric", "metric_info.json")
                metrics_path = os.path.join(scenario_dir, "metric", "metrics.json")

                if not os.path.isdir(images_dir):
                    continue

                # Load telemetry
                metric_info = {}
                if os.path.exists(metric_info_path):
                    with open(metric_info_path, "r") as f:
                        metric_info = json.load(f)

                # Load control inputs
                metrics_records = []
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r") as f:
                        metrics_data = json.load(f)
                    metrics_records = metrics_data.get("records", [])

                # Get sorted image files
                image_files = sorted(
                    [f for f in os.listdir(images_dir) if f.endswith(".jpg")],
                    key=lambda x: int(os.path.splitext(x)[0]),
                )

                # Map metric_info keys (sorted numerically) to frame indices
                metric_keys = sorted(metric_info.keys(), key=lambda x: int(x))

                for frame_idx, img_file in enumerate(image_files):
                    frame_num = int(os.path.splitext(img_file)[0])
                    img_path = os.path.join(images_dir, img_file)

                    # Load boxes
                    box_path = os.path.join(boxes_dir, f"{frame_num}.json.gz")
                    boxes = []
                    if os.path.exists(box_path):
                        with gzip.open(box_path, "rt") as f:
                            raw_boxes = json.load(f)
                        for b in raw_boxes:
                            boxes.append(
                                {
                                    "class": b.get("class", ""),
                                    "position": b.get("position", [0.0, 0.0, 0.0]),
                                    "extent": b.get("extent", [0.0, 0.0, 0.0]),
                                    "yaw": b.get("yaw", 0.0),
                                    "speed": b.get("speed", 0.0),
                                    "id": b.get("id", 0),
                                    "distance": b.get("distance", -1.0),
                                }
                            )

                    # Get telemetry for this frame
                    mi = {}
                    if frame_idx < len(metric_keys):
                        mi = metric_info.get(metric_keys[frame_idx], {})

                    # Get control for this frame
                    control = {}
                    if frame_idx < len(metrics_records):
                        control = metrics_records[frame_idx].get("control", {})

                    yield idx, {
                        "driver_id": driver_id,
                        "scenario": scenario,
                        "frame_index": frame_num,
                        "image": img_path,
                        "boxes": boxes,
                        "speed": mi.get("speed", 0.0),
                        "speed_limit": mi.get("speed_limit", 0.0),
                        "location": mi.get("location", [0.0, 0.0, 0.0]),
                        "rotation": mi.get("rotation", [0.0, 0.0, 0.0]),
                        "acceleration": mi.get("acceleration", [0.0, 0.0, 0.0]),
                        "velocity": mi.get("velocity", [0.0, 0.0, 0.0]),
                        "steer": control.get("steer", 0.0),
                        "throttle": control.get("throttle", 0.0),
                        "brake": control.get("brake", 0.0),
                        "distance_to_front_vehicle": mi.get("distance_to_front_vehicle", -1.0),
                        "lane_change_count": mi.get("lane_change_count", 0),
                        "expert_target_speed": mi.get("expert_target_speed", 0.0),
                        "expert_control_steer": mi.get("expert_control_steer", 0.0),
                        "expert_control_throttle": mi.get("expert_control_throttle", 0.0),
                        "expert_control_brake": mi.get("expert_control_brake", 0.0),
                        "target_point": mi.get("target_point", [0.0, 0.0]),
                        "target_point_next": mi.get("target_point_next", [0.0, 0.0]),
                        "driver_profile": profile_json,
                        "driving_style": driving_style,
                    }
                    idx += 1
