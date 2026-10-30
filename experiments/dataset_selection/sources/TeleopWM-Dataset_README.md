---
license: mit
language:
- en
tags:
- autonomous-driving
- teleoperation
- robotics
- carla
- world-model
- video-prediction
- predictive-display
- future-action-prediction
- imitation-learning
- mile
---

# TeleopWM-Dataset

TeleopWM-Dataset is a large-scale collection of CARLA driving rollouts used for training and evaluating **TeleopWM**, a predictive latent world model for latency-resilient vision-based teleoperation.

The dataset contains synchronized RGB observations, vehicle controls, speed measurements, and metadata used to construct short-horizon future prediction tasks for both visual rollout prediction and future-action forecasting.

## Related Resources

* Project page: https://bimilab.github.io/paper-TeleopWM/
* Model checkpoint: https://huggingface.co/bimilab/TeleopWM
* GitHub repository: https://github.com/bimilab/paper-TeleopWM
* YouTube demo: https://youtu.be/WeKqqZuwBl0

## Overview

TeleopWM-Dataset was designed for research on:

* latency-resilient teleoperation
* predictive display
* future observation prediction
* future action prediction
* world models for driving
* autonomous and teleoperated vehicle systems

The dataset follows a CARLA/MILE-style rollout format and contains driving data collected across multiple CARLA towns and driving scenarios.

## Dataset Structure

```text
mile_action_diverse/
├── train/
│   ├── Town01/
│   ├── Town03/
│   └── Town04/
│
├── val/
│   └── Town02/
│
└── test/
    └── Town05/
```

The official TeleopWM experiments use:

| Split      | Towns                  |
| ---------- | ---------------------- |
| Train      | Town01, Town03, Town04 |
| Validation | Town02                 |
| Test       | Town05                 |

This split was selected to evaluate generalization to previously unseen environments.

## Data Contents

Each rollout contains:

* RGB camera images
* vehicle controls:

  * throttle
  * steering
  * brake
* vehicle speed
* route metadata
* rollout metadata stored in:

```text
pd_dataframe.pkl
```

The TeleopWM pipeline constructs:

* 9 past frames
* 8 future frames

for predictive world-model training.

## Control Representation

Raw controls are stored as:

```text
[throttle, steer, brake]
```

TeleopWM internally converts them into:

```text
[longitudinal, scaled_steer, speed]
```

where:

```text
longitudinal = throttle - brake
```

This representation is used by the released TeleopWM model.

## Dataset Statistics

Approximate release size:

| Split      | Size   |
| ---------- | ------ |
| Train      | 71 GB  |
| Validation | 11 GB  |
| Test       | 9 GB   |
| Total      | ~90 GB |

## MILE Acknowledgement

This dataset uses a CARLA rollout format derived from the MILE ecosystem.

We acknowledge the contributions of the MILE project and the associated CARLA data-collection framework. The rollout structure, metadata conventions, and driving-data organization are based on the MILE pipeline and were extended for TeleopWM research on predictive display and future-action forecasting.

If you use this dataset, please also consider citing the original MILE work.

## Intended Use

This dataset is intended for:

* predictive world-model research
* future frame prediction
* future action prediction
* teleoperation research
* latency mitigation research
* autonomous driving research
* imitation learning research

## Out-of-Scope Use

This dataset is not intended to:

* certify safety-critical driving systems
* validate real-world autonomous vehicles without additional testing
* represent all driving environments or traffic conditions
* serve as a benchmark for real-world safety evaluation

## Citation

If you use TeleopWM-Dataset, please cite:

```bibtex
@misc{teleopwm_dataset2026,
  title={TeleopWM-Dataset: A CARLA Dataset for Latency-Resilient Vision-Based Teleoperation},
  author={Khalil, Aws and Kwon, Jaerock},
  year={2026}
}
```

## License

This dataset is released under the MIT License.