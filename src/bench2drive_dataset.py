"""Sensor observations and separately sourced motion supervision for Bench2Drive."""
from pathlib import Path
import gzip
import json
import re
import warnings

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def route_group(scenario, route_ids=None):
    """Identify a town/route independent of weather or scenario label.

    Unknown naming fails closed; supply explicit route_ids for other layouts.
    """
    if route_ids is not None and scenario in route_ids:
        return str(route_ids[scenario])
    match = re.search(r'(Town\d+(?:HD)?)[_/](?:Route)?(\d+)(?=_|/|$)', scenario)
    if not match:
        raise ValueError(f'Cannot infer route for {scenario!r}; supply route_ids')
    town = re.fullmatch(r'Town(\d+)(HD)?', match[1])
    return f'Town{int(town[1]):02d}{town[2] or ""}/Route{int(match[2])}'


def split_scenarios(scenarios, *, val_fraction=0.2, seed=2026,
                    train_scenarios=None, val_scenarios=None, route_ids=None):
    """Return sorted train/validation recording names without shared routes.

    Pass both explicit assignments, or one and its complement is inferred.
    Automatic splits sample entire route groups, not individual windows.
    """
    names = list(scenarios)
    if len(names) != len(set(names)) or len(names) < 2:
        raise ValueError('Need at least two distinct scenario recordings')
    known = set(names)
    groups = {name: route_group(name, route_ids) for name in names}
    if train_scenarios is not None or val_scenarios is not None:
        def checked(values):
            values = list(values)
            if len(values) != len(set(values)) or not set(values) <= known:
                raise ValueError('Explicit split contains duplicate or unknown recordings')
            return set(values)
        train = checked(train_scenarios) if train_scenarios is not None else None
        val = checked(val_scenarios) if val_scenarios is not None else None
        train = known - val if train is None else train
        val = known - train if val is None else val
        if train & val or train | val != known:
            raise ValueError('Explicit splits must be disjoint and cover all recordings')
    else:
        if not 0 < val_fraction < 1:
            raise ValueError('val_fraction must be between zero and one')
        unique = sorted(set(groups.values()))
        if len(unique) < 2:
            raise ValueError('Need at least two route groups')
        order = np.random.default_rng(seed).permutation(unique)
        count = min(len(order) - 1, max(1, int(np.ceil(len(order) * val_fraction))))
        held_out = set(order[:count])
        val = {name for name in names if groups[name] in held_out}
        train = known - val
    if not train or not val:
        raise ValueError('Both splits must be nonempty')
    if {groups[n] for n in train} & {groups[n] for n in val}:
        raise ValueError('The same town/route cannot appear in both splits')
    return sorted(train), sorted(val)


class Bench2DriveDataset(Dataset):
    """Return observations, candidate controls and supervision as separate fields.

    H1 (offset=1) is a provisional predictive convention, not verified collection
    timing. past_actions are *recorded labels* of observed transitions; deployment
    must supply its own actually applied command history. No future controls enter
    this history. Motion pose is used only for labels; observations contain speed.
    """
    def __init__(self, root, history=4, horizon=10, image_size=(224, 224), *,
                 action_offset=1, scenarios=None, pose_source='simulator'):
        for name, value in [('history', history), ('horizon', horizon)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if isinstance(action_offset, bool) or not isinstance(action_offset, int) or action_offset not in (0, 1):
            raise ValueError('action_offset must be 0 (H0) or 1 (H1)')
        if len(image_size) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in image_size):
            raise ValueError('image_size must be positive (width, height) integers')
        if pose_source not in ('simulator', 'gps_compass'):
            raise ValueError('pose_source must be simulator or gps_compass')
        self.root = Path(root)
        self.history, self.horizon = history, horizon
        self.image_size = tuple(image_size)
        self.action_offset, self.pose_source = int(action_offset), pose_source
        self.scenarios, self.samples, self.matching_report = {}, [], {}
        # Supports Mini's flat clips and Scenario/TownXX_route larger layouts.
        folders = {str(p.parent.relative_to(self.root)): p.parent
                   for p in sorted(self.root.rglob('anno')) if p.is_dir()}
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        if scenarios is not None:
            selected = list(scenarios)
            if len(selected) != len(set(selected)) or not set(selected) <= set(folders):
                raise ValueError('Unknown or duplicate scenario selection')
            folders = {name: folders[name] for name in sorted(selected)}
        for name, folder in folders.items():
            annotations = {}
            for path in sorted((folder / 'anno').glob('*.json.gz')):
                stem = path.name.split('.')[0]
                if not stem.isdigit():
                    raise ValueError(f'Non-numeric annotation frame: {path}')
                frame = int(stem)
                if frame in annotations:
                    raise ValueError(f'Duplicate annotation frame {frame} in {folder}')
                annotations[frame] = path
            images = {}
            image_dir = folder / 'camera/rgb_front'
            if image_dir.is_dir():
                for path in sorted(image_dir.iterdir()):
                    if path.suffix.lower() in {'.jpg', '.jpeg', '.png'} and path.stem.isdigit():
                        frame = int(path.stem)
                        if frame in images:
                            raise ValueError(f'Duplicate RGB frame {frame} in {folder}')
                        images[frame] = path
            missing_images = sorted(set(annotations) - set(images))
            missing_annotations = sorted(set(images) - set(annotations))
            self.matching_report[name] = dict(missing_images=missing_images,
                                             missing_annotations=missing_annotations)
            if missing_images or missing_annotations:
                warnings.warn(f'{name}: {len(missing_images)} missing RGB, '
                              f'{len(missing_annotations)} missing annotations; gaps are not bridged', stacklevel=2)
            frames = sorted(set(annotations) & set(images))
            if not frames:
                continue
            self.scenarios[name] = {'annotations': annotations, 'images': images}
            width = history + horizon
            for start in range(len(frames) - width + 1):
                window = frames[start:start + width]
                if all(b == a + 1 for a, b in zip(window, window[1:])):
                    self.samples.append((name, window))
        if not self.samples:
            raise ValueError('No consecutive matched windows for requested scenarios/history/horizon')

    def __len__(self):
        return len(self.samples)

    def _load_annotation(self, path):
        with gzip.open(path, 'rt') as f:
            return json.load(f)

    def _load_image(self, path):
        with Image.open(path) as image:
            image = image.convert('RGB').resize(self.image_size)
            array = np.asarray(image, dtype=np.float32) / 255.
            array = np.transpose(array, (2, 0, 1)).copy()
        return torch.from_numpy(array)

    def _pose(self, annotation):
        if self.pose_source == 'simulator':
            ego = [box for box in annotation.get('bounding_boxes', [])
                   if box.get('class') == 'ego_vehicle']
            if len(ego) != 1:
                raise ValueError('Need exactly one simulator ego_vehicle pose; '
                                 'explicitly choose gps_compass for noisy labels if unavailable')
            # Collector get_bounding_boxes(): actor.get_transform().location,
            # rotation=[pitch, roll, yaw] in degrees. Not bbox center or GPS mount.
            return np.asarray(ego[0]['location'][:2], dtype=float), np.deg2rad(float(ego[0]['rotation'][2]))
        # Collector x/y are projected GPS; compass = world yaw + pi/2 in Mini.
        # Explicit noisy fallback, never silently mixed with simulator labels.
        return np.array([annotation['x'], annotation['y']], dtype=float), float(annotation['theta']) - np.pi / 2

    def __getitem__(self, index):
        name, frames = self.samples[index]
        info = self.scenarios[name]
        annotations = [self._load_annotation(info['annotations'][i]) for i in frames]
        images = torch.stack([self._load_image(info['images'][i]) for i in frames])
        k, h, offset = self.history, self.horizon, self.action_offset
        speed = torch.tensor([[float(a['speed'])] for a in annotations], dtype=torch.float32)
        actions = torch.tensor([[float(a[key]) for key in ('steer', 'throttle', 'brake')]
                                for a in annotations], dtype=torch.float32)
        origin, yaw0 = self._pose(annotations[k - 1])
        c, s = np.cos(yaw0), np.sin(yaw0)
        targets = []
        for a in annotations[k:]:
            position, yaw = self._pose(a)
            dx, dy = position - origin
            relative_yaw = yaw - yaw0
            targets.append([float(a['speed']), c * dx + s * dy, -s * dx + c * dy,
                            np.sin(relative_yaw), np.cos(relative_yaw)])
        result = {
            'past_images': images[:k],
            'past_ego_observations': speed[:k],
            'past_actions': actions[offset:offset + k - 1],
            'future_actions': actions[k - 1 + offset:k - 1 + offset + h],
            'future_images': images[k:k + h],
            'future_motion_targets': torch.tensor(targets, dtype=torch.float32),
            'scenario': name,
            'start_frame': frames[0],
            'frame_ids': torch.tensor(frames, dtype=torch.int64),
            'action_offset': offset,
            'motion_target_source': self.pose_source,
        }
        if not all(torch.isfinite(value).all() for value in result.values()
                   if isinstance(value, torch.Tensor) and value.is_floating_point()):
            raise ValueError(f'Nonfinite values in {name}, start frame {frames[0]}')
        return result
