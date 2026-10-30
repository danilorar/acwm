"""Small runnable example of the refactored observation/target interface."""
from pathlib import Path

from torch.utils.data import DataLoader
from src.bench2drive_dataset import Bench2DriveDataset, split_scenarios


def main():
    root = Path(__file__).resolve().parent / 'data/bench2drive_mini'
    index = Bench2DriveDataset(root, history=4, horizon=10)
    train_names, val_names = split_scenarios(index.scenarios, seed=2026)
    dataset = Bench2DriveDataset(root, scenarios=train_names, action_offset=1)
    print(f'Train: {len(train_names)} recordings, {len(dataset)} windows')
    print('Held out:', val_names)
    sample = dataset[0]
    print('\nSample:', sample['scenario'])
    print('Start frame:', sample['start_frame'])
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            print(f'{key:24s}: {tuple(value.shape)}')
    batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)))
    print('\nBatch shapes:')
    for key in ['past_images', 'past_ego_observations', 'past_actions',
                'future_actions', 'future_images', 'future_motion_targets']:
        print(f'{key:24s}: {tuple(batch[key].shape)}')


if __name__ == '__main__':
    main()
