"""Paired, scenario-held-out comparison of two recorded control alignments."""
from pathlib import Path
import argparse
import csv
import json
import os
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/action-alignment-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from src.bench2drive_dataset import Bench2DriveDataset


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model():
    return nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 64),
                         nn.ReLU(), nn.Linear(64, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT / 'data/bench2drive_mini')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--split-seed', type=int, default=2026)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.003)
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('epochs must be positive and seeds must be distinct')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)

    # Reuse the production dataset's RGB/annotation matching and consecutive-ID
    # indexing, plus its gzip JSON reader. Never load images or modify the class.
    dataset = Bench2DriveDataset(args.root, history=1, horizon=1)
    cache = {name: {fid: dataset._load_annotation(path)
                   for fid, path in info['annotations'].items()}
             for name, info in dataset.scenarios.items()}
    names, frames, targets, outgoing, incoming = [], [], [], [], []
    for name, (t, next_t) in dataset.samples:
        a, b = cache[name][t], cache[name][next_t]
        def features(control):
            return [a['speed'], control['steer'], control['throttle'] - control['brake']]
        names.append(name)
        frames.append(t)
        targets.append(b['speed'] - a['speed'])
        outgoing.append(features(a))
        incoming.append(features(b))
    names, frames = np.array(names), np.array(frames)
    y = np.array(targets, dtype=np.float32)
    x = {'H0': np.array(outgoing, dtype=np.float32),
         'H1': np.array(incoming, dtype=np.float32)}
    assert len(y) and all(np.isfinite(v).all() for v in [y, *x.values()])
    unique = np.unique(names)
    if len(unique) != 10:
        raise ValueError(f'Expected 10 scenarios, found {len(unique)}')
    order = np.random.default_rng(args.split_seed).permutation(unique)
    folds = np.array_split(order, 5)
    metrics, predictions, normalizers = [], [], []
    all_predictions = {seed: {h: np.full(len(y), np.nan) for h in x} for seed in args.seeds}
    for fold, held_out in enumerate(folds):
        valid = np.isin(names, held_out)
        train = ~valid
        assert not set(names[train]) & set(names[valid])
        # Shared scaler: equal weighting of both hypotheses, training rows only.
        pool = np.concatenate([x[h][train] for h in x])
        mean, std = pool.mean(0), pool.std(0)
        std = np.where(std < 1e-6, 1., std)
        normalizers.append({'fold': fold, 'mean': mean.tolist(), 'std': std.tolist()})
        for seed in args.seeds:
            for hypothesis in x:
                torch.manual_seed(seed)
                network = model()  # Exactly identical initial weights per pair.
                optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
                inputs = torch.from_numpy((x[hypothesis] - mean) / std)
                target = torch.from_numpy(y[train, None])
                network.train()
                for _ in range(args.epochs):
                    optimizer.zero_grad()
                    loss = nn.functional.mse_loss(network(inputs[train]), target)
                    loss.backward()
                    optimizer.step()
                network.eval()
                with torch.no_grad():
                    pred = network(inputs[valid]).numpy().ravel()
                all_predictions[seed][hypothesis][valid] = pred
        print(f'Fold {fold + 1}/5 complete: {", ".join(held_out)}', flush=True)

    def score(seed, hypothesis, scope, mask, pred):
        error = pred[mask] - y[mask]
        metrics.append(dict(seed=seed, model=hypothesis, scenario=scope,
                            n=int(mask.sum()), mae=float(np.abs(error).mean()),
                            rmse=float(np.sqrt(np.mean(error ** 2)))))
    for seed in args.seeds:
        preds = {**all_predictions[seed], 'zero': np.zeros_like(y)}
        for hypothesis, pred in preds.items():
            assert np.isfinite(pred).all()
            score(seed, hypothesis, 'ALL', np.ones(len(y), dtype=bool), pred)
            for name in unique:
                score(seed, hypothesis, name, names == name, pred)
        for i in range(len(y)):
            predictions.append(dict(seed=seed, scenario=names[i], frame=int(frames[i]),
                                    actual=float(y[i]), H0=float(preds['H0'][i]),
                                    H1=float(preds['H1'][i]), zero=0.))
    write_csv(args.output / 'metrics.csv', metrics)
    write_csv(args.output / 'predictions.csv', predictions)
    summary = []
    for scope in ['ALL', *unique]:
        for hypothesis in ['H0', 'H1', 'zero']:
            rows = [r for r in metrics if r['scenario'] == scope and r['model'] == hypothesis]
            row = dict(scenario=scope, model=hypothesis, n=rows[0]['n'])
            for metric in ['mae', 'rmse']:
                values = [r[metric] for r in rows]
                row[metric + '_mean'] = float(np.mean(values))
                row[metric + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
            summary.append(row)
    write_csv(args.output / 'summary.csv', summary)
    config = dict(recording_hz=10, seeds=args.seeds, split_seed=args.split_seed,
                  epochs=args.epochs, lr=args.lr, optimizer='Adam', batch='full training fold',
                  device='cpu', torch_version=torch.__version__, transitions=len(y),
                  validation_folds=[list(f) for f in folds], normalizers=normalizers,
                  normalization='Shared training-only pooled H0/H1 input mean and std',
                  checkpoint='Final fixed epoch; no validation selection')
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    first = args.seeds[0]
    low, high = float(y.min()), float(y.max())
    for ax, hypothesis in zip(axes[0], ['H0', 'H1']):
        ax.scatter(y, all_predictions[first][hypothesis], s=8, alpha=.3)
        ax.plot([low, high], [low, high], 'k--')
        ax.set(xlabel='Actual speed change (m/s)', ylabel='Predicted speed change (m/s)',
               title=f'{hypothesis}: held-out predictions, seed {first}')
    pos = np.arange(len(unique))
    for shift, hypothesis in zip([-.25, 0, .25], ['H0', 'H1', 'zero']):
        rows = [next(r for r in summary if r['scenario'] == name and r['model'] == hypothesis)
                for name in unique]
        axes[1, 0].bar(pos + shift, [r['rmse_mean'] for r in rows], width=.25,
                        yerr=[r['rmse_std'] for r in rows], label=hypothesis)
    axes[1, 0].set_xticks(pos, [name.split('_Town')[0] for name in unique], rotation=60, ha='right', fontsize=8)
    axes[1, 0].set_ylabel('Held-out RMSE (m/s); mean ± seed SD')
    axes[1, 0].legend()
    # Display abrupt control switches in HardBreak scenario, all held out together.
    chosen = next(name for name in unique if name.startswith('HardBreak'))
    mask = names == chosen
    axes[1, 1].plot(frames[mask] / 10, y[mask], label='Actual', color='black', linewidth=1)
    for hypothesis in x:
        axes[1, 1].plot(frames[mask] / 10, all_predictions[first][hypothesis][mask],
                         label=hypothesis, alpha=.75, linewidth=1)
    axes[1, 1].set(xlabel='Recorded frame ID / 10 (s)', ylabel='Speed change (m/s)', title=f'{chosen.split("_Town")[0]}, seed {first}')
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(args.output / 'comparison.png', dpi=160)
    plt.close(fig)
    for row in summary[:3]:
        print(f"{row['model']}: MAE {row['mae_mean']:.6f} ± {row['mae_std']:.6f}; RMSE {row['rmse_mean']:.6f} ± {row['rmse_std']:.6f}")


if __name__ == '__main__':
    main()
