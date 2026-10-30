"""Scenario-held-out, action-conditioned autoregressive speed diagnostic."""
from pathlib import Path
import argparse
import csv
import itertools
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
sys.path.insert(0, str(PROJECT))
from src.bench2drive_dataset import Bench2DriveDataset

DT, K, H = .1, 4, 10


def csv_write(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(5, 32, batch_first=True)
        self.decoder = nn.GRUCell(4, 32)
        self.delta = nn.Linear(32, 1)

    def forward(self, history, future, mean, std):
        # History tokens: observed speed, incoming known control, validity mask.
        _, hidden = self.encoder(history)
        hidden = hidden[0]
        speed = history[:, -1, 0] * std[0] + mean[0]
        result = []
        for step in range(H):
            inputs = torch.cat([((speed - mean[0]) / std[0])[:, None], future[:, step]], 1)
            hidden = self.decoder(inputs, hidden)
            speed = (speed + self.delta(hidden).squeeze(1)).clamp_min(0)
            result.append(speed)
        return torch.stack(result, 1)


def load(root):
    ds = Bench2DriveDataset(root, history=K, horizon=H)
    cache = {name: {fid: ds._load_annotation(path) for fid, path in info['annotations'].items()}
             for name, info in ds.scenarios.items()}
    names, starts, speeds, controls = [], [], [], []
    for name, frames in ds.samples:
        rows = [cache[name][fid] for fid in frames]
        names.append(name)
        starts.append(frames[0])
        speeds.append([r['speed'] for r in rows])
        controls.append([[r['steer'], r['throttle'], r['brake']] for r in rows])
    v, a = np.array(speeds, np.float32), np.array(controls, np.float32)
    # Four observed speeds; three controls of transitions entirely in history.
    # The first token has no preceding observation/control, hence mask=0.
    history, future = {}, {}
    for offset, align in enumerate(['H0', 'H1']):
        hist = np.zeros((len(v), K, 5), np.float32)
        hist[:, :, 0] = v[:, :K]
        hist[:, 1:, 1:4] = a[:, offset:offset + K - 1]
        hist[:, 1:, 4] = 1
        history[align] = hist
        future[align] = a[:, K - 1 + offset:K - 1 + offset + H]
        assert future[align].shape == (len(v), H, 3)
        # Indices 0..3 are the only history observations; controls never exceed 3.
        assert offset + K - 2 <= K - 1
    # Fit physics on unique one-step transitions from training scenarios later.
    one_step = {}
    for name, rows in cache.items():
        ids = sorted(set(rows) & set(ds.scenarios[name]['images']))
        pairs = [(i, i + 1) for i in ids if i + 1 in rows and i + 1 in ds.scenarios[name]['images']]
        one_step[name] = np.array([[rows[i]['speed'], rows[j]['speed'],
                                    rows[i]['throttle'], rows[i]['brake'],
                                    rows[j]['throttle'], rows[j]['brake']]
                                   for i, j in pairs], dtype=float)
    arrays = [v, a, *history.values(), *future.values()]
    assert all(np.isfinite(x).all() for x in arrays)
    return np.array(names), np.array(starts), v, a, history, future, one_step


def fit_physics(rows, align):
    # Convex sign-constrained least squares via all eight active sets.
    # dv/dt = bias + throttle_gain*T - brake_gain*B - drag*v.
    offset = 2 if align == 'H0' else 4
    base = np.column_stack([rows[:, offset], -rows[:, offset + 1], -rows[:, 0]])
    target = (rows[:, 1] - rows[:, 0]) / DT
    best = None
    for mask in itertools.product([False, True], repeat=3):
        active = np.flatnonzero(mask)
        design = np.column_stack([np.ones(len(rows)), base[:, active]])
        beta = np.linalg.lstsq(design, target, rcond=None)[0]
        if (beta[1:] < -1e-10).any():
            continue
        coef = np.zeros(4)
        coef[0], coef[active + 1] = beta[0], np.maximum(beta[1:], 0)
        error = np.mean((design @ beta - target) ** 2)
        if best is None or error < best[0]:
            best = error, coef
    return best[1]


def physics_rollout(speed, actions, coef):
    speed = speed.copy()
    result = []
    for step in range(H):
        acceleration = coef[0] + coef[1] * actions[:, step, 1] - coef[2] * actions[:, step, 2] - coef[3] * speed
        speed = np.maximum(0, speed + DT * acceleration)
        result.append(speed.copy())
    return np.stack(result, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT / 'data/bench2drive_mini')
    parser.add_argument('--output', type=Path, default=HERE / 'results')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--lr', type=float, default=.003)
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Positive epochs and distinct seeds required')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    names, starts, v, a, hist, future, one_step = load(args.root)
    folds = json.loads((PROJECT / 'experiments/action_alignment/results/config.json').read_text())['validation_folds']
    assert sorted(sum(folds, [])) == list(np.unique(names))
    target, current = v[:, K:], v[:, K - 1]
    # Identical observed-event subsets for both hypotheses; controls do not define events.
    increments = np.diff(v[:, K - 1:], axis=1)
    strong = (current >= 2) & (increments.min(1) <= -.5)  # >=5 m/s² observed deceleration
    onset = (current >= 2) & (increments[:, 0] <= -.5)
    models = ['GRU_H0', 'GRU_H1', 'physics_H0', 'physics_H1', 'constant']
    predictions = {seed: {m: np.full(target.shape, np.nan, np.float32) for m in models} for seed in args.seeds}
    responses, settings, curves = [], [], []
    controls = {'throttle': [0, 1, 0], 'coast': [0, 0, 0], 'brake': [0, 0, 1]}
    for fold, held_out in enumerate(folds):
        valid = np.isin(names, held_out)
        train = ~valid
        assert not set(names[train]) & set(names[valid])
        # Shared scaler, exclusively training observations/commands.
        train_speed = v[train, :K].reshape(-1)
        pool = np.concatenate([hist[h][train, 1:, 1:4].reshape(-1, 3) for h in hist] +
                              [future[h][train].reshape(-1, 3) for h in future])
        mean = np.r_[train_speed.mean(), pool.mean(0)].astype(np.float32)
        std = np.r_[train_speed.std(), pool.std(0)].astype(np.float32)
        std[std < 1e-6] = 1
        mean_t, std_t = torch.tensor(mean), torch.tensor(std)
        prepared = {}
        for align in hist:
            history = hist[align].copy()
            history[:, :, :4] = (history[:, :, :4] - mean) / std
            history[:, 0, 1:4] = 0  # Missing initial control has a separate zero validity flag.
            prepared[align] = (torch.from_numpy(history), torch.from_numpy((future[align] - mean[1:]) / std[1:]))
        fit_rows = np.concatenate([one_step[n] for n in sorted(one_step) if n not in held_out])
        coefficients = {h: fit_physics(fit_rows, h) for h in hist}
        settings.append(dict(fold=fold, held_out=held_out, train_windows=int(train.sum()),
                             validation_windows=int(valid.sum()), mean=mean.tolist(), std=std.tolist(),
                             physics={h: c.tolist() for h, c in coefficients.items()}))
        for seed in args.seeds:
            for align in hist:
                torch.manual_seed(seed)
                net = Dynamics()
                optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
                history, actions = prepared[align]
                th, ta = history[train], actions[train]
                ty = torch.from_numpy(target[train])
                net.train()
                for epoch in range(args.epochs):
                    optimizer.zero_grad()
                    pred = net(th, ta, mean_t, std_t)
                    loss = nn.functional.mse_loss(pred, ty)
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5.)
                    optimizer.step()
                    if epoch in [0, args.epochs - 1]:
                        curves.append(dict(fold=fold, seed=seed, model='GRU_' + align,
                                           epoch=epoch + 1, training_mse=float(loss.detach())))
                net.eval()
                with torch.no_grad():
                    predictions[seed]['GRU_' + align][valid] = net(history[valid], actions[valid], mean_t, std_t).numpy()
                    for candidate, command in controls.items():
                        raw = np.tile(command, (int(valid.sum()), H, 1)).astype(np.float32)
                        candidate_pred = net(history[valid], torch.tensor((raw - mean[1:]) / std[1:]), mean_t, std_t).numpy()
                        selected = current[valid] >= 2
                        for i, global_i in enumerate(np.flatnonzero(valid)):
                            if selected[i]:
                                responses.append(dict(seed=seed, model='GRU_' + align, scenario=names[global_i],
                                                      start_frame=int(starts[global_i]), candidate=candidate,
                                                      initial_speed=float(current[global_i]),
                                                      speed_1s=float(candidate_pred[i, -1])))
            for align in hist:
                predictions[seed]['physics_' + align][valid] = physics_rollout(current[valid], future[align][valid], coefficients[align])
            predictions[seed]['constant'][valid] = np.repeat(current[valid, None], H, axis=1)
            print(f'Fold {fold + 1}/5 seed {seed}: complete ({time.monotonic() - started:.1f}s)', flush=True)
    # Every window is predicted only by models trained on other scenarios.
    metric_rows = []
    masks = {'all': np.ones(len(v), bool), 'strong_braking': strong, 'immediate_braking': onset}
    for seed in args.seeds:
        for model in models:
            pred = predictions[seed][model]
            assert np.isfinite(pred).all()
            for scope in ['ALL', *np.unique(names)]:
                for subset, subset_mask in masks.items():
                    mask = subset_mask & ((names == scope) if scope != 'ALL' else True)
                    if not mask.any():
                        continue
                    for step in [0, *range(1, H + 1)]:
                        error = (pred[mask] - target[mask]) if step == 0 else (pred[mask, step - 1] - target[mask, step - 1])
                        metric_rows.append(dict(seed=seed, model=model, scenario=scope, subset=subset,
                                                step=step, seconds='full' if step == 0 else step * DT,
                                                n_windows=int(mask.sum()), mae=float(np.abs(error).mean()),
                                                rmse=float(np.sqrt(np.mean(error ** 2)))))
    csv_write(args.output / 'metrics.csv', metric_rows)
    summary = []
    keys = list(dict.fromkeys((r['model'], r['scenario'], r['subset'], r['step']) for r in metric_rows))
    for model, scope, subset, step in keys:
        rows = [r for r in metric_rows if (r['model'], r['scenario'], r['subset'], r['step']) == (model, scope, subset, step)]
        row = dict(model=model, scenario=scope, subset=subset, step=step, n_windows=rows[0]['n_windows'])
        for measure in ['mae', 'rmse']:
            values = [r[measure] for r in rows]
            row[measure + '_mean'] = float(np.mean(values))
            row[measure + '_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
        summary.append(row)
    csv_write(args.output / 'summary.csv', summary)
    csv_write(args.output / 'control_responses.csv', responses)
    csv_write(args.output / 'training.csv', curves)
    # Store arrays for independent metric verification and alternative plots.
    np.savez_compressed(args.output / 'rollouts.npz', scenario=names, start_frame=starts,
                        actual=target, current=current, strong_braking=strong, immediate_braking=onset,
                        **{f'{m}_seed{s}': p for s in args.seeds for m, p in predictions[s].items()})
    config = dict(history=K, horizon=H, dt=DT, seeds=args.seeds, epochs=args.epochs,
                  lr=args.lr, batch='full training fold', optimizer='Adam', gradient_clip=5,
                  hidden_size=32, input_normalization='shared training-only', device='cpu',
                  torch_version=torch.__version__, windows=len(v), strong_braking_windows=int(strong.sum()),
                  immediate_braking_windows=int(onset.sum()), folds=settings,
                  runtime_seconds=time.monotonic() - started)
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    def summary_row(model, subset='all', step=0, scenario='ALL'):
        return next(r for r in summary if (r['model'], r['scenario'], r['subset'], r['step']) == (model, scenario, subset, step))
    colors = dict(zip(models, ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'gray']))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model in models:
        for ax, measure in zip(axes, ['mae', 'rmse']):
            rows = [summary_row(model, step=i) for i in range(1, H + 1)]
            means = np.array([r[measure + '_mean'] for r in rows])
            sd = np.array([r[measure + '_std'] for r in rows])
            ax.plot(np.arange(1, H + 1) * DT, means, label=model, color=colors[model])
            ax.fill_between(np.arange(1, H + 1) * DT, means - sd, means + sd, alpha=.12, color=colors[model])
            ax.set(xlabel='Forecast horizon (s)', ylabel=f'Speed {measure.upper()} (m/s)')
            ax.grid(alpha=.3)
    axes[0].legend(fontsize=8)
    fig.suptitle('Scenario-held-out autoregressive rollouts; mean ± seed SD')
    fig.tight_layout()
    fig.savefig(args.output / 'horizon_errors.png', dpi=160)
    plt.close(fig)
    # Examples selected by targets only, once per scenario; no model-error selection.
    examples = []
    for prefix in ['HardBreak', 'ControlLoss', 'VehicleTurning', 'Accident_Town']:
        mask = np.array([name.startswith(prefix) for name in names]) & (current >= 2)
        candidates = np.flatnonzero(mask)
        examples.append(candidates[np.argmax(current[candidates] - target[candidates, -1])])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, index in zip(axes.flat, examples):
        ax.plot(np.arange(H + 1) * DT, np.r_[current[index], target[index]], 'k-o', markersize=3, label='Actual')
        for model in models:
            pred = np.mean([predictions[s][model][index] for s in args.seeds], 0)
            ax.plot(np.arange(H + 1) * DT, np.r_[current[index], pred], label=model, color=colors[model])
        ax.set(title=f'{names[index].split("_Town")[0]}: cutoff frame {starts[index] + K - 1}',
               xlabel='Forecast horizon (s)', ylabel='Speed (m/s)')
        ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Largest observed 1-second speed drop per selected scenario; predictions averaged across seeds')
    fig.tight_layout()
    fig.savefig(args.output / 'rollout_examples.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model in models:
        rows = [summary_row(model, 'strong_braking', i) for i in range(1, H + 1)]
        axes[0].plot(np.arange(1, H + 1) * DT, [r['rmse_mean'] for r in rows], label=model, color=colors[model])
    axes[0].set(title=f'Strong braking windows (n={strong.sum()})', xlabel='Horizon (s)', ylabel='Speed RMSE (m/s)')
    axes[0].legend(fontsize=8)
    for model in models:
        row = summary_row(model, 'immediate_braking', 1)
        axes[1].bar(model, row['mae_mean'], yerr=row['mae_std'], color=colors[model])
    axes[1].set(title=f'Immediate strong braking (n={onset.sum()})', ylabel='0.1-second speed MAE (m/s)')
    axes[1].tick_params(axis='x', rotation=25)
    fig.tight_layout()
    fig.savefig(args.output / 'braking_analysis.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    response_summary = []
    for model in ['GRU_H0', 'GRU_H1']:
        means = []
        for command in controls:
            rows = [r for r in responses if r['model'] == model and r['candidate'] == command]
            delta = [r['speed_1s'] - r['initial_speed'] for r in rows]
            means.append(np.mean(delta))
            response_summary.append(dict(model=model, candidate=command, mean_delta_1s=float(np.mean(delta))))
        ax.plot(list(controls), means, 'o-', label=model)
    ax.set(ylabel='Mean predicted 1-second speed change (m/s)', title='Candidate controls: identical observed histories, initial speed ≥2 m/s')
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(args.output / 'control_response.png', dpi=160)
    plt.close(fig)
    csv_write(args.output / 'control_response_summary.csv', response_summary)
    for model in models:
        row = summary_row(model)
        print(f"{model}: full MAE={row['mae_mean']:.4f}±{row['mae_std']:.4f}, RMSE={row['rmse_mean']:.4f}±{row['rmse_std']:.4f}")
    print(f'Results: {args.output}', flush=True)


if __name__ == '__main__':
    main()
