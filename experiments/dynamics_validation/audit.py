"""Check indexing/results and describe observational action coverage, without retraining."""
from pathlib import Path
import csv
import json
import numpy as np
import torch
from run import load, Dynamics, PROJECT, HERE, K, H


def main():
    output = HERE / 'results'
    names, starts, speed, controls, histories, future, _ = load(PROJECT / 'data/bench2drive_mini')
    # Directly verify every history/future slice and the common speed targets.
    for offset, align in enumerate(['H0', 'H1']):
        np.testing.assert_array_equal(histories[align][:, :, 0], speed[:, :K])
        np.testing.assert_array_equal(histories[align][:, 1:, 1:4], controls[:, offset:offset + K - 1])
        np.testing.assert_array_equal(future[align], controls[:, K - 1 + offset:K - 1 + offset + H])
        # Perturb every value strictly after observation cutoff: history must stay identical.
        changed = controls.copy()
        changed[:, K:] = 999
        np.testing.assert_array_equal(changed[:, offset:offset + K - 1], histories[align][:, 1:, 1:4])
    data = np.load(output / 'rollouts.npz')
    np.testing.assert_array_equal(data['actual'], speed[:, K:])
    np.testing.assert_array_equal(data['scenario'], names)
    np.testing.assert_array_equal(data['start_frame'], starts)
    assert len(set(zip(names, starts))) == len(names)
    config = json.loads((output / 'config.json').read_text())
    for seed in config['seeds']:
        torch.manual_seed(seed)
        a = Dynamics()
        torch.manual_seed(seed)
        b = Dynamics()
        assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    with (output / 'metrics.csv').open() as f:
        metrics = list(csv.DictReader(f))
    masks = {'all': np.ones(len(names), bool), 'strong_braking': data['strong_braking'],
             'immediate_braking': data['immediate_braking']}
    for row in metrics:
        mask = masks[row['subset']] & ((names == row['scenario']) if row['scenario'] != 'ALL' else True)
        pred = data[f"{row['model']}_seed{row['seed']}"]
        step = int(row['step'])
        error = pred[mask] - data['actual'][mask]
        if step:
            error = error[:, step - 1]
        assert mask.sum() == int(row['n_windows'])
        np.testing.assert_allclose(np.abs(error).mean(), float(row['mae']), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.sqrt((error ** 2).mean()), float(row['rmse']), rtol=1e-5, atol=1e-6)
    # Compare the previous MLP only on the identical first target transitions.
    previous = PROJECT / 'experiments/action_alignment/results/predictions.csv'
    with previous.open() as f:
        old = {(int(r['seed']), r['scenario'], int(r['frame'])): r for r in csv.DictReader(f)}
    reference = []
    for seed in config['seeds']:
        matched = [old[(seed, name, int(start) + K - 1)] for name, start in zip(names, starts)]
        np.testing.assert_allclose([float(r['actual']) for r in matched], data['actual'][:, 0] - data['current'], atol=2e-6)
        for align in ['H0', 'H1']:
            for subset, mask in masks.items():
                error = np.array([float(r[align]) - float(r['actual']) for r in matched])[mask]
                reference.append(dict(seed=seed, model='previous_MLP_' + align, subset=subset,
                                      n_windows=int(mask.sum()), mae=float(np.abs(error).mean()),
                                      rmse=float(np.sqrt((error ** 2).mean()))))
    with (output / 'previous_mlp_reference.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(reference[0]))
        writer.writeheader()
        writer.writerows(reference)
    print('Verified every reported metric, matched frame IDs, paired initialization and no future controls in history.')
    with (output / 'control_responses.csv').open() as f:
        rows = list(csv.DictReader(f))
    paired = {}
    for row in rows:
        key = row['model'], row['seed'], row['scenario'], row['start_frame']
        paired.setdefault(key, {})[row['candidate']] = float(row['speed_1s'])
    report = {}
    for model in ['GRU_H0', 'GRU_H1']:
        pairs = [p for key, p in paired.items() if key[0] == model]
        violations = sum(not(p['brake'] <= p['coast'] + .05 and p['coast'] <= p['throttle'] + .05) for p in pairs)
        report[model] = dict(paired_cases=len(pairs), monotonicity_violations=violations,
                             violation_fraction=violations / len(pairs), tolerance_mps=.05)
    # Count unique frames rather than repeatedly counting overlapping windows.
    cache = {}
    for name in np.unique(names):
        import gzip
        folder = PROJECT / 'data/bench2drive_mini' / name / 'anno'
        for path in sorted(folder.glob('*.json.gz')):
            with gzip.open(path, 'rt') as f:
                cache[(name, int(path.name.split('.')[0]))] = json.load(f)
    moving = [r for r in cache.values() if r['speed'] >= 2]
    coast = [r for r in moving if r['throttle'] <= .01 and r['brake'] <= .01]
    report['coverage'] = dict(unique_annotations=len(cache), moving_annotations=len(moving),
                               moving_coast_annotations=len(coast), moving_coast_fraction=len(coast) / len(moving),
                               simultaneous_throttle_brake=sum(r['throttle'] > .01 and r['brake'] > .01 for r in cache.values()))
    for align in future:
        a = future[align]
        mask = speed[:, K - 1] >= 2
        report['coverage'][align + '_moving_windows'] = int(mask.sum())
        for candidate, matches in {
            'throttle': ((a[:, :, 1] >= .9) & (a[:, :, 2] <= .01)).all(1),
            'coast': ((a[:, :, 1] <= .01) & (a[:, :, 2] <= .01)).all(1),
            'brake': ((a[:, :, 1] <= .01) & (a[:, :, 2] >= .9)).all(1),
        }.items():
            report['coverage'][align + '_' + candidate + '_sustained_windows'] = int((mask & matches).sum())
    events = []
    for (name, frame), row in cache.items():
        nxt = cache.get((name, frame + 1))
        if nxt is not None and row['speed'] >= 2 and nxt['speed'] - row['speed'] <= -.5:
            events.append((row, nxt))
    report['abrupt_deceleration'] = dict(
        unique_transitions=len(events),
        brake_at_t_ge_half=sum(a['brake'] >= .5 for a, b in events),
        brake_at_next_ge_half=sum(b['brake'] >= .5 for a, b in events),
        note='Observed deceleration need not be commanded braking; collision/grade/lag may contribute.')
    (output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
