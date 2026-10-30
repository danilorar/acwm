from pathlib import Path
import argparse
import gzip
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=float, help='Verified saved-frame rate, not simulator tick rate')
    args = parser.parse_args()
    if args.fps is not None and args.fps <= 0:
        parser.error('--fps must be positive')
    root = Path(__file__).resolve().parent
    scenario = root / 'data/bench2drive_mini/HardBreakRoute_Town01_Route30_Weather3'
    paths = sorted((scenario / 'anno').glob('*.json.gz'))
    if len(paths) < 3:
        raise ValueError(f'Not enough annotations in {scenario}')
    frames = np.array([int(p.name.split('.')[0]) for p in paths])
    if not np.all(np.diff(frames) == 1):
        raise ValueError('Missing frames: FFT requires uniformly sampled data')
    records = []
    for path in paths:
        with gzip.open(path, 'rt') as f:
            records.append(json.load(f))
    signals = {key: np.array([r[key] for r in records], dtype=float)
               for key in ['speed', 'throttle', 'brake', 'steer']}
    output = root / 'outputs/control_analysis'
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(frames, signals['speed'])
    axes[0].set_ylabel('Speed (m/s)')
    for key in ['throttle', 'brake']:
        axes[1].plot(frames, signals[key], label=key.capitalize())
    axes[1].set_ylabel('Control')
    axes[1].legend()
    axes[2].plot(frames, signals['steer'])
    axes[2].set_ylabel('Steering')
    axes[2].set_xlabel('Recorded frame ID')
    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(output / 'controls.png', dpi=160)
    plt.close(fig)

    # Remove a linear trend and apply a Hann window to reduce spectral leakage.
    fs = args.fps if args.fps is not None else 1.0
    unit = 'Hz' if args.fps is not None else 'cycles / recorded frame'
    freq = np.fft.rfftfreq(len(frames), d=1 / fs)
    window = np.hanning(len(frames))
    t = np.arange(len(frames))
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    print('Frames:', len(records))
    simultaneous = (signals['throttle'] > .01) & (signals['brake'] > .01)
    print(f'Simultaneous throttle and brake: {simultaneous.sum()} ({simultaneous.mean():.2%})')
    for ax, (key, values) in zip(axes, signals.items()):
        detrended = values - np.polyval(np.polyfit(t, values, 1), t)
        psd = abs(np.fft.rfft(detrended * window)) ** 2 / (fs * np.sum(window ** 2))
        if len(frames) % 2 == 0:
            psd[1:-1] *= 2
        else:
            psd[1:] *= 2
        ax.plot(freq[1:], psd[1:])
        ax.set_ylabel(f'{key}\nPSD')
        ax.grid(True)
        peak = 1 + np.argmax(psd[1:])
        print(f'{key:10s} min={values.min():.3f} max={values.max():.3f} mean={values.mean():.3f}; spectral peak={freq[peak]:.5f} {unit}')
    axes[-1].set_xlabel(f'Frequency ({unit})')
    fig.suptitle('Linear detrend + Hann-window periodogram (PSD units: signal units squared per frequency unit)')
    fig.tight_layout()
    fig.savefig(output / 'spectra.png', dpi=160)
    plt.close(fig)
    print('Plots saved in:', output)


if __name__ == '__main__':
    main()
