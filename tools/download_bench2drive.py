"""Restore the frozen Bench2Drive cohort; no transfer unless --download is passed."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

PROJECT = Path(__file__).resolve().parents[1]
REPO = 'rethinklab/Bench2Drive-V0.0.4'


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verified(path, entry):
    return path.is_file() and path.stat().st_size == entry['size'] and sha256(path) == entry['sha256']


def main():
    """Restore selected archives only; selection/evaluation filtering is already frozen."""
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--download',action='store_true')
    args=parser.parse_args();base=PROJECT/'data/bench2drive';base.mkdir(parents=True,exist_ok=True)
    selection=json.loads((PROJECT/'data/config/bench_selection.json').read_text())
    save_json(base/'manifests/selection.json',selection)
    print('Frozen Bench2Drive selection:',len(selection['selected']),'routes')
    if not args.download:return
    from huggingface_hub import hf_hub_download
    for entry in selection['selected']:
        destination=base/'archives'/entry['path']
        if verified(destination,entry):continue
        if shutil.disk_usage(base).free<entry['size']+20*1024**3:raise OSError('Disk reserve reached')
        hf_hub_download(REPO,entry['path'],repo_type='dataset',revision=selection['revision'],local_dir=base/'archives')
        if not verified(destination,entry):raise ValueError('Archive checksum mismatch: '+entry['path'])


if __name__=='__main__':main()
