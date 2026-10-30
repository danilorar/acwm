"""Plan by default; pass --download only after reviewing the transfer/storage plan."""
from pathlib import Path
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import re
import shutil
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET

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


def entry_from_file(file):
    path = file['path']
    match = re.fullmatch(r'([^/]+)_(Town\d+(?:HD)?)_Route(\d+)_Weather(\d+)_(.+)\.tar\.gz', path)
    if not match:
        return None
    lfs = file.get('lfs') or {}
    digest = lfs.get('sha256')
    if not digest or not file.get('size'):
        raise ValueError(f'Missing authoritative size/SHA256: {path}')
    return dict(path=path, size=file['size'], sha256=digest, scenario_type=match[1],
                town=match[2], route=int(match[3]), weather=int(match[4]))


def catalog_from_hub():
    from huggingface_hub import HfApi
    info = HfApi().dataset_info(REPO, files_metadata=True)
    files = []
    for item in info.siblings:
        lfs = item.lfs
        files.append(dict(path=item.rfilename, size=item.size,
                          lfs=dict(sha256=lfs.sha256) if lfs is not None else None))
    return dict(repository=REPO, revision=info.sha, files=files)


def select(entries, count, seed, previous=()):
    groups = defaultdict(list)
    for entry in entries:
        groups[entry['scenario_type']].append(entry)
    selected = list(previous)
    selected_paths = {entry['path'] for entry in selected}
    route_keys = {(entry['town'], entry['route']) for entry in selected}
    # Existing prefixes survive expansion. Each round prioritizes least-covered types.
    while len(selected) < count:
        counts = Counter(entry['scenario_type'] for entry in selected)
        choices = []
        for name in sorted(groups):
            candidates = [e for e in groups[name] if e['path'] not in selected_paths and (e['town'],e['route']) not in route_keys]
            if not candidates:
                continue
            same = [e for e in selected if e['scenario_type'] == name]
            towns = Counter(e['town'] for e in same)
            weather = Counter(e['weather'] for e in same)
            rank = lambda e: (towns[e['town']], weather[e['weather']], hashlib.sha256(f"{seed}:{e['path']}".encode()).hexdigest())
            choices.append((counts[name], name, min(candidates, key=rank)))
        if not choices:
            raise ValueError('Not enough unique eligible routes for requested selection')
        entry = min(choices, key=lambda x: (x[0], x[1]))[2]
        selected.append(entry)
        selected_paths.add(entry['path'])
        route_keys.add((entry['town'],entry['route']))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-routes', type=int, default=132)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--destination', type=Path, default=PROJECT/'data/bench2drive_train')
    parser.add_argument('--download', action='store_true', help='Execute the reviewed transfer; default only plans')
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--workers', type=int, default=4, help='Concurrent archive transfers')
    parser.add_argument('--extraction-factor', type=float, default=3., help='Conservative retained extraction allowance / compressed size; estimate, not measured')
    parser.add_argument('--reserve-gib', type=float, default=20.)
    parser.add_argument('--only', nargs='+', help='Download selected filenames only; keep full manifest unchanged')
    parser.add_argument('--metadata', type=Path, help='Previously queried catalog JSON for offline/reproducible planning')
    args = parser.parse_args()
    if args.num_routes < 1 or args.retries < 1 or args.workers < 1 or args.extraction_factor <= 0 or args.reserve_gib < 0:
        parser.error('Invalid count, retry or storage settings')
    base = args.destination.resolve()
    for directory in ['archives','extracted','manifests']:(base/directory).mkdir(parents=True, exist_ok=True)
    manifest_path = base/'manifests/selection.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous:
        if previous['seed'] != args.seed or args.num_routes < len(previous['selected']):
            raise ValueError('Existing selection must retain its seed and prefix; use a new destination for a different selection')
        # Keep revision pinned for repeatability; expanding queries the same revision.
        catalog_path = base/'manifests/catalog.json'
        catalog = json.loads(catalog_path.read_text())
    else:
        catalog = json.loads(args.metadata.read_text()) if args.metadata else catalog_from_hub()
        if catalog.get('repository',REPO) != REPO:
            raise ValueError('Only the specified official 0.0.4 training repository is allowed')
        catalog['repository'] = REPO
        save_json(base/'manifests/catalog.json',catalog)
    entries = [e for f in catalog['files'] if (e := entry_from_file(f)) is not None]
    evaluation_xml = PROJECT/'external/Bench2Drive/leaderboard/data/bench2drive_0.0.4_val.xml'
    evaluation = {(route.attrib['town'],int(route.attrib['id'])) for route in ET.parse(evaluation_xml).getroot().iter('route')}
    eligible = [e for e in entries if (e['town'],e['route']) not in evaluation]
    if args.num_routes >= len(entries):
        raise ValueError('Refusing to select the complete collection')
    chosen = select(eligible,args.num_routes,args.seed,previous['selected'] if previous else ())
    plan = dict(repository=REPO,revision=catalog['revision'],seed=args.seed,selected=chosen,
                excluded_evaluation_route_matches=len(entries)-len(eligible),
                evaluation_xml=str(evaluation_xml.relative_to(PROJECT)))
    save_json(manifest_path,plan)
    compressed = sum(e['size'] for e in chosen)
    remaining = sum(max(0,e['size']-(base/'archives'/e['path']).stat().st_size) if (base/'archives'/e['path']).is_file() else e['size'] for e in chosen)
    free = shutil.disk_usage(base).free
    # Includes full compressed payload, retained-data allowance, HF partial/cache margin and reserve.
    required = remaining + args.extraction_factor*compressed + args.workers*max(e['size'] for e in chosen) + args.reserve_gib*1024**3
    report = dict(routes=len(chosen),scenario_distribution=dict(Counter(e['scenario_type'] for e in chosen)),
                  towns=dict(Counter(e['town'] for e in chosen)),weather=dict(Counter(e['weather'] for e in chosen)),
                  download_bytes=compressed,remaining_download_bytes=remaining,free_bytes=free,
                  estimated_additional_required_bytes=int(required),extraction_factor=args.extraction_factor,
                  extraction_estimate_note='Allowance only: HF metadata does not publish expanded tar member sizes; extraction checks actual retained sizes before writing.',
                  reserve_gib=args.reserve_gib,workers=args.workers,destination=str(base),fits=required<=free)
    save_json(base/'manifests/storage_plan.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='scenario_distribution'},indent=2),flush=True)
    if not report['fits']:
        raise RuntimeError('Plan does not fit. Reduce --num-routes in a new destination/selection; no transfer started.')
    if not args.download:
        print('Plan saved. No archives downloaded. Review before running --download.')
        return
    selected = chosen
    if args.only:
        unknown = set(args.only)-{e['path'] for e in chosen}
        if unknown:raise ValueError(f'--only files not in selection: {unknown}')
        selected = [e for e in chosen if e['path'] in args.only]
    from huggingface_hub import hf_hub_download
    status_path=base/'manifests/download_status.json'
    status=json.loads(status_path.read_text()) if status_path.exists() else {}
    failures=[]
    lock=threading.Lock()
    def record(entry,value):
        with lock:
            status[entry['path']]=value
            save_json(status_path,status)
    def transfer(item):
        i,entry=item
        path=base/'archives'/entry['path']
        if verified(path,entry):
            record(entry,dict(status='verified',sha256=entry['sha256'],size=entry['size']))
            print('Verified existing:',entry['path'],flush=True)
            return
        for attempt in range(args.retries):
            try:
                if shutil.disk_usage(base).free < args.workers*max(e['size'] for e in selected)+args.reserve_gib*1024**3:
                    raise OSError('Insufficient free space plus concurrent-transfer margin and reserve')
                print(f'Downloading {i+1}/{len(selected)}: {entry["path"]}',flush=True)
                hf_hub_download(repo_id=REPO,repo_type='dataset',filename=entry['path'],
                                revision=plan['revision'],local_dir=base/'archives',
                                force_download=path.exists())
                if not verified(path,entry):raise ValueError('Downloaded size/SHA256 mismatch')
                record(entry,dict(status='verified',sha256=entry['sha256'],size=entry['size']))
                print('Verified:',entry['path'],flush=True)
                return
            except Exception as exc:
                failure=dict(path=entry['path'],attempt=attempt+1,error=str(exc),time=time.time())
                with lock:
                    with (base/'manifests/download_failures.jsonl').open('a') as f:f.write(json.dumps(failure)+'\n')
                if attempt+1==args.retries:
                    with lock:failures.append(failure)
                    record(entry,dict(status='failed',error=str(exc)))
                else:time.sleep(min(5*(attempt+1),15))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(transfer,enumerate(selected)))
    if failures:raise RuntimeError(f'{len(failures)} archive downloads failed; rerun the same command to retry')
    print(f'Verified {len(selected)} selected archives.',flush=True)


if __name__=='__main__':main()
