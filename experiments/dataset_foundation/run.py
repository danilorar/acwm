"""Run the complete diagnostic after the separately authorized data preparation."""
from pathlib import Path
import argparse,subprocess,sys
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--skip-training',action='store_true');args=p.parse_args()
steps=[['prepare_splits.py'],['verify.py'],['coverage.py','--verify-images']]
# verify needs an explicit hypothesis file. Until validation has selected offsets,
# validation training runs first; it never consumes held-out test labels for selection.
if not args.skip_training:
 steps=[['prepare_splits.py'],['dynamics.py'],['verify.py'],['coverage.py','--verify-images'],['summarize.py'],['source_checks.py'],['candidate_checks.py']]
else:steps.extend([['summarize.py'],['source_checks.py'],['candidate_checks.py']])
steps.extend([['inventory.py'],['make_report.py']])
for step in steps:subprocess.run([sys.executable,str(HERE/step[0]),*step[1:]],check=True,cwd=HERE.parents[1])
