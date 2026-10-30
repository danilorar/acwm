# Think Before Act

Bench2Drive data preparation and diagnostics for an action-conditioned visual
world model. The repository contains the dataset implementation, download and
extraction tools, tests, diagnostic experiments and their recorded results.
Visual-encoder and Transformer training have not been implemented.

## Setup

```bash
git clone --recurse-submodules https://github.com/danilorar/think_b4_act.git
cd think_b4_act
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

The CPU PyTorch install is sufficient for dataset tests and diagnostics. Choose
an appropriate GPU build separately when GPU training is introduced.

Bench2Drive is pinned as an upstream submodule. An optional compatibility patch
updates its old Mini-download CLI calls:

```bash
git -C external/Bench2Drive apply ../../patches/bench2drive-download-mini.patch
```

Raw datasets, archives, virtual environments and checkpoints are excluded from
Git. Selection/catalog/split manifests and compact reports are included for
reproducibility. Downloading data requires substantial storage; inspect the plan
before starting transfers.

- [Dataset interface and tests](docs/bench2drive_dataset.md)
- [Training-data expansion and coverage](docs/bench2drive_expansion.md)
- [Data audit](experiments/data_audit/README.md)
- [One-step action alignment](experiments/action_alignment/README.md)
- [Multi-step dynamics validation](experiments/dynamics_validation/README.md)

The existing experiment reports preserve their historical assumptions; the
refactored dataset interface is documented separately. Follow the data preparation
instructions to restore the local datasets before running real-data integrations.

```bash
python -m unittest discover -s tests -v
python test_dataset.py
```

Synthetic tests work without Mini; Mini integration tests skip when data is absent.
The demo and recorded experiment reruns require the corresponding local data.
Action timing remains a provisional hypothesis; limited coasting and binary brake
coverage do not establish arbitrary-action MPC validity.
