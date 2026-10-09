# NSOTree

NSOTree (Neural Survival Oblique Tree) is an experimental PyTorch implementation
of neural survival trees for Cox proportional-hazards modeling. The repository
contains the tree networks, PyCox model wrappers, data-loading utilities,
training scripts, sample survival datasets, and generated tree visualizations.

## Repository layout

```text
NSOTree2/           Previous implementation preserved from the original repo
models/             Neural tree networks and PyCox wrappers
utils/              Dataset loading and tree utilities
datasets/           Simulated and real survival datasets
tree_plots/         Example generated tree visualizations
train_nstree.py     Main NSOTree training experiment
train_cont.py       Continuous-tree training and fine-tuning experiment
test.py, test2.py   Development and visualization scripts
```

## Requirements

- Python 3.8-3.10 is recommended.
- A CUDA-capable GPU and a CUDA-enabled PyTorch build are required by the
  current training and tree-network code.

Create an environment and install the dependencies:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For platform-specific PyTorch installation commands, see the
[PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Data

The repository includes the simulated linear and Gaussian survival datasets
used by the default scripts, together with FLCHAIN, METABRIC, and SUPPORT CSV
files. The data loader also supports datasets exposed by `pycox.datasets`.

Available names accepted by `load_dataset` include:

- `simulated_linear`
- `simulated_gaussian`
- `support`
- `metabric`
- `flchain`
- `gbsg`

The `_2d` simulated variants require corresponding HDF5 files that are not
included in this repository.

## Usage

Run the main NSOTree experiment from the repository root:

```bash
python train_nstree.py
```

Run the continuous-tree experiment:

```bash
python train_cont.py
```

Experiment settings such as the dataset, tree depth, dropout, batch size,
epochs, and learning rate are defined near the bottom of each training script.
Training reports the time-dependent concordance index and, where enabled,
Jacobian and leaf-input ranks.

## Notes

This is research code. The training scripts currently assume CUDA and have not
been packaged as a stable Python library. Generated model checkpoints, Python
caches, editor settings, and new plot outputs are excluded from version control.
