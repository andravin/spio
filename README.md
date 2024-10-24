# Spio

Efficient CUDA kernels for training convolutional neural networks with PyTorch.

## Installation from Source

First, ensure you have a C compiler installed. On Ubuntu:

```bash
sudo apt update
sudo apt install build-essential
```

Clone the repository:

```bash
git clone https://github.com/andravin/spio.git
cd spio
```

Optionally, create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the package from source using pip:

```bash
pip install --upgrade pip
pip install .
```

Optionally, run the unit tests. This can take a while,
because Spio tests every configuration of every kernel. It goes a bit faster
if we set the SPIO_WORKERS environment variable to use all CPU cores for compiling kernels:

```bash
cd tests
SPIO_WORKERS=$(nproc) pytest .
```

Note: the tests and scripts cannot be run from the top-level spio directory because
that would cause Python to find the local spio package instead of the installed package.
Only the installed package has the compiled spio.cuda.driver Cython extension, so using
the local package would cause an import error. That is why `cd tests` before `pytest .` is essential.
