# Spio

Efficient CUDA kernels for training convolutional neural networks with PyTorch.

## Installation

Install the package using pip:

```bash
pip install spio
```

## Installation from Source

First, ensure you have a C compiler installed. On Ubuntu:

```bash
sudo apt update
sudo apt install build-essential
```

Clone the repository:

```bash
git clone https://github.com/yourusername/spio.git
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

Because you installed from source, you can run the tests:

```bash
pytest tests/
```

Kernel compilation dominates the test time. Set the SPIO_WORKERS environment
variable equal to the number of processes to use for compilation. The default value is four.

```
export SPIO_WORKERS=`nproc`
pytest tests/
```
