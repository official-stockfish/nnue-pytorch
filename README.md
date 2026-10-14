# NNUE PyTorch

## Setup

### Docker

Use Docker with the PyTorch container. This eliminates the need for local
Python environment setup and C++ compilation. An alternative is Conda or
Micromamba if Docker is not available, or if you want native Apple Silicon MPS
acceleration. While Docker is available on Apple for CPU-Only testing, it does
not support native MPS acceleration.

#### Prerequisites

For AMD Users:
- Docker
- Up-to-date ROCm driver

For NVIDIA Users:
- Docker
- Up-to-date NVIDIA driver
- NVIDIA Container Toolkit

For Apple Silicon Users (MPS):
- Native MPS acceleration does not work with Docker.
- See below for recommended setup or test with CPU only.

For CPU only (for testing purposes):
- Docker

For driver requirements, check [Running ROCm Docker containers
(AMD)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html)
or the [PyTorch container release notes
(Nvidia)](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-04.html#rel-25-04).

The container includes CUDA 12.x / ROCm 6.4.3 and all required dependencies.
Your local CUDA/ROCm toolkit version doesn't matter.

### Running the container

Use the provided script to build and start the container:

```
./run_docker.sh
```

You'll be prompted to select the target GPU vendor (or CPU only for testing)
and the path to your data directory, which will be mounted into the container.
Once inside the container, you can run training commands directly. Also
supports non-interactive workflows if all necessary arguments are given through
the CLI.

_Building the container will take it's time and disk space (~30-60GB)_

### Setup for Apple Silicon
- Up-to-date Package manager conda or micromamba is recommended.
- Create environment and activate (works the same with micromamba):
    ```
    conda create -n nnue_pytorch -c pytorch -c conda-forge \
        python=3.12 \
        pytorch \
        torchvision \
        torchaudio \
        compilers \
        llvm-openmp \
        jpeg \
        libjpeg-turbo \
        cmake \
        make
    conda activate nnue_pytorch
    ```
- Afterwards run:
    ```
    pip install --no-cache-dir -r requirements.txt
    ./setup_script.sh
    ```

## Network training and testing

For reproducible training and engine testing, follow the
[local training workflow guide](https://github.com/vondele/nettest#local-execution).
See [Network training](docs/training.md) for hardware requirements and links to
workflow configuration and CI instructions.

GPU training needs at least 16 GB system RAM and 8 GB VRAM, with 8 GB VRAM
being tight. The full dataset set in [the threats training recipe](https://github.com/vondele/nettest/blob/master/threats.yaml) requires 800+ GB
of storage; a single small binpack can be used with much less storage, but
produces weaker networks.

For trainer development, run local code and checks with `run_docker.sh`. See
[Developing and testing with Docker](docs/development.md) for CPU and GPU
workflows, the training pipeline test, Stockfish evaluation cross-checks, and
logging. Direct `train.py` usage is documented in the
[wiki](https://github.com/official-stockfish/nnue-pytorch/wiki/Basic-training-procedure-(train.py)).

## Thanks

* Sopel - for the amazing fast sparse data loader
* connormcmonigle - https://github.com/connormcmonigle/seer-nnue, and loss function advice.
* syzygy - http://www.talkchess.com/forum3/viewtopic.php?f=7&t=75506
* https://github.com/DanielUranga/TensorFlowNNUE
* https://hxim.github.io/Stockfish-Evaluation-Guide/
* dkappe - Suggesting ranger (https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer)
