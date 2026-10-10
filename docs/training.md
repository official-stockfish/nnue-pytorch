# Network training

For reproducible training and engine testing, follow the upstream guides:

- [Run a training workflow locally](https://github.com/vondele/nettest#local-execution).
- [Configure training stages and restarts](https://github.com/vondele/nettest#recipe-description).
- [Select trainer revisions, engines, and datasets](https://github.com/vondele/nettest#external-tools-and-data).
- [Run a workflow in CI](https://github.com/vondele/nettest#execution-in-the-ci-environment).

## Hardware and datasets

GPU training requires at least 16 GB of system RAM and 8 GB of GPU VRAM.
8 GB of VRAM is tight; more gives room for larger batches and networks.

The complete dataset set in the
[threats training recipe](https://github.com/vondele/nettest/blob/master/threats.yaml)
requires 800+ GB of storage, plus space for checkpoints, caches, and outputs.
Training can also use a single small binpack, which requires much less storage
but produces weaker networks. Use fast storage for data loading and allow
additional CPU and memory resources for engine testing.

## Local trainer development

Use `run_docker.sh` to develop and test this checkout. See
[Developing and testing with Docker](development.md) for checks, the training
pipeline test, evaluation cross-checks, and logging.
