# Developing and testing with Docker

Run commands from the nnue-pytorch repository root. `run_docker.sh` builds the
selected image, mounts this checkout at `/workspace/nnue-pytorch`, mounts your
data directory at `/data`, compiles the native data loader through
`setup_script.sh`, and starts a shell or executes a command. Source edits on
the host are immediately visible in the container, and logs and checkpoints
written to either mount persist on the host.

## Start a development shell

```bash
./run_docker.sh NVIDIA /absolute/path/to/data
# Or use AMD for ROCm, or CPU for CPU testing.
```

Inside the container:

```bash
python train.py --help
python serialize.py --help
pytest
```

Use the trainer's help for the current options. For complete recipe-based
training and engine testing, see [training with Nettest](training.md).

## Run checks without a shell

These commands match the repository's CPU CI workflow. The first builds the
image and compiles the loader; subsequent commands can reuse the image:

```bash
./run_docker.sh CPU . --non-interactive --exec pytest
./run_docker.sh CPU . --skip-build --non-interactive --exec ruff check .
./run_docker.sh CPU . --skip-build --non-interactive --exec \
    python -u tests/test_training_pipeline_run.py --device cpu \
    --test-dir /data/logs/training/runs/local_pipeline_cpu
```

The pipeline test uses `.pgo/small.binpack` and exercises training, checkpoint
contents, restarting from a model and checkpoint, serialization, and
feature-transformer optimization. Choose a fresh output directory for each run.
If it already exists, the script asks whether to delete it; CI uses `-y` to
replace its disposable test output automatically.

For a focused check, replace `pytest` with, for example, `pytest
tests/test_lambda_scheduler.py`. CPU checks do not exercise CUDA kernels; run
relevant device tests with an NVIDIA or AMD container when changing
accelerator-specific code. The pipeline test accepts `--device 0` for the first
CUDA/ROCm GPU.

`--skip-build` reuses an existing image; rebuild after changing Dockerfiles or
requirements. `--skip-setup` skips native loader compilation and should only be
used after setup succeeds and while loader sources and build settings are
unchanged. Keep `--exec` last: every following argument belongs to the command.
`--exec` does not itself disable the interactive shell, so use
`--non-interactive` for automated checks and command exit statuses.

## Check Stockfish evaluation

After the pipeline test, run `cross_check_eval.py` with a Stockfish binary
whose network format and architecture match the trainer. For example, place it
at `Stockfish/src/stockfish` in this checkout and run:

```bash
./run_docker.sh CPU . --skip-build --skip-setup --non-interactive --exec \
    python -u cross_check_eval.py \
    --net /data/logs/training/runs/local_pipeline_cpu/training_logs/version_2/checkpoints/last.nnue \
    --checkpoint /data/logs/training/runs/local_pipeline_cpu/training_logs/version_2/checkpoints/last.ckpt \
    --data .pgo/small.binpack --engine ./Stockfish/src/stockfish --device cpu
```

See `.github/workflows/cpu-testrun.yml` for the compatible engine revision used
by CI. Nettest recipes also support cross-checks and fastchess matches for full
network testing.

## Logging

Training writes TensorBoard events under the chosen `--default-root-dir`. To
inspect logs from the host, with TensorBoard installed:

```bash
tensorboard --logdir=logs
```

Open http://localhost:6006/. If training outputs are in your data mount, point
TensorBoard at that host directory. `run_docker.sh` does not publish container
ports.

