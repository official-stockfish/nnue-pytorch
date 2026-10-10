# Training with Nettest

Use [Nettest](https://github.com/vondele/nettest) for reproducible network
training and testing. A YAML recipe defines dataset downloads, trainer and
engine revisions, training stages and restarts, checkpoint conversion,
feature-transformer optimization, and engine testing. Nettest caches downloaded
data and completed steps so later stages can be iterated on without repeating
the entire run.

Start from
[testing.yaml](https://github.com/vondele/nettest/blob/master/testing.yaml) for
experimentation or
[threats.yaml](https://github.com/vondele/nettest/blob/master/threats.yaml) for
a full training recipe. Review the recipe before running: even the testing
recipe downloads real datasets and trains networks; it is not a small unit
test.

## Run a recipe locally

The following uses Nettest's NVIDIA container. Install Docker, an NVIDIA
driver, and the NVIDIA Container Toolkit first. Create host directories for the
data cache, scratch files, and CI artifacts; adjust the paths below for your
machine. Put the data cache on fast storage.

```bash
git clone https://github.com/vondele/nettest.git
cd nettest
mkdir -p /mnt/ssd/data /mnt/ssd/scratch /mnt/ssd/cidir
docker build -t nettest.docker -f ci/docker/Dockerfile.NVIDIA .
docker run --rm -u "$(id -u):$(id -g)" -it \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --gpus all --cap-add=sys_nice \
    -v /mnt/ssd/data:/workspace/data \
    -v /mnt/ssd/scratch:/workspace/scratch \
    -v /mnt/ssd/cidir:/workspace/cidir \
    -v "$PWD":/workspace/nettest \
    nettest.docker python -m nettest.execute_recipe \
    --executor local --recipe nettest/testing.yaml
```

The Nettest container runs from `/workspace`. Mounting the checkout at
`/workspace/nettest` lets you edit recipes on the host. Add `--environment
nettest/environments/local.yaml` to use a resource allocation configuration;
consult the [Nettest README](https://github.com/vondele/nettest#readme) for
remote execution and CI workflows.

GPU training requires at least 16 GB of system RAM and 8 GB of GPU VRAM.
8 GB of VRAM is tight; more gives room for larger batches and networks.

The complete set of binpacks listed in Nettest's `threats.yaml` requires
800+ GB of storage, with additional space needed for checkpoints, caches, and
other outputs. This is the full recipe's dataset requirement. Training can
also use a single small binpack, which needs much less disk space but produces
weaker networks. Adjust the recipe's dataset selection to suit your storage
and training goals. Data caching benefits from fast storage, and engine
testing consumes additional CPU time and memory.

## Select the trainer and engine revisions

The recipe's `trainer` mapping selects the GitHub owner and commit of `nnue-pytorch`:

```yaml
trainer: &trainer
  owner: YOUR_GITHUB_OWNER
  sha: YOUR_TRAINER_COMMIT_SHA
```

Training steps refer to this mapping; keep the cross-check trainer consistent
with the trained model. Pin compatible Stockfish revisions in the recipe's
reference and testing engine configuration, and keep feature sets, layer sizes,
and serialization options consistent across training, conversion, and
cross-checks.

Nettest fetches the pinned trainer from GitHub into its scratch cache. Mounting
a local nnue-pytorch checkout does not make a recipe use uncommitted changes.
Develop and test local edits with `run_docker.sh` as described in
[development.md](development.md); for a subsequent recipe run, select an
available trainer commit in your fork. Nettest fetches dependencies over HTTPS
by default and can use SSH when GitHub SSH authentication is available.

## Inspect results

Nettest prints the generated network paths and test results. Checkpoints and
intermediate outputs are kept under the mounted scratch directory, and exported
networks are copied to the mounted CI artifact directory. Inspect the logs and
engine results before comparing recipes; training loss alone does not establish
playing strength.

