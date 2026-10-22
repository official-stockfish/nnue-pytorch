"""Multi-process DDP tests, run on the CPU with the gloo backend.

The trainer only wraps the model in DistributedDataParallel when more than one
process takes part, which no other test does. These tests start two.
"""

import glob
import hashlib
import os
import socket
import subprocess
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from model import NNUE, NNUEConfig
from trainer.engine import SimpleTrainer, init_distributed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORLD_SIZE = 2
# A deadlock between the ranks would otherwise block the test run forever.
TIMEOUT_SECONDS = 600

pytestmark = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed with the gloo backend is not available",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kill_process_tree(pid: int) -> None:
    # torchrun starts every worker in a session of its own, so killing the
    # process group of torchrun would leave the workers running
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = parent.children(recursive=True) + [parent]
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=30)


def _data_loader_built() -> bool:
    patterns = (
        os.path.join(REPO_ROOT, "build", "**", "*training_data_loader.so"),
        os.path.join(
            REPO_ROOT, "data_loader", "cpp", "build", "**", "*training_data_loader.so"
        ),
    )
    return any(glob.glob(p, recursive=True) for p in patterns)


def _synthetic_batches(seed, num_batches, batch_size, num_inputs, max_active):
    """Batches in the layout of the native data loader, with random contents."""
    gen = torch.Generator().manual_seed(seed)
    num_active = min(max_active, 24)
    batches = []
    for _ in range(num_batches):
        us = torch.randint(0, 2, (batch_size, 1), generator=gen).float()
        indices = []
        for _ in range(2):
            idx = torch.full((batch_size, max_active), -1, dtype=torch.int32)
            idx[:, :num_active] = torch.randint(
                0,
                num_inputs,
                (batch_size, num_active),
                generator=gen,
                dtype=torch.int32,
            )
            indices.append(idx)
        outcome = torch.randint(0, 3, (batch_size, 1), generator=gen).float() / 2
        score = torch.randn(batch_size, 1, generator=gen) * 300
        c_us = torch.randint(0, 68640, (batch_size,), generator=gen)
        c_them = torch.randint(0, 68640, (batch_size,), generator=gen)
        batches.append((us, 1.0 - us, *indices, outcome, score, c_us, c_them))
    return batches


def _params_digest(module: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for p in module.parameters():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _gather(obj):
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out


def _sync_worker(rank, world_size, port, root_dir, num_batches):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torch.set_num_threads(1)
    rank, world_size, local_rank = init_distributed("gloo", device_type="cpu")
    try:
        torch.manual_seed(0)
        config = NNUEConfig()
        config.model_config.L1 = 128
        # RangerLite hands back its lookahead weights at the end of an epoch,
        # which equal the initial ones until its first merge (5 steps)
        config.optimizer_config.optimizer_name = "adamw"
        nnue = NNUE(config, max_epoch=1, num_batches_per_epoch=num_batches)
        optimizer, schedulers = nnue.configure_optimizers()
        if isinstance(optimizer, list):
            (optimizer,) = optimizer

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            trainer = SimpleTrainer(
                model=nnue,
                optimizer=optimizer,
                schedulers=schedulers,
                max_epochs=1,
                check_val_every_n_epoch=1,
                gradient_clip_val=2.0,
                log_every_n_steps=1,
                default_root_dir=root_dir,
                callbacks=[],
                logger=None,
                device="cpu",
                rank=rank,
                world_size=world_size,
                local_rank=local_rank,
            )
        assert isinstance(trainer._ddp_model, DDP)
        # torch 2.13 deprecated broadcast_buffers in favour of forward_sync_buffers
        deprecations = [
            str(w.message) for w in caught if "broadcast_buffers" in str(w.message)
        ]
        assert not deprecations, deprecations

        initial = _params_digest(nnue.model)
        ft = nnue.model.input
        # Every rank trains on its own data, as with the sharded data loader
        batches = _synthetic_batches(
            1000 + rank, num_batches, 64, ft.NUM_INPUTS, ft.MAX_ACTIVE_FEATURES
        )
        scores = _gather(torch.cat([b[5] for b in batches]))
        assert not torch.equal(scores[0], scores[1]), "ranks got the same data"

        trainer.fit(batches)

        assert trainer.global_step == num_batches
        digests = _gather(_params_digest(nnue.model))
        assert digests[rank] != initial, "training did not change the weights"
        assert len(set(digests)) == 1, (
            f"weights differ between ranks after training: {digests}"
        )
        losses = _gather(trainer.callback_metrics["train_loss_epoch"])
        assert len(set(losses)) == 1, f"epoch loss differs between ranks: {losses}"
    finally:
        dist.destroy_process_group()


def test_ddp_ranks_stay_in_sync(tmp_path):
    """Ranks that train on different data must end up with identical weights.

    That only holds if DDP averages the gradients, so this also fails when the
    gradient synchronization breaks, not only when a rank errors out.
    """
    ctx = mp.start_processes(
        _sync_worker,
        args=(WORLD_SIZE, _free_port(), str(tmp_path), 3),
        nprocs=WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + TIMEOUT_SECONDS
    while not ctx.join(timeout=5):
        if time.monotonic() > deadline:
            for p in ctx.processes:
                p.kill()
            pytest.fail(f"DDP workers did not finish within {TIMEOUT_SECONDS} s")


@pytest.mark.skipif(not _data_loader_built(), reason="data loader library not built")
def test_train_script_two_ranks(tmp_path):
    """Run train.py under torchrun with two CPU ranks, including validation."""
    with open(os.path.join(REPO_ROOT, ".pgo", "small.binpack"), "rb") as f:
        chunk = f.read()
    # The loader gives each rank every WORLD_SIZE-th chunk of a file, and
    # .pgo/small.binpack is a single chunk. Repeat it so that no rank is
    # left without data.
    data = tmp_path / "train.binpack"
    data.write_bytes(chunk * WORLD_SIZE)
    root_dir = tmp_path / "run"

    batch_size, epoch_size, max_epochs = 256, 2048, 2
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={WORLD_SIZE}",
        "train.py",
        str(data),
        "--accelerator=cpu",
        "--l1=128",
        f"--batch-size={batch_size}",
        f"--epoch-size={epoch_size}",
        "--validation-size=512",
        f"--max_epochs={max_epochs}",
        "--num_workers=1",
        "--threads=1",
        f"--default_root_dir={root_dir}",
    ]
    log_path = tmp_path / "train.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            proc.wait(timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc.pid)
            pytest.fail(
                f"two-rank training did not finish within {TIMEOUT_SECONDS} s:\n"
                + log_path.read_text()[-4000:]
            )
    output = log_path.read_text()

    assert proc.returncode == 0, output[-4000:]
    # Only the master process prints the optimizer configuration
    assert output.count("[OptimizerConfig] Using") == 1, output[-4000:]
    assert (root_dir / "training_finished").exists()

    checkpoints = glob.glob(
        str(root_dir / "training_logs" / "*" / "checkpoints" / "last.ckpt")
    )
    assert len(checkpoints) == 1, checkpoints
    ckpt = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    # --batch-size is the global batch, which train.py splits across the ranks
    steps_per_epoch = epoch_size // batch_size
    assert ckpt["global_step"] == max_epochs * steps_per_epoch
