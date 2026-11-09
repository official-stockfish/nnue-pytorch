"""Layer-stack tables: which raw stacks (from --ls-keys) are stored, and which stored stack every
other raw stack uses.

A table is a list `remap` with remap[raw] = the raw stack whose weights `raw` uses; stacks with remap[s] == s are
stored. In configs and checkpoints it is written as "raw:fallback" pairs for the remapped stacks only.
"""

from functools import partial
from pathlib import Path

import numpy as np

KEY_DISTANCE = {
    "pawnhalf": [[0, 1, 2], [1, 0, 1], [2, 1, 0]],
    "manyoffi": [[0, 1], [1, 0]],
}


def table_to_str(remap: list[int]) -> str:
    return ",".join(f"{s}:{f}" for s, f in enumerate(remap) if s != f)


def table_from_str(text: str, raw_count: int) -> list[int]:
    remap = list(range(raw_count))
    for pair in filter(None, text.split(",")):
        s, f = (int(v) for v in pair.split(":"))
        remap[s] = f
    return remap


def fallback_table(
    counts: np.ndarray, keys: list[str], sizes: dict[str, int], min_count: int
) -> list[int]:
    """Route stacks seen fewer than min_count times to the nearest well-seen stack: same queen key and piece-count
    band with the fewest and smallest changes of the other keys, else the nearest piece-count band (ties: the more
    frequent)."""

    def split(s):
        v = {}
        for k in reversed(keys):
            s, v[k] = divmod(s, sizes[k])
        return v

    others = [k for k in keys if k not in ("piececount", "queen")]

    def score(v, t):
        w = split(t)
        band = abs(w.get("piececount", 0) - v.get("piececount", 0))
        band += 0 if w.get("queen") == v.get("queen") else 100
        changed = sum(v[k] != w[k] for k in others)
        dist = sum(KEY_DISTANCE[k][v[k]][w[k]] for k in others)
        return band, changed, dist, -int(counts[t])

    good = [s for s in range(len(counts)) if counts[s] >= min_count]
    assert good, "no layer stack reaches the minimum count"
    remap = list(range(len(counts)))
    for s in range(len(counts)):
        if counts[s] < min_count:
            remap[s] = min(good, key=partial(score, split(s)))
    return remap


def table_from_state_dict(state_dict) -> str | None:
    """The layer-stack table saved in a checkpoint, or None."""
    for k, v in state_dict.items():
        if k.endswith("layer_stacks.stack_fallback"):
            return table_to_str(v.tolist())
    return None


def use_checkpoint_stack_table(nnue_config, checkpoint: dict) -> None:
    """Before building a model to load a checkpoint into: take the checkpoint's layer-stack table."""
    mc = nnue_config.model_config
    if not mc.ls_remap_table:
        table = table_from_state_dict(checkpoint["state_dict"])
        if table is not None:
            mc.ls_remap_table = table


def resolve_data_path(path: str, datasets: list[str]) -> str:
    """A binpack path as given, or relative to the data root of the training binpacks (data/<owner>/<repo>/<file>)."""
    p = Path(path)
    if p.exists():
        return str(p)
    for d in datasets:
        for root in Path(d).resolve().parents:
            if (root / path).exists():
                return str(root / path)
    raise FileNotFoundError(
        f"--ls-remap-data {path} not found (also tried relative to the training data)"
    )


def count_stacks(
    binpack: str, ls_keys: str, raw_count: int, threads: int
) -> np.ndarray:
    """Raw layer-stack counts over a binpack (no position filtering), from the data loader's stack index."""
    from data_loader.config import DataloaderSkipConfig
    from data_loader.stream import (
        create_sparse_batch_stream,
        destroy_sparse_batch,
        destroy_sparse_batch_stream,
        fetch_next_sparse_batch,
    )

    skip = DataloaderSkipConfig(
        filtered=False,
        wld_filtered=False,
        random_fen_skipping=0,
        early_fen_skipping=-1,
        soft_early_fen_skipping=0,
        pc_y0=1.0,
        pc_y1=1.0,
        pc_y2=1.0,
        pc_y3=1.0,
        pc_y4=1.0,
    )
    stream = create_sparse_batch_stream(
        b"K32H3", max(1, threads), [binpack], 65536, False, skip, ls_keys=ls_keys
    )
    counts = np.zeros(raw_count, np.int64)
    while b := fetch_next_sparse_batch(stream):
        counts += np.bincount(
            b.contents.get_tensors("cpu")[-1].numpy(), minlength=raw_count
        )
        destroy_sparse_batch(b)
    destroy_sparse_batch_stream(stream)
    return counts
