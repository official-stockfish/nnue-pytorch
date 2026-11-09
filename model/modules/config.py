from dataclasses import dataclass
from typing import Annotated

import tyro


# 3 layer fully connected network
@dataclass(kw_only=True)
class LayerStacksConfig:
    L1: Annotated[int, tyro.conf.arg(name="l1")] = 1024
    """Size of first hidden layer."""
    L2: Annotated[int, tyro.conf.arg(name="l2")] = 32
    """Size of second hidden layer."""
    L3: Annotated[int, tyro.conf.arg(name="l3")] = 32
    """Size of third hidden layer."""
    ls_factors: Annotated[str, tyro.conf.arg(name="ls-factors")] = (
        "piececount+queen:pawnhalf:manyoffi"
    )
    """Factorized layer stacks: ':'-separated factors, each a --ls-keys key or keys joined by '+' (e.g.
    "piececount+queen"), shared by the stacks with the same values of those keys, added to every stack layer during
    training and folded into the stacks at export (the exported net is unchanged in format). Empty = off."""
    ls_keys: Annotated[str, tyro.conf.arg(name="ls-keys")] = (
        "piececount:queen:pawnhalf:manyoffi"
    )
    """Layer-stack keys, ':'-separated, in mixed radix with the last key varying fastest: "piececount" (piece-count
    band, (pieces - 1) / 4: 8), "queen" (2 if the side to move has a queen, plus 1 if the opponent has one: 4),
    "pawnhalf" (pawns on both board halves a-d / e-h, on one half only, or none: 3), "manyoffi" (1 if some side has
    two or more officers N, B, R, Q: 2). Stacks = the product of the key sizes. The data loader computes the stack
    index."""
    ls_remap_data: Annotated[str, tyro.conf.arg(name="ls-remap-data")] = ""
    """Binpack to build the layer-stack table from at the start of training: every stack is counted over it (no
    position filtering), and stacks seen fewer than --ls-remap-min times use the nearest well-seen stack instead of
    being stored. A relative path is also looked up under the data root of the training binpacks. The net carries
    the table, with which the engine maps raw stacks to stored ones; it is kept in the checkpoint too, so later
    steps (resume, conversion, crosscheck) reuse it. Without a table every stack is stored."""
    ls_remap_min: Annotated[int, tyro.conf.arg(name="ls-remap-min")] = 50
    """Minimum count in --ls-remap-data for a stack of its own."""
    ls_remap_table: Annotated[str, tyro.conf.arg(name="ls-remap-table")] = ""
    """The layer-stack table itself, as "stack:fallback" pairs (normally filled in from --ls-remap-data, a
    checkpoint or a net)."""
