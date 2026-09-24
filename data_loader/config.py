import ctypes
from dataclasses import dataclass


@dataclass
class DataloaderSkipConfig:
    filtered: bool = True
    """If disabled, no smart fen skipping will be done."""
    wld_filtered: bool = True
    """If disabled, no WLD-based fen skipping will be done."""
    random_fen_skipping: int = 0
    """Skip a random fraction of positions. 0 = disabled."""
    early_fen_skipping: int = -1
    """Skip positions from the start of the game. -1 = disabled."""
    soft_early_fen_skipping: int = 20
    """Apply soft probability filter up to this ply. <= 0 = disabled. When active parameters `ply_*` control the skip probability for early positions (linear interpolation)."""
    simple_eval_skipping: int = -1
    """Skip positions based on simple eval. -1 = disabled."""
    param_index: int = 0
    """Indexing for parameter scans."""
    pc_y0: float = 0.0
    """Piece count spline y0 parameter (x=0). Parameters `pc_y*` control the desired relative frequency of positions with certain piececounts x. Uses spline interpolation inbetween."""
    pc_y1: float = 0.4
    """Piece count spline y1 parameter (x=8)."""
    pc_y2: float = 1.0
    """Piece count spline y2 parameter (x=16)."""
    pc_y3: float = 1.0
    """Piece count spline y3 parameter (x=24)."""
    pc_y4: float = 0.75
    """Piece count spline y4 parameter (x=32)."""
    ply_x1: float = 0.0
    """Ply soft filter control point x1."""
    ply_y1: float = 0.1
    """Ply soft filter control point y1."""
    ply_x2: float = 6.0
    """Ply soft filter control point x2."""
    ply_y2: float = 0.15
    """Ply soft filter control point y2."""
    ply_x3: float = 10.0
    """Ply soft filter control point x3."""
    ply_y3: float = 0.25
    """Ply soft filter control point y3."""
    ply_x4: float = 18.0
    """Ply soft filter control point x4."""
    ply_y4: float = 0.75
    """Ply soft filter control point y4."""


@dataclass
class DataloaderDDPConfig:
    rank: int = 0
    world_size: int = 1

class CDataloaderSkipConfig(ctypes.Structure):
    _fields_ = [
        ("filtered", ctypes.c_bool),
        ("random_fen_skipping", ctypes.c_int),
        ("wld_filtered", ctypes.c_bool),
        ("early_fen_skipping", ctypes.c_int),
        ("soft_early_fen_skipping", ctypes.c_int),
        ("simple_eval_skipping", ctypes.c_int),
        ("param_index", ctypes.c_int),
        ("pc_y0", ctypes.c_double),
        ("pc_y1", ctypes.c_double),
        ("pc_y2", ctypes.c_double),
        ("pc_y3", ctypes.c_double),
        ("pc_y4", ctypes.c_double),
        ("ply_x1", ctypes.c_double),
        ("ply_y1", ctypes.c_double),
        ("ply_x2", ctypes.c_double),
        ("ply_y2", ctypes.c_double),
        ("ply_x3", ctypes.c_double),
        ("ply_y3", ctypes.c_double),
        ("ply_x4", ctypes.c_double),
        ("ply_y4", ctypes.c_double),
    ]

    def __init__(self, config: DataloaderSkipConfig):
        super().__init__(
            filtered=config.filtered,
            random_fen_skipping=config.random_fen_skipping,
            wld_filtered=config.wld_filtered,
            early_fen_skipping=config.early_fen_skipping,
            soft_early_fen_skipping=config.soft_early_fen_skipping,
            simple_eval_skipping=config.simple_eval_skipping,
            param_index=config.param_index,
            pc_y0=config.pc_y0,
            pc_y1=config.pc_y1,
            pc_y2=config.pc_y2,
            pc_y3=config.pc_y3,
            pc_y4=config.pc_y4,
            ply_x1=config.ply_x1,
            ply_y1=config.ply_y1,
            ply_x2=config.ply_x2,
            ply_y2=config.ply_y2,
            ply_x3=config.ply_x3,
            ply_y3=config.ply_y3,
            ply_x4=config.ply_x4,
            ply_y4=config.ply_y4,
        )


class CDataloaderDDPConfig(ctypes.Structure):
    _fields_ = [
        ("rank", ctypes.c_int),
        ("world_size", ctypes.c_int),
    ]

    def __init__(self, config: DataloaderDDPConfig):
        super().__init__(
            rank=config.rank,
            world_size=config.world_size,
        )


@dataclass
class DataloaderHllConfig:
    """Configuration for HyperLogLog unique position counting.

    initial_hll: serialized HLL state for restart from checkpoint (bytes or None)
    initial_total: total position count (after filtering) at restart
    initial_preskip: preskip position count (before filtering) at restart
    """
    initial_hll: bytes | None = None
    initial_total: int = 0
    initial_preskip: int = 0


class CDataloaderHllConfig(ctypes.Structure):
    _fields_ = [
        ("initial_hll", ctypes.c_void_p),
        ("initial_hll_size", ctypes.c_size_t),
        ("initial_total", ctypes.c_uint64),
        ("initial_preskip", ctypes.c_uint64),
    ]

    def __init__(self, config: DataloaderHllConfig):
        self._hll_buf = None
        ptr = None
        if config.initial_hll is not None:
            self._hll_buf = (ctypes.c_uint8 * len(config.initial_hll)).from_buffer_copy(
                config.initial_hll
            )
            ptr = ctypes.cast(self._hll_buf, ctypes.c_void_p)
        super().__init__(
            initial_hll=ptr,
            initial_hll_size=len(config.initial_hll) if config.initial_hll is not None else 0,
            initial_total=config.initial_total,
            initial_preskip=config.initial_preskip,
        )
