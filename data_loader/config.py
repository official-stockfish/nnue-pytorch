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
    """Skip positions from the start of the game `pos.ply <= early_fen_skipping`. -1 = disabled."""
    early_fen_skipping_ply_factor: float = 0.0
    """Multiply probabilitys of early fens by `early_fen_skipping_ply_factor^(early_fen_skipping - pos.ply + 1)`. 0.0 = skip all early fens."""
    simple_eval_skipping: int = -1
    """Skip positions based on simple eval. -1 = disabled."""
    param_index: int = 0
    """Indexing for parameter scans."""
    pc_y1: float = 1.0
    """Piecewise quadratic interpolation y1 parameter."""
    pc_y2: float = 2.0
    """Piecewise quadratic interpolation y2 parameter."""
    pc_y3: float = 1.0
    """Piecewise quadratic interpolation y3 parameter."""

    @staticmethod
    def get_dataloader_skip_config_from_args(args) -> "DataloaderSkipConfig":
        return DataloaderSkipConfig(
            filtered=args.filtered,
            random_fen_skipping=args.random_fen_skipping,
            wld_filtered=args.wld_filtered,
            early_fen_skipping=args.early_fen_skipping,
            early_fen_skipping_ply_factor=args.early_fen_skipping_ply_factor,
            simple_eval_skipping=args.simple_eval_skipping,
            param_index=args.param_index,
            pc_y1=args.pc_y1,
            pc_y2=args.pc_y2,
            pc_y3=args.pc_y3,
        )


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
        ("simple_eval_skipping", ctypes.c_int),
        ("param_index", ctypes.c_int),
        ("early_fen_skipping_ply_factor", ctypes.c_double),
        ("pc_y1", ctypes.c_double),
        ("pc_y2", ctypes.c_double),
        ("pc_y3", ctypes.c_double),
    ]

    def __init__(self, config: DataloaderSkipConfig):
        super().__init__(
            filtered=config.filtered,
            random_fen_skipping=config.random_fen_skipping,
            wld_filtered=config.wld_filtered,
            early_fen_skipping=config.early_fen_skipping,
            simple_eval_skipping=config.simple_eval_skipping,
            param_index=config.param_index,
            pc_y1=config.pc_y1,
            pc_y2=config.pc_y2,
            pc_y3=config.pc_y3,
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
