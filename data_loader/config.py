import ctypes
from dataclasses import dataclass


@dataclass
class DataloaderSkipConfig:
    filtered: bool = False
    random_fen_skipping: int = 0
    wld_filtered: bool = False
    early_fen_skipping: int = -1
    simple_eval_skipping: int = -1
    param_index: int = 0
    pc_y1: float = 1.0
    pc_y2: float = 2.0
    pc_y3: float = 1.0


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
