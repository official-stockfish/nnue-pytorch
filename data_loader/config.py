import argparse
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

    @staticmethod
    def add_dataloader_skip_args(parser):
        parser.add_argument(
            "--no-smart-fen-skipping",
            action="store_true",
            dest="no_smart_fen_skipping",
            help="If used then no smart fen skipping will be done. By default smart fen skipping is done.",
        )

        parser.add_argument(
            "--random-fen-skipping",
            default=3,
            type=int,
            dest="random_fen_skipping",
            help="skip fens randomly on average random_fen_skipping before using one.",
        )

        parser.add_argument(
            "--no-wld-fen-skipping",
            action="store_true",
            dest="no_wld_fen_skipping",
            help="If used then no wld fen skipping will be done. By default wld fen skipping is done.",
        )

        parser.add_argument(
            "--early-fen-skipping",
            type=int,
            default=-1,
            dest="early_fen_skipping",
            help="Skip n plies from the start.",
        )

        parser.add_argument(
            "--simple-eval-skipping",
            type=int,
            default=-1,
            dest="simple_eval_skipping",
            help="Skip positions that have abs(simple_eval(pos)) < n",
        )

        parser.add_argument(
            "--param-index",
            type=int,
            default=0,
            dest="param_index",
            help="Indexing for parameter scans.",
        )

        parser.add_argument(
            "--pc-y1",
            type=float,
            default=1.0,
            dest="pc_y1",
            help="piece count parameter y1 (default=1.0)",
        )
        parser.add_argument(
            "--pc-y2",
            type=float,
            default=2.0,
            dest="pc_y2",
            help="piece count parameter y2 (default=2.0)",
        )
        parser.add_argument(
            "--pc-y3",
            type=float,
            default=1.0,
            dest="pc_y3",
            help="piece count parameter y3 (default=1.0)",
        )

    @staticmethod
    def get_dataloader_skip_config_from_args(args) -> "DataloaderSkipConfig":
        return DataloaderSkipConfig(
            filtered=not args.no_smart_fen_skipping,
            random_fen_skipping=args.random_fen_skipping,
            wld_filtered=not args.no_wld_fen_skipping,
            early_fen_skipping=args.early_fen_skipping,
            simple_eval_skipping=args.simple_eval_skipping,
            param_index=args.param_index,
            pc_y1=args.pc_y1,
            pc_y2=args.pc_y2,
            pc_y3=args.pc_y3,
        )


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


def make_action(class_name, field_name):
    class SetConfig(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(class_name, field_name, values)

    return SetConfig
