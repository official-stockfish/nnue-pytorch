import subprocess
import re
import tyro

import chess

import data_loader
import model as M

from dataclasses import dataclass, field
from typing import Optional, Literal, Annotated
from tyro.conf import OmitArgPrefixes


@dataclass(frozen=True)
class CrossCheckConfig:
    # Flags and Options
    engine: str
    """Path to the engine binary to use for evaluation."""

    data: str
    """Path to the .bin or .binpack dataset to use for evaluation."""

    net: str
    """Path to the .nnue net to evaluate."""

    checkpoint: Optional[str] = None
    """Optional checkpoint (used instead of nnue for local eval)."""

    device: Literal["cuda", "mps", "cpu"] = "cuda"
    """Device for the NNUE model."""

    net_type: Literal["big", "small"] = "big"
    """Which net to evaluate: 'big' uses EvalFile, 'small' uses EvalFileSmall"""

    count: int = 2**10
    """Number of positions to process."""


@dataclass(frozen=True)
class CliConfig:
    cross_check_config: OmitArgPrefixes[CrossCheckConfig]
    nnue_lightning_config: OmitArgPrefixes[M.NNUELightningConfig]


def read_model(
    nnue_path,
    config: M.NNUELightningConfig,
    quantize_config: M.QuantizationConfig,
):
    with open(nnue_path, "rb") as f:
        reader = M.NNUEReader(f, config.features, config.model_config, quantize_config)
        return reader.model


def make_fen_batch_provider(data_path, batch_size):
    return data_loader.FenBatchProvider(
        data_path,
        True,
        1,
        batch_size,
        data_loader.DataloaderSkipConfig(
            random_fen_skipping=10,
        ),
    )


def eval_model_batch(model, batch: data_loader.SparseBatchPtr, device: str):
    (
        us,
        them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        outcome,
        score,
        psqt_indices,
        layer_stack_indices,
    ) = batch.contents.get_tensors(device)

    evals = [
        v.item()
        for v in model.forward(
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            psqt_indices,
            layer_stack_indices,
        ) * model.quantization.nnue2score
    ]
    return evals


re_nnue_eval_big = re.compile(
    r"\(Big net\) NNUE evaluation\s+([-+]?\d+)\s+\(side to move, internal units\)"
)
re_nnue_eval_small = re.compile(
    r"\(Small net\) NNUE evaluation\s+([-+]?\d+)\s+\(side to move, internal units\)"
)


def compute_basic_eval_stats(evals):
    min_engine_eval = min(evals)
    max_engine_eval = max(evals)
    avg_engine_eval = sum(evals) / len(evals)
    avg_abs_engine_eval = sum(abs(v) for v in evals) / len(evals)

    return min_engine_eval, max_engine_eval, avg_engine_eval, avg_abs_engine_eval


def compute_correlation(engine_evals, model_evals):
    if len(engine_evals) != len(model_evals):
        raise Exception(
            "number of engine evals doesn't match the number of model evals. Got {} engine evals and {} model evals.".format(
                len(engine_evals), len(model_evals)
            )
        )

    min_engine_eval, max_engine_eval, avg_engine_eval, avg_abs_engine_eval = (
        compute_basic_eval_stats(engine_evals)
    )
    min_model_eval, max_model_eval, avg_model_eval, avg_abs_model_eval = (
        compute_basic_eval_stats(model_evals)
    )

    print("Min engine/model eval: {} / {}".format(min_engine_eval, min_model_eval))
    print("Max engine/model eval: {} / {}".format(max_engine_eval, max_model_eval))
    print("Avg engine/model eval: {} / {}".format(avg_engine_eval, avg_model_eval))
    print(
        "Avg abs engine/model eval: {} / {}".format(
            avg_abs_engine_eval, avg_abs_model_eval
        )
    )

    relative_model_error = sum(
        abs(model - engine) / (abs(engine) + 0.001)
        for model, engine in zip(model_evals, engine_evals)
    ) / len(engine_evals)
    relative_engine_error = sum(
        abs(model - engine) / (abs(model) + 0.001)
        for model, engine in zip(model_evals, engine_evals)
    ) / len(engine_evals)
    min_diff = min(
        abs(model - engine) for model, engine in zip(model_evals, engine_evals)
    )
    max_diff = max(
        abs(model - engine) for model, engine in zip(model_evals, engine_evals)
    )
    print("Relative engine error: {}".format(relative_engine_error))
    print("Relative model error: {}".format(relative_model_error))
    print(
        "Avg abs difference: {}".format(
            sum(abs(model - engine) for model, engine in zip(model_evals, engine_evals))
            / len(engine_evals)
        )
    )
    print("Min difference: {}".format(min_diff))
    print("Max difference: {}".format(max_diff))


def eval_engine_batch(engine_path, net_path, fens, net_type="big"):
    if not fens:
        return []
    engine = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    option_name = "EvalFile" if net_type == "big" else "EvalFileSmall"
    parts = ["uci", "setoption name {} value {}".format(option_name, net_path)]
    for fen in fens:
        parts.append("position fen {}".format(fen))
        parts.append("eval")
    parts.append("quit")
    query = "\n".join(parts)
    out = engine.communicate(input=query)[0]
    pattern = re_nnue_eval_big if net_type == "big" else re_nnue_eval_small
    evals = re.findall(pattern, out)
    if len(evals) != len(fens):
        raise Exception(
            "number of evals returned by the engine doesn't match the number of fens. Got {} evals and {} fens. Output was:\n{}".format(
                len(evals), len(fens), out
            )
        )
    return [int(v) for v in evals]


def filter_fens(fens):
    # We don't want fens where a king is in check, as these cannot be evaluated by the engine.
    filtered_fens = []
    for fen in fens:
        board = chess.Board(fen=fen)
        if not board.is_check():
            filtered_fens.append(fen)
    return filtered_fens


def main():
    args = tyro.cli(CliConfig)

    cross_check_config = args.cross_check_config
    nnue_lightning_config = args.nnue_lightning_config

    batch_size = 1024

    if cross_check_config.checkpoint:
        model = M.NNUE.load_from_checkpoint(
            cross_check_config.checkpoint,
            config=nnue_lightning_config,
            quantize_config=M.QuantizationConfig(),
        )
    else:
        model = read_model(
            cross_check_config.net,
            config=nnue_lightning_config,
            quantize_config=M.QuantizationConfig(),
        )
    model.to(cross_check_config.device)
    model.eval()
    # --checkpoint - returns a Lightning NNUE wrapping a NNUEModel
    # --net - returns the NNUEModel directly
    inner_model = model.model if isinstance(model, M.NNUE) else model
    input_feature_name = inner_model.input_feature_name
    fen_batch_provider = make_fen_batch_provider(cross_check_config.data, batch_size)

    model_evals = []
    engine_evals = []

    done = 0
    print("Processed {} positions.".format(done))
    while done < cross_check_config.count:
        fens = filter_fens(next(fen_batch_provider))

        b = data_loader.get_sparse_batch_from_fens(
            input_feature_name, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens)
        )
        model_evals += eval_model_batch(inner_model, b, cross_check_config.device)
        data_loader.destroy_sparse_batch(b)

        engine_evals += eval_engine_batch(
            cross_check_config.engine,
            cross_check_config.net,
            fens,
            cross_check_config.net_type
        )

        done += len(fens)
        print("Processed {} positions.".format(done))

    compute_correlation(engine_evals, model_evals)


if __name__ == "__main__":
    main()
