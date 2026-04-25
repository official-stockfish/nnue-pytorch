import subprocess
import re
import math

import tyro

import chess

import data_loader
import model as M

from dataclasses import dataclass
from typing import Optional, Literal
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
        )
        * model.quantization.nnue2score
    ]
    return evals


re_nnue_eval_big = re.compile(
    r"\(Big net\) NNUE evaluation\s+([-+]?\d+)\s+\(side to move, internal units\)"
)
re_nnue_eval_small = re.compile(
    r"\(Small net\) NNUE evaluation\s+([-+]?\d+)\s+\(side to move, internal units\)"
)


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


def calculate_qf(score, offset, scaling):
    q = (score - offset) / scaling
    qm = (-score - offset) / scaling
    return 0.5 * (1.0 + sigmoid(q) - sigmoid(qm))


def get_percentile(data, percentile):
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = (len(sorted_data) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    weight = index - lower
    return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


def compute_basic_eval_stats(evals):
    if not evals:
        return 0, 0, 0, 0
    min_val = min(evals)
    max_val = max(evals)
    avg_val = sum(evals) / len(evals)
    avg_abs_val = sum(abs(v) for v in evals) / len(evals)
    return min_val, max_val, avg_val, avg_abs_val


def compute_correlation(engine_evals, model_evals, fens):
    if len(engine_evals) != len(model_evals):
        raise Exception(f"Mismatch: {len(engine_evals)} vs {len(model_evals)}")

    # Trainer parameters from your configuration
    IN_OFFSET = 280.0
    IN_SCALING = 353.0

    data = []
    abs_errors = []
    q_errors = []

    for e, m, f in zip(engine_evals, model_evals, fens):
        ae = abs(m - e)
        q_sf = calculate_qf(e, IN_OFFSET, IN_SCALING)
        q_py = calculate_qf(m, IN_OFFSET, IN_SCALING)
        qe = abs(q_py - q_sf)

        abs_errors.append(ae)
        q_errors.append(qe)
        data.append(
            {
                "fen": f,
                "sf": e,
                "py": m,
                "abs_err": ae,
                "rel_err": ae / (abs(e) if e != 0 else 1.0),
                "q_sf": q_sf,
                "q_py": q_py,
                "q_err": qe,
            }
        )

    # R^2 Calculation for Scores
    mean_sf = sum(engine_evals) / len(engine_evals)
    ss_res = sum((d["sf"] - d["py"]) ** 2 for d in data)
    ss_tot = sum((d["sf"] - mean_sf) ** 2 for d in data)
    r_squared_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # R^2 Calculation for Q (Expected Score)
    mean_q_sf = sum(d["q_sf"] for d in data) / len(data)
    ss_res_q = sum((d["q_sf"] - d["q_py"]) ** 2 for d in data)
    ss_tot_q = sum((d["q_sf"] - mean_q_sf) ** 2 for d in data)
    r_squared_q = 1 - (ss_res_q / ss_tot_q) if ss_tot_q != 0 else 0.0

    # Summary Stats
    en_min, en_max, en_avg, en_abs_avg = compute_basic_eval_stats(engine_evals)

    W = 115
    print("\n" + "=" * W)
    print(f"{'CROSS-CHECK EVALUATION SUMMARY':^{W}}")
    print("=" * W)
    print(
        f"{'Metric':<30} | {'Score (Internal Units)':>38} | {'Q (Expected Score)':>38}"
    )
    print("-" * W)

    # 1. Values Summary
    print(
        f"{'Average Absolute Value':<30} | {en_abs_avg:>38.2f} | {sum(d['q_py'] for d in data) / len(data):>38.4f}"
    )
    print(
        f"{'Min / Max Value':<30} | {en_min:>17.1f} / {en_max:<18.1f} | {min(d['q_py'] for d in data):>17.4f} / {max(d['q_py'] for d in data):<18.4f}"
    )
    print("-" * W)

    # 2. Correlation and Errors Summary
    print(
        f"{'Correlation (R^2)':<30} | {r_squared_score:>38.6f} | {r_squared_q:>38.6f}"
    )
    print(
        f"{'Avg Absolute Error (Mean)':<30} | {sum(abs_errors) / len(data):>38.2f} | {sum(q_errors) / len(data):>38.6f}"
    )

    # Quantiles (Sigma intervals)
    for label, p in [
        ("1-Sigma Error (68.3%)", 0.6827),
        ("2-Sigma Error (95.5%)", 0.9545),
        ("3-Sigma Error (99.7%)", 0.9973),
    ]:
        score_p = get_percentile(abs_errors, p)
        q_p = get_percentile(q_errors, p)
        print(f"{label:<30} | {score_p:>38.2f} | {q_p:>38.6f}")

    print("=" * W)

    # Detailed Top 5 Offenders
    def print_top(title, key, col_name, fmt, is_pct=False):
        print(f"\n>>> {title}")
        top = sorted(data, key=lambda x: x[key], reverse=True)[:5]
        print(
            f"{col_name:>12} | {'SF Score':>10} | {'Py Score':>10} | {'SF Q':>8} | {'Py Q':>8} | {'FEN'}"
        )
        print("-" * W)
        for d in top:
            v = d[key] * 100 if is_pct else d[key]
            v_str = f"{v:{fmt}}" + ("%" if is_pct else "")
            print(
                f"{v_str:>12} | {d['sf']:>10.1f} | {d['py']:>10.2f} | {d['q_sf']:>8.4f} | {d['q_py']:>8.4f} | {d['fen']}"
            )

    print_top("TOP 5 LARGEST ABSOLUTE ERRORS", "abs_err", "Abs Err", "12.2f")
    print_top(
        "TOP 5 LARGEST RELATIVE ERRORS", "rel_err", "Rel Err", "11.2f", is_pct=True
    )
    print_top("TOP 5 LARGEST Q ERRORS", "q_err", "Q Err", "12.6f")
    print("\n" + "=" * W + "\n")


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
    all_fens = []

    done = 0
    print("Processed {} positions.".format(done))
    while done < cross_check_config.count:
        fens = filter_fens(next(fen_batch_provider))
        all_fens += fens

        b = data_loader.get_sparse_batch_from_fens(
            input_feature_name, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens)
        )
        model_evals += eval_model_batch(inner_model, b, cross_check_config.device)
        data_loader.destroy_sparse_batch(b)

        engine_evals += eval_engine_batch(
            cross_check_config.engine,
            cross_check_config.net,
            fens,
            cross_check_config.net_type,
        )

        done += len(fens)
        print("Processed {} positions.".format(done))

    compute_correlation(engine_evals, model_evals, all_fens)


if __name__ == "__main__":
    main()
