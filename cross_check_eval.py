import math
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Literal

import chess
import torch
import tyro
from tyro.conf import OmitArgPrefixes

import data_loader
import model as M


@dataclass(frozen=True)
class CrossCheckConfig:
    # Flags and Options
    engine: str
    """Path to the engine binary to use for evaluation."""

    data: str
    """Path to the .bin or .binpack dataset to use for evaluation."""

    net: str | None = None
    """Path to the .nnue net to evaluate."""

    random_weights: bool = False
    """Generate a random network for cross-checking instead of using a trained net."""

    checkpoint: str | None = None
    """Optional checkpoint (used instead of nnue for local eval)."""

    device: Literal["cuda", "mps", "cpu"] = "cuda"
    """Device for the NNUE model."""

    count: int = 8 * 2**10
    """Number of positions to process."""

    seed: int | None = None
    """Random seed for reproducibility when generating random weights."""

    max_error: float | None = None
    """Maximum allowed absolute error between quantized NNUE and engine evaluations."""

    min_r2: float | None = None
    """Minimum allowed R^2 score between quantized NNUE and engine evaluations."""


@dataclass(frozen=True)
class CliConfig:
    cross_check_config: OmitArgPrefixes[CrossCheckConfig]
    nnue_config: OmitArgPrefixes[M.NNUEConfig]


def read_model(
    nnue_path,
    config: M.NNUEConfig,
):
    with open(nnue_path, "rb") as f:
        reader = M.NNUEReader(f, config.features, config.model_config)
        return reader.model


def make_fen_batch_provider(data_path, batch_size):
    return data_loader.FenBatchProvider(
        data_path,
        True,
        1,
        batch_size,
        data_loader.DataloaderSkipConfig(
            random_fen_skipping=5,
            soft_early_fen_skipping=-1,
        ),
    )


def eval_model_batch(
    model: M.NNUEModel,
    batch: data_loader.SparseBatchPtr,
    device: str,
    fake_quantize: bool,
):
    (
        us,
        them,
        white_indices,
        black_indices,
        _outcome,
        _score,
        piece_count,
        queen_bucket,
    ) = batch.contents.get_tensors(device)

    evals = [
        v.item()
        for v in model.forward(
            us,
            them,
            white_indices,
            black_indices,
            piece_count,
            queen_bucket,
            fake_quantize_acts=fake_quantize,
            fake_quantize_weights=fake_quantize,
        )
        * model.quantization.nnue2score
    ]
    return evals


re_nnue_eval = re.compile(
    r"NNUE evaluation\s+([-+]?\d+)\s+\(side to move, internal units\)"
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


def compute_correlation(cmp_evals, ref_evals, fens, title, cmp_name, ref_name):
    if len(ref_evals) != len(cmp_evals):
        raise ValueError(f"Mismatch: {len(ref_evals)} vs {len(cmp_evals)}")

    # Trainer parameters from your configuration
    IN_OFFSET = 280.0
    IN_SCALING = 353.0

    data = []
    abs_errors = []
    q_errors = []

    for e, m, f in zip(ref_evals, cmp_evals, fens):
        ae = abs(m - e)
        # relative error, with a floor to avoid division by zero and to not exaggerate small evals too much
        ae_rel = ae / (max(abs(e), 1 / 32))
        q_ref = calculate_qf(e, IN_OFFSET, IN_SCALING)
        q_cmp = calculate_qf(m, IN_OFFSET, IN_SCALING)
        qe = abs(q_cmp - q_ref)

        abs_errors.append(ae)
        q_errors.append(qe)
        data.append(
            {
                "fen": f,
                "ref": e,
                "cmp": m,
                "abs_err": ae,
                "rel_err": ae_rel,
                "q_ref": q_ref,
                "q_cmp": q_cmp,
                "q_err": qe,
            }
        )

    # R^2 Calculation for Scores
    mean_sf = sum(ref_evals) / len(ref_evals)
    ss_res = sum((d["ref"] - d["cmp"]) ** 2 for d in data)
    ss_tot = sum((d["ref"] - mean_sf) ** 2 for d in data)
    r_squared_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # R^2 Calculation for Q (Expected Score)
    mean_q_ref = sum(d["q_ref"] for d in data) / len(data)
    ss_res_q = sum((d["q_ref"] - d["q_cmp"]) ** 2 for d in data)
    ss_tot_q = sum((d["q_ref"] - mean_q_ref) ** 2 for d in data)
    r_squared_q = 1 - (ss_res_q / ss_tot_q) if ss_tot_q != 0 else 0.0

    # Summary Stats
    en_min, en_max, _, en_abs_avg = compute_basic_eval_stats(ref_evals)
    py_min, py_max, _, py_abs_avg = compute_basic_eval_stats(cmp_evals)

    W = 115
    print("\n" + "=" * W)
    print(f"{title:^{W}}")
    print("=" * W)
    print(
        f"{'Metric':<30} | {'Score (Internal Units)':>38} | {'Q (Expected Score)':>38}"
    )
    print("-" * W)

    # 1. Values Summary
    print(
        f"{f'Average Absolute Value ({ref_name})':<30} | {en_abs_avg:>38.2f} | {sum(d['q_ref'] for d in data) / len(data):>38.4f}"
    )
    print(
        f"{f'Min / Max Value ({ref_name})':<30} | {en_min:>17.1f} / {en_max:<18.1f} | {min(d['q_ref'] for d in data):>17.4f} / {max(d['q_ref'] for d in data):<18.4f}"
    )
    print("-" * W)

    print(
        f"{f'Average Absolute Value ({cmp_name})':<30} | {py_abs_avg:>38.2f} | {sum(d['q_cmp'] for d in data) / len(data):>38.4f}"
    )
    print(
        f"{f'Min / Max Value ({cmp_name})':<30} | {py_min:>17.1f} / {py_max:<18.1f} | {min(d['q_cmp'] for d in data):>17.4f} / {max(d['q_cmp'] for d in data):<18.4f}"
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
    def print_top(title_text, key, col_name, fmt, is_pct=False):
        print(f"\n>>> {title_text}")
        top = sorted(data, key=lambda x: x[key], reverse=True)[:5]
        print(
            f"{col_name:>12} | {ref_name + ' Score':>10} | {cmp_name + ' Score':>10} | {ref_name + ' Q':>8} | {cmp_name + ' Q':>8} | {'FEN'}"
        )
        print("-" * W)
        for d in top:
            v = d[key] * 100 if is_pct else d[key]
            v_str = f"{v:{fmt}}" + ("%" if is_pct else "")
            print(
                f"{v_str:>12} | {d['ref']:>10.2f} | {d['cmp']:>10.2f} | {d['q_ref']:>8.4f} | {d['q_cmp']:>8.4f} | {d['fen']}"
            )

    print_top("TOP 5 LARGEST ABSOLUTE ERRORS", "abs_err", "Abs Err", "12.2f")
    print_top(
        "TOP 5 LARGEST RELATIVE ERRORS", "rel_err", "Rel Err", "11.2f", is_pct=True
    )
    print_top("TOP 5 LARGEST Q ERRORS", "q_err", "Q Err", "12.6f")
    print("\n" + "=" * W + "\n")

    return {
        "abs_errors": abs_errors,
        "max_abs_error": max(abs_errors) if abs_errors else 0.0,
        "mean_abs_error": sum(abs_errors) / len(data) if data else 0.0,
        "r_squared_score": r_squared_score,
        "r_squared_q": r_squared_q,
    }


def eval_engine_batch(engine_path, net_path, fens):
    if not fens:
        return []
    engine = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    parts = ["uci", f"setoption name EvalFile value {net_path}"]
    for fen in fens:
        parts.append(f"position fen {fen}")
        parts.append("eval")
    parts.append("quit")
    query = "\n".join(parts)
    out = engine.communicate(input=query)[0]
    evals = re.findall(re_nnue_eval, out)
    if len(evals) != len(fens):
        raise ValueError(
            f"number of evals returned by the engine doesn't match the number of fens. Got {len(evals)} evals and {len(fens)} fens. Output was:\n{out}"
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


def init_random_weights(model: M.NNUEModel) -> None:
    """Initialize model weights across full dynamic range, hitting clipping bounds."""
    with torch.no_grad():
        touched = set()

        for f in model.input.features:
            min_w, max_w = model.quantization.get_ft_weight_bounds(
                f.EXPORT_WEIGHT_DTYPE
            )
            f.weight.data.uniform_(min_w * 1.2, max_w * 1.2)
            touched.add(id(f.weight))
            if hasattr(f, "virtual_weight") and f.virtual_weight is not None:
                f.virtual_weight.data.uniform_(min_w * 1.2, max_w * 1.2)
                touched.add(id(f.virtual_weight))

        model.input.bias.data.uniform_(-1.0, 1.0)
        touched.add(id(model.input.bias))

        for group in model.weight_clipping:
            min_w = group["min_weight"]
            max_w = group["max_weight"]
            for p in group["params"]:
                p.data.uniform_(min_w * 1.2, max_w * 1.2)
                touched.add(id(p))
        for ls in [
            model.layer_stacks.l1,
            model.layer_stacks.l2,
            model.layer_stacks.output,
        ]:
            ls.bias_base.data.uniform_(-0.5, 0.5)
            touched.add(id(ls.bias_base))
            ls.adapter_down.data.uniform_(-0.1, 0.1)
            touched.add(id(ls.adapter_down))
            ls.down_proj.data.uniform_(-0.1, 0.1)
            touched.add(id(ls.down_proj))
            ls.up_proj.data.uniform_(-0.1, 0.1)
            touched.add(id(ls.up_proj))
            ls.bias_adapter.data.uniform_(-0.1, 0.1)
            touched.add(id(ls.bias_adapter))

        for p in model.router.parameters():
            p.data.uniform_(-1.0, 1.0)
            touched.add(id(p))
        model.router.update_lookup_table()

        all_param_ids = {id(p) for p in model.parameters()}
        untouched = all_param_ids - touched
        if untouched:
            untouched_names = [
                name for name, p in model.named_parameters() if id(p) in untouched
            ]
            raise RuntimeError(
                f"init_random_weights missed parameters: {untouched_names}"
            )

        model.clip_weights(include_input=True)


def run_smoke_test(
    engine_path: str,
    net_path: str,
    nnue_model: M.NNUEModel,
    ckpt_model: M.NNUEModel | None,
    device: str,
) -> None:
    fen = chess.STARTING_FEN
    fens = [fen]
    b = data_loader.get_sparse_batch_from_fens(
        nnue_model.input_feature_name, fens, [0], [1], [0]
    )
    nnue_eval = eval_model_batch(nnue_model, b, device, False)[0]
    nnue_quant = eval_model_batch(nnue_model, b, device, True)[0]

    ckpt_eval = None
    ckpt_quant = None
    if ckpt_model is not None:
        ckpt_eval = eval_model_batch(ckpt_model, b, device, False)[0]
        ckpt_quant = eval_model_batch(ckpt_model, b, device, True)[0]

    data_loader.destroy_sparse_batch(b)

    sf_evals = eval_engine_batch(engine_path, net_path, fens)
    if not sf_evals:
        raise RuntimeError("Stockfish returned no evaluation for starting position.")
    sf_eval = sf_evals[0]

    diff_quant = nnue_quant - sf_eval

    W = 80
    print("=" * W)
    print(f"[Smoke Test] Evaluating starting position: {fen}")
    print(f"  Stockfish Eval      : {sf_eval:+d}")
    print(f"  NNUE (PyTorch) Eval : {nnue_eval:+.2f}")
    print(f"  NNUE (Quant) Eval   : {nnue_quant:+.2f}")
    if ckpt_model is not None and ckpt_eval is not None and ckpt_quant is not None:
        print(f"  CKPT Eval           : {ckpt_eval:+.2f}")
        print(f"  CKPT (Quant) Eval   : {ckpt_quant:+.2f}")
    print(f"  Diff (Quant - SF)   : {diff_quant:+.2f}")

    if math.isnan(nnue_quant) or abs(diff_quant) > 5.0:
        print(
            f"[Smoke Test] FAILED: Evaluation difference too large ({diff_quant:+.2f})!"
        )
        print("=" * W)
        raise RuntimeError(
            f"Smoke test failed: Quantized NNUE eval ({nnue_quant}) and Stockfish eval ({sf_eval}) "
            f"differ by {diff_quant:.2f} (> 5.0) on starting position."
        )

    print("[Smoke Test] SUCCESS: Starting position evaluation verified!")
    print("=" * W + "\n")


def main():
    args = tyro.cli(CliConfig)

    cross_check_config = args.cross_check_config
    nnue_config = args.nnue_config

    if cross_check_config.random_weights and cross_check_config.net:
        raise ValueError("Cannot specify both --random-weights and --net.")

    if cross_check_config.seed is not None:
        torch.manual_seed(cross_check_config.seed)
        print(f"Set random seed: {cross_check_config.seed}")

    batch_size = 1024

    ckpt_model = None
    if cross_check_config.checkpoint:
        checkpoint = torch.load(
            cross_check_config.checkpoint, map_location="cpu", weights_only=False
        )
        ckpt = M.NNUE(config=nnue_config)
        ckpt.load_state_dict(checkpoint["state_dict"])
        ckpt.to(cross_check_config.device)
        ckpt.eval()
        # --checkpoint - returns a plain NNUE wrapping a NNUEModel
        ckpt_model = ckpt.model

    temp_net_path = None
    try:
        if cross_check_config.random_weights:
            init_model = M.NNUEModel(
                feature_name=nnue_config.features,
                config=nnue_config.model_config,
            )
            init_random_weights(init_model)
            with tempfile.NamedTemporaryFile(suffix=".nnue", delete=False) as temp_file:
                net_path = temp_file.name
            temp_net_path = net_path

            writer = M.NNUEWriter(init_model, verbose=False)
            with open(net_path, "wb") as f:
                f.write(writer.buf)

            nnue = read_model(net_path, config=nnue_config)
        elif cross_check_config.net:
            net_path = cross_check_config.net
            nnue = read_model(net_path, config=nnue_config)
        else:
            raise ValueError("Either --net or --random-weights must be specified.")

        nnue.to(cross_check_config.device)
        nnue.eval()
        # --net - returns the NNUEModel directly
        nnue_model = nnue

        # Run smoke test upfront before initializing dataloader
        run_smoke_test(
            cross_check_config.engine,
            net_path,
            nnue_model,
            ckpt_model,
            cross_check_config.device,
        )

        input_feature_name = nnue_model.input_feature_name
        fen_batch_provider = make_fen_batch_provider(
            cross_check_config.data, batch_size
        )

        ckpt_evals = []
        ckpt_quantized_evals = []
        nnue_evals = []
        nnue_quantized_evals = []
        engine_evals = []
        all_fens = []

        done = 0
        print(f"Processed {done} positions.")
        while done < cross_check_config.count:
            fens = filter_fens(next(fen_batch_provider))
            all_fens += fens

            b = data_loader.get_sparse_batch_from_fens(
                input_feature_name,
                fens,
                [0] * len(fens),
                [1] * len(fens),
                [0] * len(fens),
            )
            if ckpt_model:
                ckpt_evals += eval_model_batch(
                    ckpt_model, b, cross_check_config.device, False
                )
                ckpt_quantized_evals += eval_model_batch(
                    ckpt_model, b, cross_check_config.device, True
                )

            nnue_evals += eval_model_batch(
                nnue_model, b, cross_check_config.device, False
            )
            nnue_quantized_evals += eval_model_batch(
                nnue_model, b, cross_check_config.device, True
            )
            data_loader.destroy_sparse_batch(b)

            engine_evals += eval_engine_batch(
                cross_check_config.engine,
                net_path,
                fens,
            )

            done += len(fens)
            print(f"Processed {done} positions.")

        q_stats = None
        if ckpt_model:
            compute_correlation(
                ckpt_evals, nnue_evals, all_fens, "CKPT VS NNUE", "CKPT", "NNUE"
            )
            compute_correlation(
                ckpt_quantized_evals,
                nnue_quantized_evals,
                all_fens,
                "CKPT (Q) VS NNUE (Q)",
                "CKPT (Q)",
                "NNUE (Q)",
            )
            compute_correlation(
                ckpt_evals, engine_evals, all_fens, "CKPT VS SF", "CKPT", "SF"
            )
            compute_correlation(
                ckpt_quantized_evals,
                engine_evals,
                all_fens,
                "QUANTIZED CKPT VS SF",
                "CKPT (Q)",
                "SF",
            )
            compute_correlation(
                nnue_evals, engine_evals, all_fens, "NNUE VS SF", "NNUE", "SF"
            )
            q_stats = compute_correlation(
                nnue_quantized_evals,
                engine_evals,
                all_fens,
                "QUANTIZED NNUE VS SF",
                "NNUE (Q)",
                "SF",
            )
        else:
            compute_correlation(
                nnue_evals, engine_evals, all_fens, "NNUE VS SF", "NNUE", "SF"
            )
            q_stats = compute_correlation(
                nnue_quantized_evals,
                engine_evals,
                all_fens,
                "QUANTIZED NNUE VS SF",
                "NNUE (Q)",
                "SF",
            )

        if cross_check_config.max_error is not None and q_stats is not None:
            max_err = q_stats["max_abs_error"]
            if max_err > cross_check_config.max_error:
                raise RuntimeError(
                    f"Cross-check error threshold exceeded: max absolute error {max_err:.2f} > {cross_check_config.max_error:.2f}"
                )

        if cross_check_config.min_r2 is not None and q_stats is not None:
            r2 = q_stats["r_squared_score"]
            if r2 < cross_check_config.min_r2:
                raise RuntimeError(
                    f"Cross-check R^2 threshold not met: R^2 {r2:.4f} < {cross_check_config.min_r2:.4f}"
                )
    finally:
        if temp_net_path and os.path.exists(temp_net_path):
            os.remove(temp_net_path)


if __name__ == "__main__":
    main()
