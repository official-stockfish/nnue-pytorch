import sys

import numpy as np
import torch

from ..modules.router import (
    NUM_WEIGHT_EXPERTS,
    decompose_router_index,
)


def king_sq_name(k: int) -> str:
    """Returns human-readable oriented mirrored king square label."""
    if k == 32:
        return "e1[c]"
    r = k // 4
    f = k % 4
    file_map = ["a", "g", "c", "e"]
    return f"{file_map[f]}{r + 1}"


def enemy_king_sq_name(sq: int) -> str:
    """Returns human-readable enemy king square label."""
    if sq == 64:
        return "e8[c]"
    return f"{chr(ord('a') + sq % 8)}{sq // 8 + 1}"


@torch.no_grad()
def print_routing_profile(model, file=None) -> None:
    """Prints the compact Expert Routing Profile card."""
    if file is None:
        file = sys.stdout

    router = model.router
    layer_stacks = model.layer_stacks

    lut = router.get_lookup_table()
    total_contexts = lut.numel()

    all_ctx = torch.arange(total_contexts, device=lut.device)
    mat_id, k_us, k_them = decompose_router_index(all_ctx)
    mat_feats = router.material_decode_table[mat_id]

    lut_np = (lut & 0x3F).cpu().numpy()  # Extract 6 bits for weight experts
    t_np = mat_feats[:, 0].cpu().numpy() + 1  # 1..16 pieces
    q_np = mat_feats[:, 2].cpu().numpy()  # 0 or 1
    k_us_np = k_us.cpu().numpy()  # 0..32
    k_them_np = k_them.cpu().numpy()  # 0..64

    delta_w_layers = layer_stacks.get_expert_deltas(router.codebook)
    is_coalesced = getattr(layer_stacks.l1, "is_coalesced", False)
    coalesced_deltas = []
    if is_coalesced:
        for layer in [layer_stacks.l1, layer_stacks.l2, layer_stacks.output]:
            w = layer.coalesced_weight.reshape(
                NUM_WEIGHT_EXPERTS, layer.out_features, -1
            )
            w_mean = w.mean(dim=0, keepdim=True)
            coalesced_deltas.append(w - w_mean)

    w_exp = 6
    w_ctx = 14
    w_pc = 14
    w_q = 7
    w_kus = 34
    w_kth = 22
    w_dw = 8

    print(f"\nExpert Routing Profile ({total_contexts:,} contexts):", file=file)
    header = (
        f"{'Expert':^{w_exp}} | {'Contexts (%)':^{w_ctx}} | {'PC (mean/mode)':^{w_pc}} | "
        f"{'Queen %':^{w_q}} | {'Top King (Our)':^{w_kus}} | {'Top King (Enemy)':^{w_kth}} | "
        f"{'||ΔW||_F':^{w_dw}}"
    )
    sep = (
        f"{'-' * w_exp}-+-{'-' * w_ctx}-+-{'-' * w_pc}-+-{'-' * w_q}-+-"
        f"{'-' * w_kus}-+-{'-' * w_kth}-+-{'-' * w_dw}"
    )
    print(header, file=file)
    print(sep, file=file)

    for k in range(NUM_WEIGHT_EXPERTS):
        mask = lut_np == k
        count = int(np.sum(mask))
        pct = (count / float(total_contexts)) * 100.0

        if count > 0:
            pc_subset = t_np[mask]
            pc_mean = float(np.mean(pc_subset))
            pc_mode = int(np.bincount(pc_subset).argmax())
            queen_pct = float(np.mean(q_np[mask])) * 100.0

            # Always print top 3 friendly king squares
            k_counts = np.bincount(k_us_np[mask], minlength=33)
            top_ks = np.argsort(-k_counts)[:3]
            top_squares = [
                f"{king_sq_name(int(sq))} ({(k_counts[sq] / count) * 100:.0f}%)"
                for sq in top_ks
                if k_counts[sq] > 0
            ]
            sq_str = ", ".join(top_squares) if top_squares else "-"

            # Top 2 enemy king squares
            kth_counts = np.bincount(k_them_np[mask], minlength=65)
            top_kth = np.argsort(-kth_counts)[:2]
            top_enemy_squares = [
                f"{enemy_king_sq_name(int(sq))} ({(kth_counts[sq] / count) * 100:.0f}%)"
                for sq in top_kth
                if kth_counts[sq] > 0
            ]
            enemy_sq_str = ", ".join(top_enemy_squares) if top_enemy_squares else "-"
        else:
            pc_mean = 0.0
            pc_mode = 0
            queen_pct = 0.0
            sq_str = "-"
            enemy_sq_str = "-"

        # ||ΔW||_F (calculated from low-rank deltas if available, or deviation from mean if coalesced)
        if (
            delta_w_layers
            and float(sum(torch.norm(dw).item() for dw in delta_w_layers)) > 0
        ):
            norm_sq = sum(
                float(torch.norm(dw[k], p="fro") ** 2) for dw in delta_w_layers
            )
            dw_norm = float(np.sqrt(norm_sq))
        elif coalesced_deltas:
            norm_sq = sum(
                float(torch.norm(cd[k], p="fro") ** 2) for cd in coalesced_deltas
            )
            dw_norm = float(np.sqrt(norm_sq))
        else:
            dw_norm = 0.0

        exp_str = f" #{k:02d}  "
        ctx_str = f"{count:6,d} ({pct:4.1f}%)"
        pc_str = f"  {pc_mean:4.1f} / {pc_mode:2d}   "
        q_str = f"  {queen_pct:3.0f}% "
        kus_str = f"{sq_str:<{w_kus}}"
        kth_str = f"{enemy_sq_str:<{w_kth}}"
        dw_str = f"  {dw_norm:5.2f} "

        print(
            f"{exp_str} | {ctx_str} | {pc_str} | {q_str} | {kus_str} | {kth_str} | {dw_str}",
            file=file,
        )
    print(file=file)
