import argparse
import hashlib
import os

import torch

import model as M


def main():
    parser = argparse.ArgumentParser(
        description="Converts files between ckpt and nnue format."
    )
    parser.add_argument("source", help="Source file (can be .ckpt, .pt or .nnue)")
    parser.add_argument("target", help="Target file (can be .pt or .nnue)")
    parser.add_argument(
        "--out-sha",
        action="store_true",
        dest="out_sha",
        help="Ignore target file name and save as nn-<sha>.nnue. If target is a directory, the file is placed there; otherwise it goes to dirname(target) or CWD.",
    )
    parser.add_argument(
        "--description",
        default=None,
        type=str,
        dest="description",
        help="The description string to include in the network. Only works when serializing into a .nnue file.",
    )
    parser.add_argument(
        "--ft_compression",
        default="leb128",
        type=str,
        dest="ft_compression",
        help="Compression method to use for FT weights and biases. Either 'none' or 'leb128'. Only allowed if saving to .nnue.",
    )
    parser.add_argument(
        "--ft_perm",
        default=None,
        type=str,
        dest="ft_perm",
        help="Path to a file that defines the permutation to use on the feature transformer.",
    )
    parser.add_argument(
        "--ft_optimize",
        action="store_true",
        dest="ft_optimize",
        help="Whether to perform full feature transformer optimization (ftperm.py) on the resulting network. This process is very time consuming.",
    )
    parser.add_argument(
        "--ft_optimize_data",
        default=None,
        type=str,
        dest="ft_optimize_data",
        help="Path to the dataset to use for FT optimization.",
    )
    parser.add_argument(
        "--ft_optimize_count",
        default=10000,
        type=int,
        dest="ft_optimize_count",
        help="Number of positions to use for FT optimization.",
    )
    parser.add_argument(
        "--no-cupy",
        action="store_false",
        dest="use_cupy",
        help="Disable CUPY usage if not enough GPU memory is available. This will use numpy instead, which is slower.",
    )
    parser.add_argument(
        "--device", type=int, default="0", help="Device to use for cupy"
    )
    parser.add_argument("--l1", type=int, default=M.ModelConfig().L1)
    M.add_feature_args(parser)
    args = parser.parse_args()

    feature_set = M.get_feature_set_from_name(args.features)

    print("Converting %s to %s" % (args.source, args.target))

    # Treat --out-sha as targeting .nnue even if target doesn't end with .nnue
    target_is_nnue = args.out_sha or args.target.endswith(".nnue")

    if args.source.endswith(".ckpt"):
        nnue = M.NNUE.load_from_checkpoint(
            args.source,
            feature_set=feature_set,
            config=M.ModelConfig(L1=args.l1),
            quantize_config=M.QuantizationConfig(),
            map_location=torch.device("cpu"),
        )
        nnue.eval()
    elif args.source.endswith(".pt"):
        nnue = torch.load(args.source, weights_only=False)
    elif args.source.endswith(".nnue"):
        with open(args.source, "rb") as f:
            nnue = M.NNUE(
                feature_set, M.ModelConfig(L1=args.l1), M.QuantizationConfig()
            )
            reader = M.NNUEReader(
                f, feature_set, M.ModelConfig(L1=args.l1), M.QuantizationConfig()
            )
            nnue.model = reader.model
            if args.description is None:
                args.description = reader.description
    else:
        raise Exception("Invalid network input format.")

    if args.ft_compression != "none" and not target_is_nnue:
        args.ft_compression = "none"
        # raise Exception('Compression only allowed for .nnue target.')

    if args.ft_compression not in ["none", "leb128"]:
        raise Exception("Invalid compression method.")

    if args.ft_optimize and args.ft_perm is not None:
        raise Exception("Options --ft_perm and --ft_optimize are mutually exclusive.")

    if args.ft_perm is not None and target_is_nnue:
        import ftperm

        if not args.source.endswith(".nnue"):
            M.coalesce_ft_weights_inplace(nnue.model.feature_set, nnue.model.input)
            nnue.model.layer_stacks.coalesce_layer_stacks_inplace()

        ftperm.ft_permute(nnue.model, args.ft_perm)

    if args.ft_optimize and target_is_nnue:
        import ftperm

        if args.ft_optimize_data is None:
            raise Exception(
                "Invalid dataset path for FT optimization. (--ft_optimize_data)"
            )
        if args.ft_optimize_count is None or args.ft_optimize_count < 1:
            raise Exception(
                "Invalid number of positions to optimize FT with. (--ft_optimize_count)"
            )

        if args.use_cupy:
            if args.device is not None:
                ftperm.set_cupy_device(args.device)

        if not args.source.endswith(".nnue"):
            M.coalesce_ft_weights_inplace(nnue.model.feature_set, nnue.model.input)
            nnue.model.layer_stacks.coalesce_layer_stacks_inplace()

        ftperm.ft_optimize(
            nnue.model,
            args.ft_optimize_data,
            args.ft_optimize_count,
            use_cupy=args.use_cupy,
        )

    if args.target.endswith(".ckpt"):
        raise Exception("Cannot convert into .ckpt")
    elif args.target.endswith(".pt"):
        torch.save(nnue, args.target)
    elif target_is_nnue:
        if os.path.isdir(args.target):
            out_dir = os.path.abspath(args.target)
        else:
            out_dir = os.path.abspath(os.path.dirname(args.target) or os.getcwd())
        os.makedirs(out_dir, exist_ok=True)

        writer = M.NNUEWriter(
            nnue.model, args.description, ft_compression=args.ft_compression
        )
        buf = writer.buf

        if args.out_sha:
            sha = hashlib.sha256(buf).hexdigest()
            final_path = os.path.join(out_dir, f"nn-{sha[:12]}.nnue")
            with open(final_path, "wb") as f:
                f.write(buf)
            print(f"Wrote {final_path}")
        else:
            with open(args.target, "wb") as f:
                f.write(buf)
            print(f"Wrote {args.target}")
    else:
        raise Exception("Invalid network output format.")


if __name__ == "__main__":
    main()
