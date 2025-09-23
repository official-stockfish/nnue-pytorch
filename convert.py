import argparse

import torch

import model as M

def main():
    parser = argparse.ArgumentParser(
        description="Converts files between ckpt and nnue format."
    )
    parser.add_argument("source", help="Source file (can be .ckpt, .pt or .nnue)")
    parser.add_argument("target", help="Target file (can be .pt or .nnue)")
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

    if args.source.endswith(".ckpt"):
        nnue = M.NNUE.load_from_checkpoint(
            args.source,
            feature_set=feature_set,
            config=M.ModelConfig(L1=args.l1),
            map_location=torch.device("cpu"),
        )
        nnue.eval()
    elif args.source.endswith(".pt"):
        nnue = torch.load(args.source, weights_only=False)
    elif args.source.endswith(".nnue"):
        with open(args.source, "rb") as f:
            nnue = M.NNUE(feature_set, M.ModelConfig(L1=args.l1))
            reader = NNUEReader(f, feature_set, M.ModelConfig(L1=args.l1))
            nnue.model = reader.model
            if args.description is None:
                args.description = reader.description
    else:
        raise Exception("Invalid network input format.")

    if args.ft_compression != "none" and not args.target.endswith(".nnue"):
        args.ft_compression = "none"
        # raise Exception('Compression only allowed for .nnue target.')

    if args.ft_compression not in ["none", "leb128"]:
        raise Exception("Invalid compression method.")

    if args.ft_optimize and args.ft_perm is not None:
        raise Exception("Options --ft_perm and --ft_optimize are mutually exclusive.")

    if args.ft_perm is not None and args.target.endswith(".nnue"):
        import ftperm

        ftperm.ft_permute(nnue.model, args.ft_perm)

    if args.ft_optimize and args.target.endswith(".nnue"):
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
    elif args.target.endswith(".nnue"):
        writer = NNUEWriter(
            nnue.model, args.description, ft_compression=args.ft_compression
        )
        with open(args.target, "wb") as f:
            f.write(writer.buf)
    else:
        raise Exception("Invalid network output format.")


if __name__ == "__main__":
    main()
