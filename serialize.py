import hashlib
import os
import torch
import tyro

from dataclasses import dataclass, field
from typing import Optional, Literal, Annotated, Union
from tyro.conf import OmitArgPrefixes, Positional

from data_loader import DataloaderSkipConfig

import model as M


@dataclass(frozen=True)
class SerializeConfig:
    # Flags and Options
    out_sha: bool = False
    """Ignore target file name and save as nn-<sha>.nnue. If target is a directory,
    the file is placed there; otherwise it goes to dirname(target) or CWD."""

    description: Optional[str] = None
    """The description string to include in the network.
    Only works when serializing into a .nnue file."""

    ft_compression: Literal["none", "leb128"] = "leb128"
    """Compression method to use for FT weights and biases.
    Either 'none' or 'leb128'. Only allowed if saving to .nnue."""

    ft_perm: Optional[str] = None
    """Path to a file that defines the permutation to use on the feature transformer."""

    ft_optimize: bool = False
    """Whether to perform full feature transformer optimization (ftperm.py)
    on the resulting network. This process is very time consuming."""

    ft_optimize_data: Optional[str] = None
    """Path to the dataset to use for FT optimization."""

    ft_optimize_count: int = 10000
    """Number of positions to use for FT optimization."""

    use_cupy: Annotated[bool, tyro.conf.arg(name="cupy")] = True
    """Disable CUPY usage if not enough GPU memory is available.
    This will use numpy instead, which is slower."""

    device: Union[int, Literal["cpu", "mps"]] = 0
    """Device to use for ft_optimize acceleration."""

    loader_num_workers: int = 4
    """Number of workers to use for data loading during FT optimization."""

    dataloader_config: OmitArgPrefixes[DataloaderSkipConfig] = field(
        default_factory=DataloaderSkipConfig
    )


@dataclass(frozen=True)
class CliConfig:
    # Positional arguments
    source: Positional[str]
    """Source file (can be .ckpt, .pt or .nnue)"""

    target: Positional[str]
    """Target file (can be .pt or .nnue)"""
    serialize_config: OmitArgPrefixes[SerializeConfig] = field(
        default_factory=SerializeConfig
    )
    nnue_lightning_config: OmitArgPrefixes[M.NNUELightningConfig] = field(
        default_factory=M.NNUELightningConfig
    )


def main():
    args = tyro.cli(CliConfig)
    serialize_config = args.serialize_config
    nnue_lightning_config = args.nnue_lightning_config
    feature_name = nnue_lightning_config.features

    print("Converting %s to %s" % (args.source, args.target))

    # Treat --out-sha as targeting .nnue even if target doesn't end with .nnue
    target_is_nnue = serialize_config.out_sha or args.target.endswith(".nnue")

    model_description = serialize_config.description
    ft_compression = serialize_config.ft_compression
    if args.source.endswith(".ckpt"):
        nnue = M.NNUE.load_from_checkpoint(
            args.source,
            config=nnue_lightning_config,
            quantize_config=M.QuantizationConfig(),
            map_location=torch.device("cpu"),
        )
        nnue.eval()
    elif args.source.endswith(".pt"):
        nnue = torch.load(args.source, weights_only=False)
    elif args.source.endswith(".nnue"):
        with open(args.source, "rb") as f:
            nnue = M.NNUE(
                config=nnue_lightning_config,
                quantize_config=M.QuantizationConfig(),
            )
            reader = M.NNUEReader(
                f,
                feature_name,
                config=nnue_lightning_config.model_config,
                quantize_config=M.QuantizationConfig(),
            )
            nnue.model = reader.model
            if serialize_config.description is None:
                model_description = reader.description
    else:
        raise Exception("Invalid network input format.")

    if ft_compression != "none" and not target_is_nnue:
        print("Warning: Compression method for non `.nnue` target ignored.")
        ft_compression = "none"

    if ft_compression not in ["none", "leb128"]:
        raise Exception("Invalid compression method.")

    if serialize_config.ft_optimize and serialize_config.ft_perm is not None:
        raise Exception("Options --ft_perm and --ft_optimize are mutually exclusive.")

    if serialize_config.ft_perm is not None and target_is_nnue:
        import ftperm

        if not args.source.endswith(".nnue"):
            nnue.model.input.coalesce()
            nnue.model.layer_stacks.coalesce_layer_stacks_inplace()

        ftperm.ft_permute(nnue.model, serialize_config.ft_perm)

    if serialize_config.ft_optimize and target_is_nnue:
        import ftperm

        if serialize_config.ft_optimize_data is None:
            raise Exception(
                "Invalid dataset path for FT optimization. (--ft_optimize_data)"
            )
        if (
            serialize_config.ft_optimize_count is None
            or serialize_config.ft_optimize_count < 1
        ):
            raise Exception(
                "Invalid number of positions to optimize FT with. (--ft_optimize_count)"
            )

        if not args.source.endswith(".nnue"):
            nnue.model.input.coalesce()
            nnue.model.layer_stacks.coalesce_layer_stacks_inplace()

        ftperm.ft_optimize(
            nnue.model,
            serialize_config.ft_optimize_data,
            serialize_config.ft_optimize_count,
            use_cupy=serialize_config.use_cupy,
            device=serialize_config.device,
            loader_num_workers=serialize_config.loader_num_workers,
            loader_config=serialize_config.dataloader_config,
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
            nnue.model, model_description, ft_compression=ft_compression
        )
        buf = writer.buf

        if serialize_config.out_sha:
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
