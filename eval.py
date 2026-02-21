import argparse

from torch.utils.data import DataLoader

import data_loader
import model as M


def main():
    parser = argparse.ArgumentParser(
        description="Converts files between ckpt and nnue format."
    )
    parser.add_argument("source", help="Source file (.nnue)")
    parser.add_argument("datasets")
    parser.add_argument("--l1", type=int, default=M.ModelConfig().L1)
    M.add_feature_args(parser)
    args = parser.parse_args()

    feature_name = args.features

    train_infinite = data_loader.SparseBatchDataset(
        feature_name,
        [args.datasets],
        -1,
        num_workers=1,
        config=data_loader.DataloaderSkipConfig(),
    )
    train = DataLoader(
        data_loader.FixedNumBatchesDataset(train_infinite, (100000 + 100 - 1) // 100),
        batch_size=None,
        batch_sampler=None,
    )

    with open(args.source, "rb") as f:
        reader = M.NNUEReader(
            f, feature_name, M.ModelConfig(L1=args.l1), M.QuantizationConfig()
        )
        model = reader.model

    print(iter(train))
    model()


if __name__ == "__main__":
    main()
