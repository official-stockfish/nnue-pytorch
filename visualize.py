import argparse
import features
import model as M
import numpy as np
import torch
import matplotlib.pyplot as plt

from serialize import NNUEReader


class NNUEVisualizer():
    def __init__(self, model):
        self.model = model

    def coalesce_ft_weights(self, model, layer):
        weight = layer.weight.data
        indices = model.feature_set.get_virtual_to_real_features_gather_indices()
        weight_coalesced = weight.new_zeros(
            (weight.shape[0], model.feature_set.num_real_features))
        for i_real, is_virtual in enumerate(indices):
            weight_coalesced[:, i_real] = sum(
                weight[:, i_virtual] for i_virtual in is_virtual)

        return weight_coalesced

    def plot_input_weights(self, net_name, vmin, vmax, save_dir=None):
        # Coalesce weights and transform them to Numpy domain.
        weights = self.coalesce_ft_weights(self.model, self.model.input)
        weights = weights.transpose(0, 1).flatten().numpy()

        hd = 256  # Output feature dimension.
        numx = 32  # Number of output features per row.

        # Derived/fixed constants.
        numy = hd//numx
        widthx = 128
        widthy = 304
        totalx = numx * widthx
        totaly = numy * widthy
        totaldim = totalx*totaly

        # Generate image.
        print("Generating input weights plot...", end="", flush=True)

        # [Thanks to https://github.com/hxim/Stockfish-Evaluation-Guide
        # upon which the following code is based.]
        img = np.zeros(totaldim)
        for j in range(weights.size):
            # Calculate piece and king placement.
            pi = (j // hd - 1) % 641
            ki = (j // hd - 1) // 641
            piece = pi // 64
            rank = (pi % 64) // 8

            if (pi == 640 or (rank == 0 or rank == 7) and (piece == 0 or piece == 1)):
                # Ignore unused weights for "Shogi piece drop" and pawns on first/last rank.
                continue

            kipos = [ki % 8, ki // 8]
            pipos = [pi % 8, rank]
            inpos = [(7-kipos[0])+pipos[0]*8,
                     kipos[1]+(7-pipos[1])*8]
            d = - 8 if piece < 2 else 48 + (piece // 2 - 1) * 64
            jhd = j % hd
            x = inpos[0] + widthx * ((jhd) % numx) + (piece % 2)*64
            y = inpos[1] + d + widthy * (jhd // numx)
            ii = x + y * totalx

            img[ii] = weights[j]

        if vmin >= 0:
            img = np.abs(img)
            title_template = "abs(input weights) [{NETNAME}]"
        else:
            title_template = "input weights [{NETNAME}]"

        print(" done")

        # Plot image.
        plt.figure(figsize=(16, 9))
        cmap = 'coolwarm' if vmin < 0 else 'viridis'
        plt.matshow(img.reshape((totaldim//totalx, totalx)),
                    fignum=0, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(fraction=0.046, pad=0.04)

        line_options = {'color': 'black', 'linewidth': 0.5}
        for i in range(1, numx):
            plt.axvline(x=widthx*i-0.5, **line_options)

        for j in range(1, numy):
            plt.axhline(y=widthy*j-0.5, **line_options)

        plt.xlim([0, totalx])
        plt.ylim([totaly, 0])
        plt.xticks(ticks=widthx*np.arange(1, numx) - 0.5)
        plt.yticks(ticks=widthy*np.arange(1, numy) - 0.5)
        plt.axis('off')
        plt.title(title_template.format(NETNAME=net_name))
        plt.tight_layout()

        # Save figure.
        if save_dir:
            from os.path import join
            destname = join(save_dir, "input-weights.jpg")
            print("Saving input weights plot to {}".format(destname))
            plt.savefig(destname)


def main():
    parser = argparse.ArgumentParser(
        description="Visualizes networks in ckpt, pt and nnue format.")
    parser.add_argument(
        "source", help="Source file (can be .ckpt, .pt or .nnue)")
    parser.add_argument(
        "--input-weights-vmin", default=-0.5, type=float, help="Minimum of color map range for input weights (absolute values are plotted if this is positive or zero).")
    parser.add_argument(
        "--input-weights-vmax", default=0.5, type=float, help="Maximum of color map range for input weights.")
    parser.add_argument("--save-dir", type=str, required=False,
                        help="Save the plots in this directory.")
    parser.add_argument("--dont-show", action="store_true",
                        help="Don't show the plots.")
    parser.add_argument("--net-name", type=str, required=False,
                        help="Override the network name used in plot titles (default = network basename).")
    features.add_argparse_args(parser)
    args = parser.parse_args()

    assert args.features == 'HalfKP'
    feature_set = features.get_feature_set_from_name(args.features)

    print("Visualizing {}".format(args.source))

    if args.source.endswith(".pt") or args.source.endswith(".ckpt"):
        if args.source.endswith(".pt"):
            nnue = torch.load(args.source)
        else:
            nnue = M.NNUE.load_from_checkpoint(
                args.source, feature_set=feature_set)
        nnue.eval()
    elif args.source.endswith(".nnue"):
        with open(args.source, 'rb') as f:
            reader = NNUEReader(f, feature_set)
        nnue = reader.model
    else:
        raise Exception("Invalid filetype: " + str(args))

    if args.net_name:
        net_name = args.net_name
    else:
        from os.path import basename
        net_name = basename(args.source)

    visualizer = NNUEVisualizer(nnue)
    visualizer.plot_input_weights(
        net_name, args.input_weights_vmin, args.input_weights_vmax, args.save_dir)

    if not args.dont_show:
        plt.show()


if __name__ == '__main__':
    main()
