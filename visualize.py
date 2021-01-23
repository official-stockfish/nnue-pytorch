import argparse
import features
import model as M
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

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

    def plot_input_weights(self, net_name, vmin, vmax, order_neurons=False, save_dir=None, save_prefix=None):
        # Coalesce weights and transform them to Numpy domain.
        weights = self.coalesce_ft_weights(self.model, self.model.input)
        weights = weights.transpose(0, 1).flatten().numpy()

        hd = 256  # Output feature dimension.
        self.M = hd
        numx = 32  # Number of output features per row.

        self.ordered_input_neurons = np.arange(hd, dtype=int)

        if order_neurons:
            # Order input neuron by the L1-norm of their associated weights.
            neuron_weights_norm = np.zeros(hd)
            for i in range(hd):
                neuron_weights_norm[i] = np.sum(np.abs(weights[i::256]))

            inv_order = np.flip(np.argsort(neuron_weights_norm))

            for i in range(hd):
                self.ordered_input_neurons[inv_order[i]] = i

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
            jhd = self.ordered_input_neurons[j % hd]
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
            destname = join(
                save_dir, "{}input-weights.jpg".format("" if save_prefix is None else save_prefix + "_"))
            print("Saving input weights plot to {}".format(destname))
            plt.savefig(destname)

    def plot_fc_weights(self, net_name, vmin, vmax, save_dir=None, save_prefix=None):
        # L1.
        l1_weights_ = self.model.l1.weight.data.numpy()

        N = l1_weights_.size // (2*self.M)

        l1_weights = np.zeros((2*N, self.M))

        for i in range(N):
            l1_weights[2*i] = l1_weights_[i][self.ordered_input_neurons]
            l1_weights[2*i+1] = l1_weights_[i][self.M +
                                               self.ordered_input_neurons]

        if vmin >= 0:
            title_template = "abs(L1 weights) [{NETNAME}]"
        else:
            title_template = "L1 weights [{NETNAME}]"

        cmap = 'coolwarm' if vmin < 0 else 'viridis'
        plt.figure(figsize=(16, 9))
        gs = GridSpec(100, 100)
        plt.subplot(gs[:50, :])
        plt.matshow(np.abs(l1_weights) if vmin >= 0 else l1_weights,
                    fignum=0, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.title(title_template.format(NETNAME=net_name))

        line_options = {'color': 'gray', 'linewidth': 0.5}
        for i in range(1, self.M):
            #plt.axvline(x=i-0.5, **line_options)
            pass

        for j in range(1, N):
            plt.axhline(y=2*j-0.5, **line_options)
            # pass

        # L2.
        l2_weights = self.model.l2.weight.data.numpy()

        if vmin >= 0:
            title_template = "abs(L2 weights) [{NETNAME}]"
        else:
            title_template = "L2 weights [{NETNAME}]"

        cmap = 'coolwarm' if vmin < 0 else 'viridis'
        plt.subplot(gs[55:75, 40:60])
        plt.matshow(np.abs(l2_weights) if vmin >= 0 else l2_weights,
                    fignum=0, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.title(title_template.format(NETNAME=net_name))

        # Output.
        output_weights = self.model.output.weight.data.numpy()

        if vmin >= 0:
            title_template = "abs(output weights) [{NETNAME}]"
        else:
            title_template = "output weights [{NETNAME}]"

        cmap = 'coolwarm' if vmin < 0 else 'viridis'
        plt.subplot(gs[75:, :])
        plt.matshow(np.abs(output_weights) if vmin >= 0 else output_weights,
                    fignum=0, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        plt.title(title_template.format(NETNAME=net_name))

        # Save figure.
        if save_dir:
            from os.path import join
            destname = join(
                save_dir, "{}fc-weights.jpg".format("" if save_prefix is None else save_prefix + "_"))
            print("Saving FC weights plot to {}".format(destname))
            plt.savefig(destname)

        plt.figure()
        title_template = "L1 weights histogram [{NETNAME}]"
        plt.hist(l1_weights.flatten(), bins=(np.arange(-128, 129)-0.5)/64)
        plt.title(title_template.format(NETNAME=net_name))

        # Save figure.
        if save_dir:
            from os.path import join
            destname = join(save_dir, "{}l1-weights-histogram.jpg".format(
                "" if save_prefix is None else save_prefix + "_"))
            print("Saving L1 weights histogram to {}".format(destname))
            plt.savefig(destname)

        plt.figure()
        title_template = "L2 weights histogram [{NETNAME}]"
        plt.hist(l2_weights.flatten(), bins=(np.arange(-128, 129)-0.5)/64)
        plt.title(title_template.format(NETNAME=net_name))

        # Save figure.
        if save_dir:
            from os.path import join
            destname = join(save_dir, "{}l2-weights-histogram.jpg".format(
                "" if save_prefix is None else save_prefix + "_"))
            print("Saving L2 weights histogram to {}".format(destname))
            plt.savefig(destname)


def main():
    parser = argparse.ArgumentParser(
        description="Visualizes networks in ckpt, pt and nnue format.")
    parser.add_argument(
        "source", help="Source file (can be .ckpt, .pt or .nnue)")
    parser.add_argument(
        "--input-weights-vmin", default=-1, type=float, help="Minimum of color map range for input weights (absolute values are plotted if this is positive or zero).")
    parser.add_argument(
        "--input-weights-vmax", default=1, type=float, help="Maximum of color map range for input weights.")
    parser.add_argument(
        "--order-input-neurons", action="store_true",
        help="Order the neurons of the input layer by the L1-norm (sum of absolute values) of their weights.")
    parser.add_argument(
        "--fc-weights-vmin", default=-2, type=float, help="Minimum of color map range for fully-connected layer weights (absolute values are plotted if this is positive or zero).")
    parser.add_argument(
        "--fc-weights-vmax", default=2, type=float, help="Maximum of color map range for fully-connected layer weights.")
    parser.add_argument("--save-dir", type=str, required=False,
                        help="Save the plots in this directory.")
    parser.add_argument("--save-prefix", type=str, required=False,
                        help="Prefix used for the name of the saved files (default = network name).")
    parser.add_argument("--dont-show", action="store_true",
                        help="Don't show the plots.")
    parser.add_argument("--net-name", type=str, required=False,
                        help="Override the network name used in plot titles (default = network basename).")
    features.add_argparse_args(parser)
    args = parser.parse_args()

    assert args.features in ['HalfKP', 'HalfKP^']
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

    if args.save_prefix:
        save_prefix = args.save_prefix
    else:
        save_prefix = net_name

    if args.order_input_neurons:
        net_name = "reordered " + net_name

    visualizer = NNUEVisualizer(nnue)
    visualizer.plot_input_weights(
        net_name, args.input_weights_vmin, args.input_weights_vmax, args.order_input_neurons, args.save_dir, save_prefix)
    visualizer.plot_fc_weights(
        net_name, args.fc_weights_vmin, args.fc_weights_vmax, args.save_dir, save_prefix)

    if not args.dont_show:
        plt.show()


if __name__ == '__main__':
    main()
