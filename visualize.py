import argparse
import chess
import features
import model as M
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from serialize import NNUEReader


class NNUEVisualizer():
    def __init__(self, model, ref_model, args):
        self.model = model
        self.ref_model = ref_model
        self.args = args

        import matplotlib as mpl
        self.dpi = 100
        mpl.rcParams["figure.figsize"] = (
            self.args.default_width//self.dpi, self.args.default_height//self.dpi)
        mpl.rcParams["figure.dpi"] = self.dpi

    def _process_fig(self, name):
        if self.args.save_dir:
            from os.path import join
            destname = join(
                self.args.save_dir, "{}{}.jpg".format("" if self.args.label is None else self.args.label + "_", name))
            print("Saving {}".format(destname))
            plt.savefig(destname)

    def coalesce_ft_weights(self, model, layer):
        weight = layer.weight.data
        indices = model.feature_set.get_virtual_to_real_features_gather_indices()
        weight_coalesced = weight.new_zeros(
            (weight.shape[0], model.feature_set.num_real_features))
        for i_real, is_virtual in enumerate(indices):
            weight_coalesced[:, i_real] = sum(
                weight[:, i_virtual] for i_virtual in is_virtual)

        return weight_coalesced

    def plot_input_weights(self):
        # Coalesce weights and transform them to Numpy domain.
        weights = self.coalesce_ft_weights(self.model, self.model.input)
        weights = weights.transpose(0, 1).flatten().numpy()

        if self.args.ref_model:
            ref_weights = self.coalesce_ft_weights(
                self.ref_model, self.ref_model.input)
            ref_weights = ref_weights.transpose(0, 1).flatten().numpy()
            weights -= ref_weights

        hd = M.L1  # Number of input neurons.
        self.M = hd

        # Preferred ratio of number of input neurons per row/col.
        preferred_ratio = 4

        # Number of input neurons per row.
        # Find a factor of hd such that the aspect ratio
        # is as close to the preferred ratio as possible.
        factor, smallest_diff = 0, hd
        for n in range(1, hd+1):
            if hd % n == 0:
                ratio = hd / (n*n)
                diff = abs(preferred_ratio-ratio)
                if diff < smallest_diff:
                    factor = n
                    smallest_diff = diff

        numx = hd // factor

        inv_sorted_input_neurons = np.arange(hd, dtype=int)

        if self.args.sort_input_neurons:
            # Sort input neurons by the L1-norm of their associated weights.
            neuron_weights_norm = np.zeros(hd)
            for i in range(hd):
                neuron_weights_norm[i] = np.sum(np.abs(weights[i::hd]))

            self.sorted_input_neurons = np.flip(
                np.argsort(neuron_weights_norm))

            for i in range(hd):
                inv_sorted_input_neurons[self.sorted_input_neurons[i]] = i
        else:
            self.sorted_input_neurons = inv_sorted_input_neurons

        # Derived/fixed constants.
        numy = hd//numx
        widthx = 128
        widthy = 304
        totalx = numx * widthx
        totaly = numy * widthy
        totaldim = totalx*totaly

        if not self.args.no_input_weights:
            # Generate image.
            print("Generating input weights plot...", end="", flush=True)

            # [Thanks to https://github.com/hxim/Stockfish-Evaluation-Guide
            # upon which the following code is based.]
            img = np.zeros(totaldim)
            default_order = self.args.input_weights_order == "piece-centric-flipped-king"
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

                if default_order:
                    # Piece centric, but with flipped king position.
                    # Same order as used by https://github.com/hxim/Stockfish-Evaluation-Guide.
                    # See also https://github.com/glinscott/nnue-pytorch/issues/42#issuecomment-753604393.
                    inpos = [(7-kipos[0])+pipos[0]*8,
                             kipos[1]+(7-pipos[1])*8]
                    d = - 8 if piece < 2 else 48 + (piece // 2 - 1) * 64
                else:
                    # King centric.
                    inpos = [8*kipos[0]+pipos[0],
                             8*(7-kipos[1])+(7-pipos[1])]
                    d = -2*(7-kipos[1]) - 1 if piece < 2 else 48 + \
                        (piece // 2 - 1) * 64

                jhd = inv_sorted_input_neurons[j % hd]
                x = inpos[0] + widthx * ((jhd) % numx) + (piece % 2)*64
                y = inpos[1] + d + widthy * (jhd // numx)
                ii = x + y * totalx

                img[ii] = weights[j]

            if self.args.input_weights_auto_scale:
                vmin = None
                vmax = None
            else:
                vmin = self.args.input_weights_vmin
                vmax = self.args.input_weights_vmax

            extra_info = ""
            if self.args.sort_input_neurons:
                extra_info += "sorted"
                if not default_order:
                    extra_info += ", " + self.args.input_weights_order
            else:
                if not default_order:
                    extra_info += self.args.input_weights_order
            if len(extra_info) > 0:
                extra_info = "; " + extra_info

            if self.args.input_weights_auto_scale or self.args.input_weights_vmin < 0:
                title_template = "input weights [{LABEL}" + extra_info + "]"
                hist_title_template = "input weights histogram [{LABEL}]"
                cmap = 'coolwarm'
            else:
                img = np.abs(img)
                title_template = "abs(input weights) [{LABEL}" + \
                    extra_info + "]"
                hist_title_template = "abs(input weights) histogram [{LABEL}]"
                cmap = 'viridis'

            print(" done")

            # Input weights.
            scalex = (numx / numy) / preferred_ratio
            plt.figure(figsize=((scalex*self.args.default_width) //
                                self.dpi, self.args.default_height//self.dpi))
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
            plt.title(title_template.format(LABEL=self.args.label))
            plt.tight_layout()

            def format_coord(x, y):
                x, y = int(round(x)), int(round(y))

                x_ = x % widthx
                y_ = y % widthy
                piece_type = (y_+16)//64
                piece_name = "{} {}".format(
                    "white" if x_ // (widthx//2) == 0 else "black", chess.piece_name(piece_type+1))

                x_ = x_ % (widthx//2)
                y_ = (y_+16) % 64 if y_ >= 48 else y_+8
                if default_order:
                    # Piece centric, flipped king.
                    piece_square_name = chess.square_name(x_//8 + 8*(7-y_//8))
                    king_square_name = chess.square_name(
                        7-(x_ % 8) + 8*(y_ % 8))
                else:
                    # King centric.
                    if piece_type == 0:
                        piece_square_name = chess.square_name(
                            x_ % 8 + 8*(6-((y_-8) % 6)))
                        king_square_name = chess.square_name(
                            x_//8 + 8*(7-(y_-8)//6))
                    else:
                        piece_square_name = chess.square_name(
                            x_ % 8 + 8*(7-(y_ % 8)))
                        king_square_name = chess.square_name(
                            x_//8 + 8*(7-y_//8))

                neuron_id = int(numx * (y // widthy) + x // widthx)
                if self.args.sort_input_neurons:
                    neuron_label = "sorted neuron {} (original {})".format(
                        neuron_id, self.sorted_input_neurons[neuron_id])
                else:
                    neuron_label = "neuron {}".format(neuron_id)

                return "{}, {} on {}, white king on {}".format(neuron_label, piece_name, piece_square_name, king_square_name)

            ax = plt.gca()
            ax.format_coord = format_coord

            self._process_fig("input-weights")

            if not self.args.no_hist:
                # Input weights histogram.
                plt.figure()
                plt.hist(img, log=True, bins=(
                    np.arange(int(np.min(img)*127)-1, int(np.max(img)*127)+3)-0.5)/127)
                plt.title(hist_title_template.format(LABEL=self.args.label))
                plt.tight_layout()
                self._process_fig("input-weights-histogram")

    def plot_fc_weights(self):
        if not self.args.no_fc_weights:
            # L1 weights.
            l1_weights_ = self.model.l1.weight.data.numpy()

            if self.args.ref_model:
                l1_weights_ -= self.ref_model.l1.weight.data.numpy()

            N = l1_weights_.size // (2*self.M)

            l1_weights = np.zeros((2*N, self.M))

            for i in range(N):
                l1_weights[2*i] = l1_weights_[i][self.sorted_input_neurons]
                l1_weights[2*i+1] = l1_weights_[i][self.M +
                                                   self.sorted_input_neurons]

            if self.args.fc_weights_auto_scale:
                vmin = None
                vmax = None
            else:
                vmin = self.args.fc_weights_vmin
                vmax = self.args.fc_weights_vmax

            extra_info_l1 = ""
            if self.args.sort_input_neurons:
                extra_info_l1 += "; sorted input neurons"

            if self.args.fc_weights_auto_scale or self.args.fc_weights_vmin < 0:
                l1_title_template = "L1 weights [{LABEL}" + extra_info_l1 + "]"
                l2_title_template = "L2 weights [{LABEL}]"
                output_title_template = "output weights [{LABEL}]"
                plot_abs = False
                cmap = 'coolwarm'
            else:
                l1_title_template = "abs(L1 weights) [{LABEL}" + \
                    extra_info_l1 + "]"
                l2_title_template = "abs(L2 weights) [{LABEL}]"
                output_title_template = "abs(output weights) [{LABEL}]"
                plot_abs = True
                cmap = 'viridis'

            plt.figure()
            gs = GridSpec(100, 100)
            plt.subplot(gs[:50, :])
            plt.matshow(np.abs(l1_weights) if plot_abs else l1_weights,
                        fignum=0, vmin=vmin, vmax=vmax, cmap=cmap)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            plt.title(l1_title_template.format(LABEL=self.args.label))

            line_options = {'color': 'gray', 'linewidth': 0.5}

            for j in range(1, N):
                plt.axhline(y=2*j-0.5, **line_options)

            # L2 weights.
            l2_weights = self.model.l2.weight.data.numpy()

            if self.args.ref_model:
                l2_weights -= self.ref_model.l2.weight.data.numpy()

            plt.subplot(gs[55:75, 40:60])
            plt.matshow(np.abs(l2_weights) if plot_abs else l2_weights,
                        fignum=0, vmin=None if vmin == float(
                "-inf") else vmin, vmax=vmax, cmap=cmap)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            plt.title(l2_title_template.format(LABEL=self.args.label))

            # Output weights.
            output_weights = self.model.output.weight.data.numpy()

            if self.args.ref_model:
                output_weights -= self.ref_model.output.weight.data.numpy()

            plt.subplot(gs[75:, :])
            plt.matshow(np.abs(output_weights) if plot_abs else output_weights,
                        fignum=0, vmin=vmin, vmax=vmax, cmap=cmap)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            plt.title(output_title_template.format(LABEL=self.args.label))
            self._process_fig("fc-weights")

            if not self.args.no_hist:
                # L1 weights histogram.
                plt.figure()
                title_template = "L1 weights histogram [{LABEL}]"
                plt.hist(l1_weights.flatten(), log=True, bins=(
                    np.arange(int(np.min(l1_weights)*64)-1, int(np.max(l1_weights)*64)+3)-0.5)/64)
                plt.title(title_template.format(LABEL=self.args.label))
                plt.tight_layout()
                self._process_fig("l1-weights-histogram")

                # L2 weights histogram.
                plt.figure()
                title_template = "L2 weights histogram [{LABEL}]"
                plt.hist(l2_weights.flatten(), log=True, bins=(
                    np.arange(int(np.min(l2_weights)*64)-1, int(np.max(l2_weights)*64)+3)-0.5)/64)
                plt.title(title_template.format(LABEL=self.args.label))
                plt.tight_layout()
                self._process_fig("l2-weights-histogram")

    def plot_biases(self):
        if not self.args.no_biases:
            input_biases = self.model.input.bias.data.numpy()[
                self.sorted_input_neurons]
            l1_biases = self.model.l1.bias.data.numpy()
            l2_biases = self.model.l2.bias.data.numpy()
            output_bias = self.model.output.bias.data.numpy()

            if self.args.ref_model:
                input_biases -= self.ref_model.input.bias.data.numpy()[
                    self.sorted_input_neurons]
                l1_biases -= self.ref_model.l1.bias.data.numpy()
                l2_biases -= self.ref_model.l2.bias.data.numpy()
                output_bias -= self.ref_model.output.bias.data.numpy()

            extra_info = ""
            if self.args.sort_input_neurons:
                extra_info += "; sorted input neurons"

            plt.figure()
            title_template = "biases [{LABEL}" + extra_info + "]"
            plt.subplot(2, 1, 1)
            plt.plot(input_biases, '+', label='input')
            plt.title(title_template.format(LABEL=self.args.label))
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(l1_biases, 'x', label='L1')
            plt.plot(l2_biases, '+', label='L2')
            plt.plot(output_bias, 'o', label='output')
            plt.legend()
            plt.tight_layout()
            self._process_fig("biases")


def load_model(filename, feature_set):
    if filename.endswith(".pt") or filename.endswith(".ckpt"):
        if filename.endswith(".pt"):
            model = torch.load(filename)
        else:
            model = M.NNUE.load_from_checkpoint(
                filename, feature_set=feature_set)
        model.eval()
    elif filename.endswith(".nnue"):
        with open(filename, 'rb') as f:
            reader = NNUEReader(f, feature_set)
        model = reader.model
    else:
        raise Exception("Invalid filetype: " + str(filename))

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Visualizes networks in ckpt, pt and nnue format.")
    parser.add_argument(
        "model", help="Source model (can be .ckpt, .pt or .nnue)")
    parser.add_argument(
        "--ref-model", type=str, required=False,
        help="Visualize the difference between the given reference model (can be .ckpt, .pt or .nnue).")
    parser.add_argument(
        "--ref-features", type=str, required=False,
        help="The reference feature set to use (default = same as source model).")
    parser.add_argument(
        "--input-weights-vmin", default=-1, type=float,
        help="Minimum of color map range for input weights (absolute values are plotted if this is positive or zero).")
    parser.add_argument(
        "--input-weights-vmax", default=1, type=float,
        help="Maximum of color map range for input weights.")
    parser.add_argument(
        "--input-weights-auto-scale", action="store_true",
        help="Use auto-scale for the color map range for input weights. This ignores input-weights-vmin and input-weights-vmax.")
    parser.add_argument(
        "--input-weights-order", type=str, choices=["piece-centric-flipped-king", "king-centric"], default="piece-centric-flipped-king",
        help="Order of the input weights for each input neuron.")
    parser.add_argument(
        "--sort-input-neurons", action="store_true",
        help="Sort the neurons of the input layer by the L1-norm (sum of absolute values) of their weights.")
    parser.add_argument(
        "--fc-weights-vmin", default=-2, type=float,
        help="Minimum of color map range for fully-connected layer weights (absolute values are plotted if this is positive or zero).")
    parser.add_argument(
        "--fc-weights-vmax", default=2, type=float,
        help="Maximum of color map range for fully-connected layer weights.")
    parser.add_argument(
        "--fc-weights-auto-scale", action="store_true",
        help="Use auto-scale for the color map range for fully-connected layer weights. This ignores fc-weights-vmin and fc-weights-vmax.")
    parser.add_argument(
        "--no-hist", action="store_true",
        help="Don't generate any histograms.")
    parser.add_argument(
        "--no-biases", action="store_true",
        help="Don't generate plots for biases.")
    parser.add_argument(
        "--no-input-weights", action="store_true",
        help="Don't generate plots or histograms for input weights.")
    parser.add_argument(
        "--no-fc-weights", action="store_true",
        help="Don't generate plots or histograms for fully-connected layer weights.")
    parser.add_argument(
        "--default-width", default=1600, type=int,
        help="Default width of all plots (in pixels).")
    parser.add_argument(
        "--default-height", default=900, type=int,
        help="Default height of all plots (in pixels).")
    parser.add_argument(
        "--save-dir", type=str, required=False,
        help="Save the plots in this directory.")
    parser.add_argument(
        "--dont-show", action="store_true",
        help="Don't show the plots.")
    parser.add_argument(
        "--label", type=str, required=False,
        help="Override the label used in plot titles and as prefix of saved files.")
    features.add_argparse_args(parser)
    args = parser.parse_args()

    supported_features = ('HalfKP', 'HalfKP^')
    assert args.features in supported_features
    feature_set = features.get_feature_set_from_name(args.features)

    from os.path import basename
    label = basename(args.model)

    model = load_model(args.model, feature_set)

    if args.ref_model:
        if args.ref_features:
            assert args.ref_features in supported_features
            ref_feature_set = features.get_feature_set_from_name(
                args.ref_features)
        else:
            ref_feature_set = feature_set

        ref_model = load_model(args.ref_model, ref_feature_set)

        print("Visualizing difference between {} and {}".format(
            args.model, args.ref_model))

        from os.path import basename
        label = "diff " + label + "-" + basename(args.ref_model)
    else:
        ref_model = None
        print("Visualizing {}".format(args.model))

    if args.label is None:
        args.label = label

    visualizer = NNUEVisualizer(model, ref_model, args)

    visualizer.plot_input_weights()
    visualizer.plot_fc_weights()
    visualizer.plot_biases()

    if not args.dont_show:
        plt.show()


if __name__ == '__main__':
    main()
