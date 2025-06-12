import argparse
import chess
import features
import nnue_dataset
import model as M
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from serialize import NNUEReader


class NNUEVisualizer():
    def __init__(self, model, args):
        self.model = model
        self.args = args

        self.model.cuda()

        import matplotlib as mpl
        self.dpi = 100
        mpl.rcParams["figure.figsize"] = (
            self.args.default_width//self.dpi, self.args.default_height//self.dpi)
        mpl.rcParams["figure.dpi"] = self.dpi

    def _process_fig(self, name, fig=None):
        if self.args.save_dir:
            from os.path import join
            destname = join(
                self.args.save_dir, "{}{}.jpg".format("" if self.args.label is None else self.args.label + "_", name))
            print("Saving {}".format(destname))
            if fig is not None:
                fig.savefig(destname)
            else:
                plt.savefig(destname)

    def get_data(self, count, batch_size):
        fen_batch_provider = nnue_dataset.FenBatchProvider(self.args.data, True, 1, batch_size, False, 10)

        activations_by_bucket = [[] for i in range(self.model.num_ls_buckets)]
        i = 0
        while i < count:
            fens = next(fen_batch_provider)
            batch = nnue_dataset.make_sparse_batch_from_fens(self.model.feature_set, fens, [0] * len(fens), [1] * len(fens), [0] * len(fens))
            us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices = batch.contents.get_tensors('cuda')
            bucketed_preact = self.model.get_narrow_preactivations(us, them, white_indices, white_values, black_indices, black_values, psqt_indices, layer_stack_indices)

            for a, b in zip(activations_by_bucket, bucketed_preact):
                a.append(b.cpu().detach().numpy().clip(0, 1))

            i += batch_size
            print('{}/{}'.format(i, count))

        return activations_by_bucket

    def plot(self):
        bucketed_preact = self.get_data(self.args.count, self.args.batch_size)
        for i, d in enumerate(bucketed_preact):
            print('Bucket {} has {} entries.'.format(i, sum(a.shape[0] for a in d)))

        fig, axs = plt.subplots(M.L2, self.model.num_ls_buckets, sharex=True, sharey=True, figsize=(16, 16), dpi=200, gridspec_kw = {'wspace':0.05, 'hspace':0.05})

        for bucket_id, preact in enumerate(bucketed_preact):
            for i in range(M.L2):
                acts = np.concatenate([v[:,i] for v in preact]).flatten()

                ax = axs[bucket_id, i]
                ax.hist(acts, density=True, log=True, bins=128)
                ax.set_xlim([-0.01, 1.01])
                if i == 0:
                    ax.set_ylabel('Bucket {}'.format(bucket_id))
                if bucket_id == 0:
                    ax.set_xlabel('Neuron {}'.format(i))
                    ax.xaxis.set_label_position('top')

                if i == M.L2-1:
                    pass
                else:
                    ax.set_yticks([])

                if bucket_id == len(bucketed_preact)-1:
                    ax.set_xticks([0.0, 0.5, 1.0])
                else:
                    ax.set_xticks([])

        fig.savefig('{}_narrow_act.png'.format(Path(self.args.model).stem))

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
        "--default-width", default=1600, type=int,
        help="Default width of all plots (in pixels).")
    parser.add_argument(
        "--count", default=1000000, type=int,
        help="")
    parser.add_argument(
        "--batch_size", default=5000, type=int,
        help="")
    parser.add_argument(
        "--default-height", default=900, type=int,
        help="Default height of all plots (in pixels).")
    parser.add_argument(
        "--save-dir", type=str, required=False,
        help="Save the plots in this directory.")
    parser.add_argument(
        "--dont-show", action="store_true",
        help="Don't show the plots.")
    parser.add_argument("--data", type=str, help="path to a .bin or .binpack dataset")
    parser.add_argument(
        "--label", type=str, required=False,
        help="Override the label used in plot titles and as prefix of saved files.")
    features.add_argparse_args(parser)
    args = parser.parse_args()

    supported_features = ('HalfKAv2_hm', 'HalfKAv2_hm^')
    assert args.features in supported_features
    feature_set = features.get_feature_set_from_name(args.features)

    from os.path import basename
    label = basename(args.model)

    model = load_model(args.model, feature_set)

    print("Visualizing {}".format(args.model))

    if args.label is None:
        args.label = label

    visualizer = NNUEVisualizer(model, args)

    visualizer.plot()

    if not args.dont_show:
        plt.show()


if __name__ == '__main__':
    main()
