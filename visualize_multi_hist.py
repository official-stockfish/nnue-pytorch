import argparse
import features
import model as M
import numpy as np
import torch
import matplotlib.pyplot as plt

from serialize import NNUEReader


def load_model(filename, feature_set):
    if filename.endswith(".pt") or filename.endswith(".ckpt"):
        if filename.endswith(".pt"):
            model = torch.load(filename, weights_only=False)
        else:
            model = M.NNUE.load_from_checkpoint(filename, feature_set=feature_set)
        model.eval()
    elif filename.endswith(".nnue"):
        with open(filename, "rb") as f:
            reader = NNUEReader(f, feature_set)
        model = reader.model
    else:
        raise Exception("Invalid filetype: " + str(filename))

    return model


def get_bins(inputs_columns, num_bins):
    a = float("+inf")
    b = float("-inf")
    for inputs in inputs_columns:
        for inp in inputs:
            a = min(a, float(np.min(inp)))
            b = max(b, float(np.max(inp)))
    a -= 0.001
    b += 0.001
    return [a + (b - a) / num_bins * i for i in range(num_bins + 1)]


def plot_hists(
    tensors_columns,
    row_names,
    col_names,
    w=8.0,
    h=3.0,
    title=None,
    num_bins=256,
    filename="a.png",
):
    fig, axs = plt.subplots(
        len(tensors_columns[0]),
        len(tensors_columns),
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(w * len(tensors_columns), h * len(tensors_columns[0])),
        dpi=100,
    )
    if title:
        fig.suptitle(title)
    bins = get_bins(tensors_columns, num_bins)
    for i, tensors in enumerate(tensors_columns):
        print("Processing column {}/{}.".format(i + 1, len(tensors_columns)))
        for j, tensor in enumerate(tensors):
            ax = axs[j, i]
            print("    Processing tensor {}/{}.".format(j + 1, len(tensors)))
            ax.hist(tensor, log=True, bins=bins)
            if i == 0 and row_names[j]:
                ax.set_ylabel(row_names[j])
            if j == 0 and col_names[i]:
                ax.set_xlabel(col_names[i])
                ax.xaxis.set_label_position("top")
    fig.savefig(filename)


def main():
    parser = argparse.ArgumentParser(
        description="Visualizes networks in ckpt, pt and nnue format."
    )
    parser.add_argument(
        "models", nargs="+", help="Source model (can be .ckpt, .pt or .nnue)"
    )
    parser.add_argument(
        "--dont-show", action="store_true", help="Don't show the plots."
    )
    features.add_argparse_args(parser)
    args = parser.parse_args()

    supported_features = ("HalfKAv2", "HalfKAv2^", "HalfKAv2_hm", "HalfKAv2_hm^")
    assert args.features in supported_features
    feature_set = features.get_feature_set_from_name(args.features)

    from os.path import basename

    labels = []
    for m in args.models:
        label = basename(m)
        if label.startswith("nn-"):
            label = label[3:]
        if label.endswith(".nnue"):
            label = label[:-5]
        labels.append("\n".join(label.split("-")))

    models = [load_model(m, feature_set) for m in args.models]

    coalesced_ins = [M.coalesce_ft_weights(model, model.input) for model in models]
    input_weights = [
        coalesced_in[:, : M.L1].flatten().numpy() for coalesced_in in coalesced_ins
    ]
    input_weights_psqt = [
        (coalesced_in[:, M.L1 :] * 600).flatten().numpy()
        for coalesced_in in coalesced_ins
    ]
    plot_hists(
        [input_weights],
        labels,
        [None],
        w=10.0,
        h=3.0,
        num_bins=8 * 128,
        title="Distribution of feature transformer weights among different nets",
        filename="input_weights_hist.png",
    )
    plot_hists(
        [input_weights_psqt],
        labels,
        [None],
        w=10.0,
        h=3.0,
        num_bins=8 * 128,
        title="Distribution of feature transformer PSQT weights among different nets (in stockfish internal units)",
        filename="input_weights_psqt_hist.png",
    )

    layer_stacks = [model.layer_stacks for model in models]
    layers_l1 = [[] for i in range(layer_stacks[0].count)]
    layers_l2 = [[] for i in range(layer_stacks[0].count)]
    layers_l3 = [[] for i in range(layer_stacks[0].count)]
    for ls in layer_stacks:
        for i, sublayers in enumerate(ls.get_coalesced_layer_stacks()):
            l1, l2, l3 = sublayers
            layers_l1[i].append(l1.weight.flatten().numpy())
            layers_l2[i].append(l2.weight.flatten().numpy())
            layers_l3[i].append(l3.weight.flatten().numpy())
    col_names = ["Subnet {}".format(i) for i in range(layer_stacks[0].count)]
    plot_hists(
        layers_l1,
        labels,
        col_names,
        w=2.0,
        h=2.0,
        num_bins=128,
        title="Distribution of l1 weights among different nets and buckets",
        filename="l1_weights_hist.png",
    )
    plot_hists(
        layers_l2,
        labels,
        col_names,
        w=2.0,
        h=2.0,
        num_bins=32,
        title="Distribution of l2 weights among different nets and buckets",
        filename="l2_weights_hist.png",
    )
    plot_hists(
        layers_l3,
        labels,
        col_names,
        w=2.0,
        h=2.0,
        num_bins=16,
        title="Distribution of output weights among different nets and buckets",
        filename="output_weights_hist.png",
    )

    if not args.dont_show:
        plt.show()


if __name__ == "__main__":
    main()
