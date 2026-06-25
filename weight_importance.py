import argparse
import chess
import features
import model as M
import numpy as np
import torch
import nnue_dataset
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from serialize import NNUEReader

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

def do_forward_backward(model, batch, device):
    us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices = batch

    model.zero_grad()
    res = model.forward(us, them, white_indices, white_values, black_indices, black_values, psqt_indices, layer_stack_indices)
    res.backward()

def get_ft_weight_importance(model, dataset, pos_n, device):
    dataset_iter = iter(dataset)
    tot_g = None
    for i in range(pos_n):
        batch = next(dataset_iter)
        do_forward_backward(model, batch, device)
        g = model.input.weight.grad[:, :M.L1].detach()
        g = torch.abs(g)
        if tot_g is None:
            tot_g = g
        else:
            tot_g += g

        if (i + 1) % 100 == 0:
            print('Done {} out of {} evaluations...'.format(i+1, pos_n))

    tot_g = tot_g.flatten()
    val, ind = torch.sort(tot_g, descending=True)
    return ind, val

def process_ft_weight_importance(ind, val, best_n, best_pct):
    res = []
    rs = 0.0
    s = torch.sum(val)
    size = ind.shape[0]
    for i in range(size):
        x = ind[i]
        v = val[i]
        if i > best_n:
            break
        if rs > s * best_pct:
            break
        res.append((x//M.L1, x%M.L1, v))
        rs += v
    return res

def main():
    parser = argparse.ArgumentParser(
        description="Finds weights with the highest importance. Importance is measured by the absolute value of the gradient.")
    parser.add_argument(
        "model", help="Source model (can be .ckpt, .pt or .nnue)")
    parser.add_argument(
        "--best_n", type=int, default=256,
        help="Get only n most important weights")
    parser.add_argument(
        "--best_pct", type=float, default=1.0,
        help="Get only weights up to a given % [0, 1] of the total importance. Whichever of best_n or best_pct is reached faster.")
    parser.add_argument(
        "--pos_n", type=int, default=1024,
        help="The number of positions to evaluate.")
    parser.add_argument(
        "--layer", type=str, default="ft",
        help="The layer to probe. Currently only 'ft' is supported.")
    parser.add_argument(
        "--output", type=str,
        help="Optional output file.")
    parser.add_argument("--data", type=str,
        help="path to a .bin or .binpack dataset")
    features.add_argparse_args(parser)
    args = parser.parse_args()

    assert args.layer == "ft"

    supported_features = ('HalfKAv2', 'HalfKAv2_hm')
    assert args.features in supported_features
    feature_set = features.get_feature_set_from_name(args.features)

    main_device = 'cuda'
    model = load_model(args.model, feature_set)
    model.to(device=main_device)
    dataset = nnue_dataset.SparseBatchDataset(feature_set.name, args.data, 1, num_workers=1,
                                              filtered=True, random_fen_skipping=32, device=main_device)

    file = None
    if args.output:
        file = open(args.output, 'w')

    if args.layer == "ft":
        ind, val = get_ft_weight_importance(model=model, dataset=dataset, pos_n=args.pos_n, device=main_device)
        result = process_ft_weight_importance(ind, val, best_n=args.best_n, best_pct=args.best_pct)
        for i, (x, y, v) in enumerate(result):
            msg = '{}\t{}\t{}'.format(x, y, v)
            print(msg)
            if file is not None:
                file.write(msg + '\n')

    if file:
        file.close()

if __name__ == '__main__':
    main()