import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sys
import glob
import nnue_dataset
import torch
import sys
import random


def sigmoid(x, k):
    y = 1 / (1 + np.exp(-k * x))
    return y


def fit_data(x, y, sigma):
    # 1/361 is the initial guess. It's good enough to find the solution
    p0 = [1 / 361]
    popt, pcov = curve_fit(sigmoid, x, y, p0, sigma, method="dogbox")
    return popt[0]


def do_plot(data, filename):
    # plot of the eval distribution
    fig, axs = plt.subplots(2)
    fig.tight_layout(pad=2.0)
    fig.suptitle(filename)
    x = list(data.keys())
    y = [data[k][1] for k in x]
    x, y = zip(*list(sorted(zip(x, y), key=lambda x: x[0])))
    axs[0].plot(x, y)
    axs[0].set_ylabel("density")
    axs[0].set_xlabel("eval")
    axs[0].set_xscale("symlog")

    # plot of the perf% by eval and the fitted sigmoid
    x = list(data.keys())
    y = [data[k][0] / data[k][1] for k in x]
    # sigma is uncertainties, we con't care how correct it is.
    # The inverted counts are good enough.
    sigma = [1 / data[k][1] for k in x]
    k = fit_data(x, y, sigma)
    print("k: ", k)
    print("inv k: ", 1 / k)
    axs[1].scatter(x, y, label="perf")
    y = [sigmoid(xx, k) for xx in x]
    axs[1].scatter(x, y, label="sigmoid(x/{})".format(1.0 / k))
    axs[1].legend(loc="upper left")
    axs[1].set_ylabel("perf")
    axs[1].set_xlabel("eval")

    # save to a .png file
    plot_filename = ".".join(filename.split(".")[:-1]) + ".png"
    plt.savefig(plot_filename)
    print("plot saved at {}".format(plot_filename))


def gather_statistics_from_batches(batches, bucket_size):
    """
    This function takes an iterable of training batches and a bucket_size.
    It goes through all batches and collects evals and the outcomes.
    The evals are bucketed by bucket_size. Perf% is computed based on the
    evals and corresponding game outcomes.
    The result is a dictionary of the form { eval : (perf%, count) }
    """
    data = dict()
    i = 0
    for batch in batches:
        (
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            outcome,
            score,
            psqt_indices,
            layer_stack_indices,
        ) = batch
        batch_size = len(us)
        bucket = torch.round(score / bucket_size) * bucket_size
        perf = outcome
        for b, p in zip(bucket, perf):
            bucket_id = int(b)
            pp = float(p)
            if bucket_id in data:
                t = data[bucket_id]
                data[bucket_id] = (t[0] + pp, t[1] + 1)
            else:
                data[bucket_id] = (pp, 1)
        i += batch_size
        print("Loaded {} positions...".format(i))
    return data


def gather_statistics_from_data(filename, count, bucket_size):
    """
    Takes a .bin or .binpack file and produces perf% statistics
    The result is a dictionary of the form { eval : (perf%, count) }
    """
    batch_size = 8192
    cyclic = True
    smart_fen_skipping = True
    # we pass whatever feature set because we have to pass something
    # it doesn't actually matter, all we care about are the scores and outcomes
    # this is just the easiest way to do it
    dataset = nnue_dataset.SparseBatchDataset(
        "HalfKP", filename, batch_size, cyclic, smart_fen_skipping
    )
    batches = iter(dataset)
    num_batches = (count + batch_size - 1) // batch_size
    data = gather_statistics_from_batches(
        (next(batches) for i in range(num_batches)), bucket_size
    )
    return data


def show_help():
    print("Usage: python perf_sigmoid_fitter.py filename [count] [bucket_size]")
    print("count is the number of positions. Default: 1000000")
    print("bucket_size determines how the evals are bucketed. Default: 16")
    print("")
    print("This file can be used as a module")
    print("The function `gather_statistics_from_batches` can be used to determine")
    print("the sigmoid scaling factor for each batch during training")


def main():
    filename = sys.argv[1]
    count = 1000000 if len(sys.argv) < 3 else int(sys.argv[2])
    bucket_size = 16 if len(sys.argv) < 4 else int(sys.argv[3])
    data = gather_statistics_from_data(filename, count, bucket_size)
    do_plot(data, filename)


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        show_help()
    else:
        main()
