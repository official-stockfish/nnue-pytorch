from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt
import argparse
import re
import os
import collections


def find_event_files(root_dir):
    p = re.compile("events\\.out\\.tfevents.*")
    tfevent_files = []
    for path, subdirs, files in os.walk(root_dir, followlinks=False):
        for filename in files:
            m = p.match(filename)
            if m:
                tfevent_files.append(os.path.join(path, filename))
    return tfevent_files


def find_ordo_file(root_dir):
    for path, subdirs, files in os.walk(root_dir, followlinks=False):
        for filename in files:
            if filename == "ordo.out":
                return os.path.join(path, filename)
    return None


def get_list_aggregator(aggregation_mode="avg"):
    if aggregation_mode == "min":
        return lambda x: min(x)
    elif aggregation_mode == "max":
        return lambda x: max(x)
    elif aggregation_mode == "avg":
        return lambda x: sum(x) / len(x)
    else:
        raise Exception("Invalid aggregation_mode {}".format(aggregation_mode))


def aggregate_dict(values, aggregation_mode="avg"):
    """
    values must be a dict of lists
    each list is aggregated to a single scalar
    based on the aggregation_mode
    can be one of 'min', 'max', 'avg'
    """

    aggregate_list = get_list_aggregator(aggregation_mode)

    res = dict()
    for k, v in values.items():
        res[k] = aggregate_list(v)
    return res


def dict_to_xy(d):
    x = []
    y = []
    for k, v in sorted(d.items()):
        x.append(k)
        y.append(v)
    return x, y


def parse_ordo_file(filename, label):
    p = re.compile(".*nn-epoch(\\d*)\\.nnue")
    with open(filename, "r") as ordo_file:
        rows = []
        lines = ordo_file.readlines()
        for line in lines:
            if "nn-epoch" in line and label in line:
                fields = line.split()
                net = fields[1]
                epoch = int(p.match(net)[1])
                rating = float(fields[3])
                error = float(fields[4])
                rows.append((net, epoch, rating, error))

        return rows


def transpose_list_of_tuples(l):
    return list(map(list, zip(*l)))


def do_plots(out_filename, root_dirs, elo_range, loss_range, split):
    """
    1. Find tfevents files for each root directory
    2. Look for metrics
    2.1. Look for 'val_loss'
    3. Look for ordo.out
    3.1. Parse elo from ordo.
    4. Do plots.
    """

    tf_size_guidance = {
        "compressedHistograms": 10,
        "images": 0,
        "scalars": 0,
        "histograms": 1,
    }

    fig = plt.figure()
    fig.set_size_inches(18, 10)
    ax_train_loss = fig.add_subplot(311)
    ax_val_loss = fig.add_subplot(312)
    ax_elo = None

    ax_val_loss.set_xlabel("step")
    ax_val_loss.set_ylabel("val_loss")

    ax_train_loss.set_xlabel("step")
    ax_train_loss.set_ylabel("train_loss")

    for user_root_dir in root_dirs:
        # if asked to split we split the roto dir into a number of user root dirs,
        # i.e. all direct subdirectories containing tfevent files.
        # we use the ordo file in the root dir, but split the content.
        split_root_dirs = [user_root_dir]
        if split:
            split_root_dirs = []
            for item in os.listdir(user_root_dir):
                if os.path.isdir(os.path.join(user_root_dir, item)):
                    root_dir = os.path.join(user_root_dir, item)
                    if len(find_event_files(root_dir)) > 0:
                        split_root_dirs.append(root_dir)
            split_root_dirs.sort()

        for root_dir in split_root_dirs:
            print("Processing root_dir {}".format(root_dir))
            tfevents_files = find_event_files(root_dir)
            print("Found {} tfevents files.".format(len(tfevents_files)))

            val_losses = collections.defaultdict(lambda: [])
            train_losses = collections.defaultdict(lambda: [])
            for i, tfevents_file in enumerate(tfevents_files):
                print(
                    "Processing tfevents file {}/{}: {}".format(
                        i + 1, len(tfevents_files), tfevents_file
                    )
                )
                events_acc = EventAccumulator(tfevents_file, tf_size_guidance)
                events_acc.Reload()

                vv = events_acc.Scalars("val_loss")
                print("Found {} val_loss entries.".format(len(vv)))
                minloss = min([v[2] for v in vv])
                for v in vv:
                    if v[2] < minloss + loss_range:
                        step = v[1]
                        val_losses[step].append(v[2])

                vv = events_acc.Scalars("train_loss")
                minloss = min([v[2] for v in vv])
                print("Found {} train_loss entries.".format(len(vv)))
                for v in vv:
                    if v[2] < minloss + loss_range:
                        step = v[1]
                        train_losses[step].append(v[2])

            print("Aggregating data...")

            val_loss = aggregate_dict(val_losses, "min")
            x, y = dict_to_xy(val_loss)
            ax_val_loss.plot(x, y, label=root_dir)

            train_loss = aggregate_dict(train_losses, "min")
            x, y = dict_to_xy(train_loss)
            ax_train_loss.plot(x, y, label=root_dir)

            print("Finished aggregating data.")

        ordo_file = find_ordo_file(user_root_dir)
        if ordo_file:
            print("Found ordo file {}".format(ordo_file))
            if ax_elo is None:
                ax_elo = fig.add_subplot(313)
                ax_elo.set_xlabel("epoch")
                ax_elo.set_ylabel("Elo")

            for root_dir in split_root_dirs:
                rows = parse_ordo_file(ordo_file, root_dir if split else "nnue")
                if len(rows) == 0:
                    continue
                rows = sorted(rows, key=lambda x: x[1])
                epochs = []
                elos = []
                errors = []
                maxelo = max([row[2] for row in rows])
                for row in rows:
                    epoch = row[1]
                    elo = row[2]
                    error = row[3]
                    if epoch not in epochs:
                        if elo > maxelo - elo_range:
                            epochs.append(epoch)
                            elos.append(elo)
                            errors.append(error)

                print("Found ordo data for {} epochs".format(len(epochs)))

                ax_elo.errorbar(epochs, elos, yerr=errors, label=root_dir)

        else:
            print("Did not find ordo file. Skipping.")

    ax_val_loss.legend()
    ax_train_loss.legend()
    if ax_elo:
        ax_elo.legend()

    print("Saving plot at {}".format(out_filename))
    # plt.show()
    plt.savefig(out_filename, dpi=300)


def main():
    # do_plots('test_plot_out.png', ['../nnue-pytorch-training/experiment_10', '../nnue-pytorch-training/experiment_11'])

    parser = argparse.ArgumentParser(
        description="Generate plots of losses and Elo for experiments run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dirs",
        type=str,
        nargs="+",
        help="multiple root directories (containing ordo.out and tensorflow event files)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiment_loss_Elo.png",
        help="Filename of the plot generated",
    )
    parser.add_argument(
        "--elo_range",
        type=float,
        default=50.0,
        help="Limit Elo data shown to the best result - elo_range",
    )
    parser.add_argument(
        "--loss_range",
        type=float,
        default=0.004,
        help="Limit loss data shown to the best result + loss_range",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split the root dirs provided, assumes the ordo file is still at the root, and nets in that ordo file match root_dir/sub_dir/",
    )
    args = parser.parse_args()

    print(args.root_dirs)
    do_plots(
        args.output,
        args.root_dirs,
        elo_range=args.elo_range,
        loss_range=args.loss_range,
        split=args.split,
    )


if __name__ == "__main__":
    main()
