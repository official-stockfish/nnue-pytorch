import sys
import re
import os
import itertools


def parse_ordo(ordo_filename):
    ordo_scores = []

    with open(ordo_filename, "r") as ordo_file:
        lines = ordo_file.readlines()
        for line in lines:
            if "nn-epoch" in line:
                fields = line.split()
                net = fields[1]
                rating = float(fields[3])
                error = float(fields[4])
                ordo_scores.append((net, rating, error))

    return ordo_scores


def find_ckpt_files(root_dir):
    p = re.compile(".*\\.ckpt")
    ckpt_files = []
    for path, subdirs, files in os.walk(root_dir, followlinks=False):
        for filename in files:
            m = p.match(filename)
            if m:
                ckpt_files.append(os.path.join(path, filename))
    return ckpt_files


def find_nnue_files(root_dir):
    p = re.compile(".*\\.nnue")
    nnue_files = []
    for path, subdirs, files in os.walk(root_dir, followlinks=False):
        for filename in files:
            m = p.match(filename)
            if m:
                nnue_files.append(os.path.join(path, filename))
    return nnue_files


def get_net_dir(net_path):
    return os.path.dirname(net_path)


def split_nets_by_strength(nets, split_point=16):
    nets.sort(key=lambda x: -x[1])
    best_nets = nets[: min(split_point, len(nets))]
    worst_nets = nets[min(split_point, len(nets)) :]
    return best_nets, worst_nets


def get_nets_by_directory(best_nets, worst_nets, num_best_to_keep=16):
    binned_best_nets = dict()
    binned_worst_nets = dict()

    for net_name, rating, error in itertools.chain(best_nets, worst_nets):
        basedir = get_net_dir(net_name)
        if basedir not in binned_best_nets:
            binned_best_nets[basedir] = []
        if basedir not in binned_worst_nets:
            binned_worst_nets[basedir] = []

    for net_name, rating, error in worst_nets:
        basedir = get_net_dir(net_name)
        binned_worst_nets[basedir].append(net_name)

    for net_name, rating, error in best_nets:
        basedir = get_net_dir(net_name)
        binned_best_nets[basedir].append(net_name)

    return binned_best_nets, binned_worst_nets


def delete_bad_nets(root_dir, num_best_to_keep=16):
    net_epoch_p = re.compile(".*epoch([0-9]*)\\.nnue")
    ckpt_epoch_p = re.compile(".*epoch=([0-9]*).*\\.ckpt")
    ordo_filename = os.path.join(root_dir, "ordo.out")
    if not os.path.exists(ordo_filename):
        print("No ordo file found. Exiting.")
        return
    else:
        nets = parse_ordo(ordo_filename)
        best_nets, worst_nets = split_nets_by_strength(nets, num_best_to_keep)

        best_nets_by_dir, worst_nets_by_dir = get_nets_by_directory(
            best_nets, worst_nets, num_best_to_keep
        )
        for basedir, worst_nets_in_dir in worst_nets_by_dir.items():
            ckpt_files = find_ckpt_files(basedir)
            nnue_files = find_nnue_files(basedir)
            worst_epochs = [
                net_epoch_p.match(net_name)[1] for net_name in worst_nets_in_dir
            ]

            for ckpt_file in ckpt_files:
                try:
                    ckpt_epoch = ckpt_epoch_p.match(ckpt_file)[1]
                    if ckpt_epoch in worst_epochs:
                        print("Delete {}".format(ckpt_file))
                        os.remove(ckpt_file)
                except:
                    pass

                print("Keep {}".format(ckpt_file))

            for nnue_file in nnue_files:
                try:
                    nnue_epoch = net_epoch_p.match(nnue_file)[1]
                    if nnue_epoch in worst_epochs:
                        print("Delete {}".format(nnue_file))
                        os.remove(nnue_file)
                except:
                    pass

                print("Keep {}".format(nnue_file))


def show_help():
    print("Usage: python delete_bad_nets.py root_dir [num_best_to_keep]")
    print('root_dir - the directory to "cleanup"')
    print("num_best_to_keep - the number of best nets to keep. Default: 16")
    print("")
    print("It expects to find ordo.out somewhere within root_dir.")
    print("If the ordo.out is not found nothing is deleted.")
    print("It uses the ratings from the ordo file to determine which nets are best.")
    print("The engine names must contain the network name in the")
    print('following format: "nn-epoch[0-9]*\\.nnue". The network file')
    print("can be specified with a parent directory (for example")
    print('"run_0/nn-epoch100.nnue"), in which case the .ckpt file corresponding')
    print(
        'to this .nnue file will only be searched for in the parent ("run_0") directory.'
    )
    print('The .ckpt files must contain "epoch=([0-9]*).*\\.ckpt".')
    print("Both ckpt and nnue files are deleted. Only nets listed in the ordo")
    print("file can be deleted. Other nets are always kept.")
    print("The .nnue and .ckpt files are matched by epoch.")
    print("")
    print("The directory layout can be for example:")
    print("- root_dir")
    print("  - run_0")
    print("    - a/b/c/d.ckpt")
    print("    - *.nnue")
    print("  - run_1")
    print("    - a/b/c/d.ckpt")
    print("    - *.nnue")
    print("  - ordo.out")
    print("    (in this case ony lines with engine name matching")
    print('     "run_[01]/nn-epoch[0-9]*\\.nnue" will be used.)')


def main():
    if len(sys.argv) < 2:
        show_help()
        return

    root_dir = sys.argv[1]
    num_best_to_keep = sys.argv[2] if len(sys.argv) >= 3 else 16
    delete_bad_nets(root_dir, num_best_to_keep)


if __name__ == "__main__":
    main()
