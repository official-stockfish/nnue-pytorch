import re
import os
import subprocess
import sys
import time
import argparse


def convert_ckpt(root_dir):
    """ Find the list of checkpoints that are available, and convert those that have no matching .nnue """
    # run96/run0/default/version_0/checkpoints/epoch=3.ckpt, or epoch=3-step=321151.ckpt
    p = re.compile("epoch.*\.ckpt")
    ckpts = []
    for path, subdirs, files in os.walk(root_dir, followlinks=False):
        for filename in files:
            m = p.match(filename)
            if m:
                ckpts.append(os.path.join(path, filename))

    # lets move the .nnue files a bit up in the tree, and get rid of the = sign.
    # run96/run0/default/version_0/checkpoints/epoch=3.ckpt -> run96/run0/nn-epoch3.nnue
    for ckpt in ckpts:
        nnue_file_name = re.sub("default/version_[0-9]+/checkpoints/", "", ckpt)
        nnue_file_name = re.sub(r"epoch\=([0-9]+).*\.ckpt", r"nn-epoch\1.nnue", nnue_file_name)
        if not os.path.exists(nnue_file_name):
            command = "{} serialize.py {} {} ".format(sys.executable, ckpt, nnue_file_name)
            ret = os.system(command)
            if ret != 0:
                print("Error serializing!")


def find_nnue(root_dir):
    """ Find the set of nnue nets that are available for testing, going through the full subtree """
    p = re.compile("nn-epoch[0-9]*.nnue")
    nnues = []
    for path, subdirs, files in os.walk(root_dir, followlinks=False):
        for filename in files:
            m = p.match(filename)
            if m:
                nnues.append(os.path.join(path, filename))
    return nnues


def parse_ordo(root_dir, nnues):
    """ Parse an ordo output file for rating and error """
    ordo_file_name = os.path.join(root_dir, "ordo.out")
    ordo_scores = {}
    for name in nnues:
        ordo_scores[name] = (-500, 1000)

    if os.path.exists(ordo_file_name):
        ordo_file = open(ordo_file_name, "r")
        lines = ordo_file.readlines()
        for line in lines:
            if "nn-epoch" in line:
                fields = line.split()
                net = fields[1]
                rating = float(fields[3])
                error = float(fields[4])
                ordo_scores[net] = (rating, error)

    return ordo_scores


def run_match(best, root_dir, c_chess_exe, concurrency, book_file_name, stockfish_base, stockfish_test):
    """ Run a match using c-chess-cli adding pgns to a file to be analysed with ordo """
    if stockfish_test is None:
        stockfish_test = stockfish_base

    pgn_file_name = os.path.join(root_dir, "out.pgn")
    command = "{} -each tc=4+0.04 option.Hash=8 option.Threads=1 -gauntlet -games 200 -rounds 1 -concurrency {}".format(
        c_chess_exe, concurrency
    )
    command = (
        command
        + " -openings file={} order=random -repeat -resign 3 700 -draw 8 10".format(
            book_file_name
        )
    )
    command = command + " -engine cmd={} name=master".format(stockfish_base)
    for net in best:
        command = command + " -engine cmd={} name={} option.EvalFile={}".format(
            stockfish_test, net, os.path.join(os.getcwd(), net)
        )
    command = command + " -pgn {} 0 2>&1".format(
        pgn_file_name
    )

    print("Running match with c-chess-cli ... {}".format(pgn_file_name), flush=True)
    c_chess_out = open(os.path.join(root_dir, "c_chess.out"), 'w')
    process = subprocess.Popen("stdbuf -o0 " + command, stdout=subprocess.PIPE, shell=True)
    seen = {}
    for line in process.stdout:
        line = line.decode('utf-8')
        c_chess_out.write(line)
        if 'Score' in line:
            epoch_num = re.search(r'epoch(\d+)', line)
            if epoch_num.group(1) not in seen:
                sys.stdout.write('\n')
            seen[epoch_num.group(1)] = True
            sys.stdout.write('\r' + line.rstrip())
    sys.stdout.write('\n')
    c_chess_out.close()
    if process.wait() != 0:
        print("Error running match!")


def run_ordo(root_dir, ordo_exe, concurrency):
    """ run an ordo calcuation on an existing pgn file """
    pgn_file_name = os.path.join(root_dir, "out.pgn")
    ordo_file_name = os.path.join(root_dir, "ordo.out")
    command = "{} -q -G -J  -p  {} -a 0.0 --anchor=master --draw-auto --white-auto -s 100 --cpus={} -o {}".format(
        ordo_exe, pgn_file_name, concurrency, ordo_file_name
    )

    print("Running ordo ranking ... {}".format(ordo_file_name), flush=True)
    ret = os.system(command)
    if ret != 0:
        print("Error running ordo!")


def run_round(
    root_dir,
    explore_factor,
    ordo_exe,
    c_chess_exe,
    stockfish_base,
    stockfish_test,
    book_file_name,
    concurrency,
):
    """ run a round of games, finding existing nets, analyze an ordo file to pick most suitable ones, run a round, and run ordo """

    # find and convert checkpoints to .nnue
    convert_ckpt(root_dir)

    # find a list of networks to test
    nnues = find_nnue(root_dir)
    if len(nnues) == 0:
        print("No .nnue files found in {}".format(root_dir))
        time.sleep(10)
        return
    else:
        print("Found {} nn-epoch*.nnue files".format(len(nnues)))

    # Get info from ordo data if that is around
    ordo_scores = parse_ordo(root_dir, nnues)

    # provide the top 3 nets
    print("Best nets so far:")
    ordo_scores = dict(
        sorted(ordo_scores.items(), key=lambda item: item[1][0], reverse=True)
    )
    count = 0
    for net in ordo_scores:
        print("   {} : {} +- {}".format(net, ordo_scores[net][0], ordo_scores[net][1]))
        count += 1
        if count == 3:
            break

    # get top 3 with error bar added, for further investigation
    print("Measuring nets:")
    ordo_scores = dict(
        sorted(
            ordo_scores.items(),
            key=lambda item: item[1][0] + explore_factor * item[1][1],
            reverse=True,
        )
    )
    best = []
    for net in ordo_scores:
        print("   {} : {} +- {}".format(net, ordo_scores[net][0], ordo_scores[net][1]))
        best.append(net)
        if len(best) == 3:
            break

    # run these nets against master...
    run_match(best, root_dir, c_chess_exe, concurrency, book_file_name, stockfish_base, stockfish_test)

    # and run a new ordo ranking for the games played so far
    run_ordo(root_dir, ordo_exe, concurrency)


def main():
    # basic setup
    parser = argparse.ArgumentParser(
        description="Finds the strongest .nnue / .ckpt in tree, playing games.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="""The directory where to look, recursively, for .nnue or .ckpt.
                 This directory will be used to store additional files,
                 in particular the ranking (ordo.out)
                 and game results (out.pgn and c_chess.out).""",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Number of concurrently running threads",
    )
    parser.add_argument(
        "--explore_factor",
        default=1.5,
        type=float,
        help="`expected_improvement = rating + explore_factor * error` is used to pick select the next networks to run.",
    )
    parser.add_argument(
        "--ordo_exe",
        type=str,
        default="./ordo",
        help="Path to ordo, see https://github.com/michiguel/Ordo",
    )
    parser.add_argument(
        "--c_chess_exe",
        type=str,
        default="./c-chess-cli",
        help="Path to c-chess-cli, see https://github.com/lucasart/c-chess-cli",
    )
    parser.add_argument(
        "--stockfish_base",
        type=str,
        default="./stockfish",
        help="Path to stockfish master (reference version), see https://github.com/official-stockfish/Stockfish",
    )
    parser.add_argument(
        "--stockfish_test",
        type=str,
        help="(optional) Path to new stockfish binary, if not set, will use stockfish_base",
    )
    parser.add_argument(
        "--book_file_name",
        type=str,
        default="./noob_3moves.epd",
        help="Path to a suitable book, see https://github.com/official-stockfish/books",
    )
    args = parser.parse_args()

    while True:
        run_round(
            args.root_dir,
            args.explore_factor,
            args.ordo_exe,
            args.c_chess_exe,
            args.stockfish_base,
            args.stockfish_test,
            args.book_file_name,
            args.concurrency,
        )


if __name__ == "__main__":
    main()
