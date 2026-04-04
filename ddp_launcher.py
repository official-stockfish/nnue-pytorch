#!/usr/bin/env python3
import sys
import os
import argparse
from ddp_utils import setup_environment


def main():
    # 1. Validate basic usage
    if len(sys.argv) < 2:
        print(
            f"Usage: torchrun {sys.argv[0]} <train_script.py> [args...]",
            file=sys.stderr,
        )
        sys.exit(1)

    target_script = sys.argv[1]
    train_args = sys.argv[2:]

    # 2. Intercept resource arguments silently
    # add_help=False ensures we don't accidentally intercept '-h' meant for train.py
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=-1)

    # args contains our intercepted values; unknown_args contains everything else
    args, unknown_args = parser.parse_known_args(train_args)

    # 3. Apply constraints and get the optimal bounds
    actual_threads, actual_workers = setup_environment(
        requested_threads=args.threads, requested_workers=args.num_workers
    )

    # 4. Reconstruct the command line for the target script
    # We pass the newly calculated limits explicitly to the training script
    final_args = [sys.executable, target_script]
    final_args.extend(["--num-workers", str(actual_workers)])
    final_args.extend(["--threads", str(actual_threads)])
    final_args.extend(unknown_args)

    # 5. Optional diagnostic output from the lead rank
    if os.environ.get("LOCAL_RANK") == "0":
        print(
            f"[Launcher] Intercepted requests: threads={args.threads}, workers={args.num_workers}",
            flush=True,
        )
        print(
            f"[Launcher] Applied constraints : threads={actual_threads}, workers={actual_workers}",
            flush=True,
        )
        print(
            f"[Launcher] Executing target    : {' '.join(final_args[1:])}", flush=True
        )

    # 6. Replace current process with the target python script
    os.execv(sys.executable, final_args)


if __name__ == "__main__":
    main()
