import argparse
import subprocess
import shutil
import sys
from pathlib import Path
import shlex

def run_command(cmd_string):
    """Executes a shell command string and halts execution if it fails."""
    print(f"[TEST_TRAINING_PIPELINE_RUN] Run Command: {cmd_string}\n")
    args = shlex.split(cmd_string)
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed with exit code {e.returncode}")
        sys.exit(e.returncode)

def main():
    DEFAULT_TEST_DIR_STR = "./logs/training/runs/unittests_train_pipeline"
    parser = argparse.ArgumentParser(description="Execute training and serialization pipeline.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading (default 4)")
    parser.add_argument("--device", type=str, default='cpu', help="Compute device (integer x for cuda:x, 'mps' or 'cpu'; default: 'cpu')")
    parser.add_argument("--test-dir", type=str, default=DEFAULT_TEST_DIR_STR, help=f"Directory for test output (default: {DEFAULT_TEST_DIR_STR})")
    parser.add_argument("-y", "--yes", action="store_true", help="Automatically delete existing output directory without prompting")
    args = parser.parse_args()

    # --- 1. Evaluate Device Arguments ---
    try:
        int(args.device)
        is_int = True
    except ValueError:
        is_int = False

    if is_int:
        train_device_arg = f"--gpus={args.device},"
    else:
        train_device_arg = f"--accelerator={args.device}"

    serialize_device_arg = f"--device={args.device}"

    # --- 2. Evaluate Worker Arguments ---
    train_workers_arg = f"--num_workers={args.num_workers}"
    serialize_workers_arg = f"--loader_num_workers={args.num_workers}"

    # --- 3. Directory Management ---
    test_dir = Path(args.test_dir)
    test_dir_str = str(test_dir)

    if test_dir.exists():
        if not test_dir.is_dir():
            print(f"Error: Path {test_dir} exists but is a file, not a directory. Aborting.")
            sys.exit(1)

        if args.yes:
            print(f"Directory {test_dir} exists. Auto-deleting due to -y flag.")
            shutil.rmtree(test_dir)
        else:
            # Default to 'no' if the user just presses Enter
            response = input(f"WARNING: Directory {test_dir} already exists. Delete it? [y/N]: ").strip().lower()
            if response == 'y':
                print(f"Deleting {test_dir}...")
                shutil.rmtree(test_dir)
            else:
                print("Aborting. Directory must be removed before running to preserve version increment logic.")
                sys.exit(1)

    # --- 4. Define Pipeline Commands ---
    pipeline = [
        f"python -u train.py ./.pgo/small.binpack --batch-size 1024 --l1=1024 --features=Full_Threats+HalfKAv2_hm^ --epoch-size 10000 --max_epochs=2 --default_root_dir {test_dir_str} {train_device_arg} {train_workers_arg}",

        f"python -u serialize.py {test_dir_str}/lightning_logs/version_0/checkpoints/last.ckpt {test_dir_str}/lightning_logs/version_0/checkpoints/last.pt --features=Full_Threats+HalfKAv2_hm^ --l1=1024 {serialize_device_arg} {serialize_workers_arg}",

        f"python -u train.py ./.pgo/small.binpack --batch-size 2048 --l1=1024 --features=Full_Threats+HalfKAv2_hm^ --epoch-size 10000 --max_epochs=2 --default_root_dir {test_dir_str} --resume-from-model={test_dir_str}/lightning_logs/version_0/checkpoints/last.pt --validation-size=5000 {train_device_arg} {train_workers_arg}",

        f"python -u train.py ./.pgo/small.binpack --batch-size 2048 --l1=1024 --features=Full_Threats+HalfKAv2_hm^ --epoch-size 10000 --max_epochs=4 --default_root_dir {test_dir_str} --resume-from-checkpoint={test_dir_str}/lightning_logs/version_1/checkpoints/last.ckpt --validation-size=5000 {train_device_arg} {train_workers_arg}",

        f"python -u serialize.py {test_dir_str}/lightning_logs/version_2/checkpoints/last.ckpt {test_dir_str}/lightning_logs/version_2/checkpoints/last.pt --features=Full_Threats+HalfKAv2_hm^ --l1=1024 {serialize_device_arg} {serialize_workers_arg}",

        f"python -u serialize.py {test_dir_str}/lightning_logs/version_2/checkpoints/last.pt {test_dir_str}/lightning_logs/version_2/checkpoints/last.nnue --features=Full_Threats+HalfKAv2_hm^ --l1=1024 {serialize_device_arg} {serialize_workers_arg}",

        f"python -u serialize.py {test_dir_str}/lightning_logs/version_2/checkpoints/last.nnue {test_dir_str}/lightning_logs/version_2/checkpoints/last.nnue --ft_optimize_data=./.pgo/small.binpack --features=Full_Threats+HalfKAv2_hm^ --l1=1024 --ft_optimize --ft_optimize_count=1000 --ft_compression=leb128 {serialize_device_arg} {serialize_workers_arg}",

        f"python -u serialize.py {test_dir_str}/lightning_logs/version_2/checkpoints/last.nnue {test_dir_str}/lightning_logs/version_2/checkpoints/last.nnue --features=Full_Threats+HalfKAv2_hm^ --l1=1024 --ft_compression=leb128 --out-sha {serialize_device_arg} {serialize_workers_arg}"
    ]

    # --- 5. Execute ---
    for cmd in pipeline:
        run_command(cmd)

if __name__ == "__main__":
    main()
