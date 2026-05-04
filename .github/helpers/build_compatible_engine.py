import os
import shutil
import sys
import subprocess

import argparse

from typing import Optional

COMPATIBLE_ENGINE_OWNER = "official-stockfish"
COMPATIBLE_ENGINE_SHA = "1a882efc7fc22b3b16893a406e6060916022fcc4"


def clone_and_build_engine(engine_dest_path: str, repo_info: Optional[str], force_build: bool):
    if repo_info is None:
        owner = COMPATIBLE_ENGINE_OWNER
        sha = COMPATIBLE_ENGINE_SHA
    else:
        parts = repo_info.split("/", 1)
        if (
            len(parts) != 2
            or not parts[0]
            or not parts[1]
            or "/" in parts[1]
        ):
            raise ValueError(
                f"Invalid repo_info {repo_info!r}. Expected format: 'owner/sha'."
            )
        owner, sha = parts

    engine_dir = os.path.dirname(os.path.abspath(engine_dest_path))
    if engine_dir:
        os.makedirs(engine_dir, exist_ok=True)

    tmp_dir = os.path.join(engine_dir, "tmp_stockfish_build")

    # Check existence and handle prompts / CI overrides
    paths_to_check = [p for p in (tmp_dir, engine_dest_path) if os.path.exists(p)]
    if paths_to_check:
        if force_build:
            print(f"--force-build is set. Overwriting existing paths: {', '.join(paths_to_check)}")
        else:
            print(f"Warning: The following paths already exist: {', '.join(paths_to_check)}")
            if not sys.stdin.isatty():
                print("Non-interactive environment detected. Defaulting to 'no'. Skipping build.")
                return
            try:
                ans = input("Do you want to overwrite them? (y/[n]): ")
                if ans.strip().lower() != 'y':
                    print("Skipping build. Proceeding with existing files.")
                    return
            except EOFError:
                print("EOFError: No standard input available. Defaulting to 'no'. Skipping build.")
                return

        # Clean up tmp_dir if we are proceeding with overwrite
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

    repo_url = f"https://github.com/{owner}/Stockfish.git"

    print(f"Cloning {repo_url} into {tmp_dir}...")
    subprocess.run(["git", "clone", repo_url, tmp_dir], check=True)

    print(f"Checking out {sha}...")
    subprocess.run(["git", "checkout", sha], cwd=tmp_dir, check=True)

    print("Building engine (using ARCH=native)...")
    src_dir = os.path.join(tmp_dir, "src")
    subprocess.run(["make", "-j", "build", "ARCH=native"], cwd=src_dir, check=True)

    binary_name = "stockfish.exe" if os.name == "nt" else "stockfish"
    built_binary = os.path.join(src_dir, binary_name)

    if not os.path.exists(built_binary):
        raise RuntimeError("Build completed but binary was not found.")

    print(f"Moving binary to {engine_dest_path}...")
    shutil.copy(built_binary, engine_dest_path)
    shutil.rmtree(tmp_dir)

def main():
    argparser = argparse.ArgumentParser(description="Build a compatible Stockfish engine from a specific GitHub owner/sha.")
    argparser.add_argument(
        "--engine-dest-path",
        type=str,
        required=True,
        help="Path where the compatible engine binary will be placed.",
    )
    argparser.add_argument(
        "--build-engine-from-sha",
        type=str,
        default=None,
        required=False,
        help=f"If given, clones and builds engine from github repository commit given with format `owner/sha` at location given by `--engine-dest-path`. Defaults to ({COMPATIBLE_ENGINE_OWNER}/{COMPATIBLE_ENGINE_SHA}).",
    )
    argparser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        required=False,
        help="Force build and overwrite if paths already exist.",
    )
    args = argparser.parse_args()

    clone_and_build_engine(
        args.engine_dest_path,
        args.build_engine_from_sha,
        args.overwrite,
    )