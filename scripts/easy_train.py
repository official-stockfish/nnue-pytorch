#!/usr/bin/env python3

import time
import sys
import psutil
import re
import subprocess
import importlib
import importlib.metadata
import argparse
import math
import logging

EXITCODE_OK = 0
EXITCODE_MISSING_DEPENDENCIES = 2
EXITCODE_TRAINING_LIKELY_NOT_FINISHED = 3
EXITCODE_TRAINING_NOT_FINISHED = 4

logging.basicConfig()
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)
LOGGER.addHandler(logging.StreamHandler(stream=sys.stdout))
LOGGER.propagate = False


def validate_python_version():
    if sys.version_info >= (3, 7):
        LOGGER.info(f"Found python version {sys.version}. OK.")
        return True
    else:
        LOGGER.error(
            f"Found python version {sys.version} but 3.7 is required. Exiting."
        )
        return False


# Functions for checking external dependencies.


def run_for_version(name):
    process = subprocess.Popen(
        [name, "--version"],
        shell=False,
        bufsize=-1,
        universal_newlines=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    return process.stdout.read()


def validate_cmake():
    success = True
    try:
        out = run_for_version("cmake")
        parts = out.split("\n")[0].split()
        version_str = parts[-1]
        major_version = int(version_str.split(".")[0])
        minor_version = int(version_str.split(".")[1])
        success = (major_version, minor_version) >= (3, 4)
        if success:
            LOGGER.info(f"Found cmake executable version {version_str}. OK.")
        else:
            LOGGER.error(
                f"Found cmake executable version {version_str} but at least 3.4 required. Exiting."
            )
    except:
        success = False
        LOGGER.error("No cmake executable found. Exiting.")

    return success


def validate_make():
    success = True
    try:
        out = run_for_version("make")
        parts = out.split("\n")[0].split()
        version_str = parts[-1]
        major_version = int(version_str.split(".")[0])
        success = major_version >= 3
        if success:
            LOGGER.info(f"Found make executable version {version_str}. OK.")
        else:
            LOGGER.error(
                f"Found make executable version {version_str} but at least 3 required. Exiting."
            )
    except:
        success = False
        LOGGER.error("No make executable found. Exiting.")

    return success


def validate_gcc():
    success = True
    try:
        out = run_for_version("gcc")
        parts = out.split("\n")[0].split()
        for part in parts:
            try:
                version_str = (
                    part  # sometimes there are trailing strings in the version number
                )
                major_version = int(version_str.split(".")[0])
                minor_version = int(version_str.split(".")[1])
                success = (major_version, minor_version) >= (9, 2)
            except:
                continue
        if success:
            LOGGER.info(f"Found gcc executable version {version_str}. OK.")
        else:
            LOGGER.error(
                f"Found gcc executable version {version_str} but at least 9.2 required. Exiting."
            )
    except:
        success = False
        LOGGER.error("No gcc executable found. Exiting.")

    return success


def maybe_int(v):
    try:
        return int(v)
    except:
        return v


class PackageInfo:
    """
    Represents an [installed] python package.
    """

    def __init__(self, name):
        self._spec = importlib.util.find_spec(name)
        self._version_str = None
        self._version_tup = None
        try:
            if self._spec:
                self._version_str = importlib.metadata.version(name)
                self._version_tup = tuple(
                    maybe_int(v) for v in self._version_str.split(".")
                )
        except:
            pass

    @property
    def exists(self):
        return self._spec is not None

    def is_version_at_least(self, desired):
        return self._version_tup and self._version_tup >= desired

    @property
    def version(self):
        return self._version_str


# Functions for checking required python packages.


def validate_asciimatics():
    pkg = PackageInfo("asciimatics")
    if pkg.exists:
        LOGGER.info("Found asciimatics package. OK.")
        return True
    else:
        LOGGER.error(
            "No asciimatics package found. Run `pip install asciimatics`. Exiting."
        )
        return False


def validate_pytorch():
    pkg = PackageInfo("torch")
    if pkg.exists:
        if pkg.is_version_at_least((1, 7)):
            LOGGER.info(f"Found torch version {pkg.version}. OK.")
            from torch import cuda

            if cuda.is_available() and cuda.device_count() > 0:
                LOGGER.info("Found torch with CUDA. OK.")
                return True
            else:
                LOGGER.error(
                    "Found torch without CUDA but CUDA support required. Exiting"
                )
                return False
        else:
            LOGGER.error(
                f"Found torch version {pkg.version} but at least 1.8 required. Exiting."
            )
            return False
    else:
        LOGGER.error(
            "No torch package found. Install at least torch 1.8 with cuda. See https://pytorch.org/. Exiting."
        )
        return False


def validate_pytorchlightning():
    pkg = PackageInfo("lightning")
    if pkg.exists:
        LOGGER.info(f"Found lightning version {pkg.version}. OK.")
        return True
    else:
        LOGGER.error("No lightning found. Run `pip install lightning`. Exiting.")
        return False


def validate_cupy():
    pkg = PackageInfo("cupy")
    if pkg.exists:
        LOGGER.info(f"Found cupy version {pkg.version}. OK.")
        return True
    else:
        LOGGER.error(
            "No cupy found. Install cupy matching cuda version used by pytorch. See https://cupy.dev/. Exiting."
        )
        return False


def validate_gputil():
    pkg = PackageInfo("GPUtil")
    if pkg.exists:
        LOGGER.info(f"Found GPUtil version {pkg.version}. OK.")
        return True
    else:
        LOGGER.error("No GPUtil found. Run `pip install GPUtil`. Exiting.")
        return False


# Validation of required external and package dependencies.


def validate_imports():
    success = True
    success &= validate_asciimatics()
    success &= validate_pytorch()
    success &= validate_pytorchlightning()
    success &= validate_cupy()
    success &= validate_gputil()
    return success


def validate_environment_requirements():
    success = True
    try:
        success &= validate_python_version()
        success &= validate_make()
        success &= validate_cmake()
        success &= validate_gcc()
        success &= validate_imports()
    except Exception as e:
        LOGGER.error(e)
        return False
    return success


# Exit early if the requires packages have not been found
if not validate_environment_requirements():
    sys.exit(EXITCODE_MISSING_DEPENDENCIES)

# Only now import the rest of the required packages
from asciimatics.widgets import (
    Frame,
    Layout,
    Divider,
    Button,
    TextBox,
    Widget,
    VerticalDivider,
    MultiColumnListBox,
    Label,
    PopUpDialog,
)
from asciimatics.scene import Scene
from asciimatics.screen import Screen
from asciimatics.exceptions import ResizeScreenError, StopApplication
from asciimatics.event import KeyboardEvent
from threading import Thread, Lock, Event
import GPUtil
import io
import os
import requests
import zipfile
import shutil
import urllib.request
import urllib.parse
from tqdm.auto import tqdm
from pathlib import Path

# Specify which versions of ordo and c-chess-cli we want.
# We rely on specific well-tested commits because we know exactly what we need.
# repo/branch, commit id
ORDO_GIT = ("michiguel/Ordo", "17eec774f2e4b9fdd2b1b38739f55ea221fb851a")
C_CHESS_CLI_GIT = ("lucasart/c-chess-cli", "6d08fee2e95b259c486b21a886f6911b61f676af")
TIMEOUT = 600.0  # on some systems starting pytorch can be really slow


def terminate_process_on_exit(process):
    """
    Create a watchdog process that awaits the termination of this (calling) process
    and automatically terminates a given process (python's subprocess object) after.

    On Windows this is achieved by a wmic call that is deprecated in windows 10,
    and may not work in windows 11.
        See: https://stackoverflow.com/a/22559493/3763139
             https://superuser.com/a/1299350/388191

    TODO: powershell version
    TODO: linux version
    """

    if sys.platform == "win32":
        try:
            # We cannot execute from string so we write the script to a file.
            # Doesn't do anything if the file already exists.
            with open(".process_watchdog_helper.bat", "x") as file:
                file.write(""":waitforpid
tasklist /nh /fi "pid eq %1" 2>nul | find "%1" >nul
if %ERRORLEVEL%==0 (
    timeout /t 5 /nobreak >nul
    goto :waitforpid
) else (
    wmic process where processid="%2" call terminate >nul
)""")
        except:
            pass

        subprocess.Popen(
            [".process_watchdog_helper.bat", str(os.getpid()), str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif sys.platform == "linux":
        # TODO: this
        pass


# Exits the process forcefully after a specified amount of seconds with a given error code
TUI_SCREEN = None


def schedule_exit(timeout_seconds, errcode):
    def f():
        time.sleep(timeout_seconds)
        LOGGER.info("Performing a scheduled exit.")
        if TUI_SCREEN:
            if sys.platform == "win32":
                TUI_SCREEN.close(restore=True)
            else:
                # We cannot call .close directly because it tries to reset signals...
                # But resetting signals won't work from a non-main thread...
                import curses

                TUI_SCREEN._screen.keypad(0)
                curses.echo()
                curses.nocbreak()
                curses.endwin()

        os._exit(errcode)

    thread = Thread(target=f)
    thread.daemon = True
    thread.start()


if sys.platform == "win32":
    import ctypes

    WINAPI_CreateMutex = ctypes.windll.kernel32.CreateMutexA
    WINAPI_CreateMutex.argtypes = [
        ctypes.wintypes.LPCVOID,
        ctypes.wintypes.BOOL,
        ctypes.c_char_p,
    ]
    WINAPI_CreateMutex.restype = ctypes.wintypes.HANDLE

    WINAPI_WaitForSingleObject = ctypes.windll.kernel32.WaitForSingleObject
    WINAPI_WaitForSingleObject.argtypes = [
        ctypes.wintypes.HANDLE,
        ctypes.wintypes.DWORD,
    ]
    WINAPI_WaitForSingleObject.restype = ctypes.wintypes.DWORD

    WINAPI_ReleaseMutex = ctypes.windll.kernel32.ReleaseMutex
    WINAPI_ReleaseMutex.argtypes = [ctypes.wintypes.HANDLE]
    WINAPI_ReleaseMutex.restype = ctypes.wintypes.BOOL

    WINAPI_CloseHandle = ctypes.windll.kernel32.CloseHandle
    WINAPI_CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
    WINAPI_CloseHandle.restype = ctypes.wintypes.BOOL

    class SystemWideMutex:
        def __init__(self, name):
            # \ is a reserved character so we have to convert them to / to be recognized as
            # directory delimiters
            # encode as utf-8 because LPCSTR is bytes not str
            self.name = str(os.path.abspath(name)).replace("\\", "/").encode("utf-8")
            self.acquired = False
            self.file = open(self.name, "a+")
            self.handle = WINAPI_CreateMutex(None, False, self.name)
            if not self.handle:
                raise ctypes.WinError()

        def acquire(self):
            ret = WINAPI_WaitForSingleObject(self.handle, 0xFFFFFFFF)
            if ret in (0, 0x80):
                # 0 - normally acquired
                # 0x80 - acquired by other process terminating
                self.acquired = True
                return True
            else:
                raise ctypes.WinError()

        def release(self):
            ret = WINAPI_ReleaseMutex(self.handle)
            if not ret:
                raise ctypes.WinError()
            self.acquired = False

        def close(self):
            if self.handle is None:
                return

            self.file.close()

            ret = WINAPI_CloseHandle(self.handle)
            if not ret:
                raise ctypes.WinError()

            try:
                os.remove(self.name)
            except:
                pass

            self.handle = None

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release()
            self.close()
else:
    import fcntl

    class SystemWideMutex:
        def __init__(self, name):
            self.name = name
            self.acquired = False
            self.file = open(self.name, "a+")

        def acquire(self):
            fcntl.lockf(self.file, fcntl.LOCK_EX)
            self.acquired = True

        def release(self):
            fcntl.lockf(self.file, fcntl.LOCK_UN)
            self.acquired = False

        def close(self):
            if self.file is None:
                return
            os.unlink(self.name)
            self.file.close()
            self.file = None

        def __del__(self):
            self.close()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.release()


class DecayingRunningAverage:
    """
    Represents an average of a list of values with exponential decay of old values.
    Every added value has weight of decay**n, where n is the distance from the
    last element. For the last added element n==0.
    """

    def __init__(self, decay=0.995):
        self._decay = decay
        self._total = 0.0
        self._count = 0.0

    @property
    def decay(self):
        return self._decay

    @property
    def value(self):
        try:
            return self._total / self._count
        except:
            return float("NaN")

    def update(self, value):
        """
        Adds a new value at the end of the implicit running average list and
        updates the counters to reflect the change in the running average.
        """
        self._total = self._total * self._decay + value
        self._count = self._count * self._decay + 1.0


class SystemResources:
    """
    Holds information about the usage of system resources at a time point of creation.
    This includes GPU, CPU, and memory usage.
    """

    def __init__(self):
        self._gpus = dict()
        for gpu in GPUtil.getGPUs():
            self._gpus[gpu.id] = gpu
        self._cpu_usage = psutil.cpu_percent() / 100.0
        mem = psutil.virtual_memory()
        self._ram_usage_mb = mem[3] // (1024 * 1024)
        self._ram_max_mb = mem[0] // (1024 * 1024)

    @property
    def gpus(self):
        return self._gpus

    @property
    def cpu_usage(self):
        return self._cpu_usage

    @property
    def ram_usage_mb(self):
        return self._ram_usage_mb

    @property
    def ram_max_mb(self):
        return self._ram_max_mb


class SystemResourcesMonitor(Thread):
    """
    Periodically queries system resources.
    Runs as a daemon so does not need to be cleaned up.
    """

    def __init__(self, period_seconds):
        super().__init__()

        self._period_seconds = period_seconds
        self._mutex = Lock()
        self._stop_event = Event()

        self._running = True
        self._update()

        self.daemon = True
        self.start()

    def _update(self):
        self._resources = SystemResources()

    def run(self):
        while self._running:
            self._mutex.acquire()
            try:
                self._update()
            finally:
                self._mutex.release()
            self._stop_event.wait(timeout=self._period_seconds)

    @property
    def resources(self):
        """
        Returns the most recent system resources measurement.
        """
        self._mutex.acquire()
        try:
            return self._resources
        finally:
            self._mutex.release()

    def stop(self):
        self._running = False
        self._stop_event.set()


def find_latest_checkpoint(root_dir):
    """
    Recursively searches the specified directory for
    the .ckpt file with the latest creation date.
    """
    ckpts = [file for file in Path(root_dir).rglob("*.ckpt")]
    if not ckpts:
        return None

    return str(max(ckpts, key=lambda p: p.stat().st_ctime_ns))


class OrdoEntry:
    """
    Represents a single entry in an ordo file.
    Expects players to be named after network paths, if the form experiment_path/run_{}/nn-epoch{}.nnue
    """

    NET_PATTERN = re.compile(r".*?run_(\d+).*?nn-epoch(\d+)\.nnue")

    def __init__(
        self,
        line=None,
        network_path=None,
        elo=None,
        elo_error=None,
        run_id=None,
        epoch=None,
    ):
        if line:
            fields = line.split()
            self._network_path = fields[1]
            self._elo = float(fields[3])
            self._elo_error = float(fields[4])
            net_parts = OrdoEntry.NET_PATTERN.search(self._network_path)
            self._run_id = int(net_parts[1])
            self._epoch = int(net_parts[2])
        else:
            self._network_path = network_path
            self._elo = elo
            self._elo_error = elo_error
            self._run_id = run_id
            self._epoch = epoch

    @property
    def network_path(self):
        return self._network_path

    @property
    def run_id(self):
        return self._run_id

    @property
    def epoch(self):
        return self._epoch

    @property
    def elo(self):
        return self._elo

    @property
    def elo_error(self):
        return self._elo_error


def find_best_checkpoint(root_dir):
    """
    Recursively searches the specified directory the best
    .ckpt file as determined by an ordo output file that must be
    present under the path os.path.join(root_dir, 'ordo.out').
    The path to the checkpoint must have 'nn-epoch' in it,
    other checkpoints are not considered.

    Returns None if the ordo file does not exist or
    no suitable checkpoint has been found.
    """
    ckpts = [str(file) for file in Path(root_dir).rglob("*.ckpt")]
    nnues = [str(file) for file in Path(root_dir).rglob("*.nnue")]
    ordo_file_path = os.path.join(root_dir, "ordo.out")

    with open(ordo_file_path, "r") as ordo_file:
        entries = []
        lines = ordo_file.readlines()
        for line in lines:
            if "nn-epoch" in line:
                try:
                    entries.append(OrdoEntry(line=line))
                except:
                    pass

    entries.sort(key=lambda x: -x.elo + x.elo_error)

    run_id = entries[0].run_id
    epoch = entries[0].epoch
    for ckpt in ckpts:
        if f"run_{run_id}" in ckpt and f"epoch={epoch}" in ckpt:
            return ckpt
    # fallback to .nnue if no checkpoint file
    for nnue in nnues:
        if f"run_{run_id}" in nnue and f"nn-epoch{epoch}" in nnue:
            return nnue

    return None


# A global instance of the resource monitor.
# There is no need to have more than one.
RESOURCE_MONITOR = SystemResourcesMonitor(2)

# A regex pattern for a float number.
NUMERIC_CONST_PATTERN = "[-+]?(?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?"


class TrainingRun(Thread):
    """
    Manages a single pytorch training run.
    Starts it as a subprocess.
    Provides information about the current state of training.
    Runs as a separate thread and must be stopped before exiting.
    """

    # The regex pattern for extracting information from the pytorch lightning's tqdm process bar output
    ITERATION_PATTERN = re.compile(
        f"Epoch (\\d+).*?(\\d+)/(\\d+).*?({NUMERIC_CONST_PATTERN})it/s, loss=({NUMERIC_CONST_PATTERN})"
    )

    def __init__(
        self,
        gpu_id,
        run_id,
        nnue_pytorch_directory,
        training_datasets,
        validation_datasets,
        num_data_loader_threads,
        num_pytorch_threads,
        num_epochs,
        batch_size,
        random_fen_skipping,
        smart_fen_skipping,
        wld_fen_skipping,
        early_fen_skipping,
        features,
        lr,
        gamma,
        lambda_,
        network_save_period,
        save_last_network,
        seed,
        root_dir,
        epoch_size,
        validation_size,
        start_from_model=None,
        resume_training=False,
        start_lambda=None,
        end_lambda=None,
        additional_args=[],
    ):
        super().__init__()
        self._gpu_id = gpu_id
        self._run_id = run_id

        # Use abspaths because we will be running the script with a different cwd
        self._nnue_pytorch_directory = os.path.abspath(nnue_pytorch_directory)
        self._training_datasets = [os.path.abspath(d) for d in training_datasets]
        self._validation_datasets = [os.path.abspath(d) for d in validation_datasets]
        self._num_data_loader_threads = num_data_loader_threads
        self._num_pytorch_threads = num_pytorch_threads
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._random_fen_skipping = random_fen_skipping
        self._smart_fen_skipping = smart_fen_skipping
        self._wld_fen_skipping = wld_fen_skipping
        self._early_fen_skipping = early_fen_skipping
        self._features = features
        self._lr = lr
        self._gamma = gamma
        self._lambda = lambda_
        self._start_lambda = start_lambda
        self._end_lambda = end_lambda
        self._network_save_period = network_save_period
        self._save_last_network = save_last_network
        self._seed = seed
        self._root_dir = os.path.abspath(root_dir)
        self._epoch_size = epoch_size
        self._validation_size = validation_size
        self._start_from_model = start_from_model
        self._resume_training = resume_training
        self._additional_args = additional_args

        # State for the status updates
        self._current_step_in_epoch = None
        self._num_steps_in_epoch = None
        self._current_epoch = None
        self._current_loss = None
        self._momentary_iterations_per_second = None
        self._smooth_iterations_per_second = DecayingRunningAverage()
        self._has_finished = False
        self._has_started = False
        self._networks = []
        self._process = None
        self._running = False
        self._error = None

        # For speed calculation
        self._last_time = None
        self._last_step = None

    def _get_stringified_args(self):
        args = [
            f"--num-workers={self._num_data_loader_threads}",
            f"--threads={self._num_pytorch_threads}",
            f"--max_epoch={self._num_epochs}",
            f"--batch-size={self._batch_size}",
            f"--random-fen-skipping={self._random_fen_skipping}",
            f"--early-fen-skipping={self._early_fen_skipping}",
            f"--gpus={self._gpu_id},",
            f"--features={self._features}",
            f"--lr={self._lr}",
            f"--gamma={self._gamma}",
            f"--lambda={self._lambda}",
            f"--network-save-period={self._network_save_period}",
            f"--save-last-network={self._save_last_network}",
            f"--seed={self._seed}",
            f"--epoch-size={self._epoch_size}",
            f"--validation-size={self._validation_size}",
            f"--default_root_dir={self._root_dir}",
        ]

        if self._smart_fen_skipping:
            args.append("--smart-fen-skipping")
        else:
            args.append("--no-smart-fen-skipping")

        if not self._wld_fen_skipping:
            args.append("--no-wld-fen-skipping")

        if self._start_lambda:
            args.append(f"--start-lambda={self._start_lambda}")

        if self._end_lambda:
            args.append(f"--end-lambda={self._end_lambda}")

        resumed = False
        if self._resume_training:
            ckpt_path = find_latest_checkpoint(self._root_dir)
            if ckpt_path:
                args.append(f"--resume_from_checkpoint={ckpt_path}")
                resumed = True

        if self._start_from_model and not resumed:
            args.append(f"--resume-from-model={self._start_from_model}")

        for arg in self._additional_args:
            args.append(arg)

        for dataset in self._training_datasets:
            args.append(dataset)

        for dataset in self._validation_datasets:
            args.append(f"--validation-data={dataset}")

        return args

    def run(self):
        if self._resume_training and os.path.exists(
            os.path.join(self._root_dir, "training_finished")
        ):
            self._has_started = True
            self._has_finished = True
            self._running = False
            return

        self._running = True

        cmd = [sys.executable, "train.py"] + self._get_stringified_args()
        LOGGER.info(f"Running training with command: {cmd}")
        LOGGER.info(f"Also known as: {' '.join(cmd)}")
        LOGGER.info(f"Running in working directory: {self._nnue_pytorch_directory}")
        self._process = subprocess.Popen(
            cmd,
            cwd=self._nnue_pytorch_directory,
            shell=False,
            bufsize=-1,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        terminate_process_on_exit(self._process)

        reader = io.TextIOWrapper(self._process.stdout)
        while self._process.poll() is None and self._running:
            if not self._running:
                break

            # \r is properly recognized as a newline delimiter so we can just read by lines
            line = reader.readline().strip()
            if not self._has_finished:
                try:
                    matches = TrainingRun.ITERATION_PATTERN.search(line)
                    if matches:
                        self._current_epoch = int(matches.group(1))
                        self._current_step_in_epoch = int(matches.group(2))
                        self._num_steps_in_epoch = int(matches.group(3))

                        # There appears to be a pytorch lightning bug where it displays
                        # negative speed when running from checkpoint. So we work around this
                        # by computing our own speed.
                        # Only update every 10 steps to avoid the it/s to blow up.
                        # (With a higher update frequence might be affected by IO caching)
                        curr_step = self._current_step_in_epoch
                        if curr_step == self._last_step:
                            continue

                        curr_time = time.perf_counter_ns()
                        if self._last_time is None or curr_step < self._last_step:
                            self._last_time = curr_time
                            self._last_step = curr_step
                            continue

                        # self._momentary_iterations_per_second = float(matches.group(4))
                        if curr_step % 10 == 0:
                            self._momentary_iterations_per_second = (
                                curr_step - self._last_step
                            ) / ((curr_time - self._last_time) / 1e9)
                            self._smooth_iterations_per_second.update(
                                self._momentary_iterations_per_second
                            )
                            self._last_time = curr_time
                            self._last_step = curr_step

                        self._current_loss = float(matches.group(5))
                        self._has_started = True

                        # Provide some output for the cli interface.
                        if self._current_step_in_epoch % 100 == 0:
                            LOGGER.info(line)
                    else:
                        # Actually this is where most of the errors from pytorch must be handled.
                        LOGGER.info(line)
                except:
                    # Usually errors. Aside from that all output should be catched above. We want these logged.
                    LOGGER.info(line)
                    pass

                if "CUDA_ERROR_OUT_OF_MEMORY" in line or "CUDA out of memory" in line:
                    self._process.terminate()
                    self._error = "Cuda out of memory error."
                    break

        # Since _num_steps_in_epochs includes validation steps, that we cannot actually catch
        # and we don't know how to account for validation steps, and the trainer exits silently,
        # we can just estimate whether it finished with a success by using some margin...
        # NOTE: We still cannot catch when the trainer exits with no work, which for example
        #       happens when resuming from a checkpoint at the end of training.
        if (
            self._has_started
            and self._current_epoch == self._num_epochs - 1
            and self._current_step_in_epoch >= self._num_steps_in_epoch * 0.9
        ):
            self._has_finished = True

        if self._running and not self._has_finished:
            if not self._error:
                self._error = "Unknown error occured."
            LOGGER.warning(f"Training run {self._run_id} exited unexpectedly.")
            LOGGER.error(f"Error: {self._error}")
        else:
            LOGGER.info(f"Training run {self._run_id} finished.")

        self._has_started = True
        self._running = False

    def stop(self):
        self._running = False
        self.join()
        if self._process:
            self._process.terminate()
            self._process.wait()

    @property
    def gpu_id(self):
        return self._gpu_id

    @property
    def run_id(self):
        return self._run_id

    @property
    def current_step_in_epoch(self):
        return self._current_step_in_epoch

    @property
    def current_epoch(self):
        return self._current_epoch

    @property
    def num_steps_in_epoch(self):
        return self._num_steps_in_epoch

    @property
    def num_epochs(self):
        return self._num_epochs

    @property
    def current_loss(self):
        return self._current_loss

    @property
    def momentary_iterations_per_second(self):
        return self._momentary_iterations_per_second

    @property
    def smooth_iterations_per_second(self):
        return self._smooth_iterations_per_second.value

    @property
    def has_finished(self):
        return self._has_finished

    @property
    def has_started(self):
        return self._has_started

    @property
    def networks(self):
        return self._networks

    @property
    def is_running(self):
        return self._running

    @property
    def error(self):
        return self._error

    @property
    def batch_size(self):
        return self._batch_size


def requests_get_content(url, *args, **kwargs):
    try:
        result = requests.get(url, *args, **kwargs)
        result.raise_for_status()
        return result.content
    except Exception:
        raise Exception(f"GET request to {url} failed")


def get_zipfile_members_strip_common_prefix(zipfile):
    """
    Removes a common prefix from zipfile entries.
    So for example will remove the top-level directory.
    """
    parts = []
    for name in zipfile.namelist():
        if not name.endswith("/"):
            parts.append(name.split("/")[:-1])
    offset = len("/".join(os.path.commonprefix(parts)) + "/")
    for zipinfo in zipfile.infolist():
        name = zipinfo.filename
        if len(name) > offset:
            zipinfo.filename = name[offset:]
            yield zipinfo


def git_download_branch_or_commit(directory, repo, branch_or_commit):
    """
    Github proves an API to download zips of specific commits, so
    we don't need to use git clone.
    """
    url = f"http://github.com/{repo}/zipball/{branch_or_commit}"
    zipped_content = requests_get_content(url, timeout=TIMEOUT)
    zipped_input = zipfile.ZipFile(io.BytesIO(zipped_content), mode="r")
    zipped_input.extractall(
        directory, get_zipfile_members_strip_common_prefix(zipped_input)
    )


# Utility functions for dependency setup and executable location.


def make_ordo_executable_path(directory):
    path = os.path.join(directory, "ordo")
    if sys.platform == "win32":
        path += ".exe"
    return path


def is_ordo_setup(directory):
    try:
        ordo_path = make_ordo_executable_path(directory)
        with subprocess.Popen(
            [ordo_path, "--help"], stdout=subprocess.DEVNULL
        ) as process:
            if process.wait(timeout=TIMEOUT):
                return False
            return True
    except:
        return False


def setup_ordo(directory):
    if is_ordo_setup(directory):
        LOGGER.info(f"Ordo already setup in {directory}")
        return

    LOGGER.info(f"Setting up ordo in {directory}.")
    git_download_branch_or_commit(directory, *ORDO_GIT)
    if sys.platform == "win32":
        # need to append -DMINGW
        # ugly hack for a dumb makefile
        with open(os.path.join(directory, "Makefile"), "r") as makefile:
            lines = makefile.readlines()
            for i, line in enumerate(lines):
                if line.startswith("CFLAGS"):
                    lines.insert(i + 1, "CFLAGS += -DMINGW\n")
                    break

        with open(os.path.join(directory, "Makefile"), "w") as makefile:
            makefile.write("".join(lines))

    with subprocess.Popen(["make"], cwd=directory) as process:
        if process.wait():
            raise Exception("Ordo compilation failed.")

    if not is_ordo_setup(directory):
        raise Exception("Ordo does not work.")


def make_c_chess_cli_executable_path(directory):
    path = os.path.join(directory, "c-chess-cli")
    if sys.platform == "win32":
        path += ".exe"
    return path


def is_c_chess_cli_setup(directory):
    try:
        path = make_c_chess_cli_executable_path(directory)
        with subprocess.Popen([path, "-version"], stdout=subprocess.DEVNULL) as process:
            if process.wait(timeout=TIMEOUT):
                return False
            return True
    except:
        return False


def setup_c_chess_cli(directory):
    if is_c_chess_cli_setup(directory):
        LOGGER.info(f"c-chess-cli already setup in {directory}")
        return

    LOGGER.info(f"Setting up c-chess-cli in {directory}.")
    git_download_branch_or_commit(directory, *C_CHESS_CLI_GIT)

    with open(os.path.join(directory, "make.py"), "r") as makefile:
        lines = makefile.readlines()
        for i, line in enumerate(lines):
            if line.startswith("version = "):
                lines[i] = f"version = 'easy_train_custom_{C_CHESS_CLI_GIT[1]}'\n"

    with open(os.path.join(directory, "make.py"), "w") as makefile:
        makefile.write("".join(lines))

    with subprocess.Popen(
        [sys.executable, "make.py", "-c", "gcc"], cwd=directory
    ) as process:
        if process.wait():
            raise Exception("c-chess-cli compilation failed.")

    if not is_c_chess_cli_setup(directory):
        raise Exception("c-chess-cli does not work")


def make_stockfish_executable_path(directory):
    path = os.path.join(directory, "src/stockfish")
    if sys.platform == "win32":
        path += ".exe"
    return path


def is_stockfish_setup(directory):
    try:
        path = make_stockfish_executable_path(directory)
        with subprocess.Popen([path, "compiler"], stdout=subprocess.DEVNULL) as process:
            if process.wait(timeout=TIMEOUT):
                return False
            return True
    except:
        return False


def setup_stockfish(directory, repo, branch_or_commit, arch, threads=1):
    if is_stockfish_setup(directory):
        LOGGER.info(f"Stockfish already setup in {directory}.")
        return

    LOGGER.info(f"Setting up stockfish in {directory}.")
    git_download_branch_or_commit(directory, repo, branch_or_commit)

    srcdir = os.path.join(directory, "src")
    env = os.environ.copy()
    if sys.platform == "win32":
        env["MSYSTEM"] = "MINGW64"

    with subprocess.Popen(
        ["make", "build", f"ARCH={arch}", f"-j{threads}"], cwd=srcdir, env=env
    ) as process:
        if process.wait():
            raise Exception(f"stockfish {repo}/{branch_or_commit} compilation failed")

    if not is_stockfish_setup(directory):
        raise Exception(f"stockfish {repo}/{branch_or_commit} does not work")


def is_nnue_pytorch_setup(directory):
    try:
        with subprocess.Popen(
            [sys.executable, "data_loader/__init__.py"], cwd=directory
        ) as process:
            if process.wait(timeout=TIMEOUT):
                return False
            return True
    except:
        return False


def setup_nnue_pytorch(directory, repo, branch_or_commit):
    if is_nnue_pytorch_setup(directory):
        LOGGER.info(f"nnue-pytorch already setup in {directory}")
        return

    LOGGER.info(f"Setting up nnue-pytorch in {directory}")
    git_download_branch_or_commit(directory, repo, branch_or_commit)

    command = []
    if sys.platform == "linux":
        command += ["sh"]
    # It's a .bat file made for windows but works on linux too.
    # Just needs to be called with sh.
    command += [os.path.join(directory, "compile_data_loader.bat")]
    with subprocess.Popen(command, cwd=directory) as process:
        if process.wait():
            raise Exception(
                f"nnue-pytorch {repo}/{branch_or_commit} data loader compilation failed"
            )

    if not is_nnue_pytorch_setup(directory):
        raise Exception("Incorrect nnue-pytorch setup or timeout.")


class CChessCliRunningTestEntry:
    """
    Represents a single line of output from the run_games.py
    (which forwards c-chess-cli output) during network testing process.
    Calculates additional match statistics.
    """

    LINE_PATTERN = re.compile(
        r"Score.*?run_(\d+).*?nn-epoch(\d+)\.nnue:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*"
    )

    def __init__(self, line=None):
        fields = CChessCliRunningTestEntry.LINE_PATTERN.search(line)
        self._line = line
        self._run_id = int(fields[1])
        self._epoch = int(fields[2])
        self._losses = int(fields[3])  # from base perspective so reversed
        self._wins = int(fields[4])
        self._draws = int(fields[5])

    @property
    def run_id(self):
        return self._run_id

    @property
    def epoch(self):
        return self._epoch

    @property
    def wins(self):
        return self._wins

    @property
    def draws(self):
        return self._draws

    @property
    def losses(self):
        return self._losses

    @property
    def total_games(self):
        return self._wins + self._draws + self._losses

    @property
    def performance(self):
        return (self._wins + self._draws * 0.5) / self.total_games

    def _elo(self, x):
        epsilon = 1e-3
        x = max(x, epsilon)
        x = min(x, 1 - epsilon)
        return -400 * math.log10(1 / x - 1)

    @property
    def elo(self):
        return self._elo(self.performance)

    @property
    def elo_error_95(self):
        return 400 / math.sqrt(self.total_games)

    @property
    def line(self):
        return self._line


class NetworkTesting(Thread):
    """
    Manages the network testing process.
    Encapsulates run_games.py.
    Provides information about the current set of networks and their results.
    Provides information about the currently ongoing tests.
    Provides information about the current ongoing network conversions.
    Runs as a separate thread and must be stopped before exiting.
    """

    def __init__(
        self,
        nnue_pytorch_directory,
        root_dir,
        num_parallel_games=4,
        explore_factor=1.5,
        book_file_path="",
        time_per_game=None,
        time_increment_per_move=None,
        nodes_per_move=1000,
        hash=8,
        games_per_round=200,
        ordo_exe=None,
        c_chess_cli_exe=None,
        stockfish_base_exe=None,
        stockfish_test_exe=None,
        features=None,
        active=True,
        additional_args=[],
    ):
        super().__init__()

        self._nnue_pytorch_directory = os.path.abspath(nnue_pytorch_directory)
        self._root_dir = os.path.abspath(root_dir)
        self._num_parallel_games = num_parallel_games
        self._explore_factor = explore_factor
        self._book_file_path = os.path.abspath(book_file_path)
        self._time_per_game = time_per_game
        self._time_increment_per_move = time_increment_per_move
        self._nodes_per_move = nodes_per_move
        self._hash = hash
        self._games_per_round = games_per_round
        self._ordo_exe = os.path.abspath(ordo_exe) if ordo_exe else ordo_exe
        self._c_chess_cli_exe = os.path.abspath(c_chess_cli_exe)
        self._stockfish_base_exe = os.path.abspath(stockfish_base_exe)
        self._stockfish_test_exe = os.path.abspath(stockfish_test_exe)
        self._features = features
        self._active = active
        self._additional_args = additional_args

        # State for status management
        self._results = []
        self._running = False
        self._process = None
        self._current_test = None
        self._current_convert = None
        self._error = None
        self._has_finished = False  # currently never finishes
        self._has_started = False

        self._mutex = Lock()

    def _get_stringified_args(self):
        args = [
            self._root_dir,
            f"--concurrency={self._num_parallel_games}",
            f"--explore_factor={self._explore_factor}",
            f"--c_chess_exe={self._c_chess_cli_exe}",
            f"--stockfish_base={self._stockfish_base_exe}",
            f"--stockfish_test={self._stockfish_test_exe}",
            f"--book_file_name={self._book_file_path}",
            f"--hash={self._hash}",
            f"--games_per_round={self._games_per_round}",
            f"--features={self._features}",
        ]

        if self._time_per_game:
            args.append(f"--time_per_game={self._time_per_game}")

        if self._time_increment_per_move:
            args.append(f"--time_increment_per_move={self._time_increment_per_move}")

        if self._nodes_per_move:
            args.append(f"--nodes_per_move={self._nodes_per_move}")

        if self._ordo_exe:
            (args.append(f"--ordo_exe={self._ordo_exe}"),)

        for arg in self._additional_args:
            args.append(arg)

        return args

    def get_status_string(self):
        self._mutex.acquire()
        try:
            if not self._active:
                return "Network testing inactive."
            elif self._has_finished:
                return "Network testing finished."
            elif not self._has_started:
                return "Starting testing process..."
            elif not self._running:
                lines = ["Network testing has exited unexpectedly."]
                if self._error:
                    lines.append(f"Error: {self._error}")
                return "\n".join(lines)
            elif self._current_convert is not None:
                lines = [
                    "Converting network...",
                    f"Run  : {self._current_convert[0]}",
                    f"Epoch: {self._current_convert[1]}",
                ]
                return "\n".join(lines)
            elif self._current_test is not None:
                perf_pct = int(round(self._current_test.performance * 100))
                cpu_usage = RESOURCE_MONITOR.resources.cpu_usage
                lines = [
                    f"CPU load: {cpu_usage * 100:0.1f}%",
                    f"Testing run {self._current_test.run_id} epoch {self._current_test.epoch}",
                    f"+{self._current_test.wins}={self._current_test.draws}-{self._current_test.losses} [{perf_pct:0.1f}%] ({self._current_test.total_games}/{self._games_per_round})",
                    f"{self._current_test.elo:0.1f}±{self._current_test.elo_error_95:0.1f} Elo",
                ]
                return "\n".join(lines)
            else:
                return "Waiting for networks..."
        finally:
            self._mutex.release()

    def run(self):
        if not self._active:
            self._running = False
            return

        self._running = True

        cmd = [sys.executable, "run_games.py"] + self._get_stringified_args()
        LOGGER.info(f"Running network testing with command: {cmd}")
        LOGGER.info(f"Also known as: {' '.join(cmd)}")
        LOGGER.info(f"Running in working directory: {self._nnue_pytorch_directory}")
        self._process = subprocess.Popen(
            cmd,
            cwd=self._nnue_pytorch_directory,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        terminate_process_on_exit(self._process)

        try:
            reader = io.TextIOWrapper(self._process.stdout)
            while self._process.poll() is None and self._running:
                if not self._running:
                    break
                line = reader.readline().strip()

                self._mutex.acquire()
                try:
                    if not line.startswith("Score of"):
                        LOGGER.info(line)

                    if line.startswith("Finished running ordo."):
                        self._update_results_from_ordo_file(self._get_ordo_file_path())
                    elif line.startswith("Score of"):
                        try:
                            self._current_test = CChessCliRunningTestEntry(line=line)
                            if self._current_test.total_games % 100 == 0:
                                LOGGER.info(self._current_test.line)
                            self._current_convert = None
                        except:
                            self._current_test = None
                    elif line.startswith("Converting"):
                        fields = OrdoEntry.NET_PATTERN.search(line)
                        try:
                            self._current_convert = (fields[1], fields[2])
                            self._current_test = None
                            LOGGER.info(
                                f"Converting network epoch {self._current_convert[0]}, run id {self._current_convert[1]}"
                            )
                        except:
                            self._current_convert = None
                    elif line.startswith("Error running match!"):
                        self._process.terminate()
                        self._error = "Error running matches."
                        break
                    else:
                        self._current_test = None

                    self._has_started = True
                except:
                    LOGGER.info(line)
                finally:
                    self._mutex.release()
        except:
            self._process.terminate()
            self._process.wait()

        if self._running and not self._has_finished:
            LOGGER.warning("Network testing exited unexpectedly.")
            if not self._error:
                self._error = "Unknown error occured."
            LOGGER.error(f"Error: {self._error}")

        LOGGER.info("Network testing finished.")

        self._has_started = True
        self._running = False

    def stop(self):
        self._running = False
        if self._process is not None:
            self._process.terminate()
            self._process.wait()

    def _get_ordo_file_path(self):
        return os.path.join(self._root_dir, "ordo.out")

    def _update_results_from_ordo_file(self, ordo_file_path):
        new_results = []

        try:
            with open(ordo_file_path, "r") as ordo_file:
                lines = ordo_file.readlines()
                # Print the first few lines for the CLI interface.
                for line in lines[:7]:
                    LOGGER.info(line.strip())
                for line in lines:
                    try:
                        entry = OrdoEntry(line=line)
                        new_results.append(entry)
                    except:
                        pass
            self._results = new_results
        except:
            pass

    def get_ordered_results(self):
        return list(sorted(self._results, key=lambda x: -x.elo))

    @property
    def has_finished(self):
        return self._has_finished

    @property
    def is_running(self):
        return self._running

    @property
    def is_active(self):
        return self._active


def duration_string_from_seconds(seconds):
    second = int(seconds) % 60
    minute = int(seconds) // 60 % 60
    hour = int(seconds) // 3600
    return f"{hour}:{minute:02}:{second:02}"


def duration_string_from_seconds_compact(seconds):
    second = int(seconds) % 60
    minute = int(seconds) // 60 % 60
    hour = int(seconds) // 3600
    if hour > 0:
        return f"~{hour}h"
    elif minute > 0:
        return f"~{minute}m"
    else:
        return f"~{second}s"


class TrainerRunsWidget(Widget):
    """
    Displays information about the assigned training run.
    """

    def __init__(self, runs, name=None):
        super().__init__(name)

        self._runs = list(sorted(runs, key=lambda x: (x.gpu_id, x.run_id)))

        self._selected_index = 0

    def add_run(self, run):
        self._runs.append(run)
        self._runs = list(sorted(runs, key=lambda x: (x.gpu_id, x.run_id)))

    def required_height(self, offset, w):
        # A special value indicating that it should use the whole column.
        return -135792468

    def reset(self):
        pass

    def _clear_area(self):
        colour, attr, background = self._frame.palette["field"]

        height = self._h
        width = self._w - self._offset

        for i in range(height):
            self._frame.canvas.print_at(
                " " * width,
                self._x + self._offset,
                self._y + i,
                colour,
                attr,
                background,
            )

    def _get_gpu_usage(self, gpu_ids):
        gpus = RESOURCE_MONITOR.resources.gpus
        by_gpu_id = dict()
        for gpu_id in gpu_ids:
            if gpu_id in gpus:
                gpu = gpus[gpu_id]
                by_gpu_id[gpu_id] = {
                    "compute_pct": int(gpu.load * 100),
                    "memory_mb": int(gpu.memoryUsed),
                    "max_memory_mb": int(gpu.memoryTotal),
                }
        return by_gpu_id

    def _get_unique_gpu_ids(self):
        ids = set()
        for run in self._runs:
            ids.add(run.gpu_id)
        return list(ids)

    def _make_run_text(self, run):
        # TODO: Some output for the logger.
        #       Right now only the progress bar in the training run is printed occasionally.
        if run.has_finished:
            loss = run.current_loss or "unknown"
            return [f"  Run {run.run_id} - Completed; Loss: {loss}"]
        elif not run.has_started:
            return (f"  Run {run.run_id} - Starting...",)
        elif not run.is_running:
            lines = [f"  Run {run.run_id} - Exited unexpectedly."]
            if run.error:
                lines += [f"    Error: {run.error}"]
            return lines
        else:
            try:
                width = self._w - self._offset
                loss = run.current_loss
                epoch = run.current_epoch
                max_epoch = run.num_epochs - 1
                step_in_epoch = run.current_step_in_epoch
                max_step = run.num_steps_in_epoch - 1
                speed = run.smooth_iterations_per_second
                speed_knps = run.smooth_iterations_per_second * run.batch_size / 1e3

                total_steps = run.num_epochs * run.num_steps_in_epoch
                step = epoch * run.num_steps_in_epoch + step_in_epoch
                complete_pct = step / total_steps * 100
                eta_seconds = (total_steps - step) / speed
                eta_str = duration_string_from_seconds_compact(eta_seconds)

                return [
                    f"  Run {run.run_id} - {complete_pct:0.2f}% [ETA {eta_str}]",
                    f"    Speed: {speed:0.1f}it/s; {speed_knps:0.0f}kpos/s",
                    f"    Epoch: {epoch}/{max_epoch}; Step: {step_in_epoch}/{max_step}",
                    f"    Loss: {loss}",
                ]
            except:
                return [f"  Run {run.run_id} - Waiting for enough data to display..."]

    def _make_gpu_text(self, gpu_id, gpu_usage):
        # TODO: Some output for the logger
        if gpu_id in gpu_usage:
            gpu_compute_pct = gpu_usage[gpu_id]["compute_pct"]
            gpu_memory_mb = gpu_usage[gpu_id]["memory_mb"]
            gpu_max_memory_mb = gpu_usage[gpu_id]["max_memory_mb"]
            return f"GPU {gpu_id} - Usage: {gpu_compute_pct}% {gpu_memory_mb}MB/{gpu_max_memory_mb}MB "
        else:
            return f"GPU {gpu_id}"

    def update(self, frame_no):
        # TODO: scrolling

        self._clear_area()

        if self._has_focus:
            if self._selected_index is None:
                self._selected_index = 0
        else:
            self._selected_index = None

        if len(self._runs) <= 0:
            return

        height = self._h
        width = self._w - self._offset
        curr_line = 0
        prev_gpu_id = None

        gpu_usage = self._get_gpu_usage(self._get_unique_gpu_ids())
        for i, run in enumerate(self._runs):
            if curr_line >= height:
                break

            curr_gpu_id = run.gpu_id
            if prev_gpu_id != curr_gpu_id:
                if curr_line >= height:
                    break

                colour, attr, background = self._frame.palette["label"]
                text = self._make_gpu_text(curr_gpu_id, gpu_usage)
                if len(text) < width:
                    text += "-" * (len(text) - width)
                self._frame.canvas.paint(
                    text,
                    self._x + self._offset,
                    self._y + curr_line,
                    colour,
                    attr,
                    background,
                )
                curr_line += 1

                prev_gpu_id = curr_gpu_id

            colour, attr, background = self._pick_colours(
                "field", i == self._selected_index
            )
            for line in self._make_run_text(run):
                if curr_line >= height:
                    break

                self._frame.canvas.paint(
                    line[: width - 1],
                    self._x + self._offset,
                    self._y + curr_line,
                    colour,
                    attr,
                    background,
                )
                curr_line += 1

    def value(self):
        if self._selected_index:
            return self._runs[self._selected_index]
        else:
            return None

    def process_event(self, event):
        if isinstance(event, KeyboardEvent):
            if len(self._runs) > 0 and event.key_code == Screen.KEY_UP:
                # Move up one line in text - use value to trigger on_select.
                self._selected_index = max(0, self._selected_index - 1)
            elif len(self._runs) > 0 and event.key_code == Screen.KEY_DOWN:
                # Move down one line in text - use value to trigger on_select.
                self._selected_index = min(
                    len(self._runs) - 1, self._selected_index + 1
                )
            elif len(self._runs) > 0 and event.key_code == Screen.KEY_PAGE_UP:
                # Move up one page.
                self._selected_index = max(0, self._selected_index - self._h)
            elif len(self._runs) > 0 and event.key_code == Screen.KEY_PAGE_DOWN:
                # Move down one page.
                self._selected_index = min(
                    len(self._runs) - 1, self._selected_index + self._h
                )
            else:
                return event
        else:
            # Ignore other events
            return event

        # If we got here, we processed the event - swallow it.
        return None


class MainView(Frame):
    def __init__(self, screen, training_runs, network_testing):
        super().__init__(
            screen,
            screen.height,
            screen.width,
            hover_focus=False,
            can_scroll=False,
            title="Dashboard",
            reduce_cpu=True,
        )

        self._training_runs = training_runs
        self._network_testing = network_testing

        layout = Layout([300, 10, 200], fill_frame=True)
        self.add_layout(layout)

        layout.add_widget(TrainerRunsWidget(self._training_runs, "TrainerRuns"), 0)
        layout.add_widget(VerticalDivider(), 1)
        layout.add_widget(Label("Testing status:", 1), 2)
        self._network_testing_status = layout.add_widget(
            TextBox(4, line_wrap=True, readonly=True, as_string=True), 2
        )
        self._network_testing_status.disabled = True
        layout.add_widget(Divider(), 2)
        self._networks_view = layout.add_widget(
            MultiColumnListBox(
                Widget.FILL_FRAME,
                ["<4", ">4", "<6", "0", ">7", "<6"],
                [],
                add_scroll_bar=True,
                titles=["#", "Run", "Epoch", "", "Elo", "Err"],
            ),
            2,
        )

        layouta = Layout([1])
        self.add_layout(layouta)
        layouta.add_widget(Divider())

        layout2 = Layout([1, 1, 1, 1])
        self.add_layout(layout2)

        layout2.add_widget(Button("Quit", self._quit), 3)

        self.fix()

    def reset(self):
        # Do standard reset to clear out form, then populate with new data.
        super().reset()

    def _update_network_list(self):
        self._networks_view.options.clear()
        for i, entry in enumerate(self._network_testing.get_ordered_results()):
            self._networks_view.options.append(
                (
                    [
                        str(i + 1),
                        str(entry.run_id),
                        str(entry.epoch),
                        "",
                        f"{entry.elo:0.1f}",
                        f"±{entry.elo_error:0.1f}",
                    ],
                    i,
                )
            )

    def _update_network_testing_status(self):
        self._network_testing_status.value = self._network_testing.get_status_string()

    def update(self, frame_no):
        super().update(frame_no)

        self._update_network_list()
        self._update_network_testing_status()

    def _quit(self):
        self._scene.add_effect(
            PopUpDialog(
                self._screen,
                "Are you sure you want to quit?",
                ["Yes", "No"],
                has_shadow=True,
                on_close=self._quit_on_yes,
            )
        )

    @staticmethod
    def _quit_on_yes(selected):
        # Yes is the first button
        if selected == 0:
            raise StopApplication("User requested exit.")

    @property
    def frame_update_count(self):
        return 1


def app(screen, scene, training_runs, network_testing):
    global TUI_SCREEN

    try:
        TUI_SCREEN = screen

        scenes = [
            Scene([MainView(screen, training_runs, network_testing)], -1, name="Main")
        ]

        screen.play(scenes, stop_on_resize=True, start_scene=scene, allow_int=True)
    except Exception as e:
        raise e
    finally:
        TUI_SCREEN = None


def str2bool(v):
    """
    A "type" for argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def flatten_once(lst):
    return sum(lst, [])


def parse_cli_args():
    default_pytorch_threads = 2
    default_data_loader_threads = 4
    default_testing_threads = max(
        1, os.cpu_count() - default_pytorch_threads - default_data_loader_threads
    )
    default_build_threads = max(1, os.cpu_count() // 2)

    parser = argparse.ArgumentParser(
        description="Trains the network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--workspace-path",
        default="./easy_train_data",
        type=str,
        metavar="PATH",
        dest="workspace_path",
        help="Specifies the directory in which the dependencies, training, and testing will be set up.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        type=str,
        metavar="NAME",
        dest="experiment_name",
        required=True,
        help="A name of the experiment is used to identify it. The experiment's directory will have the name experiment_[experiment_name].",
    )
    parser.add_argument(
        "--training-dataset",
        type=str,
        action="append",
        nargs="+",
        metavar="PATH",
        dest="training_datasets",
        required=True,
        help="Path to a training dataset. Supports .binpack files.",
    )
    parser.add_argument(
        "--validation-dataset",
        type=str,
        action="append",
        nargs="+",
        metavar="PATH",
        dest="validation_datasets",
        help="Path to a validation dataset. Supports .binpack files.",
    )
    parser.add_argument(
        "--lambda",
        default=1.0,
        type=float,
        metavar="FLOAT",
        dest="lambda_",
        help="Interpolation coefficient for training on evaluation/result. lambda=1.0 means train on evaluations. lambda=0.0 means train on game results. Must be in range [0, 1].",
    )
    parser.add_argument(
        "--start-lambda",
        default=None,
        type=float,
        metavar="FLOAT",
        dest="start_lambda",
        help="Lambda to use at the first epoch. Defaults to --lambda if not specified.",
    )
    parser.add_argument(
        "--end-lambda",
        default=None,
        type=float,
        metavar="FLOAT",
        dest="end_lambda",
        help="Lambda to use at the last epoch. Defaults to --lambda if not specified.",
    )
    parser.add_argument(
        "--gamma",
        default=0.992,
        type=float,
        metavar="FLOAT",
        dest="gamma",
        help="Multiplicative factor applied to the learning rate after every epoch. Values lower than 1 will cause the learning rate to decrease exponentially as training progresses.",
    )
    parser.add_argument(
        "--lr",
        default=8.75e-4,
        type=float,
        metavar="FLOAT",
        dest="lr",
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--num-workers",
        default=default_data_loader_threads,
        type=int,
        dest="num_workers",
        help="Number of worker threads to use for supplying training data. Increase with large skipping rates, or underloaded GPUs.",
    )
    parser.add_argument(
        "--batch-size",
        default=16384,
        type=int,
        metavar="INTEGER",
        dest="batch_size",
        help="Number of positions per batch (1 batch = 1 iteration).",
    )
    parser.add_argument(
        "--threads",
        default=default_pytorch_threads,
        type=int,
        metavar="INTEGER",
        dest="threads",
        help="Number of threads for pytorch to use. Generally performance does not scale well with the amount of threads.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        metavar="INTEGER",
        dest="seed",
        help="The random number generator seed to use for training. Each run within a single session gets a slightly different (but deterministic) seed based on this master value.",
    )
    parser.add_argument(
        "--smart-fen-skipping",
        default=True,
        type=str2bool,
        metavar="BOOL",
        dest="smart_fen_skipping",
        help="Whether to perform smart fen skipping. This attempts to heuristically skip non-quiet positions during training.",
    )
    parser.add_argument(
        "--wld-fen-skipping",
        default=True,
        type=str2bool,
        metavar="BOOL",
        dest="wld_fen_skipping",
        help="Whether to perform position skipping during training that increases correlation between evaluations and results.",
    )
    parser.add_argument(
        "--random-fen-skipping",
        default=3,
        type=int,
        metavar="INTEGER",
        dest="random_fen_skipping",
        help="Skip on average random_fen_skipping positions during training before using one. Increases diversity for data that is not fully shuffled.",
    )
    parser.add_argument(
        "--early-fen-skipping",
        default=-1,
        type=int,
        metavar="INTEGER",
        dest="early_fen_skipping",
        help="Skip all fens from training game plies <= the given number.",
    )
    parser.add_argument(
        "--start-from-model",
        default=None,
        type=str,
        metavar="PATH",
        dest="start_from_model",
        help="Initializes training using the weights from the given .pt, .ckpt, or .nnue model.",
    )
    parser.add_argument(
        "--start-from-experiment",
        default=None,
        type=str,
        metavar="NAME",
        dest="start_from_experiment",
        help="Initializes training using the best network from a given experiment (by name). Uses the best net from ordo, falls back to last created.",
    )
    parser.add_argument(
        "--start-from-engine-test-net",
        default=False,
        type=str2bool,
        metavar="BOOL",
        dest="start_from_engine_test_net",
        help="Initializes training using the weights from the .nnue model associated with --engine-test-branch.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        metavar="INTEGER[,INTEGER]*",
        dest="gpus",
        default="0",
        help="A single GPU ID or a list of GPU IDs to use for training. Note that a single run still uses a single GPU.",
    )
    parser.add_argument(
        "--runs-per-gpu",
        default=1,
        type=int,
        metavar="INTEGER",
        dest="runs_per_gpu",
        help="Number of runs to do in parallel on each GPU. To increase the load on strong GPUs run more than one run per GPU. Doing multiple runs also means that variance has lower impact on the results.",
    )
    parser.add_argument(
        "--features",
        default=None,
        type=str,
        metavar="FEATURESET",
        help="The feature set to use. If not specified then will be inferred from the cloned nnue-pytorch repo.",
    )
    parser.add_argument(
        "--max_epoch",
        "--num-epochs",  # --max_epoch kept to match pytorch-lightning's name
        default=400,
        type=int,
        metavar="INTEGER",
        dest="max_epoch",
        help="Number of epochs to train for.",
    )
    parser.add_argument(
        "--network-save-period",
        default=20,
        type=int,
        metavar="INTEGER",
        dest="network_save_period",
        help="Number of epochs between network snapshots (checkpoints). None to disable. Note that these take a lot of space.",
    )
    parser.add_argument(
        "--save-last-network",
        default=True,
        type=str2bool,
        metavar="BOOL",
        dest="save_last_network",
        help="Whether to always save the last produced network (checkpoint).",
    )
    parser.add_argument(
        "--additional-training-arg",
        type=str,
        metavar="STRING",
        nargs="*",
        dest="additional_training_args",
        help="Additional training args passed verbatim.",
    )
    parser.add_argument(
        "--additional-testing-arg",
        type=str,
        metavar="STRING",
        nargs="*",
        dest="additional_testing_args",
        help="Additional network testing args passed verbatim.",
    )
    parser.add_argument(
        "--engine-base-branch",
        default="official-stockfish/Stockfish/master",
        type=str,
        metavar="BRANCH_OR_COMMIT",
        dest="engine_base_branch",
        help="Path to the commit/branch to use for the engine baseline. It is recommended to use a specific commit for consistency.",
    )
    parser.add_argument(
        "--engine-test-branch",
        default="official-stockfish/Stockfish/master",
        type=str,
        metavar="BRANCH_OR_COMMIT",
        dest="engine_test_branch",
        help="Path to the commit/branch to use for the engine being tested. It is recommended to use a specific commit for consistency.",
    )
    parser.add_argument(
        "--nnue-pytorch-branch",
        default="official-stockfish/nnue-pytorch/master",
        type=str,
        metavar="BRANCH_OR_COMMIT",
        dest="nnue_pytorch_branch",
        help="Path to the commit/branch to use for the trainer being tested. It is recommended to use a specific commit for consistency.",
    )
    parser.add_argument(
        "--build-engine-arch",
        default="x86-64-modern",
        type=str,
        metavar="ARCH",
        dest="build_engine_arch",
        help="ARCH to use for engine compilation, e.g. x86-64-avx2 for recent hardware.",
    )
    parser.add_argument(
        "--build-threads",
        default=default_build_threads,
        type=int,
        metavar="INTEGER",
        dest="build_threads",
        help="Number of threads to use for engine compilation.",
    )
    parser.add_argument(
        "--fail-on-experiment-exists",
        default=True,
        type=str2bool,
        metavar="BOOL",
        dest="fail_on_experiment_exists",
        help="By default an experiment must be created in an empty directory. Ignored when --resume-training is True. Care should be taken when the directory already exists as it might create consistency issue when not everything gets resetup.",
    )
    parser.add_argument(
        "--epoch-size",
        default=100000000,
        type=int,
        metavar="INTEGER",
        dest="epoch_size",
        help="Number of positions per epoch (training step).",
    )
    parser.add_argument(
        "--validation-size",
        default=1000000,
        type=int,
        metavar="INTEGER",
        dest="validation_size",
        help="Number of positions per validation step.",
    )
    parser.add_argument(
        "--tui",
        default=True,
        type=str2bool,
        metavar="BOOL",
        dest="tui",
        help="Whether to show a nice terminal user interface.",
    )
    parser.add_argument(
        "--do-network-testing",
        default=True,
        type=str2bool,
        metavar="BOOL",
        dest="do_network_testing",
        help="Whether to test networks as they are generated.",
    )
    parser.add_argument(
        "--do-network-training",
        default=True,
        type=str2bool,
        metavar="BOOL",
        dest="do_network_training",
        help="Whether to train networks.",
    )
    parser.add_argument(
        "--network-testing-threads",
        default=default_testing_threads,
        type=int,
        metavar="INTEGER",
        dest="network_testing_threads",
        help="Number of threads to use for network testing. By default the available number of threads minus default data loader and pytorch threads. The optimal value might depend on the --threads, --num-workers and other machine load.",
    )
    parser.add_argument(
        "--network-testing-explore-factor",
        default=1.5,
        type=float,
        metavar="FLOAT",
        dest="network_testing_explore_factor",
        help="Elo error estimates are multiplied by this amount to determine testing candidates.",
    )
    parser.add_argument(
        "--network-testing-book",
        default="https://github.com/official-stockfish/books/raw/master/UHO_Lichess_4852_v1.epd.zip",
        type=str,
        metavar="PATH_OR_URL",
        dest="network_testing_book",
        help="Path to a suitable book, or suitable link (URL). See https://github.com/official-stockfish/books.",
    )
    parser.add_argument(
        "--network-testing-time-per-game",
        default=None,
        type=float,
        metavar="FLOAT",
        dest="network_testing_time_per_game",
        help="Number of seconds per game for each engine.",
    )
    parser.add_argument(
        "--network-testing-time-increment-per-move",
        default=None,
        type=float,
        metavar="FLOAT",
        dest="network_testing_time_increment_per_move",
        help="Number of seconds added to the clock of an engine per move.",
    )
    parser.add_argument(
        "--network-testing-nodes-per-move",
        default=None,
        type=int,
        metavar="INTEGER",
        dest="network_testing_nodes_per_move",
        help="Number of nodes per move to use for testing. Overrides time control. Recommended over time control for better consistency.",
    )
    parser.add_argument(
        "--network-testing-hash-mb",
        default=8,
        type=int,
        metavar="INTEGER",
        dest="network_testing_hash_mb",
        help="Number of MiB of memory to use for hash allocation for each engine being tested.",
    )
    parser.add_argument(
        "--network-testing-games-per-round",
        default=20 * default_testing_threads,
        type=int,
        metavar="INTEGER",
        dest="network_testing_games_per_round",
        help="Number of games per round to use. Essentially a testing batch size.",
    )
    parser.add_argument(
        "--resume-training",
        default=True,
        type=str2bool,
        metavar="BOOL",
        dest="resume_training",
        help="Attempts to resume each run from its latest checkpoint.",
    )
    parser.add_argument(
        "--do-approximate-ordo",
        default=True,
        type=str2bool,
        metavar="BOOL",
        dest="do_approximate_ordo",
        help="If true then does not launch ordo and instead does a fast approximate computation. Workaround for ordo memory usage issues.",
    )
    parser.add_argument(
        "--auto-exit-timeout",
        default=None,
        type=str,
        metavar="DURATION",
        dest="auto_exit_timeout",
        help='Automatically exit the script after a specified time has passed since its start. Duration format "h:m:s", "m:s", or "s".',
    )
    parser.add_argument(
        "--auto-exit-timeout-on-training-finished",
        default=None,
        type=str,
        metavar="DURATION",
        dest="auto_exit_timeout_on_training_finished",
        help='Automatically exit the script after a specified time has passed after training finished. Duration format "h:m:s", "m:s", or "s"',
    )
    args = parser.parse_args()

    args.training_datasets = flatten_once(args.training_datasets)
    if args.validation_datasets:
        args.validation_datasets = flatten_once(args.validation_datasets)
    else:
        args.validation_datasets = []

    if len(args.training_datasets) == 0:
        raise Exception("No training data specified")

    if args.lambda_ < 0.0 or args.lambda_ > 1.0:
        raise Exception("lambda must be within [0, 1]")

    args.validation_datasets = args.validation_datasets or args.training_datasets
    for dataset in args.validation_datasets:
        if not Path(dataset).is_file():
            raise Exception(f"Invalid validation data set file name: {dataset}")

    for dataset in args.training_datasets:
        if not Path(dataset).is_file():
            raise Exception(f"Invalid training data set file name: {dataset}")

    # these are not required because testing is optional
    if args.engine_base_branch and args.engine_base_branch.count("/") != 2:
        raise Exception(f"Invalid base engine repo path: {args.engine_base_branch}")

    if args.engine_test_branch and args.engine_test_branch.count("/") != 2:
        raise Exception(f"Invalid test engine repo path: {args.engine_test_branch}")

    # this one is required because it has other important scripts
    if not args.nnue_pytorch_branch or args.nnue_pytorch_branch.count("/") != 2:
        raise Exception(f"Invalid test trainer repo path: {args.nnue_pytorch_branch}")

    if (
        not args.network_testing_time_per_game
        and not args.network_testing_nodes_per_move
    ):
        args.network_testing_nodes_per_move = 25000
        LOGGER.info(
            f"No time control specified. Using a default {args.network_testing_nodes_per_move} nodes per move"
        )

    if [
        args.start_from_model,
        args.start_from_engine_test_net,
        args.start_from_experiment,
    ].count(True) > 1:
        raise Exception(
            "Only one of --start-from-model, --start-from-engine-test-net, and --start-from-experiment can be specified at a time."
        )

    if args.start_from_engine_test_net and not args.engine_test_branch:
        raise Exception(
            "--start-from-engine-test-net but --engine-test-branch not given"
        )

    if args.start_from_experiment and not args.start_from_experiment.startswith(
        "experiment_"
    ):
        args.start_from_experiment = "experiment_" + args.start_from_experiment

    return args


def log_args(directory, args):
    os.makedirs(directory, exist_ok=True)

    args_dump_file_path = os.path.join(directory, "args_dump.txt")
    with open(args_dump_file_path, "w") as file:
        file.write(repr(args))

    logs_file_path = os.path.join(directory, "easy_train.log")

    LOGGER.addHandler(logging.FileHandler(logs_file_path, encoding="utf-8"))


def is_url(path):
    return (
        path.startswith("http://")
        or path.startswith("https://")
        or path.startswith("ftp://")
        or path.startswith("sftp://")
    )


class TqdmDownloadProgressBar(tqdm):
    def update_to(self, blocks_transferred=1, block_size=1, total_size=None):
        if total_size is not None:
            self.total = total_size
        return self.update(
            blocks_transferred * block_size - self.n
        )  # also sets self.n = b * bsize


class TqdmToLogger(io.StringIO):
    def __init__(self):
        super().__init__()

    def write(self, buf):
        self.buf = buf

    def flush(self):
        LOGGER.info(self.buf)


def setup_book(directory, args):
    """
    If the args.network_testing_book is a URL then it downloads the book
    and reassigns args.network_testing_book to the actual book path.
    Otherwise does nothing.
    """

    if not is_url(args.network_testing_book):
        return

    os.makedirs(directory, exist_ok=True)

    url = args.network_testing_book
    temp_filename = urllib.parse.unquote(url.split("/")[-1])
    if temp_filename.endswith(".zip"):
        filename = temp_filename[:-4]
    elif temp_filename.endswith(".epd"):
        filename = temp_filename

    if not filename.endswith(".epd"):
        LOGGER.error(
            "Cannot handle the book. Currently only .epd books are supported. If compressed with .zip the name must be a.epd.zip. No other compression format is supported right now."
        )
        raise Exception("Cannot handle opening book")

    destination_temp_file_path = os.path.abspath(os.path.join(directory, temp_filename))
    destination_file_path = os.path.abspath(os.path.join(directory, filename))
    args.network_testing_book = destination_file_path

    if not os.path.exists(destination_file_path):
        if temp_filename != filename and not os.path.exists(destination_temp_file_path):
            with (
                TqdmDownloadProgressBar(
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    miniters=1,
                    desc=temp_filename,
                    file=TqdmToLogger(),
                    mininterval=0.1,  # at least 0.1s between update so the logfile doesn't get polluted.
                ) as progress_bar
            ):
                urllib.request.urlretrieve(
                    url,
                    filename=destination_temp_file_path,
                    reporthook=progress_bar.update_to,
                    data=None,
                )
                progress_bar.total = progress_bar.n
        if temp_filename.endswith(".zip"):
            zipped = zipfile.ZipFile(destination_temp_file_path, mode="r")
            names = zipped.namelist()
            if len(names) > 1 or names[0] != filename:
                LOGGER.error(
                    f"Expected only a book with name {filename} in the archive but did not find it or found more"
                )
                raise Exception("Unexpected opening book archive content.")
            LOGGER.info(f"Extracting {temp_filename} to {filename}")
            zipped.extract(filename, directory)

    LOGGER.info("Book setup completed.")


def prepare_start_model(
    directory, model_path, run_id, nnue_pytorch_directory, features
):
    """
    Copies the specified model to the desired directory.
    Performs conversion to .pt if necessary.
    """

    os.makedirs(directory, exist_ok=True)

    LOGGER.info(f"Starting from model: {model_path}")

    destination_filename = "start_model"
    if run_id:
        destination_filename += "run_" + str(run_id)
    destination_filename += ".pt"

    destination_model_path = os.path.join(directory, destination_filename)

    if model_path.endswith(".pt"):
        shutil.copyfile(model_path, destination_model_path)
    elif model_path.endswith(".nnue") or model_path.endswith(".ckpt"):
        if model_path.endswith(".nnue") and features.endswith("^"):
            features = features[:-1]

        with subprocess.Popen(
            [
                sys.executable,
                "serialize.py",
                os.path.abspath(model_path),
                destination_model_path,
                f"--features={features}",
            ],
            cwd=nnue_pytorch_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as process:
            if process.wait():
                raise Exception("Failed to run serialize.py for start model.")

        if not os.path.exists(destination_model_path):
            raise Exception("Failed to convert start model.")

    return destination_model_path


def prepare_start_model_from_experiment(
    directory, experiment_path, run_id, nnue_pytorch_directory, features
):
    """
    Prepares start model with the best or (if no ordo found)
    last checkpoint from the given experiment.
    """
    root_dir = os.path.join(experiment_path, "training")
    best_model = find_best_checkpoint(root_dir)
    if best_model is None:
        best_model = find_latest_checkpoint(root_dir)
    if best_model is None:
        raise Exception(
            "Could not find any viable .ckpt nor .nnue files in the start experiment."
        )
    return prepare_start_model(
        directory, best_model, run_id, nnue_pytorch_directory, features
    )


def get_default_feature_set_from_nnue_pytorch(nnue_pytorch_directory):
    """
    The features __init__.py in nnue-pytorch defines the default feature set to use.
    We scrape it for the feature set name.
    Normally we could import that file and let it add the argument to argparse,
    but we setup argparse before nnue-pytorch is setup so we have to do it like that.
    """
    try:
        features_init = os.path.join(
            nnue_pytorch_directory, "model", "modules", "features", "__init__.py"
        )
        with open(features_init, "r") as features_file:
            for line in features_file:
                line = line.strip()
                if line.startswith("default="):
                    # Extract the default value from: default="HalfKAv2_hm",
                    return line.split('"')[1]
    except Exception:
        pass
    return "HalfKAv2_hm"


def parse_duration_hms_to_s(duration_str):
    """
    Parses a duration of the form [h:][m:]s
    """
    parts = duration_str.split(":")
    s = int(parts[-1])
    m = 0 if len(parts) < 2 else int(parts[-2])
    h = 0 if len(parts) < 3 else int(parts[-3])
    return h * 3600 + m * 60 + s


def spawn_training_watcher(training_runs, exit_timeout_after_finished):
    """
    Spawns a daemon thread that awaits training end and the schedules the script
    to exit after the specified amount of seconds.
    """

    def f():
        while True:
            finished = True
            success = True
            for run in training_runs:
                if not run.has_started or run.is_running:
                    finished = False
                if not run.has_finished:
                    success = False

            if finished:
                errcode = EXITCODE_OK if success else EXITCODE_TRAINING_NOT_FINISHED
                schedule_exit(exit_timeout_after_finished, errcode)
                return

            time.sleep(1)

    thread = Thread(target=f)
    thread.daemon = True
    thread.start()


def main():
    LOGGER.info("Initializing...")

    args = parse_cli_args()

    # if we ask to resume don't fail on existing directory
    if args.resume_training:
        args.fail_on_experiment_exists = False

    absolute_workspace_path = os.path.abspath(args.workspace_path)
    os.makedirs(absolute_workspace_path, exist_ok=True)

    do_network_testing = (
        args.engine_base_branch and args.engine_test_branch and args.do_network_testing
    )
    do_network_training = args.do_network_training and args.training_datasets

    # Global (workspace) setup

    with SystemWideMutex(os.path.join(absolute_workspace_path, ".lock")) as mutex:
        ordo_directory = os.path.join(absolute_workspace_path, "ordo")
        c_chess_cli_directory = os.path.join(absolute_workspace_path, "c-chess-cli")
        books_directory = os.path.join(absolute_workspace_path, "books")

        if not args.do_approximate_ordo:
            setup_ordo(ordo_directory)

        setup_c_chess_cli(c_chess_cli_directory)

        if do_network_testing:
            setup_book(books_directory, args)

    # Local (experiment) setup

    experiment_directory = os.path.join(
        absolute_workspace_path, f"experiments/experiment_{args.experiment_name}"
    )
    try:
        os.makedirs(experiment_directory, exist_ok=False)
    except FileExistsError:
        if args.fail_on_experiment_exists and os.listdir(experiment_directory):
            LOGGER.error(
                f"Directory {experiment_directory} already exists. An experiment must use a new directory."
            )
            LOGGER.error(
                "Alternatively, override this with the option --resume-training=True or --fail-on-experiment-exists=False."
            )
            return

    stockfish_base_directory = os.path.join(experiment_directory, "stockfish_base")
    stockfish_test_directory = os.path.join(experiment_directory, "stockfish_test")
    nnue_pytorch_directory = os.path.join(experiment_directory, "nnue-pytorch")
    logging_directory = os.path.join(experiment_directory, "logging")
    start_model_directory = os.path.join(experiment_directory, "start_models")

    log_args(logging_directory, args)

    if do_network_testing:
        LOGGER.info("Engines provided. Enabling network testing.")
        stockfish_base_repo = "/".join(args.engine_base_branch.split("/")[:2])
        stockfish_test_repo = "/".join(args.engine_test_branch.split("/")[:2])
        stockfish_base_branch_or_commit = args.engine_base_branch.split("/")[2]
        stockfish_test_branch_or_commit = args.engine_test_branch.split("/")[2]
        setup_stockfish(
            stockfish_base_directory,
            stockfish_base_repo,
            stockfish_base_branch_or_commit,
            args.build_engine_arch,
            args.build_threads,
        )
        setup_stockfish(
            stockfish_test_directory,
            stockfish_test_repo,
            stockfish_test_branch_or_commit,
            args.build_engine_arch,
            args.build_threads,
        )
    else:
        LOGGER.info(
            "Not doing network testing. Either engines not provided or explicitly disabled."
        )

    nnue_pytorch_repo = "/".join(args.nnue_pytorch_branch.split("/")[:2])
    nnue_pytorch_branch_or_commit = args.nnue_pytorch_branch.split("/")[2]
    setup_nnue_pytorch(
        nnue_pytorch_directory, nnue_pytorch_repo, nnue_pytorch_branch_or_commit
    )

    if args.features is None:
        args.features = get_default_feature_set_from_nnue_pytorch(
            nnue_pytorch_directory
        )

    LOGGER.info("Initialization completed.")

    # Directory layout:
    #     tmp/experiments/experiment_{name}/training/run_{i}
    #     tmp/experiments/experiment_{name}/stockfish_base
    #     tmp/experiments/experiment_{name}/stockfish_test
    #     tmp/experiments/experiment_{name}/nnue-pytorch
    #     tmp/experiments/experiment_{name}/logging
    #     tmp/c-chess-cli
    #     tmp/ordo

    start_model = None
    if args.start_from_engine_test_net:
        args.start_from_model = str(
            next(Path(os.path.join(stockfish_test_directory, "src/")).rglob("*.nnue"))
        )

    if args.start_from_model:
        start_model = prepare_start_model(
            directory=start_model_directory,
            model_path=args.start_from_model,
            run_id=None,
            nnue_pytorch_directory=nnue_pytorch_directory,
            features=args.features,
        )
    elif args.start_from_experiment:
        start_model = prepare_start_model_from_experiment(
            directory=start_model_directory,
            experiment_path=os.path.join(
                absolute_workspace_path, "experiments", args.start_from_experiment
            ),
            run_id=None,
            nnue_pytorch_directory=nnue_pytorch_directory,
            features=args.features,
        )

    training_runs = []
    if do_network_training:
        gpu_ids = [int(v) for v in args.gpus.split(",") if v]
        for gpu_id in gpu_ids:
            for j in range(args.runs_per_gpu):
                run_id = gpu_id * args.runs_per_gpu + j

                training_runs.append(
                    TrainingRun(
                        gpu_id=gpu_id,
                        run_id=run_id,
                        nnue_pytorch_directory=nnue_pytorch_directory,
                        training_datasets=args.training_datasets,
                        validation_datasets=args.validation_datasets,
                        num_data_loader_threads=args.num_workers,
                        num_pytorch_threads=args.threads,
                        num_epochs=args.max_epoch,
                        batch_size=args.batch_size,
                        random_fen_skipping=args.random_fen_skipping,
                        smart_fen_skipping=args.smart_fen_skipping,
                        wld_fen_skipping=args.wld_fen_skipping,
                        early_fen_skipping=args.early_fen_skipping,
                        features=args.features,
                        lr=args.lr,
                        gamma=args.gamma,
                        lambda_=args.lambda_,
                        start_lambda=args.start_lambda,
                        end_lambda=args.end_lambda,
                        network_save_period=args.network_save_period,
                        save_last_network=args.save_last_network,
                        seed=args.seed + run_id,
                        start_from_model=start_model,
                        root_dir=os.path.join(
                            experiment_directory, "training", f"run_{run_id}"
                        ),
                        epoch_size=args.epoch_size,
                        validation_size=args.validation_size,
                        resume_training=args.resume_training,
                        additional_args=[
                            arg for arg in args.additional_training_args or []
                        ],
                    )
                )
        LOGGER.info(
            f"Doing network training on gpus {gpu_ids}. {len(training_runs)} runs in total."
        )
    else:
        LOGGER.info("Not training networks.")

    network_testing = NetworkTesting(
        nnue_pytorch_directory=nnue_pytorch_directory,
        root_dir=os.path.join(experiment_directory, "training"),
        ordo_exe=None
        if args.do_approximate_ordo
        else make_ordo_executable_path(ordo_directory),
        c_chess_cli_exe=make_c_chess_cli_executable_path(c_chess_cli_directory),
        stockfish_base_exe=make_stockfish_executable_path(stockfish_base_directory),
        stockfish_test_exe=make_stockfish_executable_path(stockfish_test_directory),
        features=args.features,
        num_parallel_games=args.network_testing_threads,
        explore_factor=args.network_testing_explore_factor,
        book_file_path=args.network_testing_book,
        time_per_game=args.network_testing_time_per_game,
        time_increment_per_move=args.network_testing_time_increment_per_move,
        nodes_per_move=args.network_testing_nodes_per_move,
        hash=args.network_testing_hash_mb,
        games_per_round=args.network_testing_games_per_round,
        active=do_network_testing,
        additional_args=[arg for arg in args.additional_testing_args or []],
    )

    for tr in training_runs:
        tr.start()

    network_testing.start()

    if args.auto_exit_timeout:
        timeout = parse_duration_hms_to_s(args.auto_exit_timeout)
        schedule_exit(timeout, EXITCODE_TRAINING_LIKELY_NOT_FINISHED)

    if args.auto_exit_timeout_on_training_finished and args.do_network_training:
        timeout = parse_duration_hms_to_s(args.auto_exit_timeout_on_training_finished)
        spawn_training_watcher(training_runs, timeout)

    if args.tui:
        for h in LOGGER.handlers:
            if isinstance(h, logging.StreamHandler):
                LOGGER.removeHandler(h)
        last_scene = None
        while True:
            try:
                Screen.wrapper(
                    app,
                    catch_interrupt=True,
                    arguments=[last_scene, training_runs, network_testing],
                )
                break
            except ResizeScreenError as e:
                last_scene = e.scene
    else:
        while True:
            try:
                v = input()
                if v == "quit":
                    break
                else:
                    print("Type `quit` to stop.")
            except EOFError:
                # For non-interactive environments
                time.sleep(1)

    LOGGER.info("Stopping training runs.")
    for tr in training_runs:
        tr.stop()

    LOGGER.info("Stopping network testing.")
    network_testing.stop()

    any_training_error = False
    for tr in training_runs:
        if not tr.has_finished:
            any_training_error = True
            break

    if any_training_error:
        sys.exit(EXITCODE_TRAINING_NOT_FINISHED)


if __name__ == "__main__":
    main()
