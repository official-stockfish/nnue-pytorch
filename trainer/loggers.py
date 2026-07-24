import csv
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


def _make_version(root_dir: str | Path) -> str:
    """Return the next version_N string by scanning root_dir/lightning_logs."""
    log_root = Path(root_dir) / "lightning_logs"
    max_version = -1
    if log_root.exists():
        for entry in log_root.iterdir():
            if entry.is_dir():
                match = re.match(r"version_(\d+)$", entry.name)
                if match:
                    max_version = max(max_version, int(match.group(1)))
    return f"version_{max_version + 1}"


class _Logger(ABC):
    """Base interface shared by plain loggers."""

    def __init__(self, root_dir: str | Path, version: str | None = None):
        self._root_dir = Path(root_dir)
        self._version = version if version is not None else _make_version(self._root_dir)
        self._log_dir = self._root_dir / "lightning_logs" / self._version
        self._log_dir.mkdir(parents=True, exist_ok=True)

    @property
    def log_dir(self) -> str:
        return str(self._log_dir)

    @property
    def version(self) -> str:
        return self._version

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log a dictionary of metrics at the given step."""

    def log_epoch_end(self, metrics: dict[str, Any], epoch: int | None = None) -> None:
        """Optional hook called at the end of an epoch."""

    def finalize(self, status: str | None = None) -> None:
        """Flush and close any underlying writers."""

    def save(self) -> None:
        """No-op for parity with PyTorch Lightning logger API."""


class CSVLogger(_Logger):
    """Plain CSV logger that mirrors PyTorch Lightning's CSVLogger layout."""

    def __init__(self, root_dir: str | Path, version: str | None = None):
        super().__init__(root_dir, version)
        self._metrics_file = self._log_dir / "metrics.csv"
        self._columns: list[str] = []
        self._file_initialized = False

    def _initialize_columns(self, metrics: dict[str, Any]) -> None:
        # Standard ordering: epoch, step, then alphabetically sorted metrics.
        metric_keys = sorted(k for k in metrics if k not in ("epoch", "step"))
        self._columns = ["epoch", "step"] + metric_keys

    def _write_header(self) -> None:
        with open(self._metrics_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self._columns)
        self._file_initialized = True

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not metrics:
            return

        # Normalize to a fresh dict so we don't mutate the caller's data.
        row_data = dict(metrics)
        if step is not None and "step" not in row_data:
            row_data["step"] = step

        if not self._file_initialized:
            self._initialize_columns(row_data)
            self._write_header()
        else:
            # Detect any new columns and extend the header/file if needed.
            new_keys = [k for k in row_data if k not in self._columns]
            if new_keys:
                self._columns.extend(sorted(new_keys))
                # Re-write header.  This appends new columns to existing rows
                # by reading the existing rows and rewriting the file.
                rows: list[list[Any]] = []
                if self._metrics_file.exists():
                    with open(self._metrics_file, "r", newline="") as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        if header is not None:
                            old_columns = header
                            for row in reader:
                                padded = [""] * len(self._columns)
                                for idx, col in enumerate(old_columns):
                                    if col in self._columns:
                                        padded[self._columns.index(col)] = row[idx]
                                rows.append(padded)
                with open(self._metrics_file, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(self._columns)
                    writer.writerows(rows)

        row = [row_data.get(col, "") for col in self._columns]
        with open(self._metrics_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def log_epoch_end(self, metrics: dict[str, Any], epoch: int | None = None) -> None:
        if epoch is not None and "epoch" not in metrics:
            metrics = {**metrics, "epoch": epoch}
        self.log_metrics(metrics)


class _NoOpSummaryWriter:
    """Stub SummaryWriter used when tensorboard is not installed."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        pass

    def add_scalars(self, *args: Any, **kwargs: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class TensorBoardLogger(_Logger):
    """Plain TensorBoard logger, or a no-op stub if tensorboard is unavailable."""

    def __init__(self, root_dir: str | Path, version: str | None = None):
        super().__init__(root_dir, version)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            SummaryWriter = _NoOpSummaryWriter  # type: ignore[misc,assignment]

        self._writer = SummaryWriter(log_dir=str(self._log_dir))

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not metrics:
            return

        for key, value in metrics.items():
            if key in ("epoch", "step"):
                continue
            try:
                scalar_value = float(value)
            except (TypeError, ValueError):
                continue
            self._writer.add_scalar(key, scalar_value, global_step=step)
        self._writer.flush()

    def log_epoch_end(self, metrics: dict[str, Any], epoch: int | None = None) -> None:
        # TensorBoard handles step-based logging; log_epoch_end delegates to
        # log_metrics using the provided epoch as the step if no step is given.
        step = epoch
        self.log_metrics(metrics, step=step)

    def finalize(self, status: str | None = None) -> None:
        self._writer.flush()
        self._writer.close()
