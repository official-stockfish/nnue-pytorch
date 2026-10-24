import glob
import os
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_dataloader_read_balance import _build_generator

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPT = """
import perf_sigmoid_fitter

data = perf_sigmoid_fitter.gather_statistics_from_data(
    {filename!r}, {count}, 16
)
print(len(data), sum(count for _perf, count in data.values()))
"""


def _loader_built():
    return any(
        glob.glob(pattern, recursive=True)
        for pattern in (
            os.path.join(REPO_ROOT, "build", "**", "*training_data_loader.so"),
            os.path.join(
                REPO_ROOT,
                "data_loader",
                "cpp",
                "build",
                "**",
                "*training_data_loader.so",
            ),
        )
    )


@unittest.skipUnless(_loader_built(), "data loader library not built")
class TestPerfSigmoidFitter(unittest.TestCase):
    def test_gather_statistics_from_data(self):
        """The loader must accept the feature set the script asks for.

        An unknown feature set makes the native loader print an error and
        crash, so this runs in a subprocess.
        """
        generator = _build_generator()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "perf.binpack")
            subprocess.run([generator, path, "20000", "7"], check=True)
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    SCRIPT.format(filename=path, count=16384),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
                env={**os.environ, "PYTHONPATH": REPO_ROOT},
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        buckets, positions = map(int, result.stdout.split()[-2:])
        self.assertGreater(buckets, 0)
        self.assertGreaterEqual(positions, 16384)


if __name__ == "__main__":
    unittest.main()
