import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from training_marker import is_training_finished


def write_marker(root_dir, content):
    with open(os.path.join(root_dir, "training_finished"), "w") as f:
        f.write(content)


def test_no_marker_is_not_finished(tmp_path):
    assert not is_training_finished(str(tmp_path), 10)


@pytest.mark.parametrize("num_epochs", [5, 10])
def test_marker_with_at_least_the_requested_epochs_is_finished(tmp_path, num_epochs):
    write_marker(tmp_path, "10")
    assert is_training_finished(str(tmp_path), num_epochs)


def test_raising_the_epoch_count_trains_again(tmp_path):
    write_marker(tmp_path, "10")
    assert not is_training_finished(str(tmp_path), 11)


def test_marker_without_a_count_is_finished(tmp_path):
    write_marker(tmp_path, "")
    assert is_training_finished(str(tmp_path), 800)
