import os


def is_training_finished(root_dir, num_epochs):
    """
    Whether the run in root_dir already trained for num_epochs epochs.
    train.py records its --max_epochs in the training_finished marker, so raising
    the epoch count of a finished run makes it train again. A marker without a
    count (written by an older train.py) counts as finished.
    """
    marker = os.path.join(root_dir, "training_finished")
    if not os.path.exists(marker):
        return False

    try:
        with open(marker) as f:
            return int(f.read().strip()) >= num_epochs
    except ValueError:
        return True
