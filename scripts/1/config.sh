# This is a configuration file used by all scripts.
# Make sure this is setup correctly, then change APPROVED to a non-zero value.


# The path to the directory of the nnue-pytorch trainer cloned repository.
# https://github.com/glinscott/nnue-pytorch
NNUE_PYTORCH_DIR="C:/dev/nnue-pytorch"

# The path to a directory containing the base stockfish
# It needs to have a ./src/ directory inside.
# This will be built on experiment setup and used
# as a base engine during testing.
# A normal build (not PGO) is made to be consistent.
# https://github.com/official-stockfish/Stockfish
STOCKFISH_BASE_DIR="C:/dev/stockfish-base"

# The path to a directory containing the test stockfish
# It needs to have a ./src/ directory inside.
# This will be built on experiment setup and used
# as a test engine during testing. Sometimes you may
# want this to be a different directory than the base
# stockfish. The state of the directory at the point of running
# the experiment setup defines the version that will be compiled.
STOCKFISH_TEST_DIR="C:/dev/stockfish-test"

# The ARCH={} to use for compiling stockfish.
# (make build ARCH={} -j will be executed when setting up an experiment)
ARCH="x86-64-modern"

# The way to execute ordo. It is only required that "$ORDO_PATH" executes ordo
# https://github.com/michiguel/Ordo
ORDO_PATH="ordo"

# The way to execute c-chess-cli. It is only required that "$C_CHESS_CLI_PATH" executes c-chess-cli
# https://github.com/lucasart/c-chess-cli
C_CHESS_CLI_PATH="c-chess-cli"

# The directory to place experiments in. Each experiment will have a unique directory inside.
EXPERIMENTS_DIR="W:/nnue/training/experiments"

# The directory to place the plots in.
PLOTS_DIR="W:/nnue/training/plots"

# Sometimes binaries are copied while the name is changed so the script has to know
# whether to append the EXE_SUFFIX or not.
EXE_SUFFIX=".exe"

# Path to the training data. Must be one file. .bin or .binpack (preferred)
TRAINING_DATA_PATH="W:/nnue/data/d9.binpack"

# Path to the validation data. Must be one file. .bin or .binpack (preferred)
VALIDATION_DATA_PATH="W:/nnue/data/d9.binpack"

# Number of threads that torch should use for tensor operations done on the CPU
# Sometimes a higher number can speed up the training, but sometimes it makes it slower.
NUM_TORCH_THREADS=1

# Number of threads for the data loader to use. The trained utilizes a very fast custom
# data loader written in C++ and can easly provide more than a million positions per second.
# Usually 1 data loader thread is enough, but you may try increasing it.
NUM_DATALOADER_THREADS=1

# The feature set to use.
FEATURE_SET="HalfKA^"

# Batch size to use for training. Generally values below 8192 are not worth
# exploring. 16384 works best for most high end gpus but higher batch size might
# speed up the training slightly at the cost of VRAM usage.
BATCH_SIZE=16384

# The number of steps between updates to the pytorch's console output.
PROGRESS_BAR_REFRESH_RATE=20

# Training positions are being randomly skipped. Higher value = higher % of positions is skipped.
# On average 1 every $RANDOM_FEN_SKIPPING positions is used.
RANDOM_FEN_SKIPPING=3

# Interpolation factor between eval loss and wdl loss.
# 1.0 is just eval loss.
# 0.0 is just wdl loss.
LAMBDA="1.0"

# The number of epochs after which the training ends.
MAX_EPOCHS=500

# The way checkpoints are saved.
# "periodic" means that one checkpoint is produced every $CKPT_SAVE_PERIOD epochs.
# "best" means that only the checkpoint with best val_loss is kept.
CKPT_SAVE_POLICY="periodic"

# Controlls how many epochs there are between checkpoints.
CKPT_SAVE_PERIOD=10

# The path to the opening book used for playing games.
OPENING_BOOK_PATH="W:/nnue/books/noob_3moves.epd"

# The path to the baseline stockfish executable. It is copied to the experiment's
# directory at the start.
TEST_SF_BASE_PATH="W:/nnue/training/stockfish/stockfish.exe"

# The time control to use for the test games. Syntax is c-chess-cli syntax.
TEST_TC="tc=1000+10 nodes=100000"

# The number of MiB to use for hash for each engine.
TEST_HASH=32

# The number of games to run in parallel.
# If you're running fixed nodes games it can be beneficial to use hypercores too.
# If you're running timed matches then usage of hypercores is not advised.
TEST_CONCURRENCY=16

# At each step of the testing some top nets are chosen for further testing based on
# elo + error * $TEST_EXPLORE_FACTOR. $TEST_EXPLORE_FACTOR effectively controls
# the scaling for the 95% certainty elo error range when considering
# how good a net can be based on previous results.
TEST_EXPLORE_FACTOR="1.5"

# Set this to a non-zero value to tell the script the config is complete.
APPROVED=0

if [ $APPROVED -eq 0 ]
then
    echo "Config not approved. Set APPROVED to 1 when you're sure the configuration is complete."
    exit
fi
