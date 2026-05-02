#!/bin/bash

# Script that build and runs docker container for chosen accelerator.
# ./run_docker.sh <ACCELERATOR: NVIDIA/AMD/CPU> <data_path_to_be_mounted> [<flags: skip-setup/non-interactive>] [--exec <command_to_run_inside_container>]

set -e

IMAGE_BASE_NAME="nnue-pytorch"

GPU_INPUT=""
DATA_PATH=""
SKIP_SETUP="false"
INTERACTIVE="true"
EXEC_ARGS=()

# 2. Parse arguments
while [[ $# -gt 0 ]]; do
  if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
    if [ -n "$GPU_INPUT" ] && [ -n "$DATA_PATH" ]; then
      echo "Error: Too many positional arguments. Expected GPU brand and data path only."
      exit 1
    fi
    if [ -z "$GPU_INPUT" ]; then
      GPU_INPUT=$(echo "$1" | tr '[:lower:]' '[:upper:]')
    elif [ -z "$DATA_PATH" ]; then
      DATA_PATH="$1"
    fi
    shift
  else
    case $1 in
      --skip-setup)
        SKIP_SETUP="true"
        shift
        ;;
      --non-interactive)
        INTERACTIVE="false"
        shift
        ;;
      --exec)
        shift
        EXEC_ARGS=("$@")
        # Consume all remaining arguments as the execution command
        break
        ;;
      --help)
        echo "Usage: $0 [GPU_BRAND] [DATA_PATH] [--skip-setup] [--non-interactive] [--exec <command>]"
        echo "  GPU_BRAND: NVIDIA, AMD, or CPU (optional, will prompt if not provided)."
        echo "  DATA_PATH: Path to data directory to mount (optional, will prompt if not provided)."
        echo "  --skip-setup: Skip running the setup script inside the container."
        echo "  --non-interactive: Run the container in non-interactive mode (no shell). Does not prevent prompts for GPU and data path if not provided."
        echo "  --exec <command>: Command to execute inside the container instead of starting a shell."
        echo "  Note: --exec must be used at the end and does not imply --non-interactive, but typically used together."
        exit 0
        ;;
      *)
        echo "Unknown argument: $1. Use --help for usage information."
        exit 1
        ;;
    esac
  fi
done

# 3. Handle GPU selection
if [ -z "$GPU_INPUT" ]; then
  echo "Please select the target GPU brand to build for:"
  select brand in "NVIDIA" "AMD" "CPU"; do
    case $brand in
      NVIDIA ) GPU_INPUT="NVIDIA"; break;;
      AMD )    GPU_INPUT="AMD"; break;;
      CPU )    GPU_INPUT="CPU"; break;;
    esac
  done
fi

case "$GPU_INPUT" in
  NVIDIA )
    GPU_TYPE="nvidia"
    DOCKERFILE="Dockerfile.NVIDIA"
    IMAGE_TAG="${IMAGE_BASE_NAME}:nvidia"
    GPU_FLAGS="--gpus all"
    echo "Selected NVIDIA build."
    ;;
  AMD )
    GPU_TYPE="amd"
    DOCKERFILE="Dockerfile.AMD"
    IMAGE_TAG="${IMAGE_BASE_NAME}:amd"
    GPU_FLAGS="--device /dev/kfd --device /dev/dri"
    echo "Selected AMD build."
    ;;
  CPU )
    GPU_TYPE="none"
    DOCKERFILE="Dockerfile.CPU"
    IMAGE_TAG="${IMAGE_BASE_NAME}:cpu"
    GPU_FLAGS=""
    echo "Selected CPU build."
    ;;
  * )
    echo "Invalid GPU brand: $GPU_INPUT. Must be NVIDIA, AMD, or CPU."
    exit 1
    ;;
esac

echo "Building image $IMAGE_TAG from $DOCKERFILE"
docker build -t "$IMAGE_TAG" -f "$DOCKERFILE" .

# 4. Handle Data Path selection
if [ -z "$DATA_PATH" ]; then
  echo "Enter the path to your data directory to mount into the container: "
  read -r DATA_PATH
fi

echo "Using data path: $DATA_PATH"

# 5. Handle user mapping and container home directory setup
CONTAINER_HOME="$(pwd)/.container_home"
mkdir -p "$CONTAINER_HOME"

if docker info 2>/dev/null | grep -iq "rootless"; then
    echo "Rootless mode detected. Mapping container HOME to /root."
    HOST_UID=0
    HOST_GID=0
    INTERNAL_HOME="/root"
else
    echo "Standard mode detected. Mapping container HOME to /home/nnue_user."
    HOST_UID=$(id -u)
    HOST_GID=$(id -g)
    INTERNAL_HOME="/home/nnue_user"
fi


if [ "$INTERACTIVE" = "true" ]; then
  INTERACTIVE_FLAGS="-it"
else
  INTERACTIVE_FLAGS=""
fi

echo "Creating new container 'nnue-container'..."
# Note: running as root without `--user` as entrypoint script will handle
# user switching based on the mode (rootless vs standard).
docker run $INTERACTIVE_FLAGS \
  $GPU_FLAGS \
  -e HOST_UID=$HOST_UID \
  -e HOST_GID=$HOST_GID \
  -e INTERNAL_HOME="$INTERNAL_HOME" \
  -v "$CONTAINER_HOME":"$INTERNAL_HOME" \
  -v "$(pwd)":/workspace/nnue-pytorch \
  -v "$DATA_PATH":/data \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  $IMAGE_TAG \
  bash -c '
    SKIP_SETUP=$1
    INTERACTIVE=$2
    shift 2

    if [ "$SKIP_SETUP" != "true" ]; then
      echo "[RUN_DOCKER] Running setup script inside container..."
      if ! bash -e /workspace/nnue-pytorch/setup_script.sh; then
        echo "[RUN_DOCKER] Setup failed."
        exit 1
      fi
      echo "[RUN_DOCKER] Setup complete."
    fi

    if [ $# -gt 0 ]; then
      echo "[RUN_DOCKER] Executing command: $@"
      "$@"
      RESULT=$?

      if [ $RESULT -ne 0 ]; then
        echo "[RUN_DOCKER] Command failed with status $RESULT"
      fi
    fi

    if [ "$INTERACTIVE" = "true" ]; then
      echo "[RUN_DOCKER] Entering interactive shell..."
      exec bash
    else
      echo "[RUN_DOCKER] Exiting container with exit code ${RESULT:-0}."
      exit ${RESULT:-0}
    fi
  ' -- "$SKIP_SETUP" "$INTERACTIVE" "${EXEC_ARGS[@]}"
