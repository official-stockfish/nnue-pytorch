#!/bin/bash

set -e

IMAGE_BASE_NAME="nnue-pytorch"

GPU_INPUT=""
DATA_PATH=""
SKIP_SETUP="false"
INTERACTIVE="true"
EXEC_ARGS=()

# 1. Parse up to two positional arguments for interactive replacements
if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
  GPU_INPUT=$(echo "$1" | tr '[:lower:]' '[:upper:]')
  shift
fi

if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
  DATA_PATH="$1"
  shift
fi

# 2. Parse optional flags
while [[ $# -gt 0 ]]; do
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
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
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

# Checking if docker is in rootless mode.
if docker info 2>/dev/null | grep -iq "rootless"; then
    echo "Rootless mode detected."
    USER_FLAG="--user 0:0"
else
    echo "Standard mode detected."
    USER_FLAG="--user $(id -u):$(id -g)"
fi

if [ "$INTERACTIVE" = "true" ]; then
  INTERACTIVE_FLAGS="-it"
else
  INTERACTIVE_FLAGS=""
fi

echo "Creating new container 'nnue-container'..."
docker run $INTERACTIVE_FLAGS \
  $GPU_FLAGS \
  $USER_FLAG \
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
      echo "Running setup script inside container..."
      /workspace/nnue-pytorch/setup_script.sh
      echo "Setup complete."
    fi

    if [ $# -gt 0 ]; then
      echo "Executing command: $@"
      "$@"
      RESULT=$?

      if [ $RESULT -ne 0 ]; then
        echo "Command failed with status $RESULT"
      fi
    fi

    if [ "$INTERACTIVE" = "true" ]; then
      echo "Entering interactive shell..."
      exec bash
    else
      echo "Exiting container with exit code ${RESULT:-0}."
      exit ${RESULT:-0}
    fi
  ' -- "$SKIP_SETUP" "$INTERACTIVE" "${EXEC_ARGS[@]}"
