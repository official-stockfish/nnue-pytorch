#!/bin/bash

set -e

IMAGE_BASE_NAME="nnue-pytorch"

echo "Please select the target GPU brand to build for:"
select brand in "NVIDIA" "AMD"; do
  case $brand in
    NVIDIA ) GPU_TYPE="nvidia"; break;;
    AMD )    GPU_TYPE="amd"; break;;
  esac
done

if [ "$GPU_TYPE" == "nvidia" ]; then
  DOCKERFILE="Dockerfile.NVIDIA"
  IMAGE_TAG="${IMAGE_BASE_NAME}:nvidia"
  GPU_FLAGS="--gpus all"
  echo "Selected NVIDIA build."
elif [ "$GPU_TYPE" == "amd" ]; then
  DOCKERFILE="Dockerfile.AMD"
  IMAGE_TAG="${IMAGE_BASE_NAME}:amd"
  GPU_FLAGS="--device /dev/kfd --device /dev/dri"
  echo "Selected AMD build."
fi

echo "Building image $IMAGE_TAG from $DOCKERFILE"
docker build -t "$IMAGE_TAG" -f "$DOCKERFILE" .

echo "Enter the path to your data directory to mount into the container: "
read DATA_PATH

DATA_PATH=${DATA_PATH}
echo "Using data path: $DATA_PATH"

echo "Creating new container 'nnue-container'..."
docker run -it \
  $GPU_FLAGS \
  -u `id -u` \
  -v "$(pwd)":/workspace/nnue-pytorch \
  -v "$DATA_PATH":/data \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  $IMAGE_TAG
