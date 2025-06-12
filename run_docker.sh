#!/bin/bash

docker build -t nnue-pytorch .

echo "Enter the path to your data directory to mount into the container: "
read DATA_PATH

DATA_PATH=${DATA_PATH}
echo "Using data path: $DATA_PATH"

echo "Creating new container 'nnue-container'..."
docker run -it \
  --gpus all \
  -u `id -u` \
  -v "$(pwd)":/workspace/nnue-pytorch \
  -v "$DATA_PATH":/data \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nnue-pytorch

docker run -it -v /host/path:/container/path nnue-pytorch