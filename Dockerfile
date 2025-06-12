FROM nvcr.io/nvidia/pytorch:25.03-py3

RUN apt-get update && apt-get install -y \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/nnue-pytorch

COPY requirements.txt setup_script.sh ./

RUN pip install --no-cache-dir -r requirements.txt

RUN chmod +x setup_script.sh

ENTRYPOINT [ "/workspace/nnue-pytorch/setup_script.sh" ]
CMD ["/bin/bash"]