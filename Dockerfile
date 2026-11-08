# base image with cuda 12.1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# install python 3.11 and pip
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# install uv
RUN pip install uv

# create venv
ENV PATH="/.venv/bin:${PATH}"
RUN uv venv --python 3.11 /.venv

# install dependencies (pip too: the venv has none, and tools such as the
# Flash runtime used by CI run `python3 -m pip`)
RUN uv pip install pip torch --extra-index-url https://download.pytorch.org/whl/cu121 diffusers transformers accelerate safetensors xformers==0.0.23 runpod numpy==1.26.3 scipy triton huggingface-hub hf_transfer setuptools Pillow

# copy files
COPY download_weights.py schemas.py handler.py test_input.json /

# download the weights from hugging face
RUN python /download_weights.py

# run the handler
CMD python -u /handler.py