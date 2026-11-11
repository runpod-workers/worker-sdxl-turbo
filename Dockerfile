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
# xformers 0.0.23 holds torch at 2.1.x, so the libraries on top are pinned
# to a set that imports on it (the one worker-sdxl uses). diffusers 0.41+
# calls torch.accelerator, which torch 2.1 does not have.
RUN uv pip install pip torch --extra-index-url https://download.pytorch.org/whl/cu121 diffusers==0.33.1 transformers==4.52.4 accelerate==1.7.0 safetensors==0.5.3 xformers==0.0.23 runpod numpy==1.26.3 scipy triton huggingface-hub hf_transfer setuptools Pillow

# copy files
COPY download_weights.py schemas.py handler.py test_input.json /

# download the weights from hugging face
RUN python /download_weights.py

# run the handler
CMD python -u /handler.py