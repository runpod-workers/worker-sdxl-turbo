# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/base:0.6.2-cuda12.4.1

# The base image comes with many system dependencies pre-installed to help you get started quickly.
# Please refer to the base image's Dockerfile for more information before adding additional dependencies.
# IMPORTANT: The base image overrides the default huggingface cache location.


# --- Optional: System dependencies ---
# COPY builder/setup.sh /setup.sh
# RUN /bin/bash /setup.sh && \
#     rm /setup.sh


# Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# NOTE: The base image comes with multiple Python versions pre-installed.
#       It is reccommended to specify the version of Python when running your code.


# Add src files (Worker Template)
ADD src .

#RUN python3.11 /handler.py

RUN mkdir /models
RUN wget https://huggingface.co/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors -O /models/RealVisXL_V5.0_fp16.safetensors
RUN wget https://huggingface.co/SG161222/RealVisXL_V4.0/resolve/main/RealVisXL_V4.0.safetensors -O /models/RealVisXL_V4.0_fp16.safetensors

RUN python3.11 /install_models.py

CMD python3.11 -u /handler.py
