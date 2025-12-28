# USE NVIDIA BASE IMAGE (Includes CUDA 12.1 and cuDNN 8)
# This is required for TF to find the GPU libraries correctly
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install System Dependencies + Python 3.10
# (Ubuntu 22.04 comes with Python 3.10, which works perfectly with TF 2.15)
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    unzip \
    protobuf-compiler \
    gnupg \
    sudo \
    libgl1 \
    libglib2.0-0 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Symlink python3 to python (so 'python' command works)
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python packages
# NOTE: We removed 'nvidia-*' packages because they are provided by the Base Image.
# Installing them via pip causes conflicts/duplicate registration errors.
RUN pip install --no-cache-dir \
    tensorflow==2.15.0 \
    protobuf==3.20.3 \
    tf-models-official==2.15.0 \
    gdown \
    ipywidgets \
    jupyterlab \
    matplotlib \
    pillow \
    google-cloud-storage

# Clone TensorFlow models repository at specific commit
RUN git clone --depth 1 https://github.com/tensorflow/models /workspace/models && \
    cd /workspace/models && \
    git fetch --depth 1 origin ad1f7b56943998864db8f5db0706950e93bb7d81 && \
    git checkout ad1f7b56943998864db8f5db0706950e93bb7d81

# Compile protobuf files
RUN cd /workspace/models/research && \
    protoc object_detection/protos/*.proto --python_out=.

# Fix setup.py for TF 2.15
RUN sed -i 's/tf-models-official>=2.5.1/tf-models-official==2.15.0/g' \
    /workspace/models/research/object_detection/packages/tf2/setup.py && \
    cp /workspace/models/research/object_detection/packages/tf2/setup.py \
    /workspace/models/research/setup.py

# Install TensorFlow Object Detection API
RUN pip install --no-cache-dir /workspace/models/research/

# Fix TF 2.15 breaking changes in tf_slim
RUN python -c "import site; print(site.getsitepackages()[0])" > /tmp/site_packages.txt && \
    SITE_PACKAGES=$(cat /tmp/site_packages.txt) && \
    sed -i '/from __future__ import print_function/a import tensorflow as tf' ${SITE_PACKAGES}/tf_slim/data/tfexample_decoder.py && \
    sed -i 's/control_flow_ops\.case/tf.case/g' ${SITE_PACKAGES}/tf_slim/data/tfexample_decoder.py && \
    sed -i 's/control_flow_ops\.cond/tf.compat.v1.cond/g' ${SITE_PACKAGES}/tf_slim/data/tfexample_decoder.py

# Install Coral EdgeTPU Compiler
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    apt-get update && \
    apt-get install -y edgetpu-compiler && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV HOMEFOLDER=/workspace/
ENV PYTHONPATH="${PYTHONPATH}:/workspace/models/research:/workspace/models/research/slim"

# Create working directories
RUN mkdir -p /workspace/training_progress /workspace/final_output

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]