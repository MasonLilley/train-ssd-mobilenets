# USE NVIDIA BASE IMAGE (Includes CUDA 12.1 and cuDNN 8)
# This is required for TF to find the GPU libraries correctly
FROM ubuntu:latest

# Set working directory
WORKDIR /workspace

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install System Dependencies + Python 3.10
# (Ubuntu 22.04 comes with Python 3.10, which works perfectly with TF 2.15)
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
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
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip (ignore-installed to avoid conflict with Debian package)
RUN python -m pip install --upgrade pip --break-system-packages --ignore-installed

# Install Python packages
# NOTE: We removed 'nvidia-*' packages because they are provided by the Base Image.
# Installing them via pip causes conflicts/duplicate registration errors.
# Using TF 2.18 ecosystem with compatible versions (tensorflow-text only available for 2.18+)
RUN pip install --no-cache-dir \
    tensorflow==2.18.0 \
    tensorflow-text==2.18.1 \
    protobuf==5.28.0 \
    tf-models-official==2.18.0 \
    gdown \
    ipywidgets \
    jupyterlab \
    matplotlib \
    pillow \
    google-cloud-storage --break-system-packages

# Clone TensorFlow models repository - use main branch for TF 2.18 compatibility
RUN git clone https://github.com/tensorflow/models /workspace/models

# Verify directory structure and compile protobuf files
RUN if [ -d "/workspace/models/research" ]; then \
        cd /workspace/models/research && \
        protoc object_detection/protos/*.proto --python_out=.; \
    else \
        echo "ERROR: research directory not found" && exit 1; \
    fi

# Copy setup.py from the tf2 package
RUN cp /workspace/models/research/object_detection/packages/tf2/setup.py \
    /workspace/models/research/setup.py

# Install TensorFlow Object Detection API
RUN pip install --no-cache-dir /workspace/models/research/ --break-system-packages

# Fix TF 2.15 breaking changes in tf_slim
RUN python -c "import site; print(site.getsitepackages()[0])" > /tmp/site_packages.txt && \
    SITE_PACKAGES=$(cat /tmp/site_packages.txt) && \
    sed -i '/from __future__ import print_function/a import tensorflow as tf' ${SITE_PACKAGES}/tf_slim/data/tfexample_decoder.py && \
    sed -i 's/control_flow_ops\.case/tf.case/g' ${SITE_PACKAGES}/tf_slim/data/tfexample_decoder.py && \
    sed -i 's/control_flow_ops\.cond/tf.compat.v1.cond/g' ${SITE_PACKAGES}/tf_slim/data/tfexample_decoder.py

# Install Coral EdgeTPU Compiler
# RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
#     echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
#     apt-get update && \
#     apt-get install -y edgetpu-compiler && \
#     rm -rf /var/lib/apt/lists/*
# RUN mkdir -p /etc/apt/keyrings && \
#     curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
#       | gpg --dearmor -o /etc/apt/keyrings/coral.gpg && \
#     echo "deb [signed-by=/etc/apt/keyrings/coral.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
#       | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
#     apt-get update && \
#     apt-get install -y edgetpu-compiler && \
#     rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV HOMEFOLDER=/workspace/
ENV PYTHONPATH="${PYTHONPATH}:/workspace/models/research:/workspace/models/research/slim"

# Create working directories
RUN mkdir -p /workspace/training_progress /workspace/final_output

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]