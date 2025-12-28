FROM python:3.11-slim

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    protobuf-compiler \
    gnupg \
    sudo \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages with specific versions
RUN pip install --no-cache-dir \
    tensorflow[and-cuda]==2.15.0 \
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

# Start Jupyter Lab by default
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]