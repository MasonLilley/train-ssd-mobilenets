0. Have a windows computer with an NVIDIA GPU and CUDA installed (check with ```nvidia-smi```)

1. Install and run Docker Desktop

2. Clone this repo

3. In this repo, run ```docker build -t limelight-tflite .```

4. Once the docker container builds, start training with ```docker run -it --gpus all -p 6006:6006 -v "${PWD}/data:/workspace/data" -v "${PWD}/train_limelight.py:/workspace/train_limelight.py" limelight-tflite python /workspace/train_limelight.py```