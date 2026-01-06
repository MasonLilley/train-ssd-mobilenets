0. Have a windows computer with an NVIDIA GPU and CUDA installed (check with ```nvidia-smi```)

1. Install and run Docker Desktop

2. Clone this repo

3a. Put your TFRecord as a zip file named ```dataset.zip``` under ```train-ssd-mobilenets/data/dataset.zip```.

3a. Put the provided ```limelight_ssd_mobilenet_v2_640x640a_coco17_tpu-8.config``` into the ```data/``` directory as well if you wish to train a higher-res model. Also, if you wish to change model resolution, change this line in train_limelight.py:

```'model_name': 'ssd_mobilenet_v2_640x640_coco17_tpu-8',```
to ```'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',```
or vice versa.

4. In this repo, run ```docker build --no-cache -t limelight-tflite .```

5. Once the docker container builds, start training with ```docker run -it -p 6006:6006 -v "${PWD}/data:/workspace/data" -v "${PWD}/train_limelight.py:/workspace/train_limelight.py" -v "${PWD}/correct.config:/workspace/data/correct.config" limelight-tflite python /workspace/train_limelight.py```

6. Export the output file to your project from the container with ```docker ps -a```, find your Container ID, then run ```docker cp <YOUR_CONTAINER_ID>:/workspace/final_output ./output```

**how it works:**
1. builds a docker container with python3.10 and tensorflow2.15. It needs to be exactly that because python>3.11 doesn't support tf2.15 and tf2.15 is needed for the 'estimator' module as well as easy access to the object detection api.

2. runs train_limelight.py in the docker container, along with copying your dataset to the container. train_limtlight.py 