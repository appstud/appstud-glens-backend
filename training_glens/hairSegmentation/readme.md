# Training and evaluation code for hair segmentation

## Summary
This code is for training a hair segmentation model.

<p align="center">
  <img width="300" height="129" src="src/data_for_readme/test.jpg">
</p>

## Prerequisites
All training and evaluation is done inside docker. So make sure to have docker installed, nvidia drivers and nvidia-container-runtime to allow access to GPU from inside docker.

Training is done inside docker because this remove the hassle of uninstalling and installing new versions of CUDA/CuDnn everytime a new version of tensorflow is released and that is incompatible with some CUDA versions.

### Access an NVIDIA GPU

Visit the official NVIDIA drivers page to download and install the proper drivers. Reboot your system once you have done so.

Verify that your GPU is running and accessible.

### Install nvidia-container-runtime

Follow the instructions at (https://nvidia.github.io/nvidia-container-runtime/) and then run this command:

```sh
apt-get install nvidia-container-runtime
```

Ensure the nvidia-container-runtime-hook is accessible from $PATH.

```sh
which nvidia-container-runtime-hook
```

## Build docker image

To build the docker image (mytensorflow) containing all the dependencies needed for the project:

```sh
cd docker
make
```

## Download the datasets

Make sure you download the dataset and place it next to the src folder if it is not available locally from [dataset](https://drive.google.com/file/d/1q6sMKmZo_1zQnvpifQ5lK1sq4Xv4yAOZ/view?usp=sharing)

## Training evaluation



### To list available scripts
```sh
cd src/
./docker_run_script.sh list-scripts
```
### Preprocess dataset

```sh
./docker_run_script.sh run preprocess_datasets.sh 
```

### train

```sh
./docker_run_script.sh run train.sh 
```



