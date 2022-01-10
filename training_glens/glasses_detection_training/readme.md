# Training and evaluation code for glasses detection

## Summary
This code is for training and evaluation of a light neural network model for glasses detection.

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

## Download the dataset

Make sure you download the dataset if it is not available locally from [datasets](https://drive.google.com/file/d/1ubTDCEaqpBr93YpVnrFbM0vXEHs5M7-D/view?usp=sharing) and unzip it


### To list available scripts
```sh
./docker_run_script.sh list-scripts
```

## Preprocess your data

```sh
./docker_run_script.sh run prepare_dataset.sh
```
## Training
```sh
./docker_run_script.sh run train.sh
```

### Convert to tensorflow serving for deployment

```sh
./docker_run_script.sh run export_to_tfserving.sh
```



