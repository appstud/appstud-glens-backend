## Example docker-compose configurations for deploying GLENS

Some docker-compose configuration for launching some/all GLENS workers with docker-compose.

## Prerequisites
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

Restart the Docker daemon.

### GLENS
Before running docker-compose:

- Build the container images of the workers you want to use (See readme of each separate worker).
- Calibrate the cameras in case you want to estimate positions or if you want to use position estimates for tracking. 
- Supply the config.json file from your calibration experiments to redis (see inside a docker-compose.yml how it is done).

## Launch workers
```sh
docker-compose up
```





