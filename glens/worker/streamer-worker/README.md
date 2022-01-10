#### build docker image
```sh 
make
```

#### remove docker image
```sh
make clean
```
#### Allow display inside docker
```sh
xhost + 
```
#### Example of running the container on a set of ip cameras 
```sh
sudo docker run --net=host -e DISPLAY=$DISPLAY --env-file .env -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd)/streams_configuration_files:/app/streams_configuration_files -v $(pwd)/videos:/app/videos  streamer python -u streamer/stream_to_proxy.py
```

### Just for dev and testing
```sh
sudo docker run --net=host -e DISPLAY=$DISPLAY --env-file .env -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd)/streams_configuration_files:/app/streams_configuration_files -v $(pwd)/videos:/app/videos -v $(pwd)/src:/app streamer python -u streamer/stream_to_proxy.py
```
