#### build docker image 
sudo docker build . -t streamer

#### run container 
```sh
sudo docker run -e DISPLAY=$DISPLAY --link redis:redis -e CAM_ID=cam1 -e G_STREAMER_PIPELINE="rtspsrc location=rtsp://admin:Appstud2019@192.168.0.86:10554/udp/av0_0 latency=0 ! decodebin ! videoconvert ! appsink sync=False" -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/app streamer python -u stream.py
```
