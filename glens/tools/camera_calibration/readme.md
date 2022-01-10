### Calibration
```sh
python calibrate.py --g_streamer_pipeline 0 --checker_board_pattern 13 9 --length_in_meters 0.025 --output config.json
```

#### build docker image
```sh```
sudo docker built . -t streamer
```
#### example of running the calibration using docker 
```sh
sudo docker run -e DISPLAY=$DISPLAY  -it -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/app streamer python calibrateCamera.py --g_streamer_pipeline "rtspsrc location=rtsp://admin:Appstud2019@192.168.0.56:10554/udp/av0_0 latency=1 ! decodebin ! videoconvert ! appsink" --checker_board_pattern 13 9 --length_in_meter 0.03 --output config.json
```

#### save image from a source (video, ip-camera etc..) to a file for extrinsic matrix calibration
```sh
sudo docker run -e DISPLAY=$DISPLAY  -it -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/app streamer python calibrateCamera.py --g_streamer_pipeline "vid.mp4" --output calib.jpg
```
#### Calibrate extrinsic matrix 
```sh
sudo docker run -e DISPLAY=$DISPLAY -it -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/app streamer python homography_estimation_calibration.py --output config.json
```



```sh
python calibrateCameraExtrinsicWithChessboard.py --cameraIP 0 --checker_board_pattern 13 9 --length_in_meters 0.025 --output config.json
```
