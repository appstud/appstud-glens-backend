i=$1
#gst-launch-1.0 rtspsrc location=rtsp://admin:Appstud2019@192.168.0.${i}:10554/udp/av0_0 latency=0 ! decodebin ! videoconvert ! appsink drop=True

curl -v -X DESCRIBE "admin:Appstud2019@192.168.0.${i}:10554/udp/av0_0" RTSP/1.0

#gst-launch-1.0 rtspsrc location=rtsp://admin:admin@192.168.0.${i}:10554/udp/av0_0 latency=0 ! decodebin ! videoconvert ! appsink drop=True

