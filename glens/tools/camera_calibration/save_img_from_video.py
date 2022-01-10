#!/usr/bin/env python
import cv2
import numpy as np
import os
import glob
import json
import argparse
import time
def calibrate(cameraIP,output):
    if(cameraIP=="0"):
        cameraIP=0
    
    cap=cv2.VideoCapture(cameraIP)
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    while(True):
        ret,img=cap.read()
        
        key=cv2.waitKey(10000)
        if(key & 0xff==ord('q')):
            break
        if ret == True:
            # refining pixel coordinates for given 2d points.
            cv2.imshow("img",img)
            cv2.waitKey(1)
            if(key & 0xff==ord('t')):  
                cv2.imwrite(output,img)
                print("saving image")
                

if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description='Python script for camera calibration')
    parser.add_argument('--g_streamer_pipeline',type=str, nargs=1,help='IP adress of the camera to calibrate or 0 if its a webcam', default="0")
    parser.add_argument('--output',type=str, nargs=1,help='Output file to save the image',default=["calib.jpg"])
    args = parser.parse_args()
    calibrate(args.g_streamer_pipeline[0],args.output[0])

