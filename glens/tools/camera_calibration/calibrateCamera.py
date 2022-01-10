#!/usr/bin/env python
import cv2
import numpy as np
import os
import glob
import json
import argparse

def calibrate(cameraIP,length_in_meters,checkerboard,output):
    if(cameraIP=="0"):
        cameraIP=0

    # Defining the dimensions of checkerboard
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Creating vector to store vectors of 3D points for each checkerboard image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checkerboard image
    imgpoints = [] 
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0,:,:2] = length_in_meters*np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    
    
    cap=cv2.VideoCapture(cameraIP)
    #cap=cv2.VideoCapture('calibVideo.mp4')
    """
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024*1)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768*1)
    """
    count=0

    ret,img=cap.read()
    chosenFrame=np.zeros(img.shape).astype(np.uint8)
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    while(True):
        ret,img=cap.read()
        
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, checkerboard, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE) 
        #If desired number of corner are detected, we refine the pixel coordinates and display them on the images of checker board
        key=cv2.waitKey(10) 
        if(key & 0xff==ord('q')):
            break
        if ret == True:
            # refining pixel coordinates for given 2d points.
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            
            if(key & 0xff==ord('t')):  

                chosenFrame = cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
                objpoints.append(objp)
                imgpoints.append(corners2)
                count+=1
                print("Took image ",count)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
        
        cv2.putText(img,"Vary the distance/orientation of the checkerboard for better precision in the calibration" ,(10,20),cv2.FONT_HERSHEY_TRIPLEX,0.7,(0,0,255))
        cv2.putText(img,"Press q to quit, t to choose photo" ,(10,50),cv2.FONT_HERSHEY_TRIPLEX,0.7,(0,0,255))
        cv2.putText(img,"Number of chosen photos:"+str(count) ,(10,90),cv2.FONT_HERSHEY_TRIPLEX,0.7,(0,0,255))
        cv2.imshow('img',np.hstack((img,chosenFrame)))
        #cv2.imshow('img',img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    h,w = img.shape[:2]

    """
    Performing camera calibration by 
    passing the value of known 3D points (objpoints)
    and corresponding pixel coordinates of the 
    detected corners (imgpoints)
    """
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    name=input("Input the ID of the camera:")


    if(not os.path.exists(output)):
        open(output,'a').close()



    with open(output,'r',encoding='utf-8') as f:
        try:
            data=json.load(f)
        except Exception as e:
            data=dict()
            print(e)

    with open(output,'w') as f:
        if(name not in data.keys()):
            data[name]=dict()
        data[name]['alpha_u']=mtx[0][0]
        data[name]['alpha_v']=mtx[1][1]
        data[name]['c_x']=mtx[0][2]
        data[name]['c_y']=mtx[1][2]
        data[name]['distCoeff']=dist.tolist()[0]
        json.dump(data,f,ensure_ascii=False,indent=4)

    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)

if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description='Python script for camera calibration')
    parser.add_argument('--checker_board_pattern',type=int,nargs=2,help='2 numbers that defines checker board pattern for example: 9 6', default='9 6')
    parser.add_argument('--length_in_meters',type=float,nargs=1,help='Side length of a square in meters ')
    parser.add_argument('--g_streamer_pipeline',type=str, nargs=1,help='IP adress of the camera to calibrate or 0 if its a webcam', default="0")
    parser.add_argument('--output',type=str, nargs=1,help='Output file to save the data')
    args = parser.parse_args()
    calibrate(args.g_streamer_pipeline[0],args.length_in_meters[0],tuple(args.checker_board_pattern),args.output[0])

