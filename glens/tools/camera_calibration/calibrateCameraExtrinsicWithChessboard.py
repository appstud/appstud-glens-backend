import cv2
import numpy as np
import drawutils
import pdb
import json
from math import asin,acos,atan2,cos,sin
import argparse

#intrinsicMatrix=np.asmatrix(np.array([[933.38537489 ,  0. ,  642.8541392 ],[  0.,933.08605844, 367.24814261],[  0.   , 0. ,  1.   ]]))
#distCoeff=np.array([-0.14454454,  0.48699891,  0.0013472,   0.00407075, -0.54467429] )

def getEulerAnglesFromRotationMatrix(R):
    
    def isclose(x, y, rtol=1.e-5, atol=1.e-8):
        return abs(x-y) <= atol + rtol * abs(y)

    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    phi = 0.0

    #R=np.dot(np.array([[-1,0,0],[0,1,0],[0,0,-1]]),R)
    #R=np.dot(R,np.array([[1,0,0],[0,1,0],[0,0,-1]]))
    if isclose(R[0,2],-1.0):
        yaw = np.pi/2.0
        roll = atan2(R[0,1],R[0,2])
        print("yaw close to 90")
    elif isclose(R[2,0],1.0):
        yaw = -np.pi/2.0
        pitch = atan2(-R[0,1],-R[0,2])
        print("yaw close to -90")

    else:
        yaw = asin(R[0,2])
        cos_theta =cos(yaw)
        pitch = atan2(-R[1,2]/cos_theta, R[2,2]/cos_theta)
        roll = atan2(-R[0,1]/cos_theta, R[0,0]/cos_theta)

    return (180/np.pi)*roll, +(180/np.pi)*yaw,(180/np.pi)*pitch



def startCalibration(cameraIP,length_in_meters,checker_board,output,name=None,returnAngles=True):
    #Defining the dimensions of checker_board
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Creating vector to store vectors of 3D points for each checker_board image
    objpoints = []
    # Creating vector to store vectors of 2D points for each checker_board image
    imgpoints = []
    length_in_meters=0.025
    # Defining the world coordinates for 3D points
    objp = np.zeros((1, checker_board[0] * checker_board[1], 3), np.float32)
    objp[0,:,:2] = length_in_meters*np.mgrid[0:checker_board[0], 0:checker_board[1]].T.reshape(-1, 2)


    
    calibrate=False
    cv2.namedWindow("image",cv2.WINDOW_NORMAL)
    if(cameraIP=="0"):
        cap=cv2.VideoCapture(0)
    else:
        cap=cv2.VideoCapture(cameraIP)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024*1)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768*1)
        
    if(name is None):
        name=input("Please enter name of the camera in order to save data to "+output+" file and to read intrinsicMatrix:")
    
 
    with open(output,'r',encoding='utf-8') as f:
        try:
            data=json.load(f)
            print(data)
        except Exception as e:
            print(e)
            print(output+" does not exist, create it!!")
            raise Exception(output + " does not exist, create it!!")
            data=dict()
        if (name not in data.keys()):
            data[name]=dict()
            raise Exception("Camera intrinsic does not exist for this camera...calibrate it before using!")
            print(data)
        else:
            alpha_u=data[name]["alpha_u"]
            alpha_v=data[name]["alpha_v"]
            c_x=data[name]["c_x"]
            c_y=data[name]["c_y"]
            distCoeff=np.array(data[name]["distCoeff"])
            intrinsicMatrix=np.array([[alpha_u,0,c_x],[0,alpha_v,c_y],[0,0,1]])
    while(not calibrate):  
        ret,img=cap.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checker_board, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if(ret):
            corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
            if(cv2.waitKey(10)& 0xff==ord('t')):
                objpoints.append(objp)
                imgpoints.append(corners2)
                calibrate=True
            img = cv2.drawChessboardCorners(img, checker_board, corners2, ret)
        cv2.imshow('img',img)
        cv2.waitKey(1)
    objpoints=objpoints[0][0]
    imgpoints=imgpoints[0].reshape(-1,2)
    
    #ret, rvec, tvec=cv2.solvePnP(np.array(_3DPoints).astype(np.float32),np.array(_2DPoints).astype(np.float32),intrinsicMatrix,distCoeff,flags=cv2.SOLVEPNP_ITERATIVE)
    ret, rvec, tvec=cv2.solvePnP(np.array(objpoints).astype(np.float32),np.array(imgpoints).astype(np.float32),intrinsicMatrix,distCoeff,flags=cv2.SOLVEPNP_UPNP)
    
    rotationMatrix=cv2.Rodrigues(rvec)[0]
    drawutils.drawCoordinateSystems(intrinsicMatrix,np.hstack((rotationMatrix,tvec)),img,_3Dpoints=np.array([[0,0,0],[0.3, 0, 0], [0, 0.3, 0], [0, 0, -0.3]]))
    #drawutils.drawCoordinateSystems(intrinsicMatrix,np.hstack((rotationMatrix,tvec)),img,_3Dpoints=np.array([[0,0,0],[0.3, 0, 0], [0, 0.3, 0], [0, 0, +0.3]]))
    #drawutils.drawCoordinateSystems(intrinsicMatrix,np.hstack((rotationMatrix,tvec)),img,_3Dpoints=np.array(_3DPoints))


    roll,yaw,pitch=getEulerAnglesFromRotationMatrix(rotationMatrix)
    
    data[name]["roll"]=roll
    data[name]["yaw"]=yaw
    data[name]["pitch"]=pitch
    
    with open(output,'w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False,indent=4)
    print(data)
    cv2.imshow("img",img)
    cv2.waitKey(100000) 

    
    print("Rotation matrix:", rotationMatrix)
    print("Translation vector:", tvec)
    
    if(returnAngles):
        return pitch, yaw, roll, np.hstack((rotationMatrix,tvec))

if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description='Python script for estimating the 3D transformation of world coordinate system (checker boardon the ground) and camera coordinate system')
    parser.add_argument('--checker_board_pattern',type=int,nargs=2,help='2 numbers that defines checker board pattern for example: 9 6', default='9 6')
    parser.add_argument('--length_in_meters',type=float,nargs=1,help='Side length of a square in meters ')
    parser.add_argument('--camera_IP',type=str, nargs=1,help='IP adress of the camera to calibrate or 0 if its a webcam',default='0')
    parser.add_argument('--output',type=str, nargs=1,help='Output file to save the data')
    args = parser.parse_args()

    startCalibration(args.camera_IP[0],args.length_in_meters[0],tuple(args.checker_board_pattern),args.output[0],returnAngles=False)

