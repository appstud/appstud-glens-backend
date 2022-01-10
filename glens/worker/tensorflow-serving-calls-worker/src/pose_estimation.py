import cv2
import numpy as np
import pdb
from math import cos, sin,asin,atan2,acos


def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x-y) <= atol + rtol * abs(y)

#ForR=Rx.Ry.Rz
def getEulerAnglesFromRotationMatrix(R):
    if isclose(R[0,2],-1.0):
        yaw = np.pi/2.0
        roll = atan2(R[0,1],R[0,2])
    
    elif isclose(R[2,0],1.0):
        yaw = -np.pi/2.0
        pitch = atan2(-R[0,1],-R[0,2])

    else:
        yaw = asin(R[0,2])
        cos_theta =cos(yaw)
        pitch = atan2(-R[1,2]/cos_theta, R[2,2]/cos_theta)
        roll = atan2(-R[0,1]/cos_theta, R[0,0]/cos_theta)
    
    return (180/np.pi)*roll, +(180/np.pi)*yaw,(180/np.pi)*pitch


def estimatePose(bbox, yaw, roll, pitch, intrinsicMatrix, distCoeff, rigidTransformToWCS=np.eye(4)):
    """
    #Weak perspective euler angles estimation using FSANET
    yaw,roll,pitch=HeadPoseEstimator.estimateOrientation(faceImage)
    """
    #Estimate translation vector based on simple face model
    tx,ty,tz=estimateXYZ(intrinsicMatrix, bbox,roll*np.pi/180.0, pitch*np.pi/180.0)
    #Build extrinsic matrix from translation vectors and euler angles 
    extrinsicMatrix=getExtrinsicMatrix(tx,ty,tz,yaw,pitch,roll)
    #Correct estimation to be real full perspective
    extrinsicMatrix=correctRotationMatrix(intrinsicMatrix, distCoeff, extrinsicMatrix)
    
    rollCorr,yawCorr,pitchCorr=getEulerAnglesFromRotationMatrix(extrinsicMatrix[0:3,0:3]) 
        
    txCorr=extrinsicMatrix[0,3]
    tyCorr=extrinsicMatrix[1,3]
    tzCorr=extrinsicMatrix[2,3]
    #######return roll,yaw,pitch,rollCorr,yawCorr,pitchCorr,txCorr,tyCorr,tzCorr,roll,yaw,pitch,extrinsicMatrix
    
    #return roll,yaw,pitch,rollCorr,yawCorr,pitchCorr,txCorr,tyCorr,tzCorr,roll,yaw,pitch,np.dot(self.rigidTransformToWCS,extrinsicMatrix)
    return rollCorr,yawCorr, pitchCorr, txCorr, tyCorr, tzCorr, extrinsicMatrix

def projectPointsWeakPerspective(intrinsicMatrix, extrinsicMatrix, _3DPoints):
    _2DPoints=np.dot(extrinsicMatrix,np.vstack((_3DPoints.T,np.ones([1,_3DPoints.shape[0]]))))
    """_2DPoints[2,:]=extrinsicMatrix[2,3]

    _2DPoints=intrinsicMatrix.dot(_2DPoints)
    _2DPoints=_2DPoints/float(extrinsicMatrix[2,3])
    """
    alpha_u=intrinsicMatrix[0,0] 
    alpha_v=intrinsicMatrix[1,1] 
    c_x=intrinsicMatrix[0,2] 
    c_y=intrinsicMatrix[1,2]
    
    _2DPoints[0,:]=alpha_u*(_2DPoints[0,:]/float(extrinsicMatrix[2,3]))+c_x
    _2DPoints[1,:]=alpha_v*(_2DPoints[1,:]/float(extrinsicMatrix[2,3]))+c_y

    return _2DPoints[0:2,:].T.astype(np.float32)


def correctRotationMatrix(intrinsicMatrix, distCoeff, extrinsicMatrix,_3DPoints=np.array([[0,0,0],[0.08, 0, 0], [0, -0.08, 0], [0, 0, -0.3], [0, 0, -0.2]]).astype(np.float32)):
    """alpha_u=intrinsicMatrix[0,0] 
    alpha_v=intrinsicMatrix[1,1] 
    c_x=intrinsicMatrix[0,2] 
    c_y=intrinsicMatrix[1,2]
    """
    _2DPoints=projectPointsWeakPerspective(intrinsicMatrix, extrinsicMatrix,_3DPoints)
    
    ret, rvec, tvec=cv2.solvePnP(_3DPoints,_2DPoints,intrinsicMatrix,distCoeff,rvec=cv2.Rodrigues(extrinsicMatrix[0:3,0:3])[0],tvec=extrinsicMatrix[:,3],useExtrinsicGuess=True,flags=cv2.SOLVEPNP_ITERATIVE)
    #ret, rvec, tvec=cv2.solvePnP(_3DPoints,_2DPoints,intrinsicMatrix,distCoeff,rvec=np.array([[0],[0],[0]]).astype(np.float32),tvec=extrinsicMatrix[:,3],useExtrinsicGuess=True,flags=cv2.SOLVEPNP_ITERATIVE)
    R=cv2.Rodrigues(rvec)[0]
    
    return np.hstack((R,extrinsicMatrix[:,3]))
    #return np.hstack((R,tvec))

def rotationMatrixFromEulerAngles(thetax, thetay, thetaz):
    thetax = -(thetax * np.pi) / 180.0
    #thetax = (thetax * np.pi) / 180.0
    thetaz = (thetaz * np.pi) / 180.0
    #thetay =(thetay * np.pi) / 180.0
    thetay =+(thetay * np.pi) / 180.0
    #thetay =np.pi-(thetay * np.pi) / 180.0

    Rx=np.matrix([[1,0,0],[0,cos(thetax),-sin(thetax)],[0,sin(thetax),cos(thetax)]]).astype(np.float64)
    Ry=np.matrix([[cos(thetay),0,sin(thetay)],[0,1,0],[-sin(thetay),0,cos(thetay)]]).astype(np.float64)
    Rz=np.matrix([[cos(thetaz),-sin(thetaz),0],[sin(thetaz),cos(thetaz),0],[0,0,1]]).astype(np.float64)

    #return np.dot(Rz,np.dot(Ry,Rx))
    #return np.dot(np.dot(Rx,np.dot(Ry,Rz)) ,np.array([[1,0,0],[0,-1,0],[0,0,-1]]))
    #return np.dot(np.array([[-1,0,0],[0,1,0],[0,0,-1]]),np.dot(Rx,np.dot(Ry,Rz)))
    return np.dot(Rx,np.dot(Ry,Rz))

def getExtrinsicMatrix(tx,ty,tz,yaw,pitch,roll):
    return np.hstack((rotationMatrixFromEulerAngles(pitch,yaw,roll),np.array([[tx,ty,tz]]).T))


def estimateXYZ(intrinsicMatrix, bbox, roll, pitch, l=0.15):
    """bbox: coordinate of the upper corner of bounding box , followed by its width and height
       intrinsicMatrix: The intrinsicMatrix of the camera
       l: length of the face in cm fixed to be 15
       returns tx,ty,tz coordinate of the center of the face wrt to the camera cds
    """
    
    alpha_u=intrinsicMatrix[0,0] 
    alpha_v=intrinsicMatrix[1,1] 
    c_x=intrinsicMatrix[0,2] 
    c_y=intrinsicMatrix[1,2]

    ####x1,y1,x2,y2=bbox[0],bbox[1],bbox[2],bbox[3]
    x1,y1,x2,y2=bbox[0],bbox[1],bbox[2]+bbox[0],bbox[3]+bbox[1]
    #tz=self.alpha_v*(l*cos(roll)*cos(pitch))/(y2-y1)
    #dont take pitch into account because of face detection problem (bounding box not accurate with pitch especially at close distance)
    ##estimation problem when the face is looking up (bounding box not accurate...)
    tz=(alpha_v*l*cos(roll))/(y2-y1)
    #tz=(self.alpha_v*l)/(y2-y1)
    ###tx=tz*(x1-self.c_x)/self.alpha_u
    tx=tz*(x1+0.5*(x2-x1)-c_x)/alpha_u
    ###ty=tz*(y1-self.c_y)/self.alpha_v
    ty=tz*(y1+0.5*(y2-y1)-c_y)/alpha_v

    #return tx+l/4,ty+l/2,tz
    return tx,ty,tz
