import numpy as np
import pdb
from math import cos, sin,asin,atan2,acos


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

def getExtrinsicMatrix(tx,ty,tz,roll,yaw,pitch):
    return np.hstack((rotationMatrixFromEulerAngles(-pitch,yaw,roll),np.array([[tx,ty,tz]]).T))
