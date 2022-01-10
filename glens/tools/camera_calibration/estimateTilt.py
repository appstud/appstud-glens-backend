import cv2
import numpy as np
import json
from math import cos ,sin

def getIntrinsicFromFile(output,name):
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
    
    return alpha_u,alpha_v,c_x,c_y

def estimateTilt(_2Dpoints,cx,cy,alpha_u,alpha_v,h=1,r=0):
    
    tilts=np.linspace(-90,90,180)
    x=_2Dpoints[:,0]
    y=_2Dpoints[:,1]
    dotProduct=[]
    for tilt in tilts:
        tilt=(np.pi*tilt/180.0)

        X=((x-cx)*alpha_u*(r*sin(tilt)-h))/(alpha_u*(cy-y)*cos(tilt)-alpha_v*sin(tilt))
        
        Z=(((cy-y)*sin(tilt)+alpha_v*cos(tilt))*(r*sin(tilt)-h))/((cy-y)*cos(tilt)-alpha_v*sin(tilt))-r+r*cos(tilt)

        
        X=X-X[1] 
        Z=Z-Z[1] 
       
        dotProduct.append(X[0]*X[2]+Z[0]*Z[2]+h**2)
    
        #print(dotProduct,(180*tilt/np.pi))

    dotProduct=list(map(abs,dotProduct))
    print(tilts[dotProduct.index(min(dotProduct))],min(dotProduct))


if(__name__=="__main__"):
    _2Dpoints=np.array([[0,0],[100,200],[100,400]])


    alpha_u,alpha_v,cx,cy=getIntrinsicFromFile("config.json","cam1")
    
    estimateTilt(_2Dpoints,cx,cy,alpha_u,alpha_v,h=1,r=0)



  
