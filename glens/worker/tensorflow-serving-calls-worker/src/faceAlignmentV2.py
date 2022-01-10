import cv2
import os
import numpy as np
import pdb 


def getPersonsImgPatches(img,persons_bbox):
    croppedPersons=[]
    
    (h, w) = img.shape[:2]
    for b in persons_bbox:
        x1, y1, wbbox, hbbox = b[0], b[1], b[2], b[3]
        x2 = x1+wbbox
        y2 = y1+hbbox
        left=int(x1)
        right=int(x2)
        top=int(y1)
        bottom=int(y2)
        xw1 = max(left, 0)
        yw1 = max(top, 0)
        xw2 = min(right, w - 1)
        yw2 = min(bottom, h - 1)
        croppedPersons.append(cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :],(256,128)))
    
    return croppedPersons



def cropFaceForPoseEstimation(frame,bbox,ad=0.6,output_shape=(64,64)):
    
    (h, w) = frame.shape[:2]
    croppedFaces=[]
    for b in bbox:
        x1, y1, wbbox, hbbox = b[0], b[1], b[2], b[3]
        x2 = x1+wbbox
        y2 = y1+hbbox

        left=int(x1 - ad * wbbox)
        right=int(x2 + ad * wbbox)
        top=int(y1 - ad * hbbox)
        bottom=int(y2 + ad * hbbox)
        xw1 = max(left, 0)
        yw1 = max(top, 0)
        xw2 = min(right, w - 1)
        yw2 = min(bottom, h - 1)
        
        face=frame[yw1:yw2 + 1, xw1:xw2 + 1, :] 
        face= cv2.copyMakeBorder(frame[yw1:yw2 + 1, xw1:xw2 + 1, :],0 if top>=0 else -top,0 if bottom<h-1 else bottom-(h-1) ,0 if left>=0 else -left ,0 if right < w-1 else right-(w-1),borderType=cv2.BORDER_REPLICATE)

        face = cv2.resize(face, output_shape)

        try:
            croppedFaces.append(face)
        except:
            break

        croppedFaces[-1] = cv2.normalize(croppedFaces[-1], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return croppedFaces 

def getThirdPointOnEquelateralTriangle(point1,point2):
    a=((point1-point2)[1])/(point2-point1)[0]
    dist=np.linalg.norm(point1-point2)**2
    x3 = np.sqrt(((np.sqrt(3)/2)*dist)/(a**2+1))+((point1+point2)/2)[1]
    y3= a*np.sqrt(((np.sqrt(3)/2)*dist)/(a**2+1))+((point1+point2)/2)[0]
    return np.array([y3,x3])


def getTargetCoord(lEye,rEye,prevShape,targetShape=[96*3,96*3,3],l=38):
    prevH,prevW,_=prevShape
    targetH,targetW,_=targetShape
    yr,xr=rEye[1],rEye[0]
    yl,xl=lEye[1],lEye[0]


    theta=np.arctan((yr-yl)/(xr-xl))

    new_xl=xl*targetW/prevW
    new_yl=yl*targetH/prevH

    new_yr=new_yl+l*np.sin(theta)
    new_xr=new_xl+l*np.cos(theta)
    
    newLeye=np.array([new_xl,new_yl])
    newReye=np.array([new_xr,new_yr])
    newNose=getThirdPointOnEquelateralTriangle(newLeye, newReye)
    outputPts=np.float32(np.vstack((newLeye,newReye,newNose))).reshape(1,3,2).astype(np.int)
    return outputPts

def performFaceAlignment(frame,landmarks=None,leftEye=None,rightEye=None,cols=180,rows=180):
   
    l=38
    size=(67+np.int(0.6*l)-29+np.int(0.6*l),np.int(2.2*l))
    faceROI=[0 ,0, frame.shape[0], frame.shape[1]]
    eyePositions=[]   

    if(leftEye is None):
        leftEye=(landmarks[36,:]+landmarks[39,:])/2
        rightEye=(landmarks[42,:]+landmarks[45,:])/2


    eyePositions.append(np.hstack((leftEye,rightEye)))
    equilateralPoint=getThirdPointOnEquelateralTriangle(leftEye, rightEye)
    inputPts=np.float32(np.array([leftEye,rightEye,equilateralPoint])).reshape(1,3,2).astype(np.int)
    #outputPts=np.float32(np.array([(300-96)/2,(300-96)/2])+np.array([[29,32],[67,32],[48,65]])).reshape(1,3,2).astype(np.int)
    ####offset=np.array([(cols*leftEye[0])/frame.shape[1],(cols*leftEye[1])/frame.shape[0]])
    ####outputPts=np.float32(offset+np.array([[29,32],[67,32],[48,65]])).reshape(1,3,2).astype(np.int)
    
    #outputPts=getTargetCoord(leftEye,rightEye,prevShape=frame.shape,targetShape=[96*3,96*3,3],l=1.2*l)
    #
    offset=np.array([50,70]) 
    outputPts=np.float32(offset+np.array([[29,32],[29+l,32],[48,65]])).reshape(1,3,2).astype(np.int)
    try:     
        #similarity=cv2.estimateRigidTransform(inputPts,outputPts,False)
        similarity=cv2.estimateAffinePartial2D(inputPts,outputPts)[0]
        
        registeredFace = cv2.warpAffine(frame,similarity,(cols,rows),borderMode=cv2.BORDER_REFLECT)
         
        #######cv2.polylines(frame,np.int32(landmarks.reshape((-1,1,2))),True,(255,255,0),3)
        #cv2.polylines(frame,[np.int32(np.array([leftEye,rightEye,equilateralPoint]))],True,(255,255,0),3)
        #cv2.polylines(frame,np.int32(equilateralPoint.reshape((-1,1,2))),True,(255,255,255),7)
        
        #return cv2.resize(registeredFace,(96,96))
        return registeredFace,registeredFace[offset[1]:offset[1]+96,offset[0]:offset[0]+96],similarity
    except Exception as e: 
        print(e)

        pass
        return None,_,_



if(__name__=="__main__"):
    main()


