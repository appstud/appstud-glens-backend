import dlib
import cv2
import os
import numpy as np
from mtcnn import MTCNN
import pathlib


modelsPath=os.path.join(os.path.dirname(__file__),"models/")
predictor = dlib.shape_predictor(os.path.join(modelsPath,'shape_predictor_68_face_landmarks.dat'))
detector = dlib.get_frontal_face_detector()
face_cascade = cv2.CascadeClassifier(os.path.join(modelsPath,'haarcascade_frontalface_default.xml'))
prototxtPath = os.path.join(modelsPath, "deploy.prototxt")
weightsPath = os.path.join(modelsPath,"res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
mtcnn_detector=MTCNN()



def cropFaceForPoseEstimation(frame,bbox,ad=0.6,output_shape=(64,64)):
    
    (h, w) = frame.shape[:2]
    croppedFaces=[]
    for b in bbox:
        x1, y1, wbbox, hbbox = b[0], b[1], b[2], b[3]
        x2 = x1+wbbox
        y2 = y1+hbbox
        xw1 = max(int(x1 - ad * wbbox), 0)
        yw1 = max(int(y1 - ad * hbbox), 0)
        xw2 = min(int(x2 + ad * wbbox), w - 1)
        yw2 = min(int(y2 + ad * hbbox), h - 1)

        face = cv2.resize(frame[yw1:yw2 + 1, xw1:xw2 + 1, :], output_shape)

        try:
            croppedFaces.append(face)
        except:
            break

        croppedFaces[-1] = cv2.normalize(croppedFaces[-1], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return croppedFaces 


def mtcnn_face_detection(frame,minConfidence=0.7):
    detected=mtcnn_detector.detect_faces(cv2.cvtColor(np.copy(frame),cv2.COLOR_BGR2RGB))
    (h, w) = frame.shape[:2]
    faceBoundingBoxes = []
    landmarkList = []
    
    if len(detected) > 0:
        for i, d in enumerate(detected):
            if d['confidence'] > 0.95:
                x1, y1, wbbox, hbbox = d['box']
                faceBoundingBoxes.append((x1, y1, wbbox, hbbox))
                landmarkList.append(d["keypoints"])

    return landmarkList,faceBoundingBoxes,False,frame

   
                


def ssd_face_detection(frame,faceNet=faceNet,minConfidence=0.7):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faceBoundingBoxes = []
    landmarkList = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > minConfidence:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            
            x1 = startX
            y1 = startY
            wbbox = endX - startX
            hbbox = endY - startY

            x2 = x1+wbbox
            y2 = y1+hbbox
            """
            ###SSD face detector bounding box not compatible with the model for facial landmark, we must change the bbox location

            ##startY, endX = int(startY * 1.15), int(endX * 1.05)
            
            startY, endX = int(startY+hbbox* 0.1), int(endX+wbbox*0.1)
            """
            
            #faceBoundingBoxes.append((startX, startY, endX-startX, endY-startY))
            faceBoundingBoxes.append((startX, startY, wbbox, hbbox))
            rec=dlib.rectangle(int(startX),int(startY),int(endX), int(endY))
            shape = predictor(cv2.cvtColor(np.copy(face),cv2.COLOR_BGR2GRAY),rec)
            landmarkList.append(shape)
     
    return landmarkList,faceBoundingBoxes,False,frame






def trackFaceInANeighborhoodAndDetectLandmarks(image,faceROI):
     N=25
     x_top_left=max(faceROI[0]-N,0)
     x_bottom_right=min(image.shape[0],faceROI[0]+faceROI[2]+N)
     y_top_left=max(faceROI[1]-N,0)
     y_bottom_right=min(faceROI[1]+faceROI[3]+N,image.shape[1])
     landmarkList=[]
     boundingBoxList=[]
     trackingLost=True
     faceBoundingBoxes=[]
     faces=[]
     
     try:
         faceBoundingBoxes=detector(cv2.cvtColor(image[x_top_left:x_bottom_right,y_top_left:y_bottom_right,:],cv2.COLOR_BGR2GRAY), 0)
         #faceBoundingBoxes=detector(cv2.cvtColor(image[x_top_left:x_bottom_right,y_top_left:y_bottom_right,:],cv2.COLOR_BGR2RGB), 1)
         ##box=face_cascade.detectMultiScale(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), 1.1, 2,minSize=(150,150),maxSize=(205,205))
         ##box=face_cascade.detectMultiScale(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), 1.05, 2,minSize=(45,45),maxSize=(55,55))
         """
         for i in range(box.shape[0]):
             faceBoundingBoxes.append(dlib.rectangle(int(box[i,0]),int(box[i,1]),int(box[i,0]+box[i,2]), int(box[i,1]+box[i,3])))
         """
     except Exception as e:
         #print(e)
         faceBoundingBoxes=[]

     for k,d in enumerate(faceBoundingBoxes):
         shape = predictor(cv2.cvtColor(image[x_top_left:x_bottom_right,y_top_left:y_bottom_right,:],cv2.COLOR_BGR2GRAY),d)
         
         faceROI=[ y_top_left+d.left(), x_top_left+d.top(),d.width(), d.height()]
         #faces.append(image[faceROI[0]:faceROI[0]+faceROI[2],faceROI[1]:faceROI[1]+faceROI[3],:] )
         trackingLost=False
         landmarkList.append(shape)
         boundingBoxList.append(faceROI)
     
     return landmarkList,boundingBoxList,trackingLost,image



def searchForFaceInTheWholeImage(image,face_detection="mtcnn"):

    faceROI=[0, 0, image.shape[0]-1, image.shape[1]-1]
    allLandmarks=[]
    if(face_detection=="mtcnn"):
        landmarks,faceROIs,_,image=mtcnn_face_detection(image)
        if(len(faceROIs)>0):
            for landmark in landmarks:
                adaptedModel=np.zeros([5,2])
                adaptedModel[0,:]=np.array(landmark["left_eye"])
                adaptedModel[1,:]=np.array(landmark["right_eye"])
                adaptedModel[2,:]=np.array(landmark["nose"])
                adaptedModel[3,:]=np.array(landmark["mouth_left"])
                adaptedModel[4,:]=np.array(landmark["mouth_right"])

                allLandmarks.append(adaptedModel)
    else:  
        if(face_detection=="ssd"):
            landmarks,faceROIs,_,image=ssd_face_detection(image)
        else:
            landmarks,faceROIs,_,image=trackFaceInANeighborhoodAndDetectLandmarks(image,faceROI)
         
        for landmark in landmarks:
            adaptedModel=np.zeros([68,2])
            for i in range(0,68):
                adaptedModel[i,:]=np.array([landmark.part(i).x,landmark.part(i).y])
            allLandmarks.append(adaptedModel)
            """ 
            rect = (0, 0, image.shape[1], image.shape[0])
            try:
                subdiv1  = cv2.Subdiv2D(rect)
                points1 = []
                for p in adaptedModel.astype(int):
                    points1.append((int(p[0]), int(p[1])))
                    subdiv1.insert(points1[-1])

            except Exception as e:
                print(e)
                pass
            #image=draw_delaunay(image, subdiv1, (255,0,0))
            for i in range(0,len(adaptedModel)):
                draw_point(image, (int(adaptedModel[i,0]),int(adaptedModel[i,1])), (255,0,0) )
            
            return image,allLandmarks,faceROI,faces
            """
        
    return image,allLandmarks,faceROIs
    #return image,[],faceROI,[]

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


