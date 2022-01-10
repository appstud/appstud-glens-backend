import dlib
import cv2
import os
import numpy as np
#from mtcnn import MTCNN
#import tensorflow as tf
import pdb 
import logging
from facenet_pytorch import MTCNN
import torch
from utils import *

modelsPath=os.path.join(os.path.dirname(__file__),"models/")

predictor = dlib.shape_predictor(os.path.join(modelsPath,'shape_predictor_68_face_landmarks.dat'))
detector = dlib.get_frontal_face_detector()
face_cascade = cv2.CascadeClassifier(os.path.join(modelsPath,'haarcascade_frontalface_default.xml'))
prototxtPath = os.path.join(modelsPath, "deploy.prototxt")
weightsPath = os.path.join(modelsPath,"res10_300x300_ssd_iter_140000.caffemodel")
#faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
faceNet=None
mtcnn_detector=MTCNN()

"""
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess=tf.Session(config=config)
"""

log = logging.getLogger('face_detection')
log.setLevel(logging.ERROR)
configuration = GetServicesConfiguration()



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("GPU?:",torch.cuda.is_available())
"""
if (torch.cuda.is_available()):
    MAX_MEMORY_NEEDED=16096*10**6
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(MAX_MEMORY_NEEDED/total_memory)
    torch.cuda.set_per_process_memory_fraction(MAX_MEMORY_NEEDED/total_memory, 0)
"""
detector = MTCNN(thresholds=[0.85,0.8,0.7],post_process=True,min_face_size=20,keep_all=True,factor=0.6,device=device,select_largest=False)
#detector = MTCNN(thresholds=[0.8,0.7,0.6],post_process=True,min_face_size=20,keep_all=True,factor=0.7,device=device,select_largest=False)

def detect_facenet_pytorch(detector, images, batch_size):
    bboxes=[]
    landmarks=[]
    images=list(map(lambda x:cv2.cvtColor(x,cv2.COLOR_BGR2RGB),images))
    for lb in np.arange(0, len(images), batch_size):
        h,w,_=images[0].shape
        imgs = [img for img in images[lb:lb+batch_size]]
        #data=detector.detect(imgs)
        bbox,_,landmark=detector.detect(imgs,landmarks=True)
        bbox=bbox[0]
        if(bbox is not None):

            bbox=bbox.astype(int)
            for b in range(len(bbox)):

                bbox[b,0]=max(0,int(bbox[b,0]))
                bbox[b,1]=max(0,int(bbox[b,1]))
                bbox[b,2]=int(bbox[b,2])-int(bbox[b,0]) 
                bbox[b,3]=int(bbox[b,3])-int(bbox[b,1])
                if(bbox[b,2]+bbox[b,0]>w):
                    bbox[b,2]=w-bbox[b,0] 
                if(bbox[b,3]+bbox[b,1]>h):
                    bbox[b,3]=h-bbox[b,1] 
            bbox=list(bbox) 
        bboxes.append(bbox)
        landmarks.append(landmark[0])
        torch.cuda.empty_cache()
        return bboxes,landmarks





def mtcnn_face_detection(frame,minConfidence=0.7):
    detected=mtcnn_detector.detect_faces(cv2.cvtColor(np.copy(frame),cv2.COLOR_BGR2RGB))
    (h, w) = frame.shape[:2]
    faceBoundingBoxes = []
    landmarkList = []
    
    if len(detected) > 0:
        for i, d in enumerate(detected):
            if d['confidence'] > 0.95:
                x1, y1, wbbox, hbbox = d['box']
                faceBoundingBoxes.append((max(0,x1),max(0,y1), wbbox if(wbbox+x1<w) else w-x1, hbbox if(hbbox+y1<h) else h-y1))
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
            try:
                shape = predictor(cv2.cvtColor(np.copy(face),cv2.COLOR_BGR2GRAY),rec)
                landmarkList.append(shape)
            except Exception as e:
                print(e)
                faceBoundingBoxes.pop()
     
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
try:
                data=json.loads(p.get_message(timeout=0.33)['data'])
                if("mask" in data["data"].keys()):
                    mask=np.copy(stringToBGR(data["data"]["mask"]))
                    del data["data"]["mask"]
                    cv2.imshow("out",mask)
                    cv2.waitKey(1)
                
                print("Received data:", data["data"])
                #print("Received data:", mask)
            except Exception as e:
                print(e)
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



def searchForFaceInTheWholeImage(image,face_detection="py-torch-mtcnn"):

    faceROI=[0, 0, image.shape[0]-1, image.shape[1]-1]
    allLandmarks=[]
    if(face_detection=="py-torch-mtcnn"):
        faceROIs,allLandmarks=detect_facenet_pytorch(detector, [image], 1)
        if(faceROIs[0] is not None):
            faceROIs=faceROIs[0]
            allLandmarks=allLandmarks[0]
        else:
            faceROIs=[]
            allLandmarks=[]
    elif(face_detection=="mtcnn"):
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
            if(landmark is None):
                allLandmarks.append(adaptedModel)
                continue

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




def detectFaces(image,size_for_face_detection=(320,240),face_detection="py-torch-mtcnn",s=None):
    if(s is None):
        s=int(image.shape[0]/size_for_face_detection[0])+2
    draw=True
    ###draw=configuration['GLENS_RETURN_IMAGES']=="True"
    imageResized=cv2.resize(image,(int(image.shape[1]/s),int(image.shape[0]/s)))
    ###img,landmarks,faceROI=searchForFaceInTheWholeImage(np.copy(imageResized))
    _,landmarks,faceROI=searchForFaceInTheWholeImage(np.copy(imageResized),face_detection=face_detection)
    ####img=cv2.resize(img,(128,128)) 
    alignedFaces=[]
    faces=[]
    draw_image=np.copy(image)
    
    data={}
    for i,landmark in enumerate(landmarks):
       landmark=s*landmark
       landmarks[i]=landmark
       faceROI[i]=list(map(lambda x:float(s*x),faceROI[i]))
       data[str(i)]={"bbox":faceROI[i],"landmarks":landmarks[i].tolist()}
       if(draw):
           draw_image=cv2.polylines(draw_image,np.int32(landmark.reshape((-1,1,2))),True,(0,0,255),3)
           draw_image=cv2.rectangle(draw_image,(int(faceROI[i][0]),int(faceROI[i][1])),(int(faceROI[i][2]+faceROI[i][0]),int(faceROI[i][3]+faceROI[i][1])),(0,0,255),3)                                                 
   
    return draw_image,data



