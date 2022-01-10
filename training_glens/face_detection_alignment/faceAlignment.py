import dlib
import cv2
import os
import numpy as np

modelsPath=os.path.join(os.path.dirname(__file__),"models/")
predictor = dlib.shape_predictor(os.path.join(modelsPath,'shape_predictor_68_face_landmarks.dat'))
detector = dlib.get_frontal_face_detector()
face_cascade = cv2.CascadeClassifier(os.path.join(modelsPath,'haarcascade_frontalface_default.xml'))
prototxtPath = os.path.join(modelsPath, "deploy.prototxt")
weightsPath = os.path.join(modelsPath,"res10_300x300_ssd_iter_140000.caffemodel")
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)


def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Draw a point
def draw_point(img, p, color ) :
    cv2.circle( img, p, 2, color,thickness=-1 )



# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        #print getBarycentricCoordinates(t,np.array([0,0]))
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, 0)
    return img




def ssdFaceDetection(frame,faceNet=faceNet,minConfidence=0.7,drawBoundingBoxes=True):
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
    faces = []
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

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]

            # add the face and bounding boxes to their respective
            # lists
            faces.append(np.copy(face))
            faceBoundingBoxes.append((startX, startY, endX, endY))
            
            rec=dlib.rectangle(int(startX),int(startY),int(endX), int(endY))
            #faceBoundingBoxes.append(dlib.rectangle(int(startX),int(startY),int(endX), int(endY)))
            
            try:
                shape = predictor(cv2.cvtColor(np.copy(face),cv2.COLOR_BGR2GRAY),rec)

                """
                landmarks=np.zeros([68,2])
                for i in range(0,68):
                landmarks[i,:]=np.array([shape.part(i).x+y_top_left ,shape.part(i).y+x_top_left])
                """

                landmarkList.append(shape)
                if(drawBoundingBoxes):
                    cv2.rectangle(frame,(faceBoundingBoxes[-1][0],faceBoundingBoxes[-1][1]),(faceBoundingBoxes[-1][2],faceBoundingBoxes[-1][3]),(255,0,255),2)
                 
            except:
                pass

    return landmarkList,faceBoundingBoxes,False,frame,faces






def trackFaceInANeighborhoodAndDetectLandmarks(image,faceROI,drawBoundingBoxes=True):
     N=25
     x_top_left=max(faceROI[0]-N,0)
     x_bottom_right=min(image.shape[0],faceROI[0]+faceROI[2]+N)
     y_top_left=max(faceROI[1]-N,0)
     y_bottom_right=min(faceROI[1]+faceROI[3]+N,image.shape[1])
     landmarkList=[]
     boundingBoxList=[]
     if(drawBoundingBoxes):
         cv2.rectangle(image,(y_top_left,x_top_left),(y_bottom_right,x_bottom_right),(0,255,0),1)
     trackingLost=True
     faceBoundingBoxes=[]

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
         
         """
         landmarks=np.zeros([68,2])
         for i in range(0,68):
             landmarks[i,:]=np.array([shape.part(i).x+y_top_left ,shape.part(i).y+x_top_left])
         """
         faceROI=[x_top_left+d.top(), y_top_left+d.left(),d.height(), d.width()]

         trackingLost=False
         landmarkList.append(shape)
         boundingBoxList.append(faceROI)
         if(drawBoundingBoxes):
             cv2.rectangle(image,(faceROI[1],faceROI[0]),(faceROI[1]+faceROI[3],faceROI[0]+faceROI[2]),(255,0,255),2)
     return landmarkList,boundingBoxList,trackingLost,image



def searchForFaceInTheWholeImage(image):

    faceROI=[0, 0,image.shape[0]-1, image.shape[1]-1]
    landmarks,faceROI,_,image=trackFaceInANeighborhoodAndDetectLandmarks(image,faceROI)
    #landmarks,faceROI,_,image,faces=robustFaceDetection(image)
    allLandmarks=[]
    for landmark in landmarks:
        adaptedModel=np.zeros([68,2])
        for i in range(0,68):
            adaptedModel[i,:]=np.array([landmark.part(i).x,landmark.part(i).y])
        allLandmarks.append(adaptedModel)
        """
        cv2.polylines(image,np.int32(adaptedModel.reshape((-1,1,2))),True,(0,0,255),3)
  
             
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
        """
        """
        for i in range(0,len(adaptedModel)):
            draw_point(image, (int(adaptedModel[i,0]),int(adaptedModel[i,1])), (255,0,0) )
        """
        return image,allLandmarks,faceROI,faces
    
    return image,[],faceROI,[]

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

def performFaceAlignment(frame,landmarks,cols=180,rows=180):
   
    l=38
    size=(67+np.int(0.6*l)-29+np.int(0.6*l),np.int(2.2*l))
    faceROI=[0 ,0, frame.shape[0], frame.shape[1]]
    eyePositions=[]     
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
         
        cv2.polylines(frame,np.int32(landmarks.reshape((-1,1,2))),True,(255,255,0),3)
        #cv2.polylines(frame,[np.int32(np.array([leftEye,rightEye,equilateralPoint]))],True,(255,255,0),3)
        #cv2.polylines(frame,np.int32(equilateralPoint.reshape((-1,1,2))),True,(255,255,255),7)
        
        #return cv2.resize(registeredFace,(96,96))
        return registeredFace,registeredFace[offset[1]:offset[1]+96,offset[0]:offset[0]+96],similarity
    except Exception as e: 
        print(e)

        pass
        return None,_



if(__name__=="__main__"):
    main()


