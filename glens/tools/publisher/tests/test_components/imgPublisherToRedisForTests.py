import cv2
import functools
import time
import redis
import json
from utils import *
from draw_utils import *


r = redis.StrictRedis(host='localhost', port=6379)

def getIntrinsicMatrix(cam_id):
    try:
        ##camId=r.get(message["camId"])

        #dataForCamera=json.loads(r.get(message["CAM_ID"]))["cam1"]
        dataForCamera=json.loads(r.get(cam_id))[cam_id]
        alpha_u=dataForCamera['alpha_u']
        alpha_v=dataForCamera['alpha_v']
        c_x=dataForCamera['c_x']
        c_y=dataForCamera['c_y']
        distCoeff=np.array(dataForCamera["distCoeff"])
        
        intrinsicMatrix=np.array([[alpha_u,0,c_x],[0,alpha_v,c_y],[0,0,1]])
    
    except Exception as e:
        print(e)
        intrinsicMatrix=None
        distCoeff=None

    return intrinsicMatrix,distCoeff

intrinsicMatrix,distCoeff=getIntrinsicMatrix("cam1")

def testFaceDetection(cap):

    out=cv2.VideoWriter('out_face_tracking.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, ( 1914,1074))
    #out=cv2.VideoWriter('out_face_tracking.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, ( 1274,714))
    #out=cv2.VideoWriter('out_face_tracking.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, ( 3834,2154))
    #out=cv2.VideoWriter('out_face_tracking.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, ( 2304,1296))
    #out=cv2.VideoWriter('out_face_tracking.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, ( 634,354))
    p=r.pubsub()
    time.sleep(1)
    channel="results"
    p.subscribe(channel)
    CAM_ID="cam1"
    IMG_ID=0

    while(True):
        IMG_ID=(IMG_ID+1)%10 
        ret,img=cap.read()
        
        start=time.time()
        img=img[3:img.shape[0]-3,3:img.shape[1]-3]
        #img=np.zeros([3834,2154,3],dtype=np.uint8)
        img=cv2.resize(img,(1914,1074))
        print("sending img:",img.shape)
        #cv2.imshow("img",img)
        #cv2.waitKey(1)
        if(ret):
            data=json.dumps({"channel":channel,"CAM_ID":CAM_ID,"ID_IMG":IMG_ID,"current_time":start,"image":BGRToString(img)})
                
            
            r.publish("face-detection",data)
            print("Sent image to process!")

        try:

            data=json.loads(p.get_message(timeout=0.33)['data'])
            #print("camid",data["CAM_ID"])
            img=stringToBGR(data["image"])
            print(img.shape)
            draw_img=draw_bounding_box(img,data["data"])
                
            out.write(draw_img)
        except Exception as e:
            print(e)

        except KeyboardInterrupt:
            print("saving video")
            out.release()
            break

def draw_bounding_box(draw_image,data):
    for i in data.keys():
        if( isinstance(data[i],dict) and "bbox" in data[i].keys()):
            
            draw_image=draw_data(draw_image, data[i],intrinsicMatrix,draw_weak_perspective=False)

            """
            landmark=np.array(data[i]["landmarks"])
            faceROI=data[i]["bbox"]
            

            print(data[i]["age"])
            draw_image=cv2.polylines(draw_image,np.int32(landmark.reshape((-1,1,2))),True,(0,0,255),3)
            draw_image=cv2.rectangle(draw_image,(int(faceROI[0]),int(faceROI[1])),(int(faceROI[2]+faceROI[0]),int(faceROI[3]+faceROI[1])),(0,0,255),3)                                      
            """
    return draw_image



if(__name__=="__main__"):
    
    cap1=cv2.VideoCapture("bruno_mars.mp4")
    #cap1=cv2.VideoCapture("out_webcam2.mp4")
    #cap1=cv2.VideoCapture("sample.mp4")
    #cap2=cv2.VideoCapture("big_bang.mp4")
    #cap2=cv2.VideoCapture("mask.mp4")
    ##cap=cv2.VideoCapture("video2.mp4")
    cap1.set(cv2.CAP_PROP_FPS, 10)
    testFaceDetection(cap1)
