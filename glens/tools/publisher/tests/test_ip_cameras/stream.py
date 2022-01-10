import cv2
import functools
import time
import redis
import json
from utils import *
from draw_utils import *
import os
import signal
#r = redis.StrictRedis(host='localhost', port=6379)
r = redis.StrictRedis(host='redis', port=6379)

def getIntrinsicMatrix(cam_id):
    try:
        ##camId=r.get(message["camId"])

        #dataForCamera=json.loads(r.get(message["CAM_ID"]))["cam1"]
        dataForCamera=json.loads(r.get("config"))[cam_id]
        alpha_u=dataForCamera['alpha_u']
        alpha_v=dataForCamera['alpha_v']
        c_x=dataForCamera['c_x']
        c_y=dataForCamera['c_y']
        distCoeff=np.array(dataForCamera["distCoeff"])
        
        intrinsicMatrix=np.array([[alpha_u,0,c_x],[0,alpha_v,c_y],[0,0,1]])
        print(intrinsicMatrix) 
    except Exception as e:
        print(e)
        intrinsicMatrix=None
        distCoeff=None

    return intrinsicMatrix,distCoeff

exit=False
def exit_gracefully(signum,frame):
    global exit
    print('Signal handler called with signal', signum)
    exit=True
    print("RECEIVED KILLING SIGNALLLLLLLLLLLLLLLLLLLLLLLLLLLL")

def sendImages(cap,CAM_ID,processing_pipeline):
    global exit
    
    ret,img=cap.read()
    shape=img.shape
    #out=cv2.VideoWriter('out_face_tracking.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, ( 1914,1074))
    ###out=cv2.VideoWriter('out_face_tracking.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, ( 1920,1080))
    #out=cv2.VideoWriter('out_face_tracking.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, ( 2*1920,1080))
    #out=cv2.VideoWriter('out_face_tracking.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, ( 2*640,368))
    out=cv2.VideoWriter('out_face_tracking.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, ( 2*shape[1],shape[0]))
    p=r.pubsub()
    time.sleep(1)
    channel=processing_pipeline.split("|")[0].strip().split(' ')[0]
    
    p.subscribe(channel)
    IMG_ID=0
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)
    while(not exit):
        IMG_ID=(IMG_ID+1)%1000
        ret,img=cap.read()
        print(ret,img.shape)        
        start=time.time()
        time.sleep(0.033) 
        #img=img[3:img.shape[0]-3,3:img.shape[1]-3]
        #img=np.zeros([3834,2154,3],dtype=np.uint8)
        #img=cv2.resize(img,(1914,1074))
        #print("sending img:",img.shape)
        #cv2.imshow("img",img)
        #cv2.waitKey(1)
        if(ret):
            data=json.dumps({"pipeline":processing_pipeline,"CAM_ID":CAM_ID,"ID_IMG":str(IMG_ID),"current_time":start,"image":BGRToString(img)})
            r.publish(processing_pipeline.split("|")[1].strip().split(' ')[0],data)
            print("Sent image to process!",processing_pipeline.split("|")[1].strip().split(' ')[0])
        
        try:
            data=json.loads(p.get_message(timeout=0.5)['data'])
            if(data["CAM_ID"]==CAM_ID):
                #print("camid",data["CAM_ID"])
                img=stringToBGR(data["image"])
                del data["image"]
                
                draw_img,clusters_img=draw_bounding_box(img,data)
                
                print(data)
                #print("CLUSTER_img_shape", clusters_img.shape)
                #cv2.imshow(CAM_ID,draw_img)      
                #cv2.waitKey(1)
                #out.write(draw_img)
                out.write(np.hstack((draw_img,cv2.resize(clusters_img,draw_img.shape[:2][::-1]))))
        except Exception as e:
            print(e)

        except KeyboardInterrupt:
            print("saving video")
            out.release()
            break

    print("saving video")
    print("saving video")
    print("saving video")
    print("saving video")
    print("saving video")
    print("saving video")
    print("saving video")
    print("saving video")
    print("saving video")
    out.release()

def draw_bounding_box(draw_image,data):

    FPS_object_detection=[data['FPS_object_detection'] if 'FPS_object_detection' in data.keys() else 'notAvailable']
    FPS_face_detection=[data['FPS_face_detection'] if 'FPS_face_detection' in data.keys() else 'notAvailable']
    FPS_tf_models=[data['FPS_TF_MODELS'] if 'FPS_TF_MODELS' in data.keys() else 'notAvailable']
    FPS_tracking=[data['FPS_tracking'] if 'FPS_tracking' in data.keys() else 'notAvailable']
    FPS_clustering=[data['FPS_clustering'] if 'FPS_clustering' in data.keys() else 'notAvailable']
    clusters_img=np.zeros([640,420,3],dtype=np.uint8)

    data=data["data"]
    for i in data.keys():
        if (isinstance(data[i],dict) and ("bbox" in data[i].keys() or "bbox_yolo" in data[i].keys())):
            draw_image=draw_data(draw_image, data[i],i,intrinsicMatrix,draw_weak_perspective=False,FPS_clustering=FPS_clustering,FPS_object_detection=FPS_object_detection,FPS_tracking=FPS_tracking,FPS_tf_models=FPS_tf_models,FPSFaceDetection=FPS_face_detection)
    
    if("clusters" in data.keys()):
        try:
            clusters_img=draw_clusters(data["clusters"])
        except Exception as e:
            print(e)
    
    return draw_image,clusters_img




if(__name__=="__main__"):
    gstreamer_pipeline=os.getenv("G_STREAMER_PIPELINE")
    processing_pipeline=os.getenv("PROCESSING_PIPELINE")
    CAM_ID=os.getenv("CAM_ID") 
    #cv2.namedWindow(CAM_ID,cv2.WINDOW_NORMAL)
    #time.sleep(700)
    intrinsicMatrix,distCoeff=getIntrinsicMatrix(CAM_ID)
    cap=cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
    sendImages(cap,CAM_ID,processing_pipeline)
