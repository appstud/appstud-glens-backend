import cv2
import functools
import time
import redis
import json
from utils import *
from constants import PART_NAMES
from draw_utils import *
import signal

config=GetServicesConfiguration()
r = redis.StrictRedis(host=config["REDIS_HOST"], port=6379)


exit=False
def exit_gracefully(signum,frame):
    global exit
    print('Signal handler called with signal', signum)
    exit=True
    print("RECEIVED KILLING SIGNALLLLLLLLLLLLLLLLLLLLLLLLLLLL")


def format_for_drawing(data):
    pose_scores=np.ones((len(data.keys()),1))
    keypoint_scores=np.ones((len(data.keys()),len(PART_NAMES)))
    keypoint_coords=np.empty((len(data.keys()),len(PART_NAMES),2))
    bbox=[]
    bbox_persons=[]

    for i,key in enumerate(data.keys()):
        bbox.append(data[key]['bbox'])
        bbox_persons.append(data[key]['bbox_yolo'])
        for j,_ in enumerate(PART_NAMES):
            keypoint_coords[i,j,:]=np.asarray(data[key][PART_NAMES[j]]) 

    return pose_scores,keypoint_scores,keypoint_coords,bbox,bbox_persons

def publish(cap):

    out=cv2.VideoWriter('result.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, ( 640,480))
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
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    while(not exit):
        IMG_ID=(IMG_ID+1)%10 
        ret,img=cap.read()
        img=cv2.resize(img,(640,480))        
        start=time.time()
        
        if(ret):
            data=json.dumps({"pipeline":"posenet VERBOSE=true REPORT_PERF=true|results","CAM_ID":CAM_ID,"ID_IMG":str(IMG_ID),"current_time":start,"image":BGRToString(img)})
            #data=json.dumps({"pipeline":"posenet |results","CAM_ID":CAM_ID,"ID_IMG":str(IMG_ID),"current_time":start,"image":BGRToString(img)})
                
            
            r.publish("posenet",data)
        
        try:

            data=json.loads(p.get_message(timeout=0.33)['data'])
            pose_scores,keypoint_scores,keypoint_coords,bbox,bbox_persons=format_for_drawing(data['data'])
            print(data['data']) 
            
            image=stringToBGR(data["image"])
            draw_img = draw_skel_and_kp(image, pose_scores, keypoint_scores, keypoint_coords,bbox,bbox_persons,min_pose_score=0.2, min_part_score=0.2)
            cv2.putText(draw_img,"FPS: "+str(data['FPS_posenet']),(20,20),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,0,0))
            #print("camid",data["CAM_ID"])
            #print(img.shape)
            out.write(draw_img)
        except Exception as e:
            print(e)

        except KeyboardInterrupt:
            print("saving video")
            out.release()
            break

    out.release()

if(__name__=="__main__"):
    
    #cap1=cv2.VideoCapture("bruno_mars.mp4")
    #cap1=cv2.VideoCapture("UFC.mp4")
    cap1=cv2.VideoCapture("sport.mp4")
    #cap1=cv2.VideoCapture("ronaldinho.mp4")
    #cap1=cv2.VideoCapture("dance.mp4")
    #cap1=cv2.VideoCapture("TownCentreXVID.mp4")
    #cap1=cv2.VideoCapture("out_webcam2.mp4")
    #cap1=cv2.VideoCapture("sample.mp4")
    #cap2=cv2.VideoCapture("big_bang.mp4")
    #cap2=cv2.VideoCapture("mask.mp4")
    ##cap=cv2.VideoCapture("video2.mp4")
    cap1.set(cv2.CAP_PROP_FPS, 10)
    publish(cap1)
