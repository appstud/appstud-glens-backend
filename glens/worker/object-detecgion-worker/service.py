import argparse
import os
import re
import platform
import shutil
import time
import json
import base64

from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import redis

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import (
    xyxy2xywh, check_img_size, non_max_suppression, scale_coords, plot_one_box, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np
from dotenv import load_dotenv
import pdb
from tools.homography_estimation_calibration import estimate_homography,get_config_from_redis,getIntrinsicExtrinsicMatrix
load_dotenv()
from utils.glens_utils import *
import copy

configuration=GetServicesConfiguration()
r = redis.StrictRedis(host=configuration["REDIS_HOST"], port=6379)                          
#r = redis.StrictRedis(host="localhost", port=6379) 


"""def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")
"""
def load_env_variables():
    weights=os.getenv("WEIGHTS") 
    source=os.getenv("SOURCE") 
    img_size=int(os.getenv("IMG_SIZE"))
    conf_thresh=float(os.getenv("CONF_THRESH")) 
    iou_thresh=float(os.getenv("IOU_THRESH"))
    device=os.getenv("DEVICE") 
    view_img=str2bool(os.getenv("VIEW_IMG")) 
    filter_by_classes=json.loads(os.getenv("FILTER_BY_CLASSES") )
    agnostic_nms=str2bool(os.getenv("AGNOSTIC_NMS")) 
    augment=str2bool(os.getenv("AUGMENT") )
    
    print("weights {0} ,source {1}, img_size {2}, conf_thresh {3}, iou_thresh {4}, device {5},view_img {6}, filter_by_classes {7}, agnostic_nms {8}, augment {9}".format(\
            weights,source,img_size,conf_thresh,iou_thresh,device,view_img,filter_by_classes,agnostic_nms,augment))
    
    return weights,source,img_size,conf_thresh,iou_thresh,device,view_img,filter_by_classes,agnostic_nms,augment
    
weights,source,img_size,conf_thresh,iou_thresh,device,view_img,filter_by_classes,agnostic_nms,augment=load_env_variables()

if(view_img):
   cv2.namedWindow("img",cv2.WINDOW_NORMAL)

def draw_circles_labels(img,points,pose_data):
    for i,p in enumerate(points):
        cv2.circle(img,tuple(p),4,(255,0,0),thickness=-1)
        cv2.putText(img,"x,y={0}|{1}".format(format(pose_data[i,0],'.2f'),format(pose_data[i,1],'.2f')),tuple(p),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255))
    return img


def get_3D_position(H,_2Dpoints):
    #return np.dot(H,np.hstack((_2Dpoints.astype(np.float32),np.ones([_2Dpoints.shape[0],1]))).T)[:2,:].T
    #positions=np.zeros([_2Dpoints.shape[0],3])
    positions= np.dot(H,np.hstack((_2Dpoints.astype(np.float32),np.ones([_2Dpoints.shape[0],1]))).T).T
    positions=positions.astype(np.float32)/positions[:,2].reshape(-1,1)
    return positions

def run_processing(img_orig,H):
    with torch.no_grad():
        start=time.time()

        img=letterbox(img_orig,imgsz)[0][:,:,::-1].transpose(2,0,1)
        img= np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        #t1=time_synchronized()
        #print("preprocess",t1-start)
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(pred,conf_thresh , iou_thresh, classes=filter_by_classes, agnostic=agnostic_nms)
        
        #t2=time_synchronized()
        #print("dt",t2-t1)
    
    pose_data=None
    mid_points=[]
    
    iD=0
    result = dict()
    for i, det in enumerate(pred):  # detections per image
         
        if det is not None and len(det):
               
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img_orig.shape).round()
            if(device.type!="cpu"):
                det=det.cpu()
            mid_points.append(np.vstack(((det[:,0]+det[:,2])*0.5,det[:,3])).T)
            if(pose_data is None):
                #start=time.time()
                pose_data=get_3D_position(H,np.array(mid_points[-1]))
                #finish=time.time()
                #print("3Dposition",finish-start)
                
            else:
                pose_data=np.vstack((pose_data,get_3D_position(H,np.array(mid_points[-1]))))
                   
                ## Rescale boxes from img_size to im0 size
                
            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (names[int(cls)], conf)
                
                if(view_img):
                    plot_one_box(xyxy, img_orig, label=label, color=colors[int(cls)], line_thickness=3)
                result[iD] = dict()

                #result[iD]["bbox_yolo"] =list(map(lambda x:int(x.numpy()), xyxy))))
                result[iD]["bbox_yolo"] =list(map(lambda x:int(x), (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()))
                result[iD]["class"] = names[int(cls)]
                result[iD]["pose_data"]=pose_data[iD,:].tolist()
                iD = iD + 1     

            if(view_img):
                img_orig=draw_circles_labels(img_orig,mid_points[-1],pose_data)
            
    if(view_img):
        cv2.imshow("img",img_orig)
        cv2.waitKey(1)
            
    return result





def process(message):
    img=stringToBGR(message["image"])
    
    data=get_config_from_redis(message["CAM_ID"],r)
    
    H=np.array(get_config_from_redis(message["CAM_ID"],r)["Homography_img_to_3D"],dtype=np.float32)
    
    data = run_processing(img,H)
    
    result=copy.deepcopy(message)
    
    #result["image"]=message["image"]
    result["ID_IMG"] = message["ID_IMG"]+"object-detection"
    #result["CAM_ID"] = message["CAM_ID"]
    result["data"] = data
    
    return result


class options():
    def __init__(self):
        self.REPORT_PERF=False
        self.VERBOSE=False
    def reset(self):
        self.VERBOSE=False
        self.REPORT_PERF=False
    def __setattr__(self,name,value):
        if(name in ["VERBOSE","REPORT_PERF"] and type(value)!=bool):
            value=str2bool(value)
        self.__dict__[name] = value
 

def runRedisListener():
    #try:
    global r 
    CURRENT_CHANNEL='object-detection'
    p = r.pubsub()                                                              
    p.subscribe(CURRENT_CHANNEL)
    count=-1
    opts=options()
    while True:
        try:
            message = p.get_message()
        except Exception as e:
            print(e)
            continue

        if message and message["type"] == "message":
            #count=(count+1)%4
            count=4
            if(count%4==0):
                data = json.loads(message['data'])
                print("Received img " ,data["ID_IMG"])
                taken = r.getset(data["ID_IMG"], "1")


                if(taken == None):
                    opts.reset()
                    opts,destination_channel=decode_pipeline(CURRENT_CHANNEL,data['pipeline'],opts)
                    if(opts.VERBOSE or opts.REPORT_PERF):
                        start=time.time()
                        print("Processing img " , data["ID_IMG"]) 
                    #estimate_homography(cam_id=0,img=stringToBGR(data["image"]))
                    dataToSend = process(data)
                    
                    if(opts.VERBOSE or opts.REPORT_PERF):
                        finish=time.time()
                        FPS=int(1.0/(finish-start))
                        if(opts.VERBOSE):
                            print("Destination channel:",destination_channel)
                            print("t total ",FPS)
                        if(opts.REPORT_PERF):
                            dataToSend["FPS_object_detection"]=FPS
                    r.delete(data["ID_IMG"])
                    r.publish(destination_channel, json.dumps(dataToSend))
                    #r.publish("get-attributs", dataToSend)


if __name__ == '__main__':
         
    with torch.no_grad():
        device=select_device(device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        cudnn.benchmark = True  # set True to speed up constant image size inference
        model = attempt_load(weights, map_location=device)  # load FP32 model
        if(half):
            model.half()
            MAX_MEMORY_NEEDED=4096
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(MAX_MEMORY_NEEDED/total_memory, 0)
            
        
        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
        

        runRedisListener()

