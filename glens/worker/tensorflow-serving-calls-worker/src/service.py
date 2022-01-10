import numpy as np
import time
import cv2
#from predict import get_attributs
from predictV2 import get_attributs
import base64
import json
import os
import re
from enhanceQuality import dynamicRangeCompression
import datetime
import redis
from utils import stringToBGR, BGRToString, GetServicesConfiguration,decode_pipeline,str2bool
from pose_estimation import getExtrinsicMatrix
import copy
import logging 
import coloredlogs

coloredlogs.install()

configuration=GetServicesConfiguration()

r = redis.StrictRedis(host=configuration["REDIS_HOST"], port=6379)                          


class options:

    def __init__(self):
        
        self.GET_AGE_SEX_GLASSES=False
        self.GET_HAIR_COLOR=False
        self.GET_MASK=False
        self.GET_POSE=False
        self.GET_FACE_RECO=False
        self.GET_PERS_REID=False
        
        self.LOG_LEVEL="WARNING"
        self.REPORT_PERF=False
    
    def reset(self):
        self.LOG_LEVEL="WARNING"
        self.REPORT_PERF=False

        self.GET_AGE_SEX_GLASSES=False
        self.GET_HAIR_COLOR=False
        self.GET_MASK=False
        self.GET_POSE=False
        self.GET_FACE_RECO=False
        self.GET_PERS_REID=False
    def __setattr__(self,name,value):
        if(name in ["GET_AGE_SEX_GLASSES","GET_HAIR_COLOR","GET_MASK","GET_POSE","GET_FACE_RECO","GET_PERS_REID","REPORT_PERF"] and type(value)!=bool):
            value=str2bool(value)
        self.__dict__[name] = value


    def __str__(self):
        return ', '.join("%s: %s" % item for item in self.__dict__.items())

    __repr__ = __str__



def getIntrinsicExtrinsicMatrix(cam_id):
    try:    
        dataForCamera=json.loads(r.get("config"))[cam_id]
    except:
        logging.warning("no config file for camera with id {}... calibrate the camera if you asked for position estimates otherwise u can ignore this warning".format(cam_id))
        return None,None,np.eye(4),None

    if(dataForCamera['alpha_v'] is not None): 
        #dataForCamera=json.loads(r.get(message["CAM_ID"]))["cam1"]
        alpha_u=dataForCamera['alpha_u']
        alpha_v=dataForCamera['alpha_v']
        c_x=dataForCamera['c_x']
        c_y=dataForCamera['c_y']
        distCoeff=np.array(dataForCamera["distCoeff"])
        
        intrinsicMatrix=np.array([[alpha_u,0,c_x],[0,alpha_v,c_y],[0,0,1]])
    else:
        intrinsicMatrix=None
        distCoeff=None
        rigidTransformToWCS=np.eye(4)
 
    if('tx'  in dataForCamera and "roll"  in dataForCamera): 
        ###Notice the minus sign for pitch
        rigidTransformToWCS=getExtrinsicMatrix(tx=dataForCamera["tx"],ty=dataForCamera["ty"],tz=dataForCamera["tz"],yaw=dataForCamera["yaw"],pitch=-dataForCamera["pitch"],roll=dataForCamera["roll"])
        rigidTransformToWCS=np.vtack((extriniscMatrix,np.array([0,0,0,1])))
    else:
        rigidTransformToWCS=np.eye(4)
         
    if("Homography_img_to_3D" in dataForCamera):
        H=np.array(dataForCamera["Homography_img_to_3D"],dtype=np.float32)
    else:
        H=None

    return intrinsicMatrix,distCoeff,rigidTransformToWCS,H


def process(message,opts):
    cam_id=message["CAM_ID"]
    intrinsicMatrix,distCoeff,rigidTransformToWCS,H=getIntrinsicExtrinsicMatrix(cam_id)
    
    img = message["image"]
    data = handler(img,message["data"],intrinsicMatrix,distCoeff,rigidTransformToWCS,H,opts)
    
    result=copy.deepcopy(message)
    result["ID_IMG"] = message["ID_IMG"]+"tensorflow_calls"
    result["data"] =data

    return result




def handler(imgJSON,data,intrinsicMatrix,distCoeff,rigidTransformToWCS,H,opts):

    img=stringToBGR(imgJSON)
    start=time.time()
    
    img,dataToReturn=get_attributs(img,data,opts,H,intrinsicMatrix,distCoeff,rigidTransformToWCS)

    finish=time.time()
    return dataToReturn

def runRedisListener():
    #try:
    global idsToRemove
    global r 
    p = r.pubsub()     
    CURRENT_CHANNEL="tensorflow-calls"
    p.subscribe(CURRENT_CHANNEL)
    PAUSE = True
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
                data = json.loads(message["data"])
                                
                
                taken = r.getset(data["ID_IMG"], "1")
                
                opts.reset() 
                opts,destination_channel=decode_pipeline(CURRENT_CHANNEL,data["pipeline"],opts)
                if taken == None:
                    coloredlogs.set_level(getattr(logging,opts.LOG_LEVEL)) 
                    logging.debug("tensorflow calls worker: Received img " + str(data["ID_IMG"]))
                

                    if(opts.REPORT_PERF):
                        start=time.time()
                    dataToSend = process(data,opts)
                    
                    if(opts.REPORT_PERF):
                        finish=time.time()
                        FPS=int(1.0/(finish-start))
                    if(opts.REPORT_PERF):
                        dataToSend["FPS_TF_MODELS"]=FPS
                        logging.debug(f'FPS: {FPS}')
                    r.delete(data["ID_IMG"])
                    r.publish(destination_channel, json.dumps(dataToSend))
 
 
if __name__ == '__main__':
    runRedisListener()

