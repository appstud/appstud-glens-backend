import numpy as np
import time
import cv2
import base64
import json
import os
import redis
from utils import stringToBGR, BGRToString, GetServicesConfiguration,decode_pipeline,str2bool
from face_detection import detectFaces
import copy
import logging
import coloredlogs
coloredlogs.install()

configuration=GetServicesConfiguration()
r = redis.StrictRedis(host=configuration["REDIS_HOST"], port=6379)                          
#r = redis.StrictRedis(host="localhost", port=6379)                          

class options():
    def __init__(self):
        self.LOG_LEVEL="WARNING"
        self.REPORT_PERF=False


    def reset(self):
        self.LOG_LEVEL="WARNING"
        self.REPORT_PERF=False

    def __setattr__(self,name,value):
        if(name in ["REPORT_PERF"] and type(value)!=bool):
            value=str2bool(value)
        self.__dict__[name] = value


    def __str__(self):
        return ', '.join("%s: %s" % item for item in self.__dict__.items())

    __repr__ = __str__


def process(message):
  
    img = message["image"]
    data = handler(img)

    result=copy.deepcopy(message)
    
    result["data"] =data
    
    result["ID_IMG"] = message["ID_IMG"]+"face_detection"
    """result["image"]=img
    result["ID_IMG"] = message["ID_IMG"]
    result["CAM_ID"] = message["CAM_ID"]
    result["pipeline"]=message["pipeline"] 
    """
    return result


def handler(msg):

    img=stringToBGR(msg)
    start=time.time()
    
    

    imgg,data=detectFaces(np.copy(img),s=1)
    #imgg,data=detectFaces(np.copy(img),size_for_face_detection=(1080,1920),s=None)
    finish=time.time()
    """
    if(configuration['GLENS_RETURN_IMAGES']=="True"):
        base64_mask=BGRToString(imgg.astype(np.uint8),0.2)
        data["mask"]=base64_mask
    """
    return data
    ###############

"""
def removeIdsFromRedis():
    global idsToRemove
    global r 

    for elem in idsToRemove:
        if(time.time()-elem[0]>2):
            r.delete(elem[1])
            idsToRemove.remove(elem)
"""

def runRedisListener():
    #try:
    global idsToRemove
    global r 
    CURRENT_CHANNEL='face-detection'
    p = r.pubsub()                                                              
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
            count=(4)%4

            if(count%4==0):
                data = json.loads(message['data'])
                
                coloredlogs.set_level(level=getattr(logging, opts.LOG_LEVEL))
                logging.debug("Received img {} ".format(data["ID_IMG"]))
                taken = r.getset(data["ID_IMG"], "1")

                opts,destination_channel=decode_pipeline(CURRENT_CHANNEL,data["pipeline"],opts)

                if taken == None:
                    
                    logging.debug("Processing img {}".format( data["ID_IMG"]) )
                    
                    if(opts.REPORT_PERF):
                        start=time.time()
                    dataToSend = process(data)
                    
                    if(opts.REPORT_PERF):
                         finish=time.time()
                         FPS=int(1.0/(finish-start))
                    
                    if(opts.REPORT_PERF):
                        dataToSend["FPS_face_detection"]=FPS
                        logging.debug(f'FPS: {FPS}')
                    r.delete(data["ID_IMG"])
                    logging.debug(dataToSend["data"])
                    ####print("Sending img estimation for " + data["id"])
                    r.publish(destination_channel, json.dumps(dataToSend))

if __name__ == '__main__':
    runRedisListener()

