import numpy as np
import time
import cv2
import base64
import json
import os
import redis
from utils import stringToBGR, BGRToString, GetServicesConfiguration,decode_pipeline,str2bool
import copy
from posenet.posenet_factory import load_model
import logging, coloredlogs
import tensorflow as tf

coloredlogs.install()

configuration=GetServicesConfiguration()


gpus= tf.config.experimental.list_physical_devices('GPU')
if gpus:
    MAX_MEMORY=7096
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1*MAX_MEMORY)])
    
        logging.info(f'Limiting GPU memory use to {MAX_MEMORY}')
    except RuntimeError as e:
        logging.error(e,exc_info=True)
else:
    logging.error("This worker doesn't have access to a GPU!!! stopping otherwise huge latency will apply")
    assert 1==0

r = redis.StrictRedis(host=configuration["REDIS_HOST"], port=6379)                          
#r = redis.StrictRedis(host="localhost", port=6379)                          
model = 'resnet50'  # mobilenet resnet50
#model = 'mobilenet'  # mobilenet resnet50
stride = 16  # 8, 16, 32 (max 16 for mobilenet, min 16 for resnet50)
quant_bytes = 4.0  # float
multiplier = 1  # only for mobilenet
posenet = load_model(model, stride, quant_bytes, multiplier)

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
    result["ID_IMG"] = message["ID_IMG"]+"posenet"
    
    return result


def handler(msg):

    img=stringToBGR(msg)
    
    start=time.time()
    pose_scores, keypoint_scores, keypoint_coords = posenet.estimate_multiple_poses(img)
    finish=time.time()
    #posenet.print_scores("data:", pose_scores, keypoint_scores, keypoint_coords) 
    #img_poses = posenet.draw_poses(img, pose_scores, keypoint_scores, keypoint_coords)
    return posenet.save_output_in_dict(pose_scores, keypoint_scores, keypoint_coords)
    ###############


def runRedisListener():
    #try:
    global idsToRemove
    global r

    opts=options()
    CURRENT_CHANNEL='posenet'
    p = r.pubsub()                                                              
    p.subscribe(CURRENT_CHANNEL)
    PAUSE = True
    count=-1
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
                logging.debug("Received img {}".format(data["ID_IMG"]))
                taken = r.getset(data["ID_IMG"], "1")
          
                opts,destination_channel=decode_pipeline(CURRENT_CHANNEL,data["pipeline"],opts)
      
                if taken == None:
                    coloredlogs.set_level(getattr(logging,opts.LOG_LEVEL))              
                
                    logging.debug("Processing img {}".format(data["ID_IMG"]))
                    
                    if(opts.REPORT_PERF):
                        start=time.time()
                    dataToSend = process(data)
                    
                    if(opts.REPORT_PERF):
                         finish=time.time()
                         FPS=int(1.0/(finish-start))
                    
                    if(opts.REPORT_PERF):
                        dataToSend["FPS_posenet"]=FPS
                    r.delete(data["ID_IMG"])

                    logging.debug('Sending img estimation')
                    r.publish(destination_channel, json.dumps(dataToSend))

if __name__ == '__main__':
    
    runRedisListener()

