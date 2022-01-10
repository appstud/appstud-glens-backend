import cv2
import functools
import time
import redis
import json
from utils.utils import *
from utils.draw_utils import *
import os
import signal
from dataset import LoadStreams

exit=False
def exit_gracefully(signum,frame):
    global exit
    print('Signal handler called with signal', signum)
    exit=True


def testMultiStreamer(source):
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)
    data=LoadStreams(source)
    for _,data in data:
        if(exit):
                break
        for i,src in enumerate(data):
            #print(data['img'],data['CAM_ID'],data['time'])
            print(src,data[src]['CAM_ID'],data[src]['time'],data[src]['pipeline'])
            time.sleep(0.033)# wait time
            


if(__name__=="__main__"):
    # Get the configuration file for all the streams
    source=os.getenv("SOURCE")
    testMultiStreamer(source)       
