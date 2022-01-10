import numpy as np
import time
import interfaceToRedis 
from track import Tracker
import json
from utils import decode_pipeline, str2bool
from logger_utils import coloredlogs, init_logger

logger=init_logger(__name__)

#coloredlogs.install()
tracker=Tracker()


class options():

    def initialize(self):
        self.USE_TEMPORAL=False
        self.USE_RECO=False
        self.USE_PERSON_DATA=True
        self.LOG_LEVEL="DEBUG"
        self.REPORT_PERF=False
        self.MIN_E=0.01
        self.RO=0.75
        self.ALPHA=0.01
        self.LAMBDA_A=0.2
        self.W_3D=0.1
        self.ALPHA_3D=0.5
        self.W_RECO=5
        self.ALPHA_RECO=1
       
    def __init__(self):
        self.initialize()        
    
    def reset(self):
        self.initialize()


    def __setattr__(self,name,value):
        if(name in ["USE_TEMPORAL","USE_RECO","REPORT_PERF","USE_PERSON_DATA"] and type(value)!=bool):
            value=str2bool(value)
        elif(name not in ["USE_TEMPORAL","USE_RECO","LOG_LEVEL", "REPORT_PERF","USE_PERSON_DATA"]):
            value=float(value)
        self.__dict__[name] = value
    
    
    def __str__(self):
        s=""
        for k in self.__dict__.keys():
            s+=k+" : "+ str(self.__dict__[k])+ " "
        return s

        


def removeEncodingField(dictionary):
    if(isinstance(dictionary,dict)):
        if("encoding" in dictionary.keys()):
            del dictionary["encoding"]
        if("person_encoding" in dictionary.keys()):
            del dictionary["person_encoding"]
    return dictionary

def transform(dictionary,new_ids):
  
    return {new_id if new_id!="unknown" else "unknown_"+str(key) : removeEncodingField(dictionary[key]) for key, new_id in zip(dictionary.keys(),new_ids)}


def runRedisListener():
    CURRENT_CHANNEL="tracking" 
    p = interfaceToRedis.r.pubsub()                                                              
    p.subscribe(CURRENT_CHANNEL)
    count=-1
    opts=options()
    #logging.basicConfig(level=getattr(logging,"DEBUG"))
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
                taken = interfaceToRedis.r.getset(data["ID_IMG"], "1")
                if(taken == None):
                    opts.reset() 
                    opts,destination_channel=decode_pipeline(CURRENT_CHANNEL,data["pipeline"],opts)
                    
                    
                    coloredlogs.set_level(opts.LOG_LEVEL)
                    
                    #coloredlogs.set_level("DEBUG")
                    logger.debug(f'Current options {opts}')
                    CAM_ID=data["CAM_ID"] 
                    current_time=data["current_time"]
                    detected_encodings=[]
                    dets_pos=[]
                    
                    #no_position_estimates=True
                    ### for now give priority for encoding and pos for persons not face, in the future we might add support for simultaneous prediciton based on the 2 infos...
                    ### later must implement the fact that some data might be lacking 3D estimates/ if its true ignore the contribution of position in the final decision..
                    for key in data["data"].keys():
                        
                        try:
                            detected_encodings.append(data["data"][key]["person_encoding" if ("person_encoding" in data["data"][key] and opts.USE_PERSON_DATA) else "encoding"])
                        except Exception as e:
                            logger.error("Catched error {}".format(e),exc_info=True)
                        try:
                            dets_pos.append(data["data"][key]["person_pose_data" if ("person_pose_data" in data["data"][key] and opts.USE_PERSON_DATA) else "pose_data"])
                            no_position_estimates=False
                        except Exception as e:
                            """if(USE_PERSON_DATA):
                            
                                dets_pos.append([np.nan]*3) 
                            else:
                                dets_pos.append([np.nan]*6)
                            """
                            logger.error("catched error {}".format(e), exc_info=True)
                     
                    start=time.time() 
                    #if((len(dets_pos)==0 or no_position_estimates) and len(detected_encodings)==0):
                    
                    if(len(dets_pos)==0 and len(detected_encodings)==0):
                        logger.warning("Empty data maybe no detections were received...")
                        ids = tracker.update(CAM_ID,opts,current_time=current_time)
                    else:
                        if(len(dets_pos)>0):
                            dets_pos=np.array(dets_pos)[:,0:3]
                        else:
                            dets_pos=np.empty((len(detected_encodings),3))
                        
                        ids = tracker.update(CAM_ID,opts,detected_encodings,current_time,dets_pos)
                    
                    data["data"]=transform(data["data"],ids)
                    ##testing only   
                    #states,ids=tracker.getTrackerStates()
                    #data["data"]["pose_data"]=states
                    #data["data"]["ids"]=ids
        
                    finish=time.time()
                    if(opts.REPORT_PERF):
                        FPS=int(1.0/(finish-start))
                        data["FPS_tracking"]=FPS
                    interfaceToRedis.r.delete(data["ID_IMG"])
                    interfaceToRedis.r.publish(destination_channel, json.dumps(data))

if __name__ == '__main__':
    runRedisListener()

