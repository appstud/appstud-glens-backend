import re 
import numpy as np
import base64
import os
import logging

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def decode_pipeline(currentChannel,pipeline,opts):
    components=list(map(lambda x:x.strip().split(" ") ,pipeline.split("|")))
  
    for i,comp in enumerate(components):
        if(comp[0]==currentChannel): 
            if(len(components)!=i+1):
                destinationChannel=components[i+1][0]
            else:
                destinationChannel=components[0][0]
            for attr in comp[1:]:
                field,value=attr.split("=")
                if(hasattr(opts,field)):
                    setattr(opts,field,value)
                else:
                    logging.warning("field {} does not exists!!!".format(field))
            break
    return opts,destinationChannel

def GetServicesConfiguration():
    configuration = {}
    if "TENSORFLOW_HOST" not in os.environ.keys():
        configuration['TENSORFLOW_HOST'] = "tensorflow"
        #configuration['TENSORFLOW_HOST'] = "localhost"
    else:
        configuration['TENSORFLOW_HOST'] = os.getenv("TENSORFLOW_HOST")

    if "REDIS_HOST" not in os.environ.keys():
        configuration['REDIS_HOST'] = "redis"
    else:
        configuration['REDIS_HOST'] = os.getenv("REDIS_HOST")

    if("GLENS_RETURN_IMAGES" not in os.environ.keys()):
        configuration['GLENS_RETURN_IMAGES'] = False
    else:
        configuration['GLENS_RETURN_IMAGES'] = os.getenv('GLENS_RETURN_IMAGES')

    if("ATTRIBUTES" not in os.environ.keys()):
        configuration["ATTRIBUTES"]="ALL"
    else:
        configuration["ATTRIBUTES"]=os.getenv("ATTRIBUTES")

    return configuration
