import cv2
import re 
import numpy as np
import base64
import os

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
                    #print(field,str2bool(value))
                    setattr(opts,field,value)
                else:
                    print("field ",field, "does not exists!!!")
            break
    return opts,destinationChannel



def stringToBGRV2(base64_string):
    #base64_string="data:image/jpeg;base64,"+base64_string    
    #base64_string = re.sub('^data:image/.+;base64,', '', base64_string)
    #base64_string = re.sub('_9j_', '/9j/', base64_string)
    #base64_string=bytes(base64_string+"=====", encoding='utf-8').decode('utf-8')
    #base64_string=bytes(base64_string,encoding="unicode").decode('unicode')

    #print(base64_string)
    #base64_string=bytes(base64_string+"=====", encoding='utf-8')
    imgdata=np.fromstring(base64.b64decode(base64_string), np.uint8)
    #print(base64.b64decode(base64_string+"====="))
    #print(imgdata)
    #imgdata=np.fromstring(base64_string,np.uint8)
    print(imgdata)
    image=cv2.imdecode(imgdata,cv2.IMREAD_UNCHANGED)
    return image[:,:,0:3]

def stringToBGRV3(base64_string):
    base64_string = re.sub('^data:image/.+;base64,', '', base64_string)
    base64_string = re.sub('_', '/', base64_string)
    base64_string = re.sub('-', '+', base64_string)
    base64_string = re.sub('=', '.', base64_string)
    #base64.PadRight(base64.Length + (4 - base64.Length % 4) % 4, '=');    
    imgdata=np.fromstring(base64.b64decode(str(base64_string)+"===="), np.uint8)
    image=cv2.imdecode(imgdata,cv2.IMREAD_UNCHANGED)
    return image[:,:,0:3]
    



def stringToBGR(base64_string):
    base64_string = re.sub('^data:image/.+;base64,', '', base64_string)
    
    #print(base64_string)
    imgdata=np.fromstring(base64.b64decode(str(base64_string)+'==='), np.uint8)
    image=cv2.imdecode(imgdata,cv2.IMREAD_UNCHANGED)
    return image[:,:,0:3]



def BGRToString(image,encoding_coeff=0.92):
    
    encodedImage=cv2.imencode('.jpg',image, [int(cv2.IMWRITE_JPEG_QUALITY), encoding_coeff*100])[1]
    imgdata=base64.b64encode(encodedImage)
    ##print(imgdata)
    #imgdata = 'data:image/jpg;base64,'+ imgdata.decode('utf-8')
    imgdata =  imgdata.decode('utf-8')
    return imgdata

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
