import numpy as np
import os
import requests
import json
from utils import *
import logging

logging.basicConfig(level="WARNING")
configuration = GetServicesConfiguration()
HAIR_LABELS={0: 'bald', 1: 'black', 2: 'blonde', 3: 'hat', 4: 'red',5:'white'}
BEARD_MASK_LABELS={0:'beard',1:"mask on", 2: "no beard"}
MASK_LABELS={1:'no mask',0:"mask on"}


def log_exception(*err_val):
    def outer_wrapper(func):
        def inner_wrapper(*args,**kwargs):
            try:
                return func(*args,**kwargs)
            except Exception as e:
                logging.warning("Probably Tensorflow serving didn't start yet...",exc_info=True)
                
                if(len(err_val)==1):
                    return err_val[0]
                return err_val
        return inner_wrapper
    return outer_wrapper

@log_exception(None,None,None)
def callAgeSexGlassesNetwork(payload,version="1"):

    r=requests.post("http://" + configuration["TENSORFLOW_HOST"] + ":8501/v1/models/ageSexGlassesModel/versions/"+version+":predict",json=payload)
    out=json.loads(r.text)['predictions']
    glasses=[]
    gender=[]
    age=[]
    
    for imgIndex in range(len(payload['instances'])):
        if(version=="1"):
            glasses.append(out[imgIndex]['model_2/model_1/classif_glasses/Sigmoid:0'][0])
            gender.append(out[imgIndex]["model_2/classif_gender/Sigmoid:0"][0])
            age.append(out[imgIndex]["model_2/classif_age/Sigmoid:0"])
            age[-1]=np.sum(np.array(age[-1])>0.5)
            gender[-1]='Female' if gender[-1]>0.5 else 'Male'
            glasses[-1]='No' if glasses[-1]>0.5 else 'Yes'


        else:
            glasses.append("dont know")
            gender.append(out[imgIndex]["gender"][0])
            age.append(out[imgIndex]["age"][0])
            age[-1]=int(age[-1])
            gender[-1]='Female' if gender[-1]>0.5 else 'Male'

    return age, gender, glasses

@log_exception(None,None)
def callHairModel(payload):
    
    r=requests.post("http://" + configuration["TENSORFLOW_HOST"] + ":8501/v1/models/hairColorModel:predict",json=payload)
    out=json.loads(r.text)['predictions']
    colors=[]
    for imgIndex in range(len(payload['instances'])):
        colors.append(HAIR_LABELS[np.argmax(np.array(out[imgIndex]['color']))])
    masks=np.array(stringToBGRV3(out[0]['mask'])).astype(np.uint8)
    
    return colors, masks


@log_exception(None)
def callMaskModel(payload,version="2"):
    
    r=requests.post("http://" + configuration["TENSORFLOW_HOST"] + ":8501/v1/models/maskModel/versions/"+version+":predict",json=payload)
    predictions=json.loads(r.text)['predictions']
    mask=[]
    for pred in predictions:
        if(np.max(pred)>0.8):
            mask.append(MASK_LABELS[np.argmax(pred)])
        else:
            mask.append("bad position")
    
    return mask



@log_exception(None)
def callBeardMaskModel(payload):
    
    r=requests.post("http://" + configuration["TENSORFLOW_HOST"] + ":8501/v1/models/beardMaskModel:predict",json=payload)
    predictions=json.loads(r.text)['predictions']
    beard_mask=[]
    for pred in predictions:
        beard_mask.append(BEARD_MASK_LABELS[np.argmax(pred)])
    
    return beard_mask



@log_exception(None)
def callFaceRecognitionModel(payload,version="1"):
    r=requests.post("http://" + configuration["TENSORFLOW_HOST"] + ":8501/v1/models/faceRecognitionModel/versions/"+version+":predict",json=payload)
    predictions=json.loads(r.text)['predictions']
    encodings=[]
    for enc in predictions:
        encodings.append(enc)
    
    return encodings



@log_exception(None)
def callPersonReidentificationModel(payload,version="1"):
    r=requests.post("http://" + configuration["TENSORFLOW_HOST"] + ":8501/v1/models/person_reidentification:predict",json=payload)
    predictions=json.loads(r.text)['predictions']
    encodings=[]
    for enc in predictions:
        encodings.append(enc)
    
    return encodings





@log_exception(None)
def callPoseEstimationModel(payload):
    
    r=requests.post("http://" + configuration["TENSORFLOW_HOST"] + ":8501/v1/models/pose_estimation_model:predict",json=payload)
    predictions=json.loads(r.text)['predictions']
    pose=[]
    for pred in predictions:
        pose.append(pred)
    
    return pose



