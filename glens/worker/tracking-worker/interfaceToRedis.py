import struct
import redis
import numpy as np
import redis 
import json
import pdb
import time
import functools
import itertools
from utils import GetServicesConfiguration
from logger_utils import init_logger

logger=init_logger(__name__)

"""
in redis we store data for each person as follows:

       key                            Description

PERSON_|ID|_ENC_|nb| : an encoding with number nb for person with id ID
PERSON_|ID|_ELI_|nb| : Eligibility of the encoding with number nb for person with id ID
PERSON_|ID|_AGE_|nb| : Age of the encoding with number nb for person with id ID
PERSON_|ID|_LAST_ENCODING : last number of an inserted for person with id ID, used to generate new keys for future encodings of this person
PERSON_|ID|_LAST_TIME_MATCHED : last time this person was identified by the facial recognition system
PERSON_|ID|_NB_TIME_MATCHED : number of times this person was identified by the facial recognition system
PERSON_|ID|_KALMAN : KALMAN filter parameters
PERSON_|ID|_LAST_UPDATE_KALMAN : Last time KALMAN filter was updated for identity ID
PERSON_|ID|_NB_TIME_MATCHED_KALMAN :  number of times this person was identified by the tracking system (Kalman filter)
PERSON_|ID|_LAST_POSITION : last position x,y,z of person with identity ID
PERSON_|ID|_NB_FRAMES_PROCESSED_KALMAN : 
PERSON_|ID|_CAM_ID: the ID of the camera this person was first detected in
ALL_PERSONS_IDS: IDS of all persons in redis
ID_COUNTER: an id counter
PERSON_|ID|_ENC_KEYS : stores all the encoding keys of identity ID
NB_ENC: Number of encodings in redis
"""

configuration=GetServicesConfiguration()
r = redis.StrictRedis(host=configuration["REDIS_HOST"],port=6379)                          
#r.config_set('maxmemory-policy','noeviction')
r.set("ALL_PERSON_IDS",json.dumps([]))
r.set("NB_ENC",0)

############kalman filter save and load data###################

def getLastKalmanFilterUpdateTimePosition(person_IDs,CAM_ID):
    listToGetTime=[]
    listToGetPosition=[]
    

    cam_ids=getCamIds(person_IDs) 
    for i,person_ID in enumerate(person_IDs):
        listToGetTime.append("PERSON_"+str(person_ID)+"_LAST_UPDATE_KALMAN")
        listToGetPosition.append("PERSON_"+str(person_ID)+"_LAST_POSITION")
        if(CAM_ID==cam_ids[i]):
            r.incrby("PERSON_"+str(person_ID)+"_NB_FRAMES_PROCESSED_KALMAN",1) 
    
    last_update=list(map(float,r.mget(listToGetTime)))
    last_positions=list(map(json.loads,r.mget(listToGetPosition)))

    return last_update,np.array(last_positions)


def getMetaDataForIdentityKalman(person_IDs):
    keys_to_get=[]
    for person_ID in person_IDs:
        keys_to_get+=["PERSON_"+str(person_ID)+"_LAST_UPDATE_KALMAN","PERSON_"+str(person_ID)+"_NB_TIME_MATCHED_KALMAN","PERSON_"+str(person_ID)+"_NB_FRAMES_PROCESSED_KALMAN"]
    #logger.debug(time.time(),"keys_to_get",keys_to_get)
    data=list(map(float,r.mget(keys_to_get)))
    data=[{"last_time_matched":data[3*x],"nb_times_matched":data[3*x+1],"nb_frames_processed":data[3*x+2]} for x in range(len(person_IDs))] 
    
    return data


def saveKalmanFilterUpdateTime(last_update_time,person_IDs,CAM_ID):
    dictToSave={}

    cam_ids=getCamIds(person_IDs) 
    for i,person_ID in enumerate(person_IDs):
    
        dictToSave["PERSON_"+str(person_ID)+"_LAST_UPDATE_KALMAN"]=last_update_time[i]
        if(cam_ids[i]==CAM_ID):
            r.incrby("PERSON_"+str(person_ID)+"_NB_TIME_MATCHED_KALMAN",1) 
    r.mset(dictToSave)


def getKalmanStateFromRedis(person_IDs):
    listToGet=[]
    for person_ID in person_IDs:
        listToGet.append("PERSON_"+str(person_ID)+"_KALMAN")
        
    states=list(map(json.loads,r.mget(listToGet)))
    
    return states

def saveKalmanStateToRedis(jsonStates,person_IDs=[0]):
    ###TODO
    dictToSave={}
    for i,jsonState in enumerate(jsonStates):
        dictToSave["PERSON_"+str(person_IDs[i])+"_KALMAN"]=json.dumps(jsonState)
        dictToSave["PERSON_"+str(person_IDs[i])+"_LAST_POSITION"]=json.dumps(list(map(float,jsonState["x"]["data"].split(" ")))[0:3])
    r.mset(dictToSave)
    return 


#####################################################################################################

def argsort(seq,reverse=True):
    return sorted(range(len(seq)), key=seq.__getitem__,reverse=reverse)

def removeOldEncodings(nb_to_del,ids=None,INFORMATION_TO_USE=["FACE_RECOGNITION"]):
    if(ids is None):
        ids=get_all_ids()
    all_enc_keys=functools.reduce(lambda x,y:x+y,get_encoding_keys(ids))
    
    logger.debug("{} encodings in memory for the ids {}".format(len(all_enc_keys),ids))
    all_enc_age=r.mget(list(map(lambda x :x.replace("ENC","AGE"),all_enc_keys)))
    #listOfIndex=argsort(all_enc_age)
    
    listOfIndex=np.argpartition(np.array(all_enc_age),len(all_enc_age)-nb_to_del)
    all_enc_keys=[all_enc_keys[i] for i in listOfIndex]
    
    for i in range(nb_to_del):
        #removeEncoding(all_enc_keys[i],INFORMATION_TO_USE=INFORMATION_TO_USE)
        removeEncoding(all_enc_keys[-i],INFORMATION_TO_USE=INFORMATION_TO_USE)

    #r.decrby("NB_ENC",nb_to_del)

    

def control_number_of_encodings_in_redis(max_nb_enc=100,INFORMATION_TO_USE=["FACE_RECOGNITION"]):
    nb_enc=int(r.get("NB_ENC"))
    if(nb_enc>max_nb_enc):
        
        logger.warning("Controlling memory size taken by encodings... will delete {} old encodings".format(max_nb_enc//5))
        removeOldEncodings(max_nb_enc//5,None,INFORMATION_TO_USE)

def get_all_ids():
    ids=json.loads(r.get("ALL_PERSON_IDS"))
    return ids

def get_encoding_keys(person_IDs,sample="ALL",max_enc_per_identity=50):
    listToGet=[]
    listAgeToGet=[]
    listOfLen=[] 
    for person_ID in person_IDs:
        listToGet.append("PERSON_"+str(person_ID)+"_ENC_KEYS")
     
    keys=list(map(json.loads,r.mget(listToGet)))
    
    if(sample =="ALL"):
        return keys
    """elif(sample=="RANDOM LATEST"):
        for lKey in keys:
            listAgeToGet.append(list(map(lambda x:x.replace("ENC","AGE"),lKey)))
            listOfLen.append(len(lKey))
        listOfLen.insert(0,0)
        ages=functools.reduce(lambda x,y:x+y,listAgeToGet)
        ages=np.array(list(map(int,r.mget(ages))))
        for i,lKey in keys:
            if(i==0):
                ##you are getting many keys at once its  the same thing, must store all age in one list per identity implement this tomorow!
                if(listOfLen[i+1]>max_enc_per_identity):
                    np.argpartition(ages[listOfLen[i]:listOfLen[i]+listOfLen[i+1]])
                else:
                    pass
    """
        


def store_new_encoding_key(person_IDs,new_keys,nbs):
    
    keys=get_encoding_keys(person_IDs)
    dictToSave={}
    dictToSaveNbs={}
    for i,k in enumerate(keys):
        keys[i].append(new_keys[i])
        dictToSave["PERSON_"+str(person_IDs[i])+"_ENC_KEYS"]=json.dumps(keys[i])
        dictToSaveNbs["PERSON_"+str(person_IDs[i])+"_LAST_ENCODING"]=nbs[i]
     
    r.mset(dictToSave)
    updateLastInsertedEncodingKey(person_IDs,nbs,dictToSaveNbs)

def remove_encoding_key(person_ID,key,INFO=["FACE_RECOGNITION"]):
    keys=get_encoding_keys([person_ID])[-1]
    keys.remove(key)
    if(len(keys)>0):
        r.set("PERSON_"+str(person_ID)+"_ENC_KEYS",json.dumps(keys))
    else:
        removeIdentity([person_ID],INFORMATION_TO_USE=INFO)


def addPerson(encoding,create_time,camera_ID,jsonStates=None):
    all_ids=get_all_ids()
    r.setnx("ID_COUNTER",0)
    ID=r.incrby("ID_COUNTER",1)
    all_ids.append(ID)
    r.set("ALL_PERSON_IDS",json.dumps(all_ids))
    #last_time_matched=r.set("PERSON_"+str(ID)+"_LAST_TIME_MATCHED",create_time)
    r.set("PERSON_"+str(ID)+"_CAM_ID",str(camera_ID))
    logger.debug("Adding new person for camera {0} with ID {1}".format(camera_ID,ID))
    if(jsonStates is not None):
        saveKalmanStateToRedis(jsonStates,[ID])
        r.mset({"PERSON_"+str(ID)+"_NB_TIME_MATCHED_KALMAN":-1,"PERSON_"+str(ID)+"_NB_FRAMES_PROCESSED_KALMAN":0})
        saveKalmanFilterUpdateTime([create_time],[ID],str(camera_ID))
    logger.performance("aaaaaaaa") 
    if(encoding is not None): 
        #r.mset({"PERSON_"+str(ID)+"_NB_TIME_MATCHED":-1,"PERSON_"+str(ID)+"_NB_FRAMES_PROCESSED":0,"PERSON_"+str(ID)+"_ENC_KEYS":json.dumps([])})
        r.mset({"PERSON_"+str(ID)+"_NB_TIME_MATCHED":-1,"PERSON_"+str(ID)+"_NB_FRAMES_PROCESSED":0,"PERSON_"+str(ID)+"_ENC_KEYS":json.dumps([]),"PERSON_"+str(ID)+"_VERIFIED":0})
        updateLastInsertedEncodingKey([ID],[0])
        storeEncoding([ID],[encoding],create_time,str(camera_ID))


def toRedis(dict_of_enc):
    """Store given Numpy array 'a' in Redis under key 'n'"""
    for key in dict_of_enc.keys():
        h, w = dict_of_enc[key].shape
        shape = struct.pack('>II',h,w)

        dict_of_enc[key] = shape + dict_of_enc[key].astype(np.float32).tobytes()
    # Store encoded data in Redis
    r.mset(dict_of_enc)
    return

def fromRedis(encoded):
    """Retrieve Numpy array from Redis key 'n'"""
    #encoded = r.get(n)
    h, w = struct.unpack('>II',encoded[:8])
    a = np.frombuffer(encoded, dtype=np.float32, offset=8).reshape(h,w)
    return a

def getLastInsertedEncodingKey(person_IDs):
    """get the number of the last encoding inserted
    """
    listToGet=[]
    for person_ID in person_IDs:
        listToGet.append("PERSON_"+str(person_ID)+"_LAST_ENCODING")
    return list(map(int,r.mget(listToGet)))

def updateLastInsertedEncodingKey(person_IDs,nbs,dictToSave=None):
    """
    """
    if(dictToSave is None):
        dictToSave={}
        for i,person_ID in enumerate(person_IDs):
            dictToSave["PERSON_"+str(person_ID)+"_LAST_ENCODING"]=nbs[i]
        
    r.mset(dictToSave)

def getCamIds(person_IDs):
    listToGet=[]
    for person_ID in person_IDs:
        listToGet.append("PERSON_"+str(person_ID)+"_CAM_ID")
        
    cam_ids=list(map(lambda x:x.decode("utf-8"),r.mget(listToGet)))
    
    return cam_ids

def getVerifiedIdentities(person_IDs):
    listToGet=[]

    for i,ID in enumerate(person_IDs):
        listToGet.append("PERSON_"+str(person_IDs[i])+"_VERIFIED")
    
    return list(map(int,r.mget(listToGet)))


def storeEncoding(person_IDs,encoding,last_time_matched,CAM_ID):
    dict_to_set={}
    dict_to_set_enc={}
    new_encoding_keys_to_add=[]
     
    nbs=getLastInsertedEncodingKey(person_IDs)
    cam_ids=getCamIds(person_IDs) 
    r.incrby("NB_ENC",len(encoding))
    
    for i,person_ID in enumerate(person_IDs):
        nbs[i]=nbs[i]+1
        dict_to_set["PERSON_"+str(person_ID)+"_ELI_"+str(nbs[i])]=1.0
        dict_to_set["PERSON_"+str(person_ID)+"_AGE_"+str(nbs[i])]=0
        dict_to_set["PERSON_"+str(person_ID)+"_LAST_TIME_MATCHED"]=last_time_matched
        ##BE CAREFUL HERE IN CASE OF POSSIBLE DETECTION BY MULTIPLE CAMERAS AT THE BEGINING MAYBE U SHOULD REMOVE THE FIRST CONDITION
        #logger.debug("Condition on camera ID of the person {0}, current camera ID {1}, ID of the person {2}".format(cam_ids[i],CAM_ID,person_ID))
        if(nbs[i]<7 and cam_ids[i]==CAM_ID):
            r.incrby("PERSON_"+str(person_ID)+"_NB_TIME_MATCHED",1)

        new_encoding_keys_to_add.append("PERSON_"+str(person_ID)+"_ENC_"+str(nbs[i]))
        dict_to_set_enc["PERSON_"+str(person_ID)+"_ENC_"+str(nbs[i])]=encoding[i]
        #updateLastInsertedEncodingKey(person_ID,nb+1)
    
    toRedis(dict_to_set_enc)
    store_new_encoding_key(person_IDs,new_encoding_keys_to_add,nbs)
    r.mset(dict_to_set)


def getIdentitySpecificData(person_IDs):
    keys_to_get=[]
    for person_ID in person_IDs:
        keys_to_get+=["PERSON_"+str(person_ID)+"_LAST_TIME_MATCHED","PERSON_"+str(person_ID)+"_NB_TIME_MATCHED","PERSON_"+str(person_ID)+"_NB_FRAMES_PROCESSED"]
    #logger.debug(time.time(),"keys_to_get",keys_to_get)
    data=list(map(float,r.mget(keys_to_get)))
    data=[{"last_time_matched":data[3*x],"nb_times_matched":data[3*x+1],"nb_frames_processed":data[3*x+2]} for x in range(len(person_IDs))] 
    
    return data


def getAllEncodings(person_IDs,CAM_ID):
    """
    """
    enc=[]
    IDS=[]

    new_enc_keys=get_encoding_keys(person_IDs)
     
    cam_ids=getCamIds(person_IDs)
    for i,person_ID in enumerate(person_IDs):
        frames_processed=int(r.get("PERSON_"+str(person_ID)+"_NB_FRAMES_PROCESSED"))
        if(frames_processed<6 and CAM_ID==cam_ids[i]):
            r.incrby("PERSON_"+str(person_ID)+"_NB_FRAMES_PROCESSED",1)


        IDS+=len(new_enc_keys[i])*[person_ID]

   
    enc_keys=functools.reduce(lambda x,y:x+y,new_enc_keys) 
    start= time.time()
    enc=r.mget(enc_keys)
    enc=list(map(lambda x: fromRedis(x).reshape(-1),enc))
    finish=time.time()
    logger.performance("time inside get allencodings: {} - nb_retrieved_encodings= {})".format(finish-start,len(enc_keys)))
    logger.debug("number_of_encodings: {}".format(len(enc)))
    return enc_keys,enc,IDS


def updateEncodingsMetaData(enc_eli,update_eli,enc_age,min_e=0.01):
    
    #increase age for some encodings at once using mset and mget
    nb_eli=len(enc_eli)
    nb_age=len(enc_age)
    updateEli=nb_eli>0
    updateAge=nb_age>0

    enc_age=[x.replace("ENC","AGE") for x in enc_age ]
    enc_eli_new=[x.replace("ENC","ELI") for x in enc_eli ]
    start=time.time() 
    data=r.mget(enc_age+enc_eli_new)
    finish=time.time()
    logger.performance("timemGet:{}, nb_enc_keys_age_eli={}".format(finish-start,nb_age+nb_eli))
    age=data[:nb_age]
    eli=data[nb_age:]
    #eli=r.mget(enc_eli_new)
    
    
    #age=r.mget(enc_age)
    
    
    dict_age_eli={}
    for i,k_age in enumerate(enc_age):
        dict_age_eli[k_age]=int(age[i])+1
        

    
    
    for i,k_eli in enumerate(enc_eli_new):
        dict_age_eli[k_eli]=np.float32(eli[i])*update_eli[i]
        dict_age_eli[k_eli.replace("ELI","AGE")]=0

    start=time.time()
    if(updateEli or updateAge):
        r.mset(dict_age_eli)
    finish=time.time()
    
    logger.performance("timeSET: {}".format(finish-start))
    enc_to_del=itertools.compress(enc_eli,np.array(list(map(float,eli)))<min_e)
    
    for enc in enc_to_del:
        removeEncoding(enc)



def identityShouldBeRemoved(person_IDs,currentTime,data=None,max_age=150,INFORMATION_TO_USE=["TEMPORAL"]):
    last_potential_id=int(r.get("ID_COUNTER"))
    if(data is None and "FACE_RECOGNITION" in INFORMATION_TO_USE):
        data=getIdentitySpecificData(person_IDs)
    
    elif(data is None):
        
        data=getMetaDataForIdentityKalman(person_IDs)


    id_to_del=[]
    for i,d in enumerate(data):
        number_of_frames_processed=d["nb_frames_processed"]
        number_of_times_matched=d["nb_times_matched"]
        last_time_matched=d["last_time_matched"]
        if(number_of_frames_processed<=2 and number_of_times_matched!=number_of_frames_processed):
            ####
            if(last_potential_id==person_IDs[i]):
                r.decrby("ID_COUNTER",1)
                last_potential_id-=1
            
            logger.debug("Identity {} not matched in two consecutive frames... removing nb_times_matched={}, nb_frame_processed={}".format(person_IDs[i],number_of_times_matched,number_of_frames_processed))
                
            id_to_del.append(person_IDs[i])
        elif(number_of_frames_processed==6 and number_of_times_matched<3):
            #### 
            if(last_potential_id==person_IDs[i]):
                r.decrby("ID_COUNTER",1)
                last_potential_id-=1
            
            
            logger.debug("Identity {} not matched in one frame after inclusion... removing".format(person_IDs[i]))
            id_to_del.append(person_IDs[i])

        elif(number_of_frames_processed==6 and number_of_times_matched>=3):
            r.set("PERSON_"+str(person_IDs[i])+"_VERIFIED",1)
            pass
        
        elif(currentTime-last_time_matched>max_age):
            logger.debug('Removing identity age={}, ID= {}'.format(currentTime-np.float64(r.get("PERSON_"+str(person_IDs[i])+"_LAST_TIME_MATCHED")),person_IDs[i]))
            id_to_del.append(person_IDs[i])

    if(len(id_to_del)!=0):   
        removeIdentity(id_to_del,person_IDs,INFORMATION_TO_USE)
    if("FACE_RECOGNITION" in INFORMATION_TO_USE):
        control_number_of_encodings_in_redis(max_nb_enc=5000,INFORMATION_TO_USE=INFORMATION_TO_USE)

def removeIdentity(person_IDs,all_ids=None,INFORMATION_TO_USE=["TEMPORAL","FACE_RECOGNITION"]):
        
    if(all_ids is None):
        all_ids=get_all_ids()
        logger.debug("removing encodings for newly inserted {}".format(person_IDs))
    else:
        logger.debug("removing encodings for old inserted {}".format(person_IDs))
    

    if("FACE_RECOGNITION" in INFORMATION_TO_USE): 
        enc_keys=get_encoding_keys(person_IDs)
        for i,identity_keys in enumerate(enc_keys):
             
            #r.decrby("NB_ENC",len(identity_keys))
            for key in identity_keys:
                # delete the key
                r.delete(key)
                r.delete(key.replace("ENC","AGE"))
                r.delete(key.replace("ENC","ELI"))

            r.delete("PERSON_"+str(person_IDs[i])+"_LAST_TIME_MATCHED")
            r.delete("PERSON_"+str(person_IDs[i])+"_NB_TIME_MATCHED")
            r.delete("PERSON_"+str(person_IDs[i])+"_LAST_ENCODING")
            r.delete("PERSON_"+str(person_IDs[i])+"_ENC_KEYS")
    if("TEMPORAL" in INFORMATION_TO_USE): 
         for i,person_ID in enumerate(person_IDs):
             
             r.delete("PERSON_"+str(person_ID)+"_LAST_POSITION")
             r.delete("PERSON_"+str(person_ID)+"_LAST_UPDATE_KALMAN")
             r.delete("PERSON_"+str(person_ID)+"_NB_FRAMES_PROCESSED_KALMAN") 
             r.delete("PERSON_"+str(person_ID)+"_LAST_TIME_MATCHED_KALMAN")
             r.delete("PERSON_"+str(person_ID)+"_NB_TIME_MATCHED_KALMAN")
             r.delete("PERSON_"+str(person_ID)+"_KALMAN") 
 
    for i,person_ID in enumerate(person_IDs):
        all_ids.remove(person_IDs[i])
    
    """
    enc_key_to_del=[]
    for person_ID in person_IDs:
        enc_keys=get_encoding_keys(person_ID)
        enc_key_to_del+=enc_keys
        for key in enc_keys:
            # delete the key
            enc_key_to_del+=[key.replace("ENC","AGE"),key.replace("ENC","ELI")]

        enc_key_to_del+=["PERSON_"+str(person_ID)+"_LAST_TIME_MATCHED"]
        enc_key_to_del+=["PERSON_"+str(person_ID)+"_NB_TIME_MATCHED"]
        enc_key_to_del+=["PERSON_"+str(person_ID)+"_LAST_ENCODING"]
        enc_key_to_del+=["PERSON_"+str(person_ID)+"_ENC_KEYS"]
        r.delete(enc_key_to_del)
        all_ids.remove(person_ID)
    """
    r.set("ALL_PERSON_IDS",json.dumps(all_ids))

def removeEncoding(encoding_key,INFORMATION_TO_USE=["FACE_RECOGNITION"]):
    """
    """ 
    
    logger.debug(f'removing encoding {encoding_key}')
    #nb=encoding_key.split('_')[-1]
    #delete corresponding age of the encoding from redis
    r.delete(encoding_key.replace("ENC","AGE"))
    #logger.debug(encoding_key)    
    #############"be careful potential bug after if u include id for store or cam
    remove_encoding_key(int(encoding_key.split("_")[1]),encoding_key,INFORMATION_TO_USE)
    #delete corresponding eligibility of the encoding from redis
    r.delete(encoding_key.replace("ENC","ELI"))
    
    r.decrby("NB_ENC",1)
    #delete encoding from redis
    r.delete(encoding_key)
    


