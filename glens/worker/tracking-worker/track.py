from __future__ import print_function
import os
import numpy as np
import time
import argparse
import pdb
import time
from itertools import groupby
from filterpy.kalman import KalmanFilter
import time
import heapq
from scipy.spatial import distance
import interfaceToRedis as redisInterface
import itertools
import copy 
import ast
import json
from logger_utils import init_logger
#############
"""Things to work on later:
    #only store encoding if there is a match with face_recognition only?!!!
    #adaptation for multi camera: must change the conditions on removing identities nb_frames_processed etc. to make them available on one camera done
    #for scalability must maybe take a subset of available identities based on their last known x,y,z locations?
    #maybe change implementations for distance calculations in order to workon GPU as well as the sorting in function matchEncodingV2
"""
###################





logger=init_logger(__name__)

#INFO_TO_USE=["TEMPORAL","FACE_RECOGNITION"]
#INFO_TO_USE=["FACE_RECOGNITION"]
#INFO_TO_USE=["TEMPORAL"]
"""
opts.USE_TEMPORAL="TEMPORAL" in INFO_TO_USE
opts.USE_RECO="FACE_RECOGNITION" in INFO_TO_USE
"""
try:
    from numba import jit
except:
    def jit(func):
        return func

np.random.seed(0)

#LOGGING=logger.getLevelName(logging.root.level)

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0]) #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def compute_distance(a,b):
    return np.linalg.norm(a-b)


class KalmanFilterTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    
    def initializeKalmanFilterState(self,state_vector,last_time_predict=None, new_tracker=False):
        if(last_time_predict is not None):
            self.time_of_last_predict=last_time_predict
        self.kf.F = np.array([[1,0,0,self.dt,0,0],[0,1,0,0,self.dt,0],[0,0,1,0,0,self.dt],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]])

        self.kf.R *= 0.2 # uncertainty on tx ,ty, tz measurement 30 cm
        self.kf.P[0:3,0:3] *= 0.1 #covariance matrix for state vector tx,ty,tz is low compared to velocities
        self.kf.P[3:,3:] *= 0.1 #covariance matrix for velocities is high
        self.kf.Q*= 0.3
        self.kf.x[:3] = state_vector

        if(new_tracker):
            return self.kalmanFilterStateToDict()
        return

    typeOfArray=type(np.array(0))
    def __init__(self,state_vector,current_time):
        ###Temporal sampling step dt in seconds ###
        self.dt=0.033
        
        ###For temporal tracking define a Kalman filter
        self.kf = KalmanFilter(dim_x=6, dim_z=3)

        ###load default values for Kalman filter
        self.initializeKalmanFilterState(state_vector)
        
        ###The time instant of the current state of the filter 
        self.time_of_last_predict = current_time

    
    @staticmethod
    def cvtToFloat(num):
        try:
            return float(num)
        except Exception as e:
            return None
    
    @staticmethod
    def convertArrayToDict(arr):
        return {'shape':str(arr.shape),"data":" ".join(map(str,arr.reshape(-1)))}
    
    @staticmethod
    def convertDictToArray(dic):
        shape=ast.literal_eval(dic['shape'])
        return np.array(list(map(KalmanFilterTracker.cvtToFloat,dic["data"].split(" ")))).reshape(shape)

    def kalmanFilterStateToDict(self):
        #dCopy=copy.deepcopy(self.kf.__dict__)
        #typeOfArray=type(np.array(0))
        #del dCopy["inv"]
        #dCopyNew = {k: "None" if v is None else v if type(v)!=typeOfArray  else  KalmanFilterTracker.convertArrayToDict(v) for k, v in self.kf.dCopy.items()}
        dCopyNew = {k: "None" if v is None else v if type(v)!=KalmanFilterTracker.typeOfArray  else  KalmanFilterTracker.convertArrayToDict(v) for k, v in self.kf.__dict__.items()}
        dCopyNew["time_of_last_predict"]=self.time_of_last_predict
        del dCopyNew["inv"]
        stateDict=dCopyNew

        return stateDict
    
    def loadKalmanFilterStateFromDict(self,stateDict):
        data = {k:None if v=="None" else v if not isinstance(v,dict)  else KalmanFilterTracker.convertDictToArray(v) for k, v in stateDict.items()}
        self.time_of_last_predict=data["time_of_last_predict"]
        del data["time_of_last_predict"]
        for k in data.keys():
            setattr(self.kf,k,data[k])
        return

    def predict(self,current_time,stateDict=None):
        """
        Advances the state vector and returns the new x,y,z location + dictionary containing the current parameters of the filter to be saved/updated later.
        """
        if (stateDict is not None):
            self.loadKalmanFilterStateFromDict(stateDict)
        
        while(self.time_of_last_predict<=current_time-self.dt):
            self.kf.predict()
            self.time_of_last_predict += self.dt

        return self.kf.x[0:3].reshape(-1), self.kalmanFilterStateToDict()
    
    def update(self,state_vector,current_time,stateDict=None,LOGGING=False):
        """
        Updates the state vector with new x,y,z observations starting from previous stateDict params.
        """
        start=time.time()
        self.loadKalmanFilterStateFromDict(stateDict)
        finish1=time.time()
        self.kf.update(state_vector)
        finish2=time.time()
        cc=self.kalmanFilterStateToDict()
        finish3=time.time()
        
        logger.performance("loadKalmanFromDict:{}".format(finish1-start))
        logger.performance("update: {}".format(finish2-finish1))
        logger.performance("loadKalmanToDict: {}".format(finish3-finish2))
            
        return cc
        
    def get_state(self,stateDict):
        ###return current state of the filter x,y,z,v_x,v_y,v_z
        self.loadKalmanFilterStateFromDict(stateDict)
        return self.kf.x




#def matchWithEncodingsV2(encodings,encodings_keys,person_IDs,all_ids, new_observed_encodings,ro=0.75,alpha=0.01,min_e=0.4,force_RNN=True):
#def matchWithEncodingsV2(encodings,encodings_keys,person_IDs,all_ids, new_observed_encodings,ro=0.6,alpha=0.01,min_e=0.1,force_RNN=True,LOGGING=False):
def matchWithEncodingsV2(encodings,encodings_keys,person_IDs,all_ids, new_observed_encodings,ro=0.7,alpha=0.01,min_e=0.01,force_RNN=True,LOGGING=False):
     
    number_of_observations=len(new_observed_encodings)
    nb_encodings=len(person_IDs)
    nb_identities=len(all_ids)
    scores=np.zeros([nb_encodings,number_of_observations])
    scores_per_identity=np.zeros([nb_identities,number_of_observations]) 
    if(len(new_observed_encodings)==0):
        return scores_per_identity



    
    logger.debug("Matching new detections with identities:{} that have {} envcodings in mempry".format(all_ids,len(encodings)))

    if(len(encodings)==0):
        return scores
    
    encodings=np.array(encodings)
    
    if(LOGGING):
        start=time.time()
    
    ###cpu
    d=distance.cdist(encodings,np.array(new_observed_encodings))
    """
    ###gpu
    start=time.time()
    d=calculateDistanceTF(encodings.astype('float'), np.array(new_observed_encodings,dtype=float))
    finish=time.time()
    print("tf method:",finish-start,encodings.shape, np.array(new_observed_encodings,dtype=float).shape)
    """

    if(LOGGING):
        finish=time.time()
        logger.performance("timing distance:{}".format(finish-start))
    
    #min_dists_index=d.argsort()
    
    if(LOGGING):
        start=time.time()
    if(number_of_observations<=2):
        min_dists_index=d.argsort()
    else:
        min_dists_index=np.argpartition(d,2,axis=-1)
    if(number_of_observations>1): 
        ratio=d[range(nb_encodings),min_dists_index[:,0]]/d[range(nb_encodings),min_dists_index[:,1]]        
    else:
        ########### we have one observation only cannot apply reverse KNN so we apply basic distance measurements and we increase the threshold ratio####
        ratio=d
        #ro=0.9

    
    if(LOGGING):
        finish=time.time()
        logger.performance("ratio sorting:{}".format(finish-start))
    ###if reverse KNN succeeded flag the need to update the eligibility of the corresponding encoding
    index_to_change_eli=ratio<ro
    enc_eli=list(itertools.compress(encodings_keys,index_to_change_eli))
    ###else flag the need to increase the age of the encoding
    index_to_change_age=[not r for r in index_to_change_eli]
    enc_age=list(itertools.compress(encodings_keys,index_to_change_age))
    
    ###calculate the ammount of update to the eligibility
    update_eli=(ratio[index_to_change_eli]/ro)**alpha

    finish2=time.time()
    if(LOGGING):
        logger.performance("preprocess:{}".format(finish2-finish))
    ###update encodings meta data age and eligibility in redis 
    redisInterface.updateEncodingsMetaData(enc_eli,update_eli,enc_age)
    
    if(LOGGING):
        logger.performance("redis update:{}".format(time.time()-finish2))
    ###calculate the score for each identity by summing the number of matches
    for i,ID in enumerate(person_IDs):
        if(index_to_change_eli[i]):
            scores_per_identity[all_ids.index(ID),min_dists_index[i,0]]+=1
    
    logger.debug("all_ids : {}".format(all_ids)) 
    logger.debug("scores matrix : {}".format(scores_per_identity)) 
    return scores_per_identity 



def associate_detections_to_trackers(opts,detections,trackers, face_recognition_scores, tracker_last_update_time,current_time):

    """
    Assigns detections to tracked object based on face_recognition_scores and based on current predictions of positions x,y,z in trackers 
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    #print("before:",face_recognition_scores)
    face_recognition_scores=1-face_recognition_scores/(0.01+np.sum(face_recognition_scores,0))
    """    
    face_recognition_scores=(np.logical_and(face_recognition_scores==np.max(face_recognition_scores,0) , face_recognition_scores!=0)).astype(int)
    face_recognition_scores_2=np.copy(face_recognition_scores)
    #face_recognition_scores_2[:,np.sum(face_recognition_scores,0)>1]=0
    face_recognition_scores_2[np.sum(face_recognition_scores,1)>1,:]=0
    face_recognition_scores=face_recognition_scores_2 
    print("after:",face_recognition_scores)
    #face_recognition_scores=np.transpose(1-np.transpose(face_recognition_scores)/(0.01+np.sum(face_recognition_scores,1)))
    """
    face_recognition_scores=np.transpose(face_recognition_scores)

    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    affinity_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)
    affinity_matrix_3D = np.zeros((len(detections),len(trackers)),dtype=np.float32)
    #affinity_matrix_face_recognition = np.zeros((len(detections),len(trackers)),dtype=np.float32)
    affinity_matrix_face_recognition = opts.W_RECO*(1-face_recognition_scores/(opts.ALPHA_RECO))
    #affinity_matrix_face_recognition = w_face_recognition*(face_recognition_scores)
    if(opts.USE_TEMPORAL): 
        for d,det in enumerate(detections):
            for t,trk in enumerate(trackers):
                affinity_matrix_3D[d,t] = opts.W_3D*(1-compute_distance(det[0:3],trk)/opts.ALPHA_3D)*np.exp(-(-tracker_last_update_time[t]+current_time)/opts.LAMBDA_A)
                #affinity_matrix_face_recognition[d,t] = w_face_recognition*(1-compute_distance(detected_encodings[d],encoding)**2/(reduce_alpha*alpha_face_recognition))
    #################problems with temporal information coming from bad bbox trial############
    affinity_matrix_3D[affinity_matrix_3D<-opts.W_3D]=-opts.W_3D 
    affinity_matrix=affinity_matrix_face_recognition+affinity_matrix_3D
    #print("all:", affinity_matrix)  
    if min(affinity_matrix.shape) > 0:
        matched_indices = linear_assignment(-affinity_matrix)
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)


    matches = []
    for m in matched_indices:
        if(affinity_matrix[m[0], m[1]]<=0):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    #logger.debug("affinity matrix: {}".format(affinity_matrix))
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Tracker(object):
    def __init__(self):
        
        self.tracker = KalmanFilterTracker(np.array([0,0,0]).reshape(3,1),None)
    
    
    def getTrackerStates(self,ids=None):
        if(ids is None):
            ids=redisInterface.get_all_ids()
        dictStates=redisInterface.getKalmanStateFromRedis(ids)        
        return [list(self.tracker.get_state(dic).reshape(-1)[0:3]) for dic in dictStates],ids
     

    def update(self,CAM_ID, opts,detected_encodings=[],current_time=0,dets_pos=np.empty((0, 3)),time_thresh=2,dist_thresh=4):
        CAM_ID=str(CAM_ID)
        LOGGING=opts.LOG_LEVEL=="PERFORMANCE"
        redisInterface.LOGGING=LOGGING
        all_ids=redisInterface.get_all_ids()
        
       
        nb_identities=len(all_ids)
 
        if(opts.USE_TEMPORAL and nb_identities!=0):
            ###filtering identities that are too far to reduce the computation cost 
            start=time.time()
            last_updated_time,last_positions=redisInterface.getLastKalmanFilterUpdateTimePosition(all_ids,CAM_ID)
            d=distance.cdist(dets_pos,last_positions)
            mask=np.logical_or(np.any(d<dist_thresh,0)>0,current_time-np.array(last_updated_time)>time_thresh)
            all_ids=[identity for mask_id,identity in zip(mask,all_ids) if mask_id]

            nb_identities=len(all_ids)
            finish=time.time()
            logger.debug("filtering identities nb_identities to match: {}, all_ids:{}".format(nb_identities, all_ids))
            logger.performance("filtering identities nb_identities time: {}".format( finish-start))



        
        if(nb_identities==0 and (len(detected_encodings)!=0 or dets_pos.shape[0]!=0)):
            for i in range(max(len(detected_encodings),dets_pos.shape[0])):
                if(opts.USE_TEMPORAL and opts.USE_RECO):
                    
                    redisInterface.addPerson(np.array(detected_encodings[i]).reshape(1,-1),current_time,CAM_ID,[self.tracker.initializeKalmanFilterState(dets_pos[i,0:3].reshape(3,1),current_time,new_tracker=True)])
                elif(opts.USE_RECO):
                    redisInterface.addPerson(np.array(detected_encodings[i]).reshape(1,-1),current_time,CAM_ID)
                else:
                    redisInterface.addPerson(None,current_time,CAM_ID,[self.tracker.initializeKalmanFilterState(dets_pos[i,0:3].reshape(3,1),current_time,new_tracker=True)])
                
            return redisInterface.get_all_ids() 
        elif(nb_identities==0 and len(detected_encodings)==0 and dets_pos.shape[0]==0):
            return []
       

        tracker_last_update_time=[]
        trks_pos = np.zeros((nb_identities, 3))
        to_del=[]
        if(opts.USE_TEMPORAL):
            
            if(LOGGING):
                start=time.time()
            statesDicts=redisInterface.getKalmanStateFromRedis(all_ids)
            tracker_last_update_time,_=redisInterface.getLastKalmanFilterUpdateTimePosition(all_ids,CAM_ID)
            
            if(LOGGING):
                finish=time.time()
                logger.performance("getting last_time_matched and kalman states from redis time:{}".format(finish-start))
            for t,trk in enumerate(trks_pos):
                trk[:],statesDicts[t]=self.tracker.predict(current_time,statesDicts[t])

            if(LOGGING):
                finish2=time.time()
                logger.performance("kalman filter prediction time:{}".format(finish2-finish))
        else:
            ###change last update time to an old time in order to not let the final decision be affected by temporal information
            tracker_last_update_time=[current_time-1000]*nb_identities
            ###
        

        """
        for t, trk in enumerate(trks_pos):
            #pos=self.trackers[t].get_state()
            pos = self.trackers[t].predict(current_time).reshape(1,-1)[0]
            trk[:] = [pos[0], pos[1], pos[2]]
            tracker_last_update_time.append(self.trackers[t].time_of_last_update)
            if np.any(np.isnan(pos)):
                print("oupsssssss")
                to_del.append(t)
        """

        if(opts.USE_RECO and len(detected_encodings)>=0):
            if(LOGGING):
                start=time.time()
            encodings_keys,encodings,person_IDs=redisInterface.getAllEncodings(all_ids,CAM_ID)
            scores=matchWithEncodingsV2(encodings,encodings_keys,person_IDs,all_ids, detected_encodings,ro=opts.RO,alpha=opts.ALPHA,min_e=opts.MIN_E,force_RNN=True,LOGGING=LOGGING)
            #scores=matchWithEncodingsV2(encodings,encodings_keys,person_IDs,all_ids,detected_encodings,LOGGING=LOGGING)
            if(LOGGING):
                finish=time.time()
                logger.performance("matching time for all identities {}".format(finish-start))
            
            scores=np.array(scores)
            ###################
        else:
            ###put scores to zero to not let the final decision be affected by face recognition scores
            scores=np.zeros((nb_identities,dets_pos.shape[0]))
        
        if(LOGGING):
            start=time.time()
        
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(opts,dets_pos,trks_pos,scores,tracker_last_update_time,current_time)
        
        if(LOGGING):
            finish=time.time()
            logger.performance("assosication time:{}".format(finish-start))
        ##################"""
        # update matched trackers with assigned detections
        if(LOGGING):
            start=time.time()
        logger.debug("all_ids {}".format(all_ids))
        logger.debug("matched identities: {}".format([all_ids[m[1]] for m in matched]))
        if((len(matched)>=1 and opts.USE_RECO)):
            redisInterface.storeEncoding([all_ids[m[1]] for m in matched],[np.array(detected_encodings[m[0]]).reshape(1,-1) for m in matched],current_time,CAM_ID)
            # only save new encoding if the info from the appearance model is OK with it otherwise we might risk storing bad encoding for this identity... 
            #redisInterface.storeEncoding([all_ids[m[1]] for m in matched if (scores[m[1],m[0]]>0 and scores[m[1],m[0]]==np.max(scores[:,m[0]]))],[np.array(detected_encodings[m[0]]).reshape(1,-1) for m in matched if (scores[m[1],m[0]]>0 and scores[m[1],m[0]]==np.max(scores[:,m[0]]))],current_time,CAM_ID)
        
        if(opts.USE_TEMPORAL):
            if(len(matched)>0):
                for m in matched:
                    statesDicts[m[1]]=self.tracker.update(dets_pos[m[0],0:3].reshape(3,1),current_time,stateDict=statesDicts[m[1]])   
                redisInterface.saveKalmanFilterUpdateTime([current_time]*len(matched),[all_ids[m[1]] for m in matched],CAM_ID)
        
        if(LOGGING):
            finish=time.time()
            logger.performance("adding new encodings time+kalman filter update times?:{}".format(finish-start))

        for i in unmatched_dets:
            if(opts.USE_RECO and opts.USE_TEMPORAL):
                redisInterface.addPerson(np.array(detected_encodings[i]).reshape(1,-1),current_time,CAM_ID,[self.tracker.initializeKalmanFilterState(dets_pos[i,0:3].reshape(3,1),current_time,new_tracker=True)])
            ##########bug need to be fixed in case of only face recognition info? we dont insert new identities if there was no identity matched this way we avoid storing new encodings that look alike encodings of ther identities###
            elif(len(matched)>=1 and opts.USE_RECO):    
                redisInterface.addPerson(np.array(detected_encodings[i]).reshape(1,-1),current_time,CAM_ID,None)
            elif(opts.USE_TEMPORAL):
                redisInterface.addPerson(None,current_time,CAM_ID,[self.tracker.initializeKalmanFilterState(dets_pos[i,0:3].reshape(3,1),current_time,new_tracker=True)])

        
        if(opts.USE_TEMPORAL):
            redisInterface.saveKalmanStateToRedis(statesDicts,all_ids)
        

        #ids=[all_ids[m[1]] for m in matched]
        matched_dets=[m[0] for m in matched]
        #####
        if(LOGGING):
            start=time.time()
        verified=redisInterface.getVerifiedIdentities(all_ids)
        
        if(LOGGING):
            finish=time.time()
        
        if(LOGGING):
            logger.performance("verified identities:{}".format(finish-start))
        #####

        ids=[all_ids[matched[matched_dets.index(i)][1]] if (i in matched_dets and verified[matched[matched_dets.index(i)][1]]==1) else "unknown" for i in range(dets_pos.shape[0])]
        
        if(LOGGING):
            start=time.time()
        
        all_ids=redisInterface.get_all_ids()
        redisInterface.identityShouldBeRemoved(all_ids[::-1],current_time,INFORMATION_TO_USE=["TEMPORAL" if opts.USE_TEMPORAL else None,"FACE_RECOGNITION" if opts.USE_RECO else None])
        if(LOGGING):
            finish=time.time()
            logger.performance("deleting identities time: {}".format( finish-start))
        return ids

