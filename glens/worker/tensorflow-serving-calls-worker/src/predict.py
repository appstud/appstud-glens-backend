from faceAlignmentV2 import performFaceAlignment, cropFaceForPoseEstimation,getPersonsImgPatches
from face_recognition_alignment import *
import numpy as np
import cv2
import time
import os
import logging
from enhanceQuality import dynamicRangeCompression
import json
from utils import *
from interfaceToTensorflowServing import *
from draw_utils import *
from pose_estimation import estimatePose
import pdb

configuration = GetServicesConfiguration()
#modelsPath=os.path.join(os.path.dirname(__file__),"models/")




def get_3D_position(H,_2Dpoints):
    """H: Homography transorm
       _2Dpoints: 2D points in image plane
    """
    positions= np.dot(H,np.hstack((_2Dpoints.astype(np.float32),np.ones([_2Dpoints.shape[0],1]))).T).T
    positions=positions.astype(np.float32)/positions[:,2].reshape(-1,1)
    return positions

def prepareDataEntity(data):
    #pers_encodings,persons_3D_pos,persons_bbox
    dataToReturn=dict()
    iD=0
    for pred in data:
        iD=iD+1
        dataToReturn[iD]={}
        #face bbox available
        if(len(pred[7])==4):
            dataToReturn[iD]["bbox"]=pred[7]
            if(not pred[0] is None):
                dataToReturn[iD]["age"]=int(pred[0])
        
            if(not pred[1] is None):
                dataToReturn[iD]["sex"]=pred[1]

            if(not pred[2] is None):
                dataToReturn[iD]["glasses"]=pred[2]

            if(not pred[3] is None):
                dataToReturn[iD]["beard_mask"]=pred[3]

            if(not pred[4] is None):
                dataToReturn[iD]["hairColor"]=pred[4]

            if(not pred[5] is None):
                dataToReturn[iD]["pose_data"]=pred[5]

            if(not pred[6] is None):
                dataToReturn[iD]["encoding"]=pred[6]
        
        # person bbox available
        if(len(pred[11])==4):
            dataToReturn[iD]["person_bbox"]=pred[11]
            if(not pred[9] is None):
                dataToReturn[iD]["person_encoding"]=pred[9]

            if(not pred[10] is None):
                dataToReturn[iD]["person_pose_data"]=pred[10]



        if(not pred[8] is None):
            dataToReturn[iD]["landmarks"]=pred[8]

    return dataToReturn



def get_attributs(image,data,opts,H=None,intrinsicMatrix=None,distCoeff=None,rigidTransformToWCS=np.eye(4)): 
    """data: dict containing bbox and landmarks
    """
    """
    if(intrinsicMatrix is None):
        intrinsicMatrix=np.asmatrix(np.array([[933.38537489 ,  0. ,  642.8541392 ],[  0.,933.08605844, 367.24814261],[  0.   , 0. ,  1.   ]]))
        distCoeff=np.array([-0.14454454,  0.48699891,  0.0013472,   0.00407075, -0.54467429] )
    """
  
    age=[]
    sex=[]
    allDetections=[]
    faceROIs=[]
    persons_bbox=[]
    landmarks=[]
    persons_3D_pos=[]
    if(opts.GET_AGE_SEX_GLASSES):
        payload_age_sex_glasses={"instances": []}
    if(opts.GET_HAIR_COLOR):
        payload_hair_color={"instances": []}
    if(opts.GET_POSE):
        payload_pose_estimation={"instances": []}
    if(opts.GET_FACE_RECO):
        payload_face_recognition={"instances": []}
    if(opts.GET_MASK):
        payload_mask={"instances": []}
    if(opts.GET_PERS_REID):
        payload_person_reid={"instances": []}

    nb_entities=0 
    for key in data: 
        
        if( isinstance(data[key],dict) and ("bbox_yolo" in data[key] or "bbox" in data[key] )):
            nb_entities+=1
        if( isinstance(data[key],dict) and "bbox_yolo" in data[key]):
            persons_bbox.append(data[key]["bbox_yolo"])
            #maybe add the option to estimate pose_data in this worker if its not available and we need it?
            if("pose_data" in data[key]):
                persons_3D_pos.append(data[key]["pose_data"])
            else:
                logging.warning("No pose data received from previous worker! will estimate it here if I can...")
                if(H is not None and ((isinstance(data[key]["landmarks"],dict) and "leftAnkle" in data[key]["landmarks"] and "rightAnkle" in data[key]["landmarks"]) or not isinstance(data[key]["landmarks"],dict))):
                    persons_3D_pos.append(get_3D_position(H,np.array([[persons_bbox[-1][0]+persons_bbox[-1][2]*0.5, persons_bbox[-1][1]]])).tolist()[0])
                else:

                    logging.warning("could not estimate it no ankle predictions or no homography...")
                    persons_3D_pos.append([])

            if(opts.GET_PERS_REID):
                patches_for_person_reidentification=getPersonsImgPatches(image,[persons_bbox[-1]])[0]
                payload_person_reid["instances"].append({'input_image':{ "b64": BGRToString(np.copy(patches_for_person_reidentification))}})
        elif(isinstance(data[key],dict) and "bbox_yolo" not in data[key]):
            persons_bbox.append([])
            persons_3D_pos.append([])

        if("landmarks" in data[key]):
            landmarks.append(data[key]["landmarks"])
        else:
            landmarks.append(None)

        if (isinstance(data[key]["landmarks"],dict) and ("bbox" not in data[key] or "leftEye" not in data[key]["landmarks"] or "rightEye" not in data[key]["landmarks"] or "nose" not in data[key]["landmarks"])):
            faceROIs.append([])

        elif (isinstance(data[key],dict) and "bbox" in data[key]):
            faceROIs.append(data[key]["bbox"])
            #if(isinstance(data[key]["landmarks"],dict)):
            #    landmarks.append(data[key]["landmarks"].values)
            #else:
            

            if(opts.GET_AGE_SEX_GLASSES or opts.GET_HAIR_COLOR):
                ### if mtcnn is used
                if(len(landmarks[-1])==5):
                    alignedGlobal,alignedFace,_=performFaceAlignment(np.copy(image),None, np.array(landmarks[-1])[0,:],np.array(landmarks[-1])[1,:])
                ## if posenet or openpose is used
                elif(isinstance(data[key]["landmarks"],dict) and "leftEye" in data[key]["landmarks"] and "rightEye" in data[key]["landmarks"]):
                    alignedGlobal,alignedFace,_=performFaceAlignment(np.copy(image),None, leftEye=np.array(data[key]["landmarks"]["leftEye"]),rightEye=np.array(data[key]["landmarks"]["rightEye"]))
                ### if dlib face detection was used
                else:
                    alignedGlobal,alignedFace,_=performFaceAlignment(np.copy(image),np.array(landmarks[-1]))
                if(opts.GET_AGE_SEX_GLASSES):
                    payload_age_sex_glasses["instances"].append({'input_image':{ "b64": BGRToString(np.copy(alignedFace))}})
                if(opts.GET_HAIR_COLOR):
                    payload_hair_color["instances"].append({'input_image':{ "b64": BGRToString(np.copy(alignedGlobal))}})
            
            if(opts.GET_POSE):

                if(isinstance(data[key]["landmarks"],dict) and  len(faceROIs[-1]) ==4 and "leftEye" in landmarks[-1] and "rightEye" in landmarks[-1] and "nose" in landmarks[-1]):
                    faces_for_pose_estimation=cropFaceForPoseEstimation(np.copy(image),[faceROIs[-1]],ad=0.6)[0]
                    payload_pose_estimation["instances"].append({'input_image':{ "b64": BGRToString(np.copy(faces_for_pose_estimation))}})
                elif(not isinstance(data[key]["landmarks"],dict)):
                    faces_for_pose_estimation=cropFaceForPoseEstimation(np.copy(image),[faceROIs[-1]],ad=0.6)[0]
                    payload_pose_estimation["instances"].append({'input_image':{ "b64": BGRToString(np.copy(faces_for_pose_estimation))}})

            if(opts.GET_FACE_RECO):
                if(isinstance(data[key]["landmarks"],dict) and  len(faceROIs[-1]) ==4 and "leftEye" in landmarks[-1] and "rightEye" in landmarks[-1] and "nose" in landmarks[-1]):
                    faces_for_recognition=alignRotationAndScaleLightCNN(image,faceROIs=[faceROIs[-1]], face_detection="pose_estimation",landmarks=[landmarks[-1]])[0]
                    payload_face_recognition["instances"].append({'input_image':{ "b64": BGRToString(np.copy(faces_for_recognition))}})
                    #logging.warning("Encoding for face not supported for now when using posenet or openpose...Recognition might be done using the GET_PERS_REID Flag...")
                
                elif(not isinstance(data[key]["landmarks"],dict)):
                    faces_for_recognition=alignRotationAndScaleLightCNN(image,faceROIs=[faceROIs[-1]], landmarks=[np.array(landmarks[-1])])[0]
                    payload_face_recognition["instances"].append({'input_image':{ "b64": BGRToString(np.copy(faces_for_recognition))}})
           
            if(opts.GET_MASK):
                payload_mask["instances"].append({'input_image':{ "b64": BGRToString(cv2.resize(image[int(faceROIs[-1][1]):int(faceROIs[-1][1]+faceROIs[-1][3]),int(faceROIs[-1][0]):int(faceROIs[-1][0]+faceROIs[-1][2]),:],(96,96)))}})




            #alignedFace=dynamicRangeCompression(np.copy(alignedFaceTuple[1]))
            #alignedFaceForHairModel=dynamicRangeCompression(np.copy(alignedFaceTuple[0]))
            
            #alignedFace=np.copy(alignedFaceTuple[1])
            #alignedFace=np.copy(faces[i])
            
            
            #alignedFaceForPoseEstimationModel=cv2.resize(alignedFaceForHairModel,(64,64))

    if(opts.GET_PERS_REID and payload_person_reid["instances"]  ):
        pers_encodings=callPersonReidentificationModel(payload_person_reid,version="1")
    else:
        pers_encodings=[None]*nb_entities
    ###call for age, sex, glasses model
    if( opts.GET_AGE_SEX_GLASSES and payload_age_sex_glasses["instances"]):
        age,gender,glasses=callAgeSexGlassesNetwork(payload_age_sex_glasses)
    else:
        age=[None]*nb_entities
        gender=[None]*nb_entities
        glasses=[None]*nb_entities

    if(opts.GET_MASK and payload_mask["instances"]):
        beard_mask=callMaskModel(payload_mask,version="3")
    else:
        beard_mask=[None]*nb_entities

    if(opts.GET_HAIR_COLOR and payload_hair_color["instances"] ):
        hairColor, segmentationMasks=callHairModel(payload_hair_color)
    else:
        hairColor=[None]*nb_entities

    if( opts.GET_FACE_RECO and payload_face_recognition["instances"]):
        encodings=callFaceRecognitionModel(payload_face_recognition,version="3")
    else:
        encodings=[None]*nb_entities
    
    if(opts.GET_POSE and payload_pose_estimation["instances"]):
        euler_angles=callPoseEstimationModel(payload_pose_estimation)
        if(euler_angles is not None): 
            if(intrinsicMatrix is not None):
                poseData=[]
                extrinsicMatrix=[]
                draw_weak_perspective=False
                cc=0
                for bbox in faceROIs:
                    if(len(bbox)==4):
                        roll,yaw,pitch,tx,ty,tz,ex=estimatePose(bbox, euler_angles[cc][0], euler_angles[cc][2], euler_angles[cc][1], intrinsicMatrix, distCoeff, rigidTransformToWCS=rigidTransformToWCS)
                        extrinsicMatrix.append(ex)
                        poseData.append([tx,ty,tz,roll,yaw,pitch])
                        cc+=1
                    else:
                        extrinsicMatrix.append(None)
                        poseData.append(None)


                    #poseData.append([tx,ty,tz,angles[0],angles[1],angles[2]])
            else:
                extrinsicMatrix=[]
                poseData=[["n_a"]*3+e for e in euler_angles]
        else:
            poseData=[None]*nb_entities
    else:
        poseData=[None]*nb_entities
    try: 
        logging.debug(("{}"*12).format(len(age),len(gender),len(glasses),len(beard_mask), len(hairColor),len(poseData),len(encodings),len(faceROIs),len(landmarks),len(pers_encodings),len(persons_3D_pos),len(persons_bbox)))
        data=prepareDataEntity(list(zip(age,gender,glasses,beard_mask, hairColor,poseData,encodings,faceROIs,landmarks,pers_encodings,persons_3D_pos,persons_bbox)))
    except:
        return image,{}
        
    ###############################################################################
    
    return image, data
    
    
