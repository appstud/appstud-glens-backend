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
        if(pred[7] is not None):
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
            else:
                dataToReturn[iD]["pose_data"]=[]

            if(not pred[6] is None):
                dataToReturn[iD]["encoding"]=pred[6]
            else:
                dataToReturn[iD]["encoding"]=[]
        
        # person bbox available
        if(pred[11] is not None):
            dataToReturn[iD]["person_bbox"]=pred[11]
            if(not pred[9] is None):
                dataToReturn[iD]["person_encoding"]=pred[9]
            else:
                dataToReturn[iD]["person_encoding"]=[]

            if(not pred[10] is None):
                dataToReturn[iD]["person_pose_data"]=pred[10]
            else:
                dataToReturn[iD]["person_pose_data"]=[]

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
    person_encoding_exist=[]

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
    
    if(len(data.keys())==0):
        return image,{}

    for key in data: 
        ### booleans for later processing
        #Person
        isThereAPerson=isinstance(data[key],dict) and "bbox_yolo" in data[key]
        is3DPositionAvailableForPerson=isThereAPerson and  "pose_data" in data[key]
        doesDataComesFromBodyEstimationWorker=isThereAPerson and isinstance(data[key]["landmarks"],dict)
        doesDataComesFromFaceOrPersonDetection= not doesDataComesFromBodyEstimationWorker
        doesPersonBboxCoverTheWholeBody=isThereAPerson and ((doesDataComesFromBodyEstimationWorker and "leftAnkle" in data[key]["landmarks"] and "rightAnkle" in data[key]["landmarks"]) or not DoesDataComesFromBodyEstimationWorker)
        #Face
        isFaceBboxValid=doesDataComesFromBodyEstimationWorker and "leftEye" in data[key]["landmarks"] and  "rightEye" in data[key]["landmarks"] and "nose" in data[key]["landmarks"]  
        isFaceBboxValid=isFaceBboxValid or doesDataComesFromFaceOrPersonDetection
        isThereAFace=isinstance(data[key],dict) and "bbox" in data[key] and len(data[key]["bbox"])==4 and isFaceBboxValid
        
        isThereAPersonOrFace=isThereAPerson or isThereAFace
       
        bodyOrFaceLandmarksExist=isThereAPersonOrFace and "landmarks" in data[key] 
        ###################
        if(isThereAPersonOrFace):
            nb_entities+=1
        
        ### prepare person data for tf serving
        if(isThereAPerson):
            persons_bbox.append(data[key]["bbox_yolo"])
            
            if(is3DPositionAvailableForPerson):
                persons_3D_pos.append(data[key]["pose_data"])
            else:
                logging.warning("No pose data received from previous worker! will estimate it here if I can...")
                # for now I'm bypassing this condition later it's better to put it and change the tracking node to handle missing 3D position estimates for some persons...
                doesPersonBboxCoverTheWholeBody=True
                if(H is not None and doesPersonBboxCoverTheWholeBody):
                    persons_3D_pos.append(get_3D_position(H,np.array([[persons_bbox[-1][0]+persons_bbox[-1][2]*0.5, persons_bbox[-1][1]]])).tolist()[0])
                elif(H is None):
                    logging.warning("could not estimate 3D person pose: no homography ==> check CAM_ID or calibrate camera...")
                    persons_3D_pos.append(None)
                else:
                    logging.warning("could not estimate 3D person pose: no ankles predictions ==> bounding box doesn't cover the whole body...")
                    persons_3D_pos.append(None)
            
            # to be relaxed later
            if(opts.GET_PERS_REID and doesPersonBboxCoverTheWholeBody):
                patches_for_person_reidentification=getPersonsImgPatches(image,[persons_bbox[-1]])[0]
                payload_person_reid["instances"].append({'input_image':{ "b64": BGRToString(np.copy(patches_for_person_reidentification))}})
                person_encoding_exist.append(True)
            else:
                person_encoding_exist.append(False)

        else:
            persons_bbox.append(None)
            persons_3D_pos.append(None)
        
        ### prepare face data for tf serving
        if(bodyOrFaceLandmarksExist):
            landmarks.append(data[key]["landmarks"])
        else:
            landmarks.append(None)

        if (not isThereAFace):
            faceROIs.append(None)

        else:
            faceROIs.append(data[key]["bbox"])
            if(opts.GET_AGE_SEX_GLASSES or opts.GET_HAIR_COLOR):
                
                ### if posenet or openpose is used
                if(doesDataComesFromBodyEstimationWorker):
                    alignedGlobal,alignedFace,_=performFaceAlignment(np.copy(image),None, leftEye=np.array(data[key]["landmarks"]["rightEye"][::-1]),rightEye=np.array(data[key]["landmarks"]["leftEye"][::-1]))

                ### if mtcnn is used
                elif(len(landmarks[-1])==5):
                    alignedGlobal,alignedFace,_=performFaceAlignment(np.copy(image),None, np.array(landmarks[-1])[0,:],np.array(landmarks[-1])[1,:])
                ### if dlib face detection was used
                else:
                    alignedGlobal,alignedFace,_=performFaceAlignment(np.copy(image),np.array(landmarks[-1]))

                if(opts.GET_AGE_SEX_GLASSES):
                    
                    #faces_crop_age=cropFaceForPoseEstimation(np.copy(image),[faceROIs[-1]],ad=0.4,output_shape=(200,240))[0]
                    payload_age_sex_glasses["instances"].append({'input_image':{ "b64": BGRToString(np.copy(alignedFace))}})
                    #payload_age_sex_glasses["instances"].append({'input_image':{ "b64": BGRToString(np.copy(faces_crop_age))}})
                if(opts.GET_HAIR_COLOR):
                    payload_hair_color["instances"].append({'input_image':{ "b64": BGRToString(np.copy(alignedGlobal))}})
            
            if(opts.GET_POSE):
                logging.debug(faceROIs[-1])
                faces_for_pose_estimation=cropFaceForPoseEstimation(np.copy(image),[faceROIs[-1]],ad=0.6)[0]
                payload_pose_estimation["instances"].append({'input_image':{ "b64": BGRToString(np.copy(faces_for_pose_estimation))}})

            if(opts.GET_FACE_RECO):
                if(doesDataComesFromBodyEstimationWorker):
                    faces_for_recognition=alignRotationAndScaleLightCNN(image,faceROIs=[faceROIs[-1]], face_detection="pose_estimation",landmarks=[landmarks[-1]])[0]
                    payload_face_recognition["instances"].append({'input_image':{ "b64": BGRToString(np.copy(faces_for_recognition))}})
                
                else:
                    faces_for_recognition=alignRotationAndScaleLightCNN(image,faceROIs=[faceROIs[-1]], landmarks=[np.array(landmarks[-1])])[0]
                    payload_face_recognition["instances"].append({'input_image':{ "b64": BGRToString(np.copy(faces_for_recognition))}})
           
            if(opts.GET_MASK):
                payload_mask["instances"].append({'input_image':{ "b64": BGRToString(cv2.resize(image[int(faceROIs[-1][1]):int(faceROIs[-1][1]+faceROIs[-1][3]),int(faceROIs[-1][0]):int(faceROIs[-1][0]+faceROIs[-1][2]),:],(96,96)))}})



    ### call to tf serving
    if(opts.GET_PERS_REID and payload_person_reid["instances"]):
        pers_encodings=callPersonReidentificationModel(payload_person_reid,version="1")
    else:
        person_encoding_exist=[False]*nb_entities
        pers_encodings=[None]*nb_entities
    
    if( opts.GET_AGE_SEX_GLASSES and payload_age_sex_glasses["instances"]):
        age,gender,glasses=callAgeSexGlassesNetwork(payload_age_sex_glasses)
        #age,gender,glasses=callAgeSexGlassesNetwork(payload_age_sex_glasses,version="2")
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
                    if(bbox is not None):
                        roll,yaw,pitch,tx,ty,tz,ex=estimatePose(bbox, euler_angles[cc][0], euler_angles[cc][2], euler_angles[cc][1], intrinsicMatrix, distCoeff, rigidTransformToWCS=rigidTransformToWCS)
                        extrinsicMatrix.append(ex)
                        poseData.append([tx,ty,tz,roll,yaw,pitch])
                        cc+=1
                    else:
                        extrinsicMatrix.append(None)
                        poseData.append(None)


            else:
                extrinsicMatrix=[]
                poseData=[["n_a"]*3+e for e in euler_angles]
        else:
            poseData=[None]*nb_entities
    else:
        poseData=[None]*nb_entities
    
    ### not all detections will have face detections, must be careful for the missing ones
    if(doesDataComesFromBodyEstimationWorker):
        for i,bbox in enumerate(faceROIs):
            
            if(bbox is None):
                if(opts.GET_AGE_SEX_GLASSES and age is not None and payload_age_sex_glasses["instances"]):
                    age.insert(i,None)
                    gender.insert(i,None)
                    glasses.insert(i,None)
                if(opts.GET_FACE_RECO and encodings is not None and payload_face_recognition["instances"]):
                    encodings.insert(i,None)
                if(opts.GET_HAIR_COLOR and hairColor is not None and payload_hair_color["instances"]):
                    hairColor.insert(i,None)
                if(opts.GET_MASK and beard_mask is not None and payload_mask["instances"] ):
                    beard_mask.insert(i,None)




    cc=0
    final_pers_encodings=[]

    ## putting the right encoding in the right place in the list (missing encodings)
    for exist in person_encoding_exist:
        if(exist and pers_encodings is not None):
            final_pers_encodings.append(pers_encodings[cc])
            cc+=1
        else:
            final_pers_encodings.append(None)
    try:
        
        assert len(age)==len(gender)==len(glasses)==len(beard_mask)==len(hairColor)==len(poseData)==len(encodings)==len(faceROIs)==len(landmarks)==len(final_pers_encodings)==len(persons_3D_pos)==len(persons_bbox)
        data=prepareDataEntity(list(zip(age,gender,glasses,beard_mask, hairColor,poseData,encodings,faceROIs,landmarks,final_pers_encodings,persons_3D_pos,persons_bbox)))
    except AssertionError :
        print(age,gender,glasses,faceROIs)
        logging.error(("check length of these data they must be equal!!! "+ "{}"*12).format(len(age),len(gender),len(glasses),len(beard_mask), len(hairColor),len(poseData),len(encodings),len(faceROIs),len(landmarks),len(final_pers_encodings),len(persons_3D_pos),len(persons_bbox)))
        return image,{}
    except Exception as e:
        pass
    ###############################################################################
    
    return image, data
    
    
