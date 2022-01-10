from posenet.base_model import BaseModel
import posenet
import time
import cv2
import numpy as np
class PoseNet:

    def __init__(self, model: BaseModel, min_score=0.25):
        self.model = model
        self.min_score = min_score

    def estimate_multiple_poses(self, image, max_pose_detections=15,img_size_for_prediction=(1280,960)):
        s=int(image.shape[0]/img_size_for_prediction[0])+2
        image=cv2.resize(image,(int(image.shape[1]/s),int(image.shape[0]/s)))         

        heatmap_result, offsets_result, displacement_fwd_result, displacement_bwd_result, image_scale = \
            self.model.predict(image)
        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
            heatmap_result.numpy().squeeze(axis=0),
            offsets_result.numpy().squeeze(axis=0),
            displacement_fwd_result.numpy().squeeze(axis=0),
            displacement_bwd_result.numpy().squeeze(axis=0),
            output_stride=self.model.output_stride,
            max_pose_detections=max_pose_detections,
            min_pose_score=self.min_score)

        keypoint_coords *= s*image_scale

        return pose_scores, keypoint_scores, keypoint_coords

    def estimate_single_pose(self, image):
        return self.estimate_multiple_poses(image, max_pose_detections=1)

    def draw_poses(self, image, pose_scores, keypoint_scores, keypoint_coords):
        draw_image = posenet.draw_skel_and_kp(
            image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=self.min_score, min_part_score=self.min_score)

        return draw_image

    def save_output_in_dict(self, pose_scores, keypoint_scores, keypoint_coords,with_bbox=True):
        output_dict={}

        for pi in range(len(pose_scores)):
            if pose_scores[pi] == 0.:
                break
            
            output_dict[pi]={}
            
            if(with_bbox):
                #heuristics to get bbox of face and persons from body keypoints
                                
                #face_bbox
                #3 and 4 are indexesfrom left and right ears; 0,1,2 are the eyes and nose =>their mean is the center of the face
                
                if(np.mean(keypoint_scores[pi, :3])>0.5):
                    
                    width=np.sqrt(np.sum((keypoint_coords[pi,3,:]-keypoint_coords[pi,4,:])**2))
                    center=np.mean(keypoint_coords[pi,:5,:],axis=0)       
                    dist_1=np.sqrt(np.sum((keypoint_coords[pi, 0,:]-keypoint_coords[pi,4,:])**2))
                    dist_2=np.sqrt(np.sum((keypoint_coords[pi, 0,:]-keypoint_coords[pi,3,:])**2))
                    output_dict[pi]["bbox"]=(center-width//2).astype(int).tolist()[::-1]+[int(width),int(1.5*width+(min(1-dist_1/dist_2,dist_2/dist_1))*0.5)]
                
                else:
                    ##need to change this 0
                    #width=width
                    output_dict[pi]["bbox"]=[]
                #person_bbox

                width=np.sqrt(np.sum((np.mean(keypoint_coords[pi,5:7,:],axis=0) -keypoint_coords[pi,0,:])**2))
                output_dict[pi]["bbox_yolo"]=(np.array(cv2.boundingRect(keypoint_coords[pi,:,:].astype(int)))+np.array([-width,-width//2,2*width,width])).astype(int)
                output_dict[pi]["bbox_yolo"]=output_dict[pi]["bbox_yolo"][0:2][::-1].tolist()+output_dict[pi]["bbox_yolo"][2:][::-1].tolist()

            output_dict[pi]["landmarks"]={}
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                output_dict[pi]["landmarks"][posenet.PART_NAMES[ki]]=c.astype(int).tolist()
        
        return output_dict
    
    def print_scores(self, image_name, pose_scores, keypoint_scores, keypoint_coords):
        print()
        print("Results for image: %s" % image_name)
        for pi in range(len(pose_scores)):
            if pose_scores[pi] == 0.:
                break
            print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
