import cv2
import numpy as np
from utils.constants import *


def draw_keypoints(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_confidence:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
    out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
    return out_img


def get_adjacent_keypoints(img,keypoint_scores, keypoint_coords, min_confidence=0.1,draw=True):
    results = []
    img_copy=np.copy(img)
    for i,(left, right) in enumerate(CONNECTED_PART_INDICES):
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
        cv2.line(img_copy,tuple(results[-1][0]),tuple(results[-1][1]),COLOR_PARTS[i],thickness=3)
    return cv2.addWeighted(img,0.3,img_copy,0.7,0),results


def draw_skeleton(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    out_img = img
    adjacent_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_confidence)
        adjacent_keypoints.extend(new_keypoints)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img

def draw_skeleton_with_colors(img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    out_img = img
    adjacent_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_confidence)
        adjacent_keypoints.extend(new_keypoints)
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    return out_img



def draw_skel_and_kp(
        img, instance_scores, keypoint_scores, keypoint_coords,bbox,
        bbox_persons,min_pose_score=0.5, min_part_score=0.5):
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        out_img,new_keypoints = get_adjacent_keypoints(out_img,
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))
        
        ###draw_bbox
        if(len(bbox[ii])==4):
            out_img=cv2.rectangle(out_img,(bbox[ii][0],bbox[ii][1]),(bbox[ii][2]+bbox[ii][0],bbox[ii][3]+bbox[ii][1]),(255,0,0),1)
        
        if(len(bbox_persons[ii])==4):
            out_img=cv2.rectangle(out_img,(bbox_persons[ii][0],bbox_persons[ii][1]),(bbox_persons[ii][2]+bbox_persons[ii][0],bbox_persons[ii][3]+bbox_persons[ii][1]),(255,255,0),3)
    ###draw_keypoints
    #out_img = cv2.drawKeypoints(out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0), thickness=2)
    return out_img
