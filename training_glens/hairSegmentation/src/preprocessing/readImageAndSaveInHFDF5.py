import cv2
import numpy as np
import sys
import h5py
import os
import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../')))
import tqdm
from face_detection_alignment.faceAlignmentV2 import *
import pdb
import tensorflow as tf

def readImagesAndSave(textFile="../dataset/train.txt",hdf5File="../dataset/train.hdf5"):
    count=0
    with open(textFile,'r') as f, h5py.File(hdf5File, 'w') as fw:
        for l in tqdm.tqdm(f):
            line=l.strip().split(" ")
            num=line[1]
            while(len(num)<4):
                num='0'+num
            pathImage=  "../dataset/lfw_funneled/"+line[0]+"/"+line[0]+"_"+num+".jpg"
            pathGt="../dataset/parts_lfw_funneled_gt_images/"+line[0]+"_"+num+".ppm"
            img=cv2.imread(pathImage)
            _,landmarks,faceROI=searchForFaceInTheWholeImage(np.copy(img),face_detection="dlib")

            if(len(faceROI)>0):                
                
                faceROI=faceROI[0]
                gt=cv2.imread(pathGt)
                newImg=np.zeros([faceROI[2]*3,faceROI[3]*3,3],dtype=np.uint8)
                newImgGt=np.zeros([faceROI[2]*3,faceROI[3]*3,3],dtype=np.uint8)
                newImgGt[:,:,0]=255
                N=faceROI[2]
                
                newImg=cropFaceForPoseEstimation(img,[faceROI],ad=0.6,output_shape=(128,128))[0]
                newImgGt=cropFaceForPoseEstimation(gt,[faceROI],ad=0.6,output_shape=(128,128))[0]
                #newImg,_,trans=performFaceAlignment(img,landmarks[0])
                #newImgGt= cv2.warpAffine(gt,trans,newImg.shape[0:2],borderValue=(255,0,0))
                #newImgGt= cv2.warpAffine(gt,trans,newImg.shape[0:2],borderMode=cv2.BORDER_REFLECT)
                
                #img=img[max(0,faceROI[0]-N):min(faceROI[0]+faceROI[2]+N,img.shape[0]),max(faceROI[1]-N,0):min(faceROI[3]+faceROI[1]+N,img.shape[1]),:].astype(np.uint8)
                #gt=gt[max(0,faceROI[0]-N):min(faceROI[0]+faceROI[2]+N,gt.shape[0]),max(faceROI[1]-N,0):min(faceROI[3]+faceROI[1]+N,gt.shape[1]),:].astype(np.uint8)
                
                #dx,dy=np.floor((newImg.shape[0]-img.shape[0])/2),np.floor((newImg.shape[1]-img.shape[1])/2)
                #print(dx,dy)
                #newImg[int(dx):int(dx)+img.shape[0],int(dy):int(dy)+img.shape[1],:]=img
                #newImgGt[int(dx):int(dx)+img.shape[0],int(dy):int(dy)+img.shape[1],:]=gt


                fw.create_dataset(str(count), data=[newImg, newImgGt])
                count=count+1
                #cv2.imshow("image",np.vstack((img,gt)))
                #cv2.imshow("image",np.vstack((newImg,newImgGt)))
                #cv2.waitKey(1000)

            else:
                tf.print("failed")

if(__name__=="__main__"):
    readImagesAndSave()
    readImagesAndSave("../dataset/validation.txt","../dataset/validation.hdf5")
    readImagesAndSave("../dataset/test.txt","../dataset/test.hdf5")

