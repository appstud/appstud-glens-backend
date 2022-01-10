import os
import cv2
import pdb
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../')))
import tqdm
from face_detection_alignment.faceAlignmentV2 import *

def prepareImages(path_to_dataset,path_to_output):
    if(not os.path.isdir(path_to_output)):
        os.mkdir(path_to_output)
    
    for folder in os.listdir(path_to_dataset):
        print(folder)
        print(len(os.listdir(os.path.join(os.getcwd(),path_to_dataset,folder) )))
        if(not os.path.isdir(os.path.join(path_to_output,folder))):
            os.mkdir(os.path.join(path_to_output,folder))
    
   
        for img_name in tqdm.tqdm(os.listdir(os.path.join(os.getcwd(),path_to_dataset,folder))):
            
            imgPath=os.path.join(path_to_dataset,folder,img_name)
            try:
                image=cv2.imread(imgPath)
            except:
                continue
            if(image is None or type(image)==None):
                os.remove(imgPath)
                continue
            _,_,faceROI=searchForFaceInTheWholeImage(image,face_detection="dlib")
            try: 
                if(len(faceROI)==1):
                    alignedImage=cropFaceForPoseEstimation(image,[faceROI[0]],ad=0.4,output_shape=(128,128))
                    cv2.imwrite(os.path.join(path_to_dataset,folder,img_name),alignedImage[0])
                    #cv2.imshow("alignedImage",alignedImage)
                    #cv2.waitKey(1)
                else:
                    print(" no face or more than one face detected...")
            except Exception as e:
                print(e)


if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description='Python script for preprocessing dataset for hair color classification')
    parser.add_argument('--path_to_dataset',help='The path to the dataset a folder containing 2 subfolders named train and val', default='../split')
    args = parser.parse_args()


    prepareImages(os.path.join(args.path_to_dataset,"train"),os.path.join(args.path_to_dataset,"train"))
    prepareImages(os.path.join(args.path_to_dataset,"val"),os.path.join(args.path_to_dataset,"val") )
    #fixExtension("./data/train")


