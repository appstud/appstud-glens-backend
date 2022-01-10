import sys
import cv2
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../')))
from face_detection_alignment.faceAlignmentV2 import *
import tqdm
import argparse


def main():

    with open(os.path.join(path_to_dataset,'data.txt')) as fp:
        for line in tqdm.tqdm(fp):
            line=line.rstrip('\n').split(" ") 
            fullPath=os.path.join(path_to_dataset,line[0])
            img=cv2.imread(fullPath)

            _,landmarks,_=searchForFaceInTheWholeImage(np.copy(img),face_detection="dlib")
            try:
                if(landmarks[0] is not None):
                    _,alignedFace,_=performFaceAlignment(img,landmarks[0])
                    if(int(line[1])!=0):
                        cv2.imwrite(os.path.join(path_to_output,'glasses',line[0]),alignedFace)
                    else:

                        cv2.imwrite(os.path.join(path_to_output,'noGlasses',line[0]),alignedFace)
            except Exception as e:
                pass
    print("done")






if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description='Python script for preprocessing the dataset for glasses detection')
    parser.add_argument('--path_to_dataset',help='The path to the dataset', default='./MeGlass_120x120')
    parser.add_argument('--path_to_output',help='The path to the preprocessed dataset', default='./glassesDataset')
    args = parser.parse_args()

    path_to_dataset=args.path_to_dataset
    path_to_output=args.path_to_output
    if(not os.path.exists(path_to_output)):
        os.mkdir(path_to_output)
        os.mkdir(os.path.join(path_to_output,"glasses"))
        os.mkdir(os.path.join(path_to_output,"noGlasses"))
        print("creating output directory")
    
    main()


