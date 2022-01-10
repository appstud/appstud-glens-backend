import cv2
import os 
import pdb

def fixExtension(path,ext="jpeg"):
    for folder in os.listdir(path):
        for imgPath in os.listdir(os.path.join(os.getcwd(),path,folder)):
            os.rename(os.path.join(path,folder,imgPath),os.path.join(path,folder,imgPath+".jpg"))


if(__name__=="__main__"):
    fixExtension("./data/train")
    #fixExtension("./data/train")
