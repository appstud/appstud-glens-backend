import cv2
from faceAlignment import * 
import os

 
def readImagesAngAlignThem(directory):
     
    print(directory) 
    if(not os.path.exists(directory+"_aligned")):
        os.mkdir(directory+"_aligned")

    for folder in os.listdir(directory):
        targetFolder=os.path.join(directory+"_aligned",folder)
        sourceFolder=os.path.join(directory,folder)
        if(not os.path.exists(targetFolder)):
            os.mkdir(targetFolder)
        for imgName in os.listdir(sourceFolder):
            #print(imgName)
            if(imgName.endswith(".jpg")):
                img=cv2.imread(os.path.join(sourceFolder,imgName))
                
                try:
                    imageWithBox,landmarks,faceROI=searchForFaceInTheWholeImage(np.copy(img))
                    if(len(faceROI)>0 ):
                        registeredFace,faceForAge,_=performFaceAlignment(img,landmarks[0])
                        cv2.imwrite(os.path.join(targetFolder,imgName),registeredFace)
                except Exception as e:

                    print(directory) 
                    print(e)





    
if(__name__=="__main__"):
    #readImagesAngAlignThem("./celeba/train/output/train")
    #readImagesAngAlignThem("./celeba/train/output/val")
    readImagesAngAlignThem("./celeba/train")
    
    """readImagesAngAlignThem("./celeba/train/black")
    readImagesAngAlignThem("./celeba/train/white")
    readImagesAngAlignThem("./celeba/train/hat")
    readImagesAngAlignThem("./celeba/train/red")
    readImagesAngAlignThem("./celeba/train/blonde")
    readImagesAngAlignThem("./celeba/train/bald")
    """

