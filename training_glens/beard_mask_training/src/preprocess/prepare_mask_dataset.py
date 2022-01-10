import os
import cv2
import pdb
import os
import tqdm
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../src/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../../face_detection_alignment')))
from face_detection_alignment.faceAlignmentV2 import *



"""
def detectFaceAndAlignIt(image):
    ###img,landmarks,faceROI=searchForFaceInTheWholeImage(np.copy(imageResized))
    img,landmarks,faceROI,faces=searchForFaceInTheWholeImage(np.copy(image))
    alignedFaces=[]
    
    for i,landmark in enumerate(landmarks):
        alignedGlobal,alignedFace,_=performFaceAlignment(np.copy(image),landmark)
        alignedFaces.append([alignedGlobal,alignedFace])

    return img,alignedFaces,faceROI,faces
"""


def walkThroughDirectory(path,destination_directory="./aligned_dataset"):
    i=0
    if(not os.path.isdir(destination_directory)):
        os.mkdir(destination_directory)
        os.mkdir(os.path.join(destination_directory,"mask"))

    fname = []
    for root,d_names,f_names in os.walk(path):
        if(len(f_names)!=0):    
            for f in tqdm.tqdm(f_names):
                img=cv2.imread(os.path.join(root, f))
                if( img is not None and img.shape[0]*img.shape[1]>45*45):
                    #_,aligned,_,_=detectFaceAndAlignIt(img,"mtcnn")

                    _,_,faceROIs=searchForFaceInTheWholeImage(img,face_detection="mtcnn")

                       
                    for roi in faceROIs:
                        try:
                            if(roi[2]*roi[3]>46*46):
                                y1=max(0,int(roi[1]-0.2*roi[3]))
                                x1=max(0,int(roi[0]-0.2*roi[2]))

                                x2=min(img.shape[1],int(roi[0]+1.2*roi[2]))
                                y2=min(img.shape[0],int(roi[1]+1.2*roi[3]))

                                #cv2.imwrite(os.path.join(destination_directory, str(i)+'_imask.jpg'),img[roi[1]:roi[3]+roi[1],roi[0]:roi[0]+roi[2],:])
                                #cv2.imwrite(os.path.join(destination_directory, str(i)+'_imask.jpg'),img[roi[1]:roi[3]+roi[1],roi[0]:roi[0]+roi[2],:])
                                cv2.imwrite(os.path.join(os.path.join(destination_directory,"mask"), str(i)+'_imask.jpg'),img[y1:y2,x1:x2,:])
                                i+=1
                                print(i)
                                #cv2.imshow("aligned",img_to_save)
                                #cv2.waitKey(1)
                        except Exception as e:
                            print(e)
                    

"""
def prepareImages(path):
    for folder in os.listdir(path):
        print(folder)
        print(len(os.listdir(os.path.join(os.getcwd(),path,folder) )))
        
        for imgPath in os.listdir(os.path.join(os.getcwd(),path,folder)):
            
            imgPath=os.path.join(path,folder,imgPath)
            try:
                image=cv2.imread(imgPath)
            except:
                continue
            if(image is None or type(image)==None):
                os.remove(imgPath)
                continue
            _,_,faceROI=searchForFaceInTheWholeImage(image)
            try: 
                if(len(faceROI)>0):
                    N=100

                    faceROI=faceROI[0]
                    faceROI= [max(faceROI[0]-N,0), max(0,faceROI[1]-N), min(faceROI[2]+faceROI[0]+N,image.shape[0]),min(image.shape[1],N+ faceROI[1]+faceROI[3])]

                    alignedImage=image[faceROI[0]:faceROI[2],faceROI[1]:faceROI[3],:]
                    cv2.imwrite(imgPath,alignedImage)
                    cv2.imshow("alignedImage",alignedImage)
                    cv2.waitKey(1)
                else:
                    os.remove(imgPath)
            except Exception as e:
                os.remove(imgPath)
"""

if(__name__=="__main__"):
    #prepareImages("./celeba/train/output/train")
    #walkThroughDirectory("./maskImagesDataset","./mask_aligned2")
    walkThroughDirectory("../maskImagesDataset","../mask_aligned_with_margin")
    #walkThroughDirectory("./beardDataset/nomask","./nomaskwithmargin")

