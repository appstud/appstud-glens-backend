import cv2
import numpy as np
import pdb

def getWeightsOfBoundaries(gt):
    backGroundAndFace=gt[:,:,0]+gt[:,:,1]
    laplacian = np.where(np.abs(cv2.Laplacian(backGroundAndFace,cv2.CV_64F))>0.3,255,0).astype(np.uint8)
    kernel = np.ones((10,10),np.uint8)
    mask = cv2.dilate(laplacian,kernel,iterations = 1)/255.0
    
    return mask

def getWeights(gt,alpha=0.1):
    ###maskOfColor=np.zeros([256,256])
    ###mustChange

    maskOfColor=np.zeros([128,128])
    maskOfBoundary=getWeightsOfBoundaries(np.copy(gt))
    ##maskOfBoundary=np.zeros([256,256])
    
    indexOfHair=np.where(gt[:,:,2]==1)

    indexOfBeard=indexOfHair[0][np.logical_and(indexOfHair[0]>128, np.abs(indexOfHair[1]-128)<45)  ],indexOfHair[1][np.logical_and(indexOfHair[0]>128, np.abs(indexOfHair[1]-128)<45)   ]


    indexOfBackground=np.where(gt[:,:,0]==1)
    indexOfFace=np.where(gt[:,:,1]==1)

    numberOfHairPixels=len(indexOfHair[0])
    numberOfFacePixels=len(indexOfFace[0])
    numberOfBackgroundPixels=len(indexOfBackground[0])

    #noise=np.where(np.logical_and(np.logical_and(gt[:,:,0]!=1,gt[:,:,1]!=1),gt[:,:,2]!=1))


    #maskOfColor[noise[0],noise[1]]=255.0
    
    totalNum=np.float32(256*256)
    #print(gt.shape,numberOfHairPixels, numberOfFacePixels, numberOfBackgroundPixels,numberOfHairPixels+numberOfFacePixels+numberOfBackgroundPixels)
    try:
        if(numberOfHairPixels!=0):
            #maskOfColor[indexOfHair[0],indexOfHair[1]]=totalNum/numberOfHairPixels
            maskOfColor[indexOfHair[0],indexOfHair[1]]=1*0.33
            #maskOfColor[indexOfBeard[0],indexOfBeard[1]]+=0.05
        else:
            maskOfColor[indexOfHair[0],indexOfHair[1]]=0.33
    except Exception as e:
        print("Errorrrr: ",e)
        maskOfColor[indexOfHair[0],indexOfHair[1]]=1

    #maskOfColor[indexOfBackground[0],indexOfBackground[1]]=totalNum/numberOfBackgroundPixels
    maskOfColor[indexOfBackground[0],indexOfBackground[1]]=1*0.33

    #maskOfColor[indexOfFace[0],indexOfFace[1]]=totalNum/numberOfFacePixels
    maskOfColor[indexOfFace[0],indexOfFace[1]]=1*0.33

    #maskOfColor=maskOfColor/np.sum(maskOfColor,axis=-1)
    mask=(maskOfColor.astype(np.float64)+alpha*maskOfBoundary.astype(np.float64))
    
    mask=4*mask
    #mask=np.ones([256,256])+alpha*maskOfBoundary
    #cv2.imshow("mask",40*mask.astype(np.uint8))
    #cv2.waitKey(1000)
    
    return mask.astype(np.float64)


def main():
    with open("train.txt",'r') as f:
        for l in f:
            line=l.strip().split(" ")
            num=line[1]
            while(len(num)<4):
                num='0'+num
            pathImage=  "/home/appstud/Desktop/hairSegmentation/lfw-funneled/lfw_funneled/"+line[0]+"/"+line[0]+"_"+num+".jpg"
            pathGt="./parts_lfw_funneled_gt_images/"+line[0]+"_"+num+".ppm"
            img=cv2.imread(pathImage)
            gt=cv2.imread(pathGt)
            mask=getWeights(gt/255.0)
            cv2.imshow("hair_boundaries",30*mask.astype(np.uint8))
            cv2.waitKey(1000)

    

if(__name__=="__main__"):
    main()
