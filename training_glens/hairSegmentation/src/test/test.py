
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
#from model import unet, unetLightMobile
from model import unetLightMobile
#import tflite_runtime.interpreter as tflite
import cv2
#from train import generator
import numpy as np
import pdb
import dlib
import time
from enhanceQuality import dynamicRangeCompression
from faceAlignment import performFaceAlignment
from faceAlignment import *


def testOnVideoStremTLite():
    # Load TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path="converted_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']

    cap=cv2.VideoCapture(0)
    #cap=cv2.VideoCapture("video2.mp4")
    cap.set(cv2.CAP_PROP_FPS, 30)
 
    cv2.namedWindow("prediction",cv2.WINDOW_NORMAL)
    while(True):
        ret,image=cap.read()
        try:
            if(ret):
                imageWithBox,_,faceROI=searchForFaceInTheWholeImage(np.copy(image))
                if(len(faceROI)>0):

                    faceROI=faceROI[0] 
                    
                    #N=faceROI[2]
                    N=100
                    faceROI= [max(faceROI[0]-N,0), max(0,faceROI[1]-N), min(faceROI[2]+faceROI[0]+N,image.shape[0]),min(image.shape[1],N+ faceROI[1]+faceROI[3])]
                    #print("got the face:",faceROI)
                    #image=image[faceROI[0][0]-N:faceROI[0][0]+faceROI[0][2]+N,faceROI[0][1]-N:faceROI[0][3]+faceROI[0][1]+N,:]
                    imageCropped=image[faceROI[0]:faceROI[2],faceROI[1]:faceROI[3],:]
                    imageCropped=cv2.resize(imageCropped,(128,128))

                    start=time.time()
                    input_data = np.array(imageCropped/255.0, dtype=np.float32)
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    # The function `get_tensor()` returns a copy of the tensor data.
                    # Use `tensor()` in order to get a pointer to the tensor.
                    prediction = interpreter.get_tensor(output_details[0]['index'])
                    print(prediction)
                    ###prediction=model.predict(np.array([imageCropped/255.0]))[0]
                    cv2.imshow("prediction",np.hstack((imageCropped.astype(np.uint8),(255*prediction).astype(np.uint8))))
                    finish=time.time()
                    FPS=1/(finish-start)
                    cv2.putText(imageWithBox,"FPS:"+str(int(FPS)) , (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), lineType=cv2.LINE_AA)

                cv2.imshow("image",imageWithBox)
                cv2.waitKey(1)
        except Exception as e:
            print(e)
            cap.release()


    
def testOnVideoStream():
    cap=cv2.VideoCapture(0)
    #cap=cv2.VideoCapture("video2.mp4")
    cap.set(cv2.CAP_PROP_FPS, 30)
 
    ###model definition###
    #model=unet()
    model=unetLightMobile()

    ###load pretrained weights ###
    #model.load_weights("weights.000020-0.1483.checkpoint")
    cv2.namedWindow("prediction",cv2.WINDOW_NORMAL)
    #model.load_weights("weights00001500.checkpoint")
    model.load_weights("weights00000100.checkpoint")
    while(True):
        ret,img=cap.read()

        #image=dynamicRangeCompression(image)

        try:
            if(ret):
                imageWithBox,landmarks,faceROI=searchForFaceInTheWholeImage(np.copy(img))
                if(len(faceROI)>0 ):
                   
                    registeredFace,_=performFaceAlignment(img,landmarks[0])

                    faceROI=faceROI[0]
                    ##########faceROI=faceROI[0] 
                    
                    #N=faceROI[2]
                    ############N=100
                    ############faceROI= [max(faceROI[0]-N,0), max(0,faceROI[1]-N), min(faceROI[2]+faceROI[0]+N,image.shape[0]),min(image.shape[1],N+ faceROI[1]+faceROI[3])]
                    #print("got the face:",faceROI)
                    #image=image[faceROI[0][0]-N:faceROI[0][0]+faceROI[0][2]+N,faceROI[0][1]-N:faceROI[0][3]+faceROI[0][1]+N,:]
                    ###############imageCropped=image[faceROI[0]:faceROI[2],faceROI[1]:faceROI[3],:]
                    imageCropped=cv2.resize(registeredFace,(128,128))

                    #imageCropped=dynamicRangeCompression(imageCropped)
                    ##################""imageCropped=dynamicRangeCompression(cv2.resize(registeredFace,(128,128)))
                    start=time.time()
                    prediction=model.predict(np.array([imageCropped/255.0]))[0]
                    cv2.imshow("prediction",np.hstack((imageCropped.astype(np.uint8),(255*prediction).astype(np.uint8))))
                    finish=time.time()
                    FPS=1/(finish-start)
                    cv2.putText(imageWithBox,"FPS:"+str(int(FPS)) , (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), lineType=cv2.LINE_AA)

                cv2.imshow("image",imageWithBox)
                cv2.waitKey(1)
        except Exception as e:
            print(e)
            pdb.set_trace()
            cap.release()

def main():
    ###model definition###
    model=unet()

    ###load pretrained weights ### 
    ##model.load_weights("weights.000020-0.1483.checkpoint")
    model.load_weights("weights00000110.checkpoint")
    ###get test data###
    dataGen=generator("validation")
    
    while(True):
        image,gt=next(dataGen)
        ###predict using the model###
        prediction=model.predict(np.array([image])) 
        ###show the result with ground truth###
        #print(prediction[0])
        #print(prediction[0].shape,image.shape,gt.shape)
        ##cv2.imshow("images",np.hstack(((255*image).astype(np.uint8),(255*gt).astype(np.uint8),(255*prediction[0]).astype(np.uint8))))
        cv2.imshow("images",np.hstack(((255*image).astype(np.uint8),(255*prediction[0]).astype(np.uint8))))
        cv2.waitKey(1000)

if(__name__=="__main__"):
    #main()
    
    #testOnVideoStremTLite()
    testOnVideoStream()
