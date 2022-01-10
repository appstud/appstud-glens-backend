import numpy as np
import cv2
import tensorflow as tf





def cropImage(image,outputSize=(96,96)):
    
    imageShape=image.shape
    if(imageShape[0]-outputSize[1]>=0 and imageShape[1]-outputSize[0]>=0):
        topCornerX=np.random.choice(range(imageShape[0]-outputSize[1]),1)[0]
        topCornerY=np.random.choice(range(imageShape[1]-outputSize[0]),1)[0]
        return image[topCornerX:topCornerX+outputSize[1],topCornerY:topCornerY+outputSize[0],:]
    else:
        tf.print("ERROR: output size is bigger than input size")
        tf.print(image.shape)
        return cv2.resize(image,outputSize)

