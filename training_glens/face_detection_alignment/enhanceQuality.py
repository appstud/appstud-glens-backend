import numpy as np
import cv2

def dynamicRangeCompression(img):
   
    k=3
    beta=0.9

    hsvImg=cv2.cvtColor(np.copy(img),cv2.COLOR_BGR2HSV)
    V=hsvImg[:,:,2].astype(np.float64)

    avgImg1=cv2.GaussianBlur(V,(71,71),100).astype(np.float64)
    ro=(((255-k)/255.0)*avgImg1+k).astype(np.float64)
    dynamicCompressed=2.0/(1.0+np.exp((-2*V)/ro))-1

    dynamicCompressed=(255*dynamicCompressed).astype(np.float64)

    avgImg=cv2.GaussianBlur(dynamicCompressed,(5,5),6).astype(np.float64)
    diff=(dynamicCompressed-avgImg)/255.0
    IEnhanced=np.abs(diff)**beta
    IEnhanced=np.sign(diff)*IEnhanced


    IEnhanced=avgImg/255.0+IEnhanced
    IEnhanced=IEnhanced/np.max(IEnhanced)
    IEnhanced=(IEnhanced*255.0).astype(np.float64)

    enhancedImg=(IEnhanced/(V+1))[:,:,np.newaxis]*img


    return enhancedImg.astype(np.uint8)
