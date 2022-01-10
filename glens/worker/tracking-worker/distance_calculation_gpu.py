import tensorflow as tf
import numpy as np
import pdb
import time
from scipy.spatial import distance


def calculateDistanceTF(x,y):
    """calculate distance between two set of vectors x and y
    it returns a matrix D where D[i,j] is the distance between x[i,:] and y[j,:]
    """
    #x=x.astype(np.float32)
    #y=y.astype(np.float32)

    r1 = tf.reduce_sum(x*x, 1)
    r2 = tf.reduce_sum(y*y, 1)

    r1 = tf.reshape(r1, [-1, 1])
    r2 = tf.reshape(r2, [1, -1])
    

    D = r1 - 2*tf.matmul(x, y,transpose_b=True) + tf.tile(r2,[r1.shape[0],1])

    return tf.sqrt(D).numpy()


if(__name__=="__main__"):
    x_tf= tf.random.uniform([30000,512])
    y_tf= tf.random.uniform([10,512])
    x=np.random.rand(30000,512)
    y=np.random.rand(10,512)
    while(True):
       
        start=time.time()
        distanceMatTF=calculateDistanceTF(x_tf,y_tf)
        finish=time.time()

        tf.print("tf:",str(finish-start))

        start=time.time()
        distanceMatSci=distance.cdist(x,y)
        finish=time.time()

        tf.print("cdist:",str(finish-start))
