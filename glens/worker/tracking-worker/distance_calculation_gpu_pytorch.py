import numpy as np
import pdb
import time
from scipy.spatial import distance
import torch

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

def calculateDistancePytorch(x,y):
    """calculate distance between two set of vectors x and y
    it returns a matrix D where D[i,j] is the distance between x[i,:] and y[j,:]
    """
    #x=x.astype(np.float32)
    #y=y.astype(np.float32)
    
    r1 = torch.sum(x*x,1,keepdim=False)
    r2 = torch.sum(y*y,1,keepdim=False)

    r1 = torch.reshape(r1, [-1, 1])
    r2 = torch.reshape(r2, [1, -1])
    D = r1 - 2*torch.matmul(x, torch.transpose(y,0,1)) + torch.repeat_interleave(r2,repeats=r1.size()[0],dim=0)
    D=torch.sqrt(D)
    return D

def calculateDistancePytorchList(y,args):
    """calculate distance between two set of vectors x and y
    it returns a matrix D where D[i,j] is the distance between x[i,:] and y[j,:]
    """

    #x= torch.empty(100000, 512,dtype=float,device='cuda')
    #x=x.astype(np.float32)
    #y=y.astype(np.float32)
    start=time.time()
    x=torch.stack(args)
    """for i in range(xx.size(0)):
        xx[ i,:] = args[i]
    """

    #x=torch.cat(args,0)
    finish=time.time()
    print("stacking time:",finish-start)
    r1 = torch.sum(x*x,1,keepdim=False)
    r2 = torch.sum(y*y,1,keepdim=False)

    r1 = torch.reshape(r1, [-1, 1])
    r2 = torch.reshape(r2, [1, -1])
    D = r1 - 2*torch.matmul(x, torch.transpose(y,0,1)) + torch.repeat_interleave(r2,repeats=r1.size()[0],dim=0)
    D=torch.sqrt(D)
    return D




if(__name__=="__main__"):
    #x_tf= tf.random.uniform([30000,512])
    #y_tf= tf.random.uniform([10,512])
    x=np.random.rand(100,3)
    y=np.random.rand(10,3)
    x_torch_list=[]
    for i in range(x.shape[0]):
        x_torch_list.append(torch.from_numpy(x[i,:]).cuda())
    
    x_torch=torch.from_numpy(x)
    y_torch=torch.from_numpy(y).cuda()

    while(True):
       
        start=time.time()
        #distanceMatTF=calculateDistanceTF(x_tf,y_tf)
        #distanceMatPytorch=calculateDistancePytorch(x_torch,y_torch)

        calculateDistancePytorchList(y_torch,x_torch_list)
        finish=time.time()

        #print("tf:",str(finish-start))
        print("pytorch:",str(finish-start))

        start=time.time()
        distanceMatSci=distance.cdist(x,y)
        finish=time.time()

        print("cdist:",str(finish-start))
