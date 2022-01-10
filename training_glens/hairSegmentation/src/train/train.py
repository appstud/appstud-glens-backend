
import argparse
import h5py
import itertools
import tensorflow as tf
import pdb
import os
import sys
import cv2
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from helperFunctions import getWeights
import matplotlib.pyplot as plt
import scipy.io as sio
#from enhanceQuality import dynamicRangeCompression
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../')))

from models.model import unetLightMobile
tf.enable_eager_execution=True 
BATCH_SIZE=1
###must change
ARRAY_SIZE=[128,128,3]

def get_data_generator(mode="train"):
    def gen_data():
        gen=generator(mode)
        while(True): 
            image,gt=next(gen)
            """cv2.imshow("image",np.hstack((image,gt)))
            cv2.waitKey(1000)
            """
            yield image,gt

    return gen_data


def upScaleImage(image):
    padding=np.zeros([3,image.shape[1],3],dtype=np.uint8)
    padding[:,:,0]=255
    
    image=np.vstack((padding,image,padding))
   
    padding=np.zeros([image.shape[0],3,3],dtype=np.uint8)
    padding[:,:,0]=255
    

    image=np.hstack((padding,image,padding))

##########must remove this after
    image=cv2.resize(image,(128,128))
    """cv2.imshow("out",image)
    cv2.waitKey(1000)
    """

    return image



def upScaleImageVold(image):
    #img=cv2.resize(image,(256,256))
    img=cv2.resize(image,(128,128))
    """out = np.zeros(img.shape)
    idx = img.argmax(axis=-1)
    out[np.arange(img.shape[0])[:,None],np.arange(img.shape[1]),idx] = 255.0
    """
    #cv2.imshow("out",img)
    #cv2.waitKey(1000)
    return img





def augmentImages(images,flip=True):
    ###images: list of images to augment###

    augmentedImages=images
    if(flip):
        flip=np.random.choice([False,True], 1, replace=False)

    if(flip):
        for i,image in enumerate(images):
            augmentedImages[i]=cv2.flip(image,1)
    return augmentedImages 

#################
####imageGenerator=ImageDataGenerator(zoom_range=0.3,rotation_range=10,brightness_range=[0.6,1.2] , fill_mode='constant',cval=255, width_shift_range=0.1, height_shift_range=0.2, horizontal_flip=True)
####maskGenerator=ImageDataGenerator(zoom_range=0.3,rotation_range=10,brightness_range=[0.6,1.2]  ,fill_mode='constant',cval=255, width_shift_range=0.1, height_shift_range=0.2, horizontal_flip=True)
#############

imageGenerator=ImageDataGenerator(brightness_range=[0.8,1.2] , fill_mode='constant',cval=255,  horizontal_flip=True)
maskGenerator=ImageDataGenerator(brightness_range=[0.8,1.2]  ,fill_mode='constant',cval=255,  horizontal_flip=True)

"""
imageGenerator=ImageDataGenerator(brightness_range=[0.9,1.1] , fill_mode='constant',cval=255, horizontal_flip=True)
maskGenerator=ImageDataGenerator(brightness_range=[0.9,1.1]  ,fill_mode='constant',cval=255,  horizontal_flip=True)
"""

def generatorOldV(mode="train"):

    f= h5py.File(mode+".hdf5","r") 
    numberOfSamples=len(f)
    f.close()
    seed=0
    for i in itertools.cycle(range(0,numberOfSamples)):
        with h5py.File(mode+".hdf5","r") as f:
            seed=np.random.choice(range(3500),1)
            #yield(f[str(i)][0]/255.0,f[str(i)][1])
            """
            cv2.imshow("image",np.hstack((upScaleImage(f[str(i)][0]), upScaleImage(f[str(i)][1]))))
            cv2.waitKey(1000)
            """
            augmentedImages=[imageGenerator.random_transform(upScaleImage(f[str(i)][0]), seed=seed), maskGenerator.random_transform(upScaleImage(f[str(i)][1]),seed=seed) ]
            #augmentedImages=augmentImages([upScaleImage(f[str(i)][0]),upScaleImage(f[str(i)][1])])
            """ 
            cv2.imshow("image",np.hstack((augmentedImages[0],augmentedImages[1])))
            cv2.waitKey(1000)
            """
            #augmentedImages[0]=dynamicRangeCompression(augmentedImages[0])
            yield(augmentedImages[0]/255.0,augmentedImages[1]/255.0)

def cleanLabels(img):
    out = np.zeros(img.shape)
    idx = img.argmax(axis=-1)
    out[np.arange(img.shape[0])[:,None],np.arange(img.shape[1]),idx] = 255.0
    return out

def generator(mode="train"):

    f= h5py.File(os.path.join(path_to_dataset,mode+".hdf5"),"r") 
    numberOfSamples=len(f)
    f.close()
    seed=0
    for i in itertools.cycle(range(0,numberOfSamples)):
        with h5py.File(os.path.join(path_to_dataset,mode+".hdf5"),"r") as f:
            seed=np.random.choice(range(2500),1)
            #yield(f[str(i)][0]/255.0,f[str(i)][1])
            """
            cv2.imshow("image",np.hstack((upScaleImage(f[str(i)][0]), upScaleImage(f[str(i)][1]))))
            cv2.waitKey(1000)
            """
            augmentedImages=[imageGenerator.random_transform(upScaleImage(f[str(i)][0]), seed=seed), cleanLabels(maskGenerator.random_transform((upScaleImage(f[str(i)][1])),seed=seed))]
            #augmentedImages=augmentImages([upScaleImage(f[str(i)][0]),upScaleImage(f[str(i)][1])])
            
            #augmentedImages=[upScaleImage(f[str(i)][0]), cleanLabels(upScaleImage(f[str(i)][1]))]
             
            #cv2.imshow("image",np.hstack((augmentedImages[0].astype(np.uint8),augmentedImages[1].astype(np.uint8))))
            #cv2.waitKey(1000)
            
            
            #####labels=np.concatenate((getWeights(augmentedImages[1]/255.0)[...,np.newaxis], augmentedImages[1]/255.0), axis=-1)
            #must change
            labels=np.concatenate((getWeights(cv2.resize(augmentedImages[1],(128,128))/255.0)[...,np.newaxis], augmentedImages[1]/255.0), axis=-1)
  
            #labels= augmentedImages[1]/255.0 
            yield(augmentedImages[0]/255.0,labels)
            #yield(dynamicRangeCompression(augmentedImages[0])/255.0,labels)




def input_fn(mode="train"):
    dataset = tf.data.Dataset.from_generator(get_data_generator(mode=mode),(tf.float32, tf.int8), (tf.TensorShape(ARRAY_SIZE),tf.TensorShape(ARRAY_SIZE[0:2]+[4])))

    dataset = dataset.batch(batch_size=BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=2*BATCH_SIZE)

    return dataset



  
class MyCustomCallback(tf.keras.callbacks.Callback):
    def __init__(self,path_to_dataset,path_to_save_data):
        super().__init__()
        self.path_to_dataset=path_to_dataset
        self.path_to_save_data=path_to_save_data
        ##must change after
        self.img=cv2.resize(cv2.imread(os.path.join(path_to_dataset,"lfw_funneled/Ahmed_Ahmed/Ahmed_Ahmed_0001.jpg")),(128,128)).astype(np.float32)
        self.gen=get_data_generator(mode="validation")()
    #img=dynamicRangeCompression(img).astype(np.uint8)
    try:
        x=sio.loadmat(os.path.join(self.path_to_save_data,"data.mat"))
        logs={"my_metric":list(x['my_metric'][0]),"val_my_metric":list(x['val_my_metric'][0]),"loss":list(x['loss'][0]),"val_loss":list(x['val_loss'][0])}
      
    except Exception as e:
        print(e)
        logs={"my_metric":[],"val_my_metric":[],"loss":[],"val_loss":[]}
    
    print(logs)
    plt.ion()
    def plotTrainingData(self):
        plt.close('all')
        plt.figure(1)
        plt.plot(MyCustomCallback.logs['loss'])
        plt.plot(MyCustomCallback.logs['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train loss', 'validation loss'], loc='upper left')
        plt.savefig(os.path.join(self.path_to_save_data,'loss_data.png'))
     
        plt.figure(2)
        plt.plot(MyCustomCallback.logs['my_metric'])
        plt.plot(MyCustomCallback.logs['val_my_metric'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train accuracy', 'validation accuracy'], loc='upper left')
        plt.savefig(os.path.join(path_to_save_data,'acc_data.png'))
        plt.draw()
        plt.pause(0.001)
        sio.savemat(os.path.join(path_to_save_data,"data.mat"),self.logs)

    def addLogOfEpoch(self,logs):
        MyCustomCallback.logs["my_metric"].append(logs["my_metric"])
        MyCustomCallback.logs["val_my_metric"].append(logs["val_my_metric"])
        MyCustomCallback.logs["loss"].append(logs["loss"])
        MyCustomCallback.logs["val_loss"].append(logs["val_loss"])
    

    def testModelOnImage(self):
        for i in range(10):
            img,_=next(self.gen)
            pred=self.model.predict(np.array([img]))[0]
            cv2.imwrite(os.path.join(self.path_to_save_data,'test_'+str(i)+'.jpg'), np.hstack(((255*img).astype(np.uint8),(255*pred).astype(np.uint8))))
        #pred=tf.nn.softmax(self.model.predict(np.array([self.img/255.0]))[0])
        pred=self.model.predict(np.array([self.img/255.0]))[0]
        cv2.imwrite(os.path.join(self.path_to_save_data,'test.jpg'), np.hstack((self.img.astype(np.uint8),(255*pred).astype(np.uint8))))
        """cv2.imshow("test",np.hstack((MyCustomCallback.img.astype(np.uint8),(255*pred).astype(np.uint8))))
        cv2.waitKey(1)"""


    def on_epoch_end(self, epoch, logs=None):
        try:
            for k in self.logs.keys():
                self.logs[k]=self.logs[k][0:epoch-1]
        except Exception as e:
            print(e)
            pass

        self.addLogOfEpoch(logs)
        self.plotTrainingData()  
        self.testModelOnImage()


def prepare_keras_callback():
    callbacks = []


    callback_tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(path_to_save_data,'logs'),
                histogram_freq=0,
                write_graph=True,
                write_grads=True,
                write_images=True)
    callbacks.append(callback_tensorboard)
    
    ###callback_mcp = tf.keras.callbacks.ModelCheckpoint( filepath='weights.checkpoint', monitor='val_loss',verbose=1,mode='auto', save_best_only=True, period=1)
    callback_mcp = tf.keras.callbacks.ModelCheckpoint( filepath=os.path.join(path_to_save_data,'weights{epoch:08d}.checkpoint'), monitor='val_loss',verbose=1,  period=30)

    """
    callback_mcp = tf.keras.callbacks.ModelCheckpoint(
                monitor='val_loss',
                verbose=1,
                mode='auto',
                save_best_only=True,
                period=1)
    """

    callbacks.append(callback_mcp)

    callbacks.append(MyCustomCallback(path_to_dataset,path_to_save_data))
    return callbacks



if(__name__=="__main__"):
    model = unetLightMobile()
    
    parser = argparse.ArgumentParser(description='Python script for training a tensorflow model for hair segmentation')
    parser.add_argument('--path_to_dataset',help='The path to the dataset a folder containing the hdf5 files', default='../dataset')
    parser.add_argument('--name_of_result_folder',help='The path to the data such as trained models, graphs, data...', default='results')
    parser.add_argument('--epochs',help='Number of epochs for training', type=int, default=180)
    args = parser.parse_args()

    name_of_result_folder=args.name_of_result_folder
    path_to_dataset=args.path_to_dataset
    epochs=args.epochs
    path_to_save_data=os.path.join(args.path_to_dataset,args.name_of_result_folder)

    #from keras.utils import plot_model
    #plot_model(model, to_file='model.png')
    
    train_dataset = input_fn("train")
    valid_dataset = input_fn("validation")

    callbacks = prepare_keras_callback() 
    
         
    
    history = model.fit(train_dataset,
                        validation_data=valid_dataset,
                        callbacks=callbacks,
                        verbose=1,
                        steps_per_epoch=2500,
                        epochs=epochs,
                        initial_epoch=0,
                        validation_steps=500)
    
