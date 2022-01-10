import numpy as np
import pdb
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer
import scipy.io as sio
from tensorflow.keras import optimizers,losses,activations
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import argparse
import random
import sys
import mlflow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../src/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../../face_detection_alignment')))

from helperFunctions import cropImage
from models.model import *

class LearningRateReducerCb(tf.keras.callbacks.Callback):

  def on_epoch_end(self, epoch, logs={}):
    ##old_lr = self.model.optimizer.lr.read_value()
    k=0.08
    new_lr = 5*(10**-4) * np.exp(-k*(epoch))
    print("\nEpoch: {}. Reducing Learning Rate to {}".format(epoch, new_lr))
    print("\nEpoch: {}. Reducing Learning Rate to {}".format(epoch, new_lr))
    print("Epoch: {}. Reducing Learning Rate to {}".format(epoch, new_lr))
    self.model.optimizer.lr.assign(new_lr)



class MyCustomCallback(tf.keras.callbacks.Callback):
    
    def __init__(self,path_to_save_data):
        super(MyCustomCallback,self).__init__()
        self.path_to_save_data=path_to_save_data



    try:
        x=sio.loadmat(os.path.join(self.path_to_save_data,"data.mat"))
        logs={"acc":list(x['acc'][0]),"val_acc":list(x['val_acc'][0]),"loss":list(x['loss'][0]),"val_loss":list(x['val_loss'][0])}

    except Exception as e:
        print(e)
        logs={"acc":[],"val_acc":[],"loss":[],"val_loss":[]}
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
        plt.plot(MyCustomCallback.logs['acc'])
        plt.plot(MyCustomCallback.logs['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train accuracy', 'validation accuracy'], loc='upper left')
        plt.savefig(os.path.join(self.path_to_save_data,'acc_data.png'))
        #plt.draw()
        #plt.pause(0.001)
        sio.savemat("data.mat",self.logs)

    def addLogOfEpoch(self,logs):
        MyCustomCallback.logs["acc"].append(logs["acc"])
        MyCustomCallback.logs["val_acc"].append(logs["val_acc"])
        MyCustomCallback.logs["loss"].append(logs["loss"])
        MyCustomCallback.logs["val_loss"].append(logs["val_loss"])



    def on_epoch_end(self, epoch, logs=None):
        try:
            for k in self.logs.keys():
                self.logs[k]=self.logs[k][0:epoch-1]
                self.addLogOfEpoch(logs)
            self.plotTrainingData()
        except Exception as e:
            print(e)


    


def prepare_keras_callback(path_to_save_data):
    callbacks = []


    callback_tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(path_to_save_data,'logs'),
                histogram_freq=0,
                write_graph=True,
                write_grads=True,
                write_images=True)
    callbacks.append(callback_tensorboard)
    
    callback_mcp = tf.keras.callbacks.ModelCheckpoint( filepath=os.path.join(path_to_save_data,'weights.checkpoint'),save_weights_only=True, monitor='val_loss',verbose=1,mode='auto', save_best_only=True, period=1)
    #callback_mcp = tf.keras.callbacks.ModelCheckpoint( filepath=os.path.join(path_to_save_data,'weights{epoch:08d}.checkpoint'), monitor='val_loss',verbose=1,  period=10)
    callbacks.append(callback_mcp)
    callbacks.append(MyCustomCallback(path_to_save_data))
    #callbacks.append(LearningRateReducerCb())
    return callbacks




def augmentationForResolution(x):
    choose=random.choice([0,1,2,0,1])
    if(choose==0):
        s=random.choice([2,3,4,5,5,4])
        y=cv2.resize(x,(x.shape[1]//s,x.shape[0]//s),interpolation=cv2.INTER_NEAREST)
        x=cv2.resize(y,(x.shape[1],x.shape[0]),interpolation=cv2.INTER_NEAREST)
    elif(choose==1):
        
        s=random.choice([5,7])
        x=cv2.GaussianBlur(x,(s,s),s/2.0)
    else:
        pass
    return x.astype(np.float32)

def preprocess(x):
    #return dynamicRangeCompression(cv2.cvtColor(x,cv2.COLOR_RGB2BGR)).astype(np.float32)
    x=augmentationForResolution(cropImage(cv2.resize(x,(106,106))))
    #cv2.imshow("img",x.astype(np.uint8))
    #cv2.waitKey(1000)
    return cv2.cvtColor(x,cv2.COLOR_RGB2BGR).astype(np.float32)


def printTrainingConfigurtion(args,path_to_save_data,file_to_save_config="training_config.txt"):

    with open(os.path.join(path_to_save_data,file_to_save_config),'w') as f:

        msg="This script is launched with the following training configuration: \n"
        print(msg)
        f.write(msg)
        for arg in vars(args):
            msg=arg+'==>'+ str(getattr(args, arg))+"\n"
            print(msg)
            f.write(msg)

if(__name__=="__main__"):
    
    mlflow.tensorflow.autolog()

    parser = argparse.ArgumentParser(description='Python script for training a tensorflow model for beard detection')
    parser.add_argument('--path_to_dataset',nargs='?',help='The path to the dataset a folder containing 2 subfolders named train and val', default='../beardDataset/output')
    parser.add_argument('--name_of_result_folder',nargs='?',help='The path to the data such as trained models, graphs, data...', default='results')
    parser.add_argument('--batch_size',nargs='*',help='The batch size to use for training', type=int, default=16)
    parser.add_argument('--epochs',nargs='*',help='Number of epochs for training', type=int, default=50)
    args = parser.parse_args()
    
    path_to_train_dataset=os.path.join(args.path_to_dataset,'train')
    path_to_validation_dataset=os.path.join(args.path_to_dataset,'val')
    batch_size=args.batch_size
    path_to_save_data=os.path.join(args.path_to_dataset,args.name_of_result_folder)
    
    if(not os.path.isdir(path_to_save_data)):
        os.mkdir(path_to_save_data)
        
    epochs=args.epochs

    printTrainingConfigurtion(args,path_to_save_data)
    
    model=createModel("cbam")
    
    myCallbacks=prepare_keras_callback(path_to_save_data)
    
    #train_gen=ImageDataGenerator(rotation_range=20,zoom_range=0.1,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,fill_mode="nearest",rescale=1./255.0,preprocessing_function=preprocess)
    train_gen=ImageDataGenerator(rotation_range=20,brightness_range=[0.8,1.1],zoom_range=0.2,horizontal_flip=True,fill_mode="nearest",rescale=1./255.0,preprocessing_function=preprocess)
    valid_gen=ImageDataGenerator(rotation_range=20,brightness_range=[0.8,1.1],zoom_range=0.2,horizontal_flip=True,fill_mode="nearest",rescale=1./255.0,preprocessing_function=preprocess)

    #valid_gen=ImageDataGenerator(rotation_range=20,zoom_range=0.1,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,fill_mode="nearest",rescale=1./255.0,preprocessing_function=preprocess)
    #train_gen=ImageDataGenerator( horizontal_flip=True,  rescale=1./255.0,preprocessing_function=preprocess)
    #valid_gen=ImageDataGenerator( horizontal_flip=True,  rescale=1./255.0,preprocessing_function=preprocess)

    train_Gen=train_gen.flow_from_directory(class_mode='categorical',batch_size=batch_size,directory=path_to_train_dataset,target_size=(96,96))
    valid_Gen=valid_gen.flow_from_directory(class_mode='categorical',batch_size=batch_size,directory=path_to_validation_dataset,target_size=(96,96))

    labels = (train_Gen.class_indices)
    print(labels)
    labels = dict((v,k) for k,v in labels.items())
    #history=model.fit(train_Gen,steps_per_epoch=int(train_Gen.samples/batch_size),callbacks=myCallbacks, validation_data=valid_Gen,shuffle=True,epochs=epochs,class_weight={1:1,0:3})
    history=model.fit(train_Gen,steps_per_epoch=int(train_Gen.samples/batch_size),callbacks=myCallbacks, validation_data=valid_Gen,shuffle=True,epochs=epochs)
 
