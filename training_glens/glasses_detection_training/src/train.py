import os 
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#from tensorflow import keras
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer
import scipy.io as sio
from tensorflow.keras import optimizers,losses,activations
import itertools
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from enhanceQuality import dynamicRangeCompression
from model import createModel       


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
    try:
        x=sio.loadmat("data.mat")
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
        plt.savefig(os.path.join(path_to_dataset,'loss_data.png'))

        plt.figure(2)
        plt.plot(MyCustomCallback.logs['acc'])
        plt.plot(MyCustomCallback.logs['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train accuracy', 'validation accuracy'], loc='upper left')
        plt.savefig(os.path.join(path_to_dataset,'acc_data.png'))
        #plt.draw()
        #plt.pause(0.001)
        sio.savemat(os.path.join(path_to_dataset,"data.mat"),self.logs)

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


    


def prepare_keras_callback():
    callbacks = []


    callback_tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(path_to_dataset,'logs'),
                histogram_freq=0,
                write_graph=True,
                write_grads=True,
                write_images=True)
    callbacks.append(callback_tensorboard)
    callback_mcp = tf.keras.callbacks.ModelCheckpoint( filepath=os.path.join(path_to_dataset,name_of_result_folder,'weights.checkpoint'), monitor='val_loss',verbose=1,mode='auto', save_best_only=True,save_weights_only=True, period=1)
    callbacks.append(callback_mcp)
    callbacks.append(MyCustomCallback())
    #callbacks.append(LearningRateReducerCb())
    return callbacks



def preprocess(x):

    #return dynamicRangeCompression(cv2.cvtColor(x,cv2.COLOR_RGB2BGR)).astype(np.float32)
    return cv2.cvtColor(x,cv2.COLOR_RGB2BGR)


if(__name__=="__main__"):
    
    
    parser = argparse.ArgumentParser(description='Python script for training a tensorflow model for glasses detection')
    parser.add_argument('--path_to_dataset',help='The path to the dataset a folder containing 2 subfolders named train and val', default='./glassesDataset/output')
    parser.add_argument('--name_of_result_folder',help='The path to the data such as trained models, graphs, data...', default='results')
    parser.add_argument('--batch_size',help='The batch size to use for training', type=int, default=16)
    parser.add_argument('--epochs',help='Number of epochs for training', type=int, default=100)
    args = parser.parse_args()
    
    name_of_result_folder=args.name_of_result_folder
    path_to_dataset=args.path_to_dataset
    epochs=args.epochs
    batch_size=args.batch_size
    path_to_train_dataset=os.path.join(args.path_to_dataset,'train')
    path_to_validation_dataset=os.path.join(args.path_to_dataset,'val')
    batch_size=args.batch_size
    path_to_save_data=os.path.join(args.path_to_dataset,args.name_of_result_folder)

    if(not os.path.isdir(path_to_save_data)):
        os.mkdir(path_to_save_data)

    # Create model
    model=createModel()

    # Compile it: add loss, metrics and optimizer
    model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer=tf.keras.optimizers.Adam(lr=1*10**-3))
    
    # Set callbacks
    myCallbacks=prepare_keras_callback()
   
    # Data augmentation 
    train_gen=ImageDataGenerator( horizontal_flip=True,  rescale=1./255.0,preprocessing_function=preprocess)
    valid_gen=ImageDataGenerator( horizontal_flip=True,  rescale=1./255.0,preprocessing_function=preprocess)
    
    # Create data pipeline generators from directories 
    train_Gen=train_gen.flow_from_directory(class_mode='binary',batch_size=batch_size,directory=os.path.join("./glassesDataset/output/","train"),target_size=(96,96))
    valid_Gen=valid_gen.flow_from_directory(class_mode='binary',batch_size=batch_size,directory=os.path.join("./glassesDataset/output","val"),target_size=(96,96))


    labels = (train_Gen.class_indices)
    print(labels)
    labels = dict((v,k) for k,v in labels.items())
    
    history=model.fit(train_Gen,steps_per_epoch=int(39490/16.0),callbacks=myCallbacks,validation_data=valid_Gen,shuffle=True,epochs=epochs)
 
