import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../src/')))

from models.model import unetLightMobile,focal_loss
import tensorflow as tf
import pdb
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.metrics import classification_report, confusion_matrix

class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
      
    def __init__(self,valid_Gen,num_of_valid_samples,class_labels):
        super(ConfusionMatrixCallback, self).__init__()
        self.valid_Gen=valid_Gen
        self.num_of_valid_samples=num_of_valid_samples
        self.class_labels=class_labels
    
    def on_epoch_end(self, epoch, logs={}):
        #Confution Matrix and Classification Report
        Y_pred = self.model.predict(self.valid_Gen, self.num_of_valid_samples // batch_size+1)
        y_pred = np.argmax(Y_pred, axis=1)
        print('Confusion Matrix')
        print(confusion_matrix(self.valid_Gen.classes, y_pred))
        print('Classification Report')
        

        print(classification_report(self.valid_Gen.classes, y_pred, target_names=self.class_labels))





class MyCustomCallback(tf.keras.callbacks.Callback):
    
    try:
        x=sio.loadmat("data.mat")
        logs={"acc":list(x['acc'][0]),"val_acc":list(x['val_acc'][0]),"loss":list(x['loss'][0]),"val_loss":list(x['val_loss'][0])}
      
    except Exception as e:
        print(e)
        logs={"acc":[],"val_acc":[],"loss":[],"val_loss":[]}
    
    #print(logs)
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
        plt.savefig(os.path.join(path_to_save_data,'loss_data.png'))
     
        plt.figure(2)
        plt.plot(MyCustomCallback.logs['acc'])
        plt.plot(MyCustomCallback.logs['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train accuracy', 'validation accuracy'], loc='upper left')
        plt.savefig(os.path.join(path_to_save_data,'acc_data.png'))
        #plt.draw()
        #plt.pause(0.001)
        sio.savemat(os.path.join(path_to_save_data,"data.mat"),self.logs)

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
            pass

         


def prepare_keras_callback(valid_Gen,nb_valid_samples,class_labels):
    callbacks = []


    callback_tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(path_to_save_data,'logs'),
                histogram_freq=0,
                write_graph=True,
                write_grads=True,
                write_images=True)
    callbacks.append(callback_tensorboard)
    
    ###callback_mcp = tf.keras.callbacks.ModelCheckpoint( filepath='weights.checkpoint', monitor='val_loss',verbose=1,mode='auto', save_best_only=True, period=1)
    #callback_mcp = tf.keras.callbacks.ModelCheckpoint( filepath='weights{epoch:08d}.checkpoint',verbose=1,  period=10)
    callback_mcp = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(path_to_save_data,'best_model.checkpoint'), monitor='val_accuracy',verbose=1, save_weights_only=True ,period=1, save_best_only=True)
    callbacks.append(callback_mcp)
    callbacks.append(MyCustomCallback())
    callbacks.append(ConfusionMatrixCallback(valid_Gen,nb_valid_samples,class_labels))

    return callbacks

def preprocess(x):
    #return dynamicCompression(cv2.cvtColor(x,cv2.COLOR_RGB2BGR)).astype(np.float32)
    return cv2.cvtColor(x,cv2.COLOR_RGB2BGR)

def get_class_weights(base_path,classes):
    class_weights={} 
    
    total_samples=0
    for i in classes.keys():
        class_weights[i]=len([f for f in os.listdir(os.path.join(base_path,classes[i])) if f.endswith('.jpg') and os.path.isfile(os.path.join(os.path.join(base_path,classes[i]), f))])
        total_samples+=class_weights[i]

    nb_classes=len(classes.keys())
    for i in class_weights.keys():
        class_weights[i]=(1.0 / class_weights[i])*(total_samples)/nb_classes
    return class_weights


if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description='Python script for training a tensorflow model for hair color classification')
    parser.add_argument('--path_to_dataset',help='The path to the dataset a folder containing 2 subfolders named train and val', default='../split')
    parser.add_argument('--name_of_result_folder',help='The path to the data such as trained models, graphs, data...', default='results')
    parser.add_argument('--batch_size',help='The batch size to use for training', type=int, default=16)
    parser.add_argument('--epochs',help='Number of epochs for training', type=int, default=100)
    args = parser.parse_args()

    name_of_result_folder=args.name_of_result_folder
    path_to_dataset=args.path_to_dataset
    epochs=args.epochs
    batch_size=args.batch_size
    train_dir=os.path.join(args.path_to_dataset,'train')
    valid_dir=os.path.join(args.path_to_dataset,'val')
    batch_size=args.batch_size
    path_to_save_data=os.path.join(args.path_to_dataset,args.name_of_result_folder)

    ARRAY_SIZE=[128,128,3]




    if(not os.path.isdir(path_to_save_data)):
        os.mkdir(path_to_save_data)



    #model = unetLightMobile(mode='train',training=True)
    model = unetLightMobile()
    model.compile(loss=focal_loss,metrics=['accuracy'],optimizer=tf.keras.optimizers.Adam(lr=10**-4))
        
    train_gen=ImageDataGenerator( height_shift_range=0.05,width_shift_range=0.05,horizontal_flip=True,  rescale=1/255.0, preprocessing_function=preprocess)
    valid_gen=ImageDataGenerator( height_shift_range=0.05,width_shift_range=0.05,horizontal_flip=True,  rescale=1/255.0, preprocessing_function=preprocess)
    
    ########################
 
    train_Gen=train_gen.flow_from_directory(shuffle=True,class_mode='categorical',batch_size=batch_size,directory=train_dir,target_size=(128,128))
    valid_Gen=valid_gen.flow_from_directory(shuffle=False,class_mode='categorical',batch_size=batch_size,directory=valid_dir,target_size=(128,128))
    

    labels = (train_Gen.class_indices)
    
    labels = dict((v,k) for k,v in labels.items())
    #class_weight=get_class_weights(train_dir,labels)
    #print("class_weights:",class_weight)
    print("labels:",labels)
    #print(train_Gen.samples)
    
    callbacks = prepare_keras_callback(valid_Gen,valid_Gen.samples,[v for k,v in labels.items()]) 
    
    bels={0: 'bald', 1: 'black', 2: 'blonde', 3: 'hat', 4: 'red',5:'white'}
    class_weight={0: 1, 1: 1, 2: 1, 3: 1, 4: 10,5: 1}
    history=model.fit(train_Gen,class_weight=class_weight,steps_per_epoch=int(train_Gen.samples//batch_size),callbacks=callbacks,validation_data=valid_Gen,shuffle=True,epochs=300) 
    #history=model.fit(train_Gen,steps_per_epoch=int(train_Gen.samples//BATCH_SIZE),callbacks=callbacks,validation_data=valid_Gen,shuffle=True,epochs=300) 
   


