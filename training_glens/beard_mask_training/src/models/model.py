import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer
import scipy.io as sio
from tensorflow.keras import optimizers,losses,activations
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from enhanceQuality import dynamicRangeCompression
import os
from helperFunctions import cropImage
from models.extra_layers import *
import random

###tf.enable_eager_execution()
def freezeLayer(layer):
    layer.trainable = False
    if hasattr(layer, 'layers'):
        for l in layer.layers:
            freezeLayer(l)
        
def createBackboneCBAM():

    inp = tf.keras.layers.Input(shape=(96, 96, 3))
    x = tf.keras.layers.Conv2D(32, (3, 3), activation=None, padding='same')(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation=None, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x0 = tf.keras.layers.MaxPooling2D((2,2))(x)



    x = tf.keras.layers.Conv2D(64, (3, 3), activation=None, padding='same')(x0)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation=None, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x1 = tf.keras.layers.MaxPooling2D((2,2))(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation=None, padding='same')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = cbam_block(x,128)
    x = residual_block(x, 128)
    x2 = tf.keras.layers.MaxPooling2D((2,2))(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), activation=None, padding='same')(x2)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = cbam_block(x,256)
    x = residual_block(x, 256)
    x3 = tf.keras.layers.MaxPooling2D((2,2))(x)



    x = tf.keras.layers.Conv2D(512, (3, 3), activation=None, padding='same')(x3)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = cbam_block(x,512)
    x = residual_block(x, 512)
    x4 = tf.keras.layers.MaxPooling2D((2,2))(x)
    
    reshape_layer = tf.keras.layers.Flatten()(x4)
    #dense_layer = Dense(128, name='dense_layer')(reshape_layer)
    dense_layer = tf.keras.layers.Dense(200, name='dense_layer')(reshape_layer)
    out = tf.keras.layers.Dense(2, activation="softmax" ,name='beard_mask_dense2')(dense_layer)


    model=Model(inputs=inp, outputs=[out])
    print(model.summary())

    return model

def createModel(pretrained="no",training=True):
    
    
    if(pretrained=="cbam"):
        
        model=createBackboneCBAM()
        model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=tf.keras.optimizers.Adam(lr=1*10**-4))
    
    elif(pretrained=="mobilenet"): 
        base_model=tf.keras.applications.MobileNetV2(weights="imagenet",include_top=False, alpha=1.0,input_shape=(96,96,3))
        freezeLayer(base_model)
        out=tf.keras.layers.Conv2D(20,(3,3))(base_model.output)
        reshape_layer = tf.keras.layers.Flatten()(out)
        
        reshape_layer=tf.layers.Dropout(0.5)(reshape_layer)
        #dense_layer = Dense(128, name='dense_layer')(reshape_layer)
        
        dense_layer = tf.keras.layers.Dense(20, name='dense_layer')(reshape_layer)
        dense_layer=tf.layers.Dropout(0.5)(dense_layer)
        out = tf.keras.layers.Dense(2, activation="softmax" ,name='beard_mask_dense2')(dense_layer)
        model=Model(inputs=base_model.inputs, outputs=[out])
        model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=tf.keras.optimizers.Adam(lr=1*10**-5))
    
    else:
        in_a = Input(shape=(96, 96, 3))
        
        x=tf.keras.layers.Lambda(lambda x:x[:,40:,:,:],name="crop")(in_a)
        #crop=in_a
        x=tf.keras.layers.Conv2D(40,3,strides=(2,2),name="beard_mask_conv1",activation='relu')(x)
        
        x=tf.keras.layers.BatchNormalization(name="beard_mask_bn1")(x)
        #pool1=tf.keras.layers.MaxPool2D(name='beard_pool1',pool_size=(2,2))(bn1)
        #drop1=tf.keras.layers.Dropout(0.5)(bn1)
    

        x=tf.keras.layers.SeparableConv2D(60,3,name="beard_mask_conv2",activation='relu')(x)

        x=tf.keras.layers.BatchNormalization(name="beard_mask_bn2")(x)
        x=tf.keras.layers.MaxPool2D(name='beard_mask_pool2', pool_size=(2,2))(x)
        x = cbam_block(x,60)
        
        #drop2=tf.keras.layers.Dropout(0.5)(pool2)
        #drop2=pool2
        x=tf.keras.layers.SeparableConv2D(80,3,name="beard_mask_conv3",activation='relu')(x)
        x=tf.keras.layers.SeparableConv2D(3,1,name="beard_mask_conv31",activation='relu')(x)

        x=tf.keras.layers.BatchNormalization(name="beard_mask_bn3")(x)
        x=tf.keras.layers.MaxPool2D(name='beard_mask_pool3', pool_size=(2,2))(x)
        

        #drop3=tf.keras.layers.Dropout(0.5)(pool3)

        #outGlasses=tf.keras.layers.Dense(100,activation='relu')(emb_a)
        x=tf.keras.layers.Flatten(name="beard_mask_flatten1")(x)
        x=tf.keras.layers.Dense(5,activation='relu',name="beard_mask_dense1")(x)
        x=tf.keras.layers.Dense(2,activation='softmax',name='beard_mask_dense2')(x)
        
        model = Model([in_a], [x])
        
        model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer=tf.keras.optimizers.Adam(lr=10**-4))
    
        
  
    print(model.summary())
    return model


