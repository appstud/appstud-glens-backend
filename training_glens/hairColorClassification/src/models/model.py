import tensorflow as tf
import cv2
##from tensorflow import keras
import numpy as np
import pdb
#from extra_layers import *
INPUT_RESOLUTION=(128,128,3)
import tensorflow as tf


def focal_loss(onehot_labels, logits, num_cls=6,gamma=2.0, alpha=4.0):
    """
    focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: logits is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)

    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection, 130(4), 485–491.
    https://doi.org/10.1016/j.ajodo.2005.02.022

    :param labels: ground truth labels, shape of [batch_size]
    :param logits: model's output, shape of [batch_size, num_cls]
    :param gamma:
    :param alpha:
    :return: shape of [batch_size]
    """
    epsilon = 1.e-9
    #labels = tf.to_int64(labels)
    #labels = tf.convert_to_tensor(labels, tf.int64)
    #logits = tf.convert_to_tensor(logits, tf.float32)
    #num_cls = logits.shape[1]
    
    model_out = tf.add(logits, epsilon)
    #onehot_labels = tf.one_hot(labels, num_cls)
    ce = tf.multiply(onehot_labels, -tf.math.log(model_out))
    weight = tf.multiply(onehot_labels, tf.math.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max
    return reduced_fl


def freezeModel(model):
    for l in model.layers:
        print("layer not trainable:", l)
        l.trainable=False


class unetLightMobileVold(tf.keras.Model):

    def __init__(self):
        super(unetLightMobile,self).__init__()
        self.backbone=tf.keras.applications.MobileNetV2(weights="imagenet",include_top=False, alpha=1.0,input_shape=INPUT_RESOLUTION)
        freezeModel(self.backbone)
        self.dense=tf.keras.layers.Dense(6,activation='softmax')
        self.flatten=tf.keras.layers.Flatten()
        self.drop_out = tf.keras.layers.Dropout(0.5)
        self.avg_pooling=tf.keras.layers.GlobalAveragePooling2D()
        self.conv_1x1=tf.keras.layers.Conv2D(512,1)

    def __call__(self,inputs,training=True):
        x=self.backbone(inputs,training=training)
        x=self.conv_1x1(x)
        x=self.avg_pooling(x)
        x=self.flatten(x)
        x=self.drop_out(x,training)
        x=self.dense(x)
        return x



class removeFaceLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(removeFaceLayer, self).__init__()

    def call(self, inputs):
        #return tf.math.multiply(inputs[0] , tf.cast(tf.math.less(tf.slice(inputs[1],[0,0,0,1],[-1,-1,-1,1]),tf.constant([0.5])),tf.float32))
        return tf.math.multiply(inputs[0] , tf.expand_dims(tf.cast(tf.math.less(inputs[1][:,:,:,1],tf.constant([0.5])),tf.float32),-1))
        

class unetLightMobile(tf.keras.Model):

    def __init__(self):
        super(unetLightMobile,self).__init__()
        self.backbone=tf.keras.applications.MobileNetV2(weights="imagenet",include_top=False, alpha=1.0,input_shape=INPUT_RESOLUTION)
        freezeModel(self.backbone)
        self.dense=tf.keras.layers.Dense(6,activation='softmax')
        self.flatten=tf.keras.layers.Flatten()
        self.drop_out = tf.keras.layers.Dropout(0.5)
        self.avg_pooling=tf.keras.layers.GlobalAveragePooling2D()
        self.conv_1x1=tf.keras.layers.Conv2D(512,1)
        self.unet=unetLightMobileSegmentation()
        freezeModel(self.unet)
        #tf.math.multiply(t[0:1],tf.cast(tf.math.greater(t[0:1],tf.constant([5])),tf.int32))

        #self.removeFace=tf.keras.layers.Lambda(lambda x,y:tf.math.multiply(x tf.slice(y,[0,1,0],[1,-1,1])))
        #self.removeFace=tf.keras.layers.Lambda(lambda x,y:tf.math.multiply(x , tf.cast(tf.math.less(tf.slice(y,[0,1,0],[1,-1,1]),tf.constant([0.5])),tf.float32)))
        self.removeFace=removeFaceLayer()

    def __call__(self,inputs,training=True):
        x=self.unet(inputs)
        x=self.removeFace([inputs,x])
        x=self.backbone(x,training=training)
        x=self.conv_1x1(x)
        x=self.avg_pooling(x)
        x=self.flatten(x)
        x=self.drop_out(x,training)
        x=self.dense(x)
        return x


def unetLightMobileSegmentation(pretrained_weights = "models/weightsNew.checkpoint"):
    inputs = tf.keras.layers.Input((128,128,3))

    conv1 = tf.keras.layers.Conv2D(8, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name="conv1seg")(inputs)
    conv111 = tf.keras.layers.Conv2D(32, 7, activation = 'relu', strides=(2,2),padding = 'same', kernel_initializer = 'he_normal',name="conv2seg")(conv1)
    ###pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),name="pool1seg")(conv1)
    pool1=conv111
    conv2 = tf.keras.layers.SeparableConv2D(32, 3, activation = 'relu',name="conv3seg", padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv222 = tf.keras.layers.SeparableConv2D(64, 3,strides=(2,2), activation = 'relu',name="conv4seg", padding = 'same', kernel_initializer = 'he_normal')(conv2)
    #pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),name="pool2seg")(conv2)
    pool2=conv222
    conv3 = tf.keras.layers.SeparableConv2D(64, 3,name="conv5seg" , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv333 = tf.keras.layers.SeparableConv2D(128, 3,strides=(2,2) , name="conv6seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    #pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),name="pool3seg")(conv3)
    pool3 = conv333
    conv4 = tf.keras.layers.SeparableConv2D(128, 3,name="conv7seg" , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv444 = tf.keras.layers.SeparableConv2D(256, 3,name="conv8seg", activation = 'relu',strides=(2,2), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #drop4 = tf.keras.layers.Dropout(0.5)(conv444)

    drop4=conv444
    pool4=drop4
    conv5 = tf.keras.layers.SeparableConv2D(512,3 ,name="conv9seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv555 = tf.keras.layers.SeparableConv2D(512, 3, name="conv10seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #drop5 = tf.keras.layers.Dropout(0.5,name="pool5seg")(conv555)
    drop5 = conv555

    up6 = tf.keras.layers.SeparableConv2D(64, 3,name="conv11seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
    merge6 = tf.keras.layers.concatenate([conv4,up6], axis = 3)
    conv6 = tf.keras.layers.SeparableConv2D(64, 3,name="conv12seg", activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tf.keras.layers.SeparableConv2D(64, 3,name="conv13seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = tf.keras.layers.SeparableConv2D(32, 3, name="conv14seg",activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')( tf.keras.layers.UpSampling2D(size = (2,2))(conv6))
    merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)
    conv7 = tf.keras.layers.SeparableConv2D(32, 3,name="conv15seg", activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tf.keras.layers.SeparableConv2D(32, 3,name="conv16seg", activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = tf.keras.layers.SeparableConv2D(16, 3,name="conv17seg", activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv7))
    merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3)
    conv8 = tf.keras.layers.SeparableConv2D(16, 3,name="conv18seg", activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = tf.keras.layers.SeparableConv2D(16, 3, name="conv19seg",activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    conv9 = tf.keras.layers.SeparableConv2D(3, 3,name="conv23seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    up9 = tf.keras.layers.SeparableConv2D(8, 3,name="conv20seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv9))
    merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
    conv9 = tf.keras.layers.SeparableConv2D(8, 3,name="conv21seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = tf.keras.layers.SeparableConv2D(8, 3,name="conv22seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = tf.keras.layers.SeparableConv2D(3, 3,name="conv23seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    output = tf.keras.layers.SeparableConv2D(3, 1,name="conv24seg" , activation = 'softmax')(conv9)
    


    model = tf.keras.models.Model(inputs = inputs, outputs =output)
    model.load_weights(pretrained_weights,by_name=True)
    #for l in model.layers[:34]:
    #    print("layer not trainable:", l)
    #    l.trainable=False
    
    #model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])



    model.summary()
    

    """
    shapes_count = int(np.sum([np.prod(np.array([s if isinstance(s, int) else 1 for s in l.output_shape])) for l in model.layers]))

    memory = shapes_count * 4
    
    print(memory*10**-9)
    """


    return model










def unetLightMobileV1(pretrained_weights = "weightsNew.checkpoint",mode="test",training=True ):
    #inputs = tf.keras.layers.Input(input_size)
    inputs = tf.keras.layers.Input((128,128,3))

    conv1 = tf.keras.layers.Conv2D(8, 7, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',name="conv1seg")(inputs)
    conv111 = tf.keras.layers.Conv2D(32, 7, activation = 'relu', strides=(2,2),padding = 'same', kernel_initializer = 'he_normal',name="conv2seg")(conv1)
    ###pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),name="pool1seg")(conv1)
    pool1=conv111
    conv2 = tf.keras.layers.SeparableConv2D(32, 3, activation = 'relu',name="conv3seg", padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv222 = tf.keras.layers.SeparableConv2D(64, 3,strides=(2,2), activation = 'relu',name="conv4seg", padding = 'same', kernel_initializer = 'he_normal')(conv2)
    #pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),name="pool2seg")(conv2)
    pool2=conv222
    conv3 = tf.keras.layers.SeparableConv2D(64, 3,name="conv5seg" , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv333 = tf.keras.layers.SeparableConv2D(128, 3,strides=(2,2) , name="conv6seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    #pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),name="pool3seg")(conv3)
    pool3 = conv333
    conv4 = tf.keras.layers.SeparableConv2D(128, 3,name="conv7seg" , activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv444 = tf.keras.layers.SeparableConv2D(256, 3,name="conv8seg", activation = 'relu',strides=(2,2), padding = 'same', kernel_initializer = 'he_normal')(conv4)
    #drop4 = tf.keras.layers.Dropout(0.5)(conv444)

    drop4=conv444
    pool4=drop4
    conv5 = tf.keras.layers.SeparableConv2D(512,3 ,name="conv9seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv555 = tf.keras.layers.SeparableConv2D(512, 3, name="conv10seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    #drop5 = tf.keras.layers.Dropout(0.5,name="pool5seg")(conv555)
    drop5 = conv555

    up6 = tf.keras.layers.SeparableConv2D(64, 3,name="conv11seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(drop5))
    merge6 = tf.keras.layers.concatenate([conv4,up6], axis = 3)
    conv6 = tf.keras.layers.SeparableConv2D(64, 3,name="conv12seg", activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = tf.keras.layers.SeparableConv2D(64, 3,name="conv13seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = tf.keras.layers.SeparableConv2D(32, 3, name="conv14seg",activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')( tf.keras.layers.UpSampling2D(size = (2,2))(conv6))
    merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)
    conv7 = tf.keras.layers.SeparableConv2D(32, 3,name="conv15seg", activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = tf.keras.layers.SeparableConv2D(32, 3,name="conv16seg", activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = tf.keras.layers.SeparableConv2D(16, 3,name="conv17seg", activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv7))
    merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3)
    conv8 = tf.keras.layers.SeparableConv2D(16, 3,name="conv18seg", activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = tf.keras.layers.SeparableConv2D(16, 3, name="conv19seg",activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    conv9 = tf.keras.layers.SeparableConv2D(3, 3,name="conv23seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    up9 = tf.keras.layers.SeparableConv2D(8, 3,name="conv20seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(tf.keras.layers.UpSampling2D(size = (2,2))(conv9))
    merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
    conv9 = tf.keras.layers.SeparableConv2D(8, 3,name="conv21seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = tf.keras.layers.SeparableConv2D(8, 3,name="conv22seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = tf.keras.layers.SeparableConv2D(3, 3,name="conv23seg" ,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = tf.keras.layers.SeparableConv2D(3, 1,name="conv24seg" , activation = 'softmax')(conv9)
    #conv10 = tf.keras.layers.SeparableConv2D(3, 1,name="conv24seg")(conv9)




 
    merge10=tf.keras.layers.concatenate([conv10,inputs],axis=-1)
    ####noHair=tf.keras.layers.subtract([inputs,hair])
    ######merge10=tf.keras.layers.concatenate([hair,noHair],axis=-1)

    conv11 = tf.keras.layers.Conv2D(10, 3,strides=(1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge10)
    conv12 = tf.keras.layers.Conv2D(20, 3,strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
    #drop11 = tf.keras.layers.Dropout(0.5)(conv11)
    drop11 = conv12

    #drop11=tf.keras.layers.BatchNormalization()(drop11,training=training)
    #pool12 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop11)


    #conv13 = tf.keras.layers.Conv2D(15, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool12)  
    conv13 = tf.keras.layers.SeparableConv2D(40, 3,strides=(1,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop11)
    conv14 = tf.keras.layers.SeparableConv2D(40, 3,strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop11)

    #drop13 = tf.keras.layers.Dropout(0.1)(conv13)
    #drop13 = conv13
    drop13 = conv14

    #drop13=tf.keras.layers.BatchNormalization()(drop13,training=training)
     
    drop13=tf.keras.layers.SeparableConv2D(80,3,padding='same',activation='relu')(drop13)
    drop13=tf.keras.layers.SeparableConv2D(80,1,padding='same',activation='relu')(drop13)
    drop13=tf.keras.layers.GlobalAveragePooling2D()(drop13) 
    #pool14 = tf.keras.layers.MaxPooling2D(pool_size=(4, 4))(drop13)

    ###added with respect to last try
    #conv15 = tf.keras.layers.Conv2D(100, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool14)  
    ###conv15 = pool14 

    #drop13 = tf.keras.layers.Dropout(0.5)(conv13)
    ####drop16 = conv15

    ###drop17=tf.keras.layers.BatchNormalization()(drop16)
    ###pool18 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop17)


    """
    conv15 = tf.keras.layers.SeparableConv2D(30, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool14)  
    drop16 = tf.keras.layers.Dropout(0.5)(conv15)
    pool17 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop16)

    conv18 = tf.keras.layers.SeparableConv2D(60, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool17)  
    drop19 = tf.keras.layers.Dropout(0.5)(conv18)
    pool20 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(drop19)
    """


    Flatten21=tf.keras.layers.Flatten()(drop13)
    Flatten21=tf.keras.layers.Dense(6,activation='relu')(Flatten21)

    Flatten21=tf.keras.layers.BatchNormalization()(Flatten21)
    #drop22 = tf.keras.layers.Dropout(0.5)(Flatten21,training=training)
    output=tf.keras.layers.Dense(6,activation='softmax')(Flatten21)

    
    
    #model = tf.keras.models.Model(inputs = inputs, outputs = conv10)
    
    if(mode=='test'):
        model = tf.keras.models.Model(inputs = inputs, outputs =[output, conv10])
    else:
        model = tf.keras.models.Model(inputs = inputs, outputs =output)
        model.load_weights(pretrained_weights,by_name=True)
    for l in model.layers[:34]:
        print("layer not trainable:", l)
        l.trainable=False
    
    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])



    model.summary()
    

    """
    shapes_count = int(np.sum([np.prod(np.array([s if isinstance(s, int) else 1 for s in l.output_shape])) for l in model.layers]))

    memory = shapes_count * 4
    
    print(memory*10**-9)
    """


    return model








