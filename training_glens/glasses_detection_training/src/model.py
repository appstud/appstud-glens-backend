import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer

def createModel():
    # Input for anchor, positive and negative images
    in_a = Input(shape=(96, 96, 3))

    crop=tf.keras.layers.Lambda(lambda x:x[:,0:40,:,:],name="crop")(in_a)

    conv1=tf.keras.layers.Conv2D(10,3,name="glasses_conv1",activation='relu')(crop)
    bn1=tf.keras.layers.BatchNormalization(name="glasses_bn1")(conv1)
    pool1=tf.keras.layers.MaxPool2D(name='glasses_pool1',pool_size=(4,4))(bn1)

    conv2=tf.keras.layers.Conv2D(2,1,name="glasses_conv2",activation='relu')(pool1)
    bn2=tf.keras.layers.BatchNormalization(name="glasses_bn2")(conv2)
    pool2=tf.keras.layers.MaxPool2D(name='glasses_pool2', pool_size=(2,2))(bn2)


    outGlasses=tf.keras.layers.Flatten(name="glasses_flatten1")(pool2)
    outGlasses=tf.keras.layers.Dense(5,activation='relu',name="glasses_dense1")(outGlasses)
    outGlasses=tf.keras.layers.Dense(1,activation='sigmoid',name='classif_glasses')(outGlasses)

    model = Model([in_a], [outGlasses])




    print(model.summary())
    return model

