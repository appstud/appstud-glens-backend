import tensorflow as tf
import sys
import os
from model import createModel

def preprocess(x):
    """x: b64 image
    """
    imgs_map = tf.map_fn(tf.image.decode_image,x,dtype=tf.uint8) # decode jpeg
    imgs_map.set_shape((None, None, None, 3))
    
    imgs = tf.image.resize(imgs_map, [96, 96]) # resize images
    imgs = tf.reshape(imgs, (-1, 96, 96, 3)) # reshape them
    #imgs=tf.reverse(imgs, axis=[-1])
    img_float = tf.cast(imgs, dtype=tf.float32) / 255.0 # and make them to floats
    
    return img_float

class tf_serving_model(tf.keras.Model):
    def __init__(self,weights_path):
        super(tf_serving_model, self).__init__()
        
        self.model=createModel()
        self.model.load_weights(weights_path)
        self.preprocessing_layer=tf.keras.layers.Lambda(lambda x:preprocess(x))
   
    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.string,name="input_image")])
    def __call__(self,x,training=False):
        x=self.preprocessing_layer(x)
        glasses=self.model(x,training=training)
        return {"glasses":glasses}


if(__name__=="__main__"):
    model=tf_serving_model(weights_path="./glassesDataset/output/results/weights.checkpoint")

    version="1"
    export_path = './glassesModel/'+version
    tf.saved_model.save(model, export_path)



