import tensorflow as tf
import pdb
import sys
import argparse
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../src/')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../../face_detection_alignment')))
from models.model import *


def preprocess(image_string):
    """image = tf.image.decode_image(image_string, channels=3)
    image.set_shape([None, None, None])
    image = tf.image.resize_images(image, [96, 96])
    """
    #string_inp = tf.placeholder(tf.string, shape=(None,)) #string input for the base64 encoded image

    imgs_map = tf.map_fn(tf.image.decode_image,image_string,dtype=tf.uint8) # decode jpeg

    imgs_map.set_shape((None, None, None, 3))

    imgs = tf.image.resize(imgs_map, [96, 96]) # resize images

    imgs = tf.reshape(imgs, (-1, 96, 96, 3)) # reshape them

    imgs=tf.reverse(imgs, axis=[-1])

    img_float = tf.cast(imgs, dtype=tf.float32) / 255.0 # and make them to floats


    return img_float



class tf_serving_model(tf.keras.Model):
    def __init__(self,weights_path):
        super(tf_serving_model, self).__init__()

        self.model=createModel("cbam")
        self.model.load_weights(weights_path)
        self.preprocessing_layer=tf.keras.layers.Lambda(lambda x:preprocess(x))

    @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.string,name="input_image")])
    def __call__(self,x,training=False):
        x=self.preprocessing_layer(x)
        out=self.model(x,training=training)
        return out



if(__name__=="__main__"):
    
    parser = argparse.ArgumentParser(description='Python script for exporting a tensorflow model to tf serving')
    parser.add_argument('--path_to_weights',help='path to the weights of the model to export', type=str, default="../beardDataset/output/results/weights.checkpoint")
    parser.add_argument('--export_to',help='path to export the model to', type=str, default="./beardModel/1")
    args = parser.parse_args()

    

    #model=tf_serving_model('../MORPH_nonCommercial/morph_fixed_split/best_model_batch_size=8_backbone=cbam_diff_age=3_lambda_1=0.1_delta=7.0_alpha=3.0_gamma=1.2_res=(240,200,3)_.checkpoint')
    
    export_path = args.export_to
    model=tf_serving_model(weights_path=args.path_to_weights)
    tf.saved_model.save(model, export_path)



  
