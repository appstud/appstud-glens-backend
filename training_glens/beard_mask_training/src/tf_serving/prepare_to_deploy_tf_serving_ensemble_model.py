import tensorflow as tf
from trainBeard import createModel
import pdb
# The export path contains the name and the version of the model

tf.keras.backend.set_learning_phase(0) # Ignore dropout at inference

model1 =createModel(training=False) 
model2 =createModel(training=False) 

model1.load_weights('./tiny_model_1_weights.checkpoint')
model2.load_weights('./tiny_model_2_weights.checkpoint')

export_path = './maskModel/2'


def build_model(image_string):
    """image = tf.image.decode_image(image_string, channels=3)
    image.set_shape([None, None, None])
    image = tf.image.resize_images(image, [96, 96])
    """
    #string_inp = tf.placeholder(tf.string, shape=(None,)) #string input for the base64 encoded image

    imgs_map = tf.map_fn(tf.image.decode_image,image_string,dtype=tf.uint8) # decode jpeg

    imgs_map.set_shape((None, None, None, 3))

    imgs = tf.image.resize_images(imgs_map, [96, 96]) # resize images

    imgs = tf.reshape(imgs, (-1, 96, 96, 3)) # reshape them

    imgs=tf.reverse(imgs, axis=[-1])

    img_float = tf.cast(imgs, dtype=tf.float32) / 255.0 # and make them to floats

    output=0.5*(model1(img_float)+model2(img_float))




    return output

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as sess:
    
    input_ph = tf.placeholder(tf.string, shape=(None,)) #string input for the base64 encoded image
    #input_ph = tf.placeholder(tf.string, shape=[None])
    #images_tensor = tf.map_fn(prepare_image, input_ph,  dtype=tf.float32)
    out = build_model(input_ph)
    

    
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={"input_image": input_ph},

        outputs={"predictions":out})
