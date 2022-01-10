import tensorflow as tf
import numpy as np
import os
import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), '../src/')))

#from face_detection_alignment.faceAlignmentV2 import *
from models.model import unetLightMobile

INPUT_RESOLUTION=(128,128,3)

def getModel(weights_path):
    model=unetLightMobile()
    model.load_weights(weights_path)
    inp = tf.keras.layers.Input(shape=INPUT_RESOLUTION)
    x=tf.keras.layers.Lambda(lambda x: tf.reverse(x,[3]))(inp)
    out=model(x,training=False)

    model_rgb=tf.keras.Model(inputs=inp, outputs=out)

    return model_rgb


def convert_to_tflite(model_name,weights_path):
    model=getModel(weights_path)

    #### I needed to do this otherwise tflite won't detect the shape of the input for some reason...
    #input_arr = 255*np.ones((1,128,128,3))
    #outputs = model(input_arr)
    #model._set_inputs(input_arr)

    # Convert the model
    #converter=tf.lite.TFLiteConverter.from_saved_model("./tf_lite_model")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    #converter.target_spec.supported_types = [tf.float16]


    tflite_model = converter.convert()

    # Save the model.
    with open(model_name+'.tflite', 'wb') as f:
        f.write(tflite_model)


def test_tflite_model(model_path,source=0):
    # Load TFLite model and see some details about input/output
    tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    #tflite_interpreter.resize_tensor_input(input_details[0]['index'], (32, 224, 224, 3))
    #tflite_interpreter.resize_tensor_input(output_details[0]['index'], (32, 5))
    tflite_interpreter.allocate_tensors()

    print("== Input details ==")
    print("name:", input_details[0]['name'])
    print("shape:", input_details[0]['shape'])
    print("type:", input_details[0]['dtype'])

    print("\n== Output details ==")
    print("name:", output_details[0]['name'])
    print("shape:", output_details[0]['shape'])
    print("type:", output_details[0]['dtype'])
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')

    cap=cv2.VideoCapture(source)
    ret,img=cap.read()
    out = cv2.VideoWriter("out_tensorflow_lite.mp4", fourcc, 20.0, img.shape[0:2][::-1])

    while(True):
        ret,img=cap.read()

        cap.set(cv2.CAP_PROP_FPS, 30)

        start=time.time()
        _,bbox,_,frame=ssd_face_detection(img)
        finish=time.time()
        try:
            croppedFace=cv2.resize(img[bbox[0][1]:bbox[0][3]+bbox[0][1], bbox[0][0]:bbox[0][2]+bbox[0][0],:],(128,128))
            tflite_interpreter.set_tensor(input_details[0]['index'], np.array([croppedFace]).astype(np.float32)/255.0)
            tflite_interpreter.invoke()
            output = tflite_interpreter.get_tensor(output_details[0]['index'])

            cv2.putText(img,"output: "+str(output),(100,100),cv2.FONT_ITALIC, 1,(255,0,0))
            cv2.imshow("face_cropped",croppedFace)
            if(cv2.waitKey(1) & 0XFF==ord('q')):
                break
        except Exception as e:
            print(e)
        out.write(img)
        cv2.imshow("img",img)



if(__name__=="__main__"):
    convert_to_tflite("Model_hair","../split/results/best_model.checkpoint")
    #test_tflite_model(model_path="./Model_hair.tflite",source=0)
