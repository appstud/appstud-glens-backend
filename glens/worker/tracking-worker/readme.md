# Tracking worker: glens-tracking

This worker is responsible for tracking and reidentification. It can track and reidentify persons based on their face bounding boxes (obtained using glens-face-detection or posenet workers) or whole body bounding boxes (obtained using glens-object-detection or posenet workers).


## Installation

Use **make** to build a docker image of the worker (glens-tracking).

```bash
make
```

## Usage

This worker receives a json input via redis on the channel **tracking**. It can't be called directly by providing only an image: The input must be the output of tensorflow-calls with GET_PERS_REID=true or GET_FACE_RECO=true.

User provided fields: *image*, *pipeline* and *CAM_ID*.

**CAM_ID**: The camera ID used for retrieving camera specific data (calibration data).

**image**: base64 image in BGR format.

**pipeline**: field that contains information about the following workers that the image and the predictions of this worker must be sent to in addition to options for the current worker.

Refer to the CIR 2020 document If you would wish to tune the parameters of this worker and not use the default values to better understand the effect of each one of them.

Option name | Description | Possible values | Default values | Previous workers
------------ | ------------- | ------------- | ------------- | -------------
LOG_LEVEL | Sets the logging level of the logger | *WARNING* *DEBUG* *ERROR*  *VERBOSE* | *DEBUG* | tensorflow-calls 
REPORT_PERF | Whether to compute the FPS of this worker, this is useful for debugging  | *true* *false* | *false* | tensorflow-calls 
USE_RECO | Whether to use appearance descriptors (coming either from person reidentification or face recognition models) or not| *true* *false* | *false* | tensorflow-calls GET_FACE_RECO=true or tensorflow-calls GET_PERS_REID=true
USE_PERSON_DATA | Wether to use person bbox encodings or face bbox encodings if both of them are available | *false* *true* | *true* | tensorflow-calls
MIN_E | Threshold to remove encodings from redis (see the CIR 2020 for more info) | 0<*FLOAT*<1 | *0.01* | tensorflow-calls
RO | Reverse nearest neighbor threshold  | 0<*FLOAT*<1 | *0.75* | tensorflow-calls
ALPHA | Eligibility update parameter | *FLOAT*>0 | *0.01* | tensorflow-calls
LAMBDA_A | Parameter to exponentially decrease confidence in position estimates as time goes by without identification  | *FLOAT*>0 | *0.2* | tensorflow-calls
W_3D | Weights of position estimates in the final decision  | *FLOAT*>0 | *0.1* | tensorflow-calls
ALPHA_3D | Distance threshold if the distance affinity > ALPHA_3D then it is not likely that a match is correct so it is removed and marked as unmatched| *FLOAT*>0 | *0.5* |tensorflow-calls
W_RECO | Weights of appearance descriptors in the final decision | *FLOAT*>0 | *5* | tensorflow-calls
ALPHA_RECO | Distance threshold on descriptor scores if the distance affinity > ALPHA_RECO then it is not likely that a match is correct so it is removed and marked as unmatche | 0<*FLOAT*<1  | *1* | tensorflow-calls


## Example of a pipeline

An example of a pipeline that uses tracking based on face detection:
```json
{pipeline: "face-detection LOG_LEVEL=DEBUG REPORT_PERF=true | tensorflow-calls GET_FACE_RECO=true | tracking USE_RECO=true ", image: "your base64 image here", CAM_ID: "0"}
```

An example of a pipeline that uses tracking based on posenet:
```json
{pipeline: "posenet LOG_LEVEL=DEBUG REPORT_PERF=true | tensorflow-calls GET_PERS_REID=true | tracking USE_RECO=true", CAM_ID:"0", image: "your base64 image here"}
```







