# Tensorflow calls worker: glens-tensorflow-calls

This worker is responsible for calling tensorflow models hosted by the tensorflow serving REST API.
It is responsible for calling the following models :
- Face bounding box attributes:
  - Age, gender and glasses model
  - Beard detection model 
  - Hair color classification model
  - Head pose estimation model
  - Face descriptor model (Generates a descriptor for a given face) 
- Person bounding box attributes:
  - Person reidentification model (Generates a descriptor for a given face)
  - Pose estimation (Using pre-calibration data estimate the 2D position of a person with respect to coordinate system attached to the ground see [calibration])


## Installation

Use **make** to build a docker image of the worker (glens-tensorflow-calls).

```bash
make
```

## Usage

This worker receives a json input via redis on the channel **tensorflow-calls**. It can't be called directly by providing only an image: The input must be the output of either glens-face-detection, yolo or posenet.

User provided fields: *image*, *pipeline* and *CAM_ID*.

**CAM_ID**: The camera ID used for retrieving camera specific data (calibration data).

**image**: base64 image in BGR format.

**pipeline**: field that contains information about the following workers that the image and the predictions of this worker must be sent to in addition to options for the current worker.

Option name | Description | Possible values | Default values | Previous workers
------------ | ------------- | ------------- | ------------- | -------------
LOG_LEVEL | Sets the logging level of the logger | *WARNING* *DEBUG* *ERROR*  *VERBOSE* | *WARNING* | posenet, yolo, face-detection
REPORT_PERF | Whether to compute the FPS of this worker, this is useful for debugging  | *true* *false* | *false* | posenet, yolo, face-detection
GET_AGE_SEX_GLASSES | Whether to call the model for age, gender and glasses or not| *true* *false* | *false* |posenet, face-detection
GET_POSE | Whether to call the model for head pose estimation or not | *true* *false* | *false* | posenet, face-detection
GET_FACE_RECO | Whether to call the model for face description or not  | *true* *false* | *false* | posenet, face-detection
GET_HAIR_COLOR | Whether to call the model for hair color estimation or not  | *true* *false* | *false* | posenet, face-detection
GET_MASK | Whether to call the model for mask detection or not  | *true* *false* | *false* | posenet, face-detection
GET_PERS_REID | Whether to call the model for person description or not  | *true* *false* | *false* | posenet, yolo

## Example of a pipeline

An example of a pipeline that uses face-detection and send the result to the tensorflow-calls worker for age and gender estimation:
```json
{pipeline: "face-detection LOG_LEVEL=DEBUG REPORT_PERF=true | tensorflow-calls GET_AGE_SEX_GLASSES=true", image: "your base64 image here", CAM_ID: "0"}
```

An example of a pipeline that uses posenet and send the result to the tensorflow-calls worker to compute the descriptor vectors of the detected persons:
```json
{pipeline: "posenet LOG_LEVEL=DEBUG REPORT_PERF=true | tensorflow-calls GET_PERS_REID=true", CAM_ID:"0", image: "your base64 image here"}
```







