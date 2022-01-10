# Face detection worker: glens-face-detection

This worker does face detection given an input image. Different face detection algorithms are already implemented in the worker (MTCNN, SSD, dlib face detector). However only the pytorch version of MTCNN is currently being used because of its speed and accuracy.

## Installation

Use **make** to build a docker image of the worker (glens-face-detection).

```bash
make
```

## Usage

This worker receives a json input via redis on the channel **face-detection**. 

The input must contains the following fields : *image*, *pipeline* and *CAM_ID*.

**CAM_ID**: The camera ID used for retrieving camera specific data (calibration data).

**image**: base64 image in BGR format.

**pipeline**: field that contains information about the following workers that the image and the predictions of this worker must be sent to in addition to options for the current worker.

Option name | Description | Possible values | Default values 
------------ | ------------- | ------------- | -------------
LOG_LEVEL | Sets the logging level of the logger | *WARNING* *DEBUG* *ERROR*  *VERBOSE* | *WARNING*
REPORT_PERF | Whether to compute the FPS of this worker, this is useful for debugging  | *true* *false* | *false* |

## Example of a pipeline

An example of a pipeline that uses face-detection with DEBUG logging level and computes the FPS and send the result to the tensorflow-calls worker:
```json
{pipeline: "face-detection LOG_LEVEL=DEBUG REPORT_PERF=true | tensorflow-calls", image: "your base64 image here"}
```







