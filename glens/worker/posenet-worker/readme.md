# posenet worker: posenet-python-gpu

This worker is responsible for estimating body landmarks of persons.
In addition some heuristics were added to estimate person bounding box and face bounding box where possible.


## Installation

Use **make** to build a docker image of the worker (posenet-python-gpu).

```bash
make
```

## Usage

This worker receives a json input via redis on the channel **posenet**. It can be called directly by providing only an image: 

User provided fields: *image*, *pipeline* and *CAM_ID*.

**CAM_ID**: The camera ID used for retrieving camera specific data (calibration data).

**image**: base64 image in BGR format.

**pipeline**: field that contains information about the following workers that the image and the predictions of this worker must be sent to in addition to options for the current worker.



Option name | Description | Possible values | Default values 
------------ | ------------- | ------------- | -------------  
LOG_LEVEL | Sets the logging level of the logger | *WARNING* *DEBUG* *ERROR*  *VERBOSE* | *WARNING* 
REPORT_PERF | Whether to compute the FPS of this worker, this is useful for debugging  | *true* *false* | *false* 



## Example of a pipeline

An example of a pipeline that uses tracking based on posenet:
```json
{pipeline: "posenet LOG_LEVEL=DEBUG REPORT_PERF=true | tensorflow-calls GET_PERS_REID=true | tracking USE_RECO=true", CAM_ID:"0", image: "your base64 image here"}
```







