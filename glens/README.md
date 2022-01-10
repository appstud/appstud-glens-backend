
# glens-api

GLENS is an AI system for video analytics. Its architecture is based on multiple AI worker written in python that communicates between each other using redis pub-sub scheme.

This software (and all included files) are distributed through [AGPL v3](./LICENSE)

## Description

Available workers:
- [GLENS face detection](worker/face-detection-worker/readme.md)
- [GLENS posenet](worker/posenet-worker)
- [GLENS YOLO object detection](worker/object-detecgion-worker)
- [GLENS tensorflow serving models](tensorflow/)
- [GLENS tensorflow-calls](worker/tensorflow-serving-calls-worker/readme.md)
- [GLENS tracking](worker/tracking-worker) 
- [streamer](worker/streamer-worker) 
This repository contains also the following components:
- [GLENS proxy api](api/)

- [GLENS calibration procedure](tools/camera_calibration)
- [docker-compose examples for deployments](worker/deploy_glens/readme.md)


# Run Glens (light version)

### Prequisites
- Node v14.4
- Docker
- Docker-compose
- Git-LFS

### Run this commands
```
git clone
git lfs migrate export --include="*"
cd api
npm install
cd ..
docker-compose -f docker-compose_light.yml up -d
```

