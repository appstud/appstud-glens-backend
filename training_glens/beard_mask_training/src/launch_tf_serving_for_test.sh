sudo docker run -t --rm  -p 8501:8501  -v "$(pwd)/beardModel:/models/beardModel"   -e MODEL_NAME=beardModel  tensorflow/serving -v

