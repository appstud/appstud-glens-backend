#### On startup

[`worker`] > Ask for model data to Tensorflow Server

#### On image

[`tools/demo`] > [`api`] > [`redis`] > [`worker`]

#### Once processing is done

[`tools/demo`] < [`api`] < [`redis`] < [`worker`]

## Useful commands

Start redis
```sh
docker run -p 6379:6379 -d --name redis redis:alpine
```

Stop & start redis
```sh
docker stop redis && docker rm -v redis && docker run -p 6379:6379 -d --name redis redis:alpine
```

Run tensorflow server on cpu (non detached for logs)
```sh
docker run -v $PWD/tensorflow:/models --name tensorflow -p 8501:8501 -d tensorflow/serving --model_config_file=/models/models.config
```

Run tensorflow server on gpu
```sh
docker run --gpus all -v $PWD/tensorflow:/models --name tensorflow -p 8501:8501 -d tensorflow/serving --model_config_file=/models/models.config
```

Run worker (non detached for logs)
```sh
# Need to be built before with dockerfile in /worker
docker run --rm -v $PWD/worker:/app -e TENSORFLOW_HOST=172.17.0.3 -e REDIS_HOST=172.17.0.4 -e GLENS_RETURN_IMAGES=False proxy-backend-glens-redis_backend-glens python3 -u /app/service.py
```

Run demo frontend
```sh
cd tools/demo
python2 -m SimpleHTTPServer
```
