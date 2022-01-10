
#### load camera configuration file into redis
```sh
redis-cli -x SET config < tools/config.json
```

#### run redis in docker
```sh
docker run -p 6379:6379 -d --name redis redis:alpine
```

#### run worker
```sh
python detect_persons.py
```
#### run publisher
```sh
python publisher.py
```




