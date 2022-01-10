#FROM ubuntu as base_image
FROM nvidia/cuda:10.0-runtime
#FROM nvidia/cuda:10.2-runtime
#FROM nvidia/cuda:11.1-runtime

WORKDIR ./app

COPY . .
RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install --no-install-recommends  -y python3 python3-pip build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev python3-dev && \
    apt-get install -y  ffmpeg libsm6 libxext6 libatlas3-base && \
    pip3 install --upgrade setuptools && \
    pip3 install --upgrade pip && \
    pip3 install  --no-cache-dir -r requirements.txt &&\     
    rm -rf /var/lib/apt/lists/* && \  
    apt purge -y --autoremove build-essential python3-dev libboost-python-dev libgtk-3-dev cmake pkg-config libx11-dev libatlas-base-dev
# set a directory for the app
# install dependencies
# We did it as soon as possible to use a maximum the Docker cache


# copy all the remaning files to the container


# tell the port number the container should expose
###tensorflow serving port####
#EXPOSE 8501
######################

####redis port#######
#EXPOSE 6379
#####################

CMD ["python3", "-u","./service.py"]

