#FROM ubuntu as base_image
FROM tensorflow as base_image

RUN apt update && \
    DEBIAN_FRONTEND="noninteractive" apt install --no-install-recommends  -y python3 python3-pip build-essential cmake pkg-config libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev python3-dev && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --upgrade pip && \   
    pip3 install --upgrade setuptools
   

# set a directory for the app
WORKDIR ./app

# install dependencies
# We did it as soon as possible to use a maximum the Docker cache
COPY requirements.txt .
RUN  pip3 install --no-cache-dir -r requirements.txt

CMD bash
