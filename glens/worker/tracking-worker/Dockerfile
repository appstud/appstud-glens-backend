
FROM ubuntu as base_image

RUN apt update && \
   DEBIAN_FRONTEND="noninteractive" apt install --no-install-recommends  -y python3 python3-pip && \
   rm -rf /var/lib/apt/lists/* && \
   pip3 install --upgrade setuptools


# set a directory for the app
WORKDIR ./app
# install dependencies
COPY . .
RUN  pip3 install --no-cache-dir -r requirements.txt
CMD ["python3", "-u","./service.py"]
