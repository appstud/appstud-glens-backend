ARG IMAGE_VERSION

FROM tensorflow/tensorflow:$IMAGE_VERSION

RUN apt update && \
   DEBIAN_FRONTEND="noninteractive" apt install --no-install-recommends  -y python3 python3-pip libgtk-3-dev && \
   pip3 install --upgrade pip && \
   pip3 install --upgrade setuptools && \
   rm -rf /var/lib/apt/lists/* 

WORKDIR /work
COPY src/ /work/


RUN pip3 install -r requirements.txt 
ENV PYTHONPATH='/work/:$PYTHONPATH'

CMD ["python3","-u","service.py"]
