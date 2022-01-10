FROM ubuntu as base_image

RUN apt-get update
RUN apt install -y python3.6

RUN apt install -y python3-pip
RUN apt-get install -y  python3-dev 
RUN  pip3 install numpy

RUN apt-get install -qqy x11-apps
ENV DISPLAY :0


# set a directory for the app
WORKDIR ./app

# copy all the files to the container

COPY . .

 
EXPOSE 7369
# install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt


CMD ["python3","-u",  "./publisher.py"]
