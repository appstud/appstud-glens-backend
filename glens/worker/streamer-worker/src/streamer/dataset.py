from threading import Thread
import cv2
import os
import errno
import numpy as np
import time
import json
import copy
import logging


#logging.basicConfig( level=logging.WARNING)


class Params():
    """Class that loads parameters from a json file.
    

    Example:
    ```
    params = Params(json_path)
    print(params.gstreamer_pipeline)
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640):
        self.mode = 'images'
        self.img_size = img_size
        self.last_time_img_access=time.time()

        if os.path.isfile(sources):
            sources=Params(sources)
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), sources)
        
        #self.imgs = [None] * n
        self.sources = sources.dict
        n = len(self.sources.keys())
        for i, key in enumerate(self.sources.keys()):
            # Start the thread to read frames from the video stream
            s=self.sources[key]['source']
            logging.info('%g/%g: %s... ' % (i + 1, n, s))
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.sources[key]['image'] = cap.read()  # guarantee first frame
            self.sources[key]["last_time_accessed"]=time.time()
            self.sources[key]['current_time']=time.time()
            thread = Thread(target=self.update, args=([key, cap, fps]), daemon=True)
            logging.info(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()


    def update(self, key, cap, fps=10):
        # Read next stream frame in a daemon thread
        n = 0
        each_n_frames=1
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == each_n_frames:  # read every each_n_frames 
                ret, img = cap.retrieve()
                n = 0
                if(ret):
                    self.sources[key]['image']=img
                    self.sources[key]['current_time']=time.time()
                else:
                    logging.warning("problem getting video on camera {}".format(key))
           
            time.sleep(1.0/fps)  # wait time
            #time.sleep(0.1)  # wait time
            #time.sleep(0.2)  # 5FPS to account for delays of processing and sending to backend ==> might affect identification only
        del self.sources[key]
        logging.error("worker for camera {} is dead or finished".format(key))
    
    def get(self,key):
        if(self.sources[key]["last_time_accessed"]-self.sources[key]["current_time"]<0):
            self.sources[key]["last_time_accessed"]=time.time()
            return copy.deepcopy(self.sources[key])
        else:
            return False

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        
        #get only images that has not been accessed before
        info_dict= {k: v for k, v in self.sources.items() if self.sources[k]['current_time']-self.last_time_img_access>0}

        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration
        
        self.last_time_img_access=time.time()

        return self.sources, info_dict

if(__name__=="__main__"):
    """data=LoadStreams("streams.txt")

    for _,imgs in data:
        for i, img in enumerate(imgs):
            cv2.imshow("img"+str(i),img)
            cv2.waitKey(1)
    """
    LoadStreams("aouu")


