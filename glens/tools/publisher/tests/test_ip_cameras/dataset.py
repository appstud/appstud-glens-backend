from threading import Thread
import cv2
import os
import numpy as np
import time

class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640):
        self.mode = 'images'
        self.img_size = img_size

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')
            cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert cap.isOpened(), 'Failed to open %s' % s
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')  # newline


    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        each_n_frames=1
        while cap.isOpened():
            n += 1
            # _, self.imgs[index] = cap.read()
            cap.grab()
            if n == each_n_frames:  # read every each_n_frames 
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.033)  # wait time
    
    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img =img0

        # Stack
        img = np.stack(img, 0)

        # Convert
        return self.sources, img, img0, None

if(__name__=="__main__"):
    data=LoadStreams("streams.txt")

    for _,imgs,_,_ in data:
        for i in range(imgs.shape[0]):
            cv2.imshow("img"+str(i),imgs[i])
            cv2.waitKey(1)


