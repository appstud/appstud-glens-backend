import cv2
import base64
import json
import time
import numpy as np
import re
import websockets
import asyncio

def stringToBGR(base64_string):
    base64_string = re.sub('^data:image/.+;base64,', '', base64_string)
    imgdata=np.fromstring(base64.b64decode(str(base64_string)+'==='), np.uint8)
    image=cv2.imdecode(imgdata,cv2.IMREAD_UNCHANGED)
    return image[:,:,0:3]


def BGRToString(image):
    encodedImage=cv2.imencode('.jpg',image)[1]
    imgdata=base64.b64encode(encodedImage)
    imgdata = 'data:image/jpg;base64,'+ imgdata.decode('utf-8')
    return imgdata

cv2.namedWindow("img",cv2.WINDOW_NORMAL)
cv2.namedWindow("out",cv2.WINDOW_NORMAL)


@asyncio.coroutine
async def sendImage():
    cap=cv2.VideoCapture(0)
    ##cap=cv2.VideoCapture("video2.mp4")
    cap.set(cv2.CAP_PROP_FPS, 10)


    time.sleep(1) 
    

    channel="fawzi"
    uri = "ws://proxy:8765"

    print("ydddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddoyo")
    async with websockets.connect(uri) as websocket:
        print("yoyo")
        ###await websocket.send("subscribe|"+json.dumps(channel))
        while(True):
            ret,img=cap.read()
            img=cv2.resize(img,(640,480))
            cv2.imshow("img",img)
            cv2.waitKey(1)
            ret=True
            if(ret):
                #data=json.dumps({"ID":1,"img":BGRToString(img)})
                data=json.dumps({"img":BGRToString(img)})
                ##data=json.dumps({"channel":channel,"img":BGRToString(img)})
                print("Sent image to process!")

            #await websocket.send("process-image|"+data)
            await websocket.send("process-image|"+data)
            
            receivedData = await websocket.recv()
            
            print(receivedData)




          
if(__name__=="__main__"):
    asyncio.get_event_loop().run_until_complete(sendImage())
