import cv2
import functools
import time
import json
import asyncio
import websockets
import sys
from dotenv import load_dotenv
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("__file__"), './')))
from utils.utils import *
from utils.draw_utils import *
import os
import signal
from dataset import LoadStreams
import copy
import logging

exit=False

result_data={}
MAX_BUFFER=100
current_sent_images_with_no_response=0


def exit_gracefully(signum,frame):
    global exit
    print('Signal handler called with signal', signum)
    exit=True


def getIntrinsicMatrix(cam_id):
    try:
        ##camId=r.get(message["camId"])

        #dataForCamera=json.loads(r.get(message["CAM_ID"]))["cam1"]
        dataForCamera=json.loads(r.get("config"))[cam_id]
        alpha_u=dataForCamera['alpha_u']
        alpha_v=dataForCamera['alpha_v']
        c_x=dataForCamera['c_x']
        c_y=dataForCamera['c_y']
        distCoeff=np.array(dataForCamera["distCoeff"])
        
        intrinsicMatrix=np.array([[alpha_u,0,c_x],[0,alpha_v,c_y],[0,0,1]])
        print(intrinsicMatrix) 
    except Exception as e:
        print(e)
        intrinsicMatrix=None
        distCoeff=None

    return intrinsicMatrix,distCoeff


async def storeReceivedData(queue,url,streams,ping_timeout=2,retry_connection_after=100):
    global websocket, current_sent_images_with_no_response
    #async with websockets.connect(url, close_timeout=0.1) as ws:
    while(True):
        try:
            websocket=await websockets.connect(url,timeout=ping_timeout)
            logging.warning("websocket connected...")
        except:
            logging.warning("unable to connect...retrying...")
            await asyncio.sleep(retry_connection_after)
        while True:
            try:
                #data = json.loads(await asyncio.wait_for(websocket.recv(), timeout=3))
                if(current_sent_images_with_no_response<MAX_BUFFER):
                    try:

                        data = json.loads(await asyncio.wait_for(websocket.recv(), timeout=0.033))
                        queue.put_nowait((data["CAM_ID"], data))
                        current_sent_images_with_no_response-=1
                        logging.info("received:{}".format(data["CAM_ID"]))
                        logging.debug("received data predictions: {}".format({k:v for k,v in data.items() if(k!="image")}))
                    except websockets.exceptions.ConnectionClosed as e:
                        
                        logging.warning("exception in storeReceivedData ")
                        logging.warning('websocket connection lost...will try to reconnect')
                        await asyncio.sleep(retry_connection_after)
                        break  # inner loop

                    except Exception as e:

                        logging.warning("exception in storeReceivedData general except")
                        initialize_queue(queue,streams)
                else:
                    data = json.loads(await asyncio.wait_for(websocket.recv(), timeout=3))
                    queue.put_nowait((data["CAM_ID"], data))
                    current_sent_images_with_no_response-=1
                    #print("received:",data["data"])
                    logging.info("received:{}".format(data["CAM_ID"]))
                    logging.debug("received data predictions: {}".format({k:v for k,v in data.items() if(k!="image")}))


                
                logging.debug(f'inside receiving thread {current_sent_images_with_no_response}')
            except(asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                try:
                    pong = await websocket.ping()
                    await asyncio.wait_for(pong, timeout=ping_timeout)
                    
                    ###probably a lost package don't stop keep sending packets
                    while(not queue.empty()):
                        await queue.get()
                    initialize_queue(queue,streams,reset=True)
                    
                    #initialize_queue(queue,streams)
                    logging.warning('Server took too long to respond... check for log messages on the server side for a failure...Ping OK, keeping connection alive...')
                    continue    
                except Exception as e:
                    await asyncio.sleep(retry_connection_after)
                    logging.error('websocket connection lost...will try to reconnect')
                    logging.error(e)
                    break  # inner loop

                
            except Exception as e:
                logging.error(e)
                return None

def initialize_queue(queue,streams,reset=False):    
    global current_sent_images_with_no_response
    if(reset):
        current_sent_images_with_no_response=0
    ### for initialization
    iter_=iter(streams)
    s=next(iter_)
     
    for elem in s[0].keys():
        queue.put_nowait((elem,{}))
    ###

async def draw_save_results():
    #saved_resolution=(640,480,3)
    saved_resolution=(1920,1080,3)
    nb_source=len(result_data)
    if(display_or_save=="SAVE"):    
        out=cv2.VideoWriter('./videos/result.mp4',cv2.VideoWriter_fourcc(*'XVID'), 10, ( nb_source*saved_resolution[0], saved_resolution[1]))
    
    while(not exit):
        try:
            noNewImages=True
            imgs=[]
            for k,v in result_data.items():
                if(("data" in v) and ("image" in v)):
                    if(not isinstance(v["image"],np.ndarray)):
                        result_data[k]["image"],_=draw_predictions(stringToBGR(v["image"]),v)
                        noNewImages=False
                        #result_data[k]["image"],_=draw_bounding_box(stringToBGR(v["image"]),v)
                    imgs.append(cv2.resize(result_data[k]["image"],saved_resolution[:2]))
                    #imgs.append(result_data[k]["image"])
                else:
                    imgs.append(np.zeros(saved_resolution,dtype=np.uint8))
            imgs=functools.reduce(lambda a,b:np.hstack((a,b)) if a is not None else b,imgs,None)
            
            if(imgs is not None):
                if(display_or_save=="SAVE" and not noNewImages):    
                    out.write(imgs)
                elif(display_or_save=="DISPLAY"):
                    cv2.imshow("results",imgs)
                    cv2.waitKey(1)
                logging.debug("writing image")
            await asyncio.sleep(0.033) #30 FPS
        
            
        except Exception as e:
            logging.error(e)
            
    if(display_or_save=="SAVE"):    
        logging.info("saving...")
        out.release()


async def sendImages(streams, queue,url):
    global result_data, current_sent_images_with_no_response
    initialize_queue(queue,streams)
    while(True):
        logging.debug("before reading from queue...")
        received_data= await queue.get()
        #received_data= queue.get_nowait()
        
        ###drawing results
        try: 
            if("data" in received_data[1]):
                result_data[received_data[0]]=copy.deepcopy(received_data[1])
        except Exception as e:
            logging.error(e)
        logging.info(f'inside sending thread {current_sent_images_with_no_response}')
        logging.debug("after reading from queue...")
        data_to_send=streams.get(received_data[0])
        logging.debug("after reading from streams...")
        logging.debug("received in sendImages coroutines:{}".format(received_data[0]))
        try:
            if(websocket is not None and data_to_send!=False and  data_to_send is not None and type(data_to_send['image'])==type(np.array(0))):
                data_to_send['image']=BGRToString(data_to_send['image'])
                
                logging.debug("after base64 conversion...")
                del data_to_send['last_time_accessed']
                data_to_send=json.dumps({"type":'message:v1:image:process',"payload":data_to_send})
                
                logging.debug("after json conversion...")
                logging.debug("before sending images...")
                if(current_sent_images_with_no_response<MAX_BUFFER):
                    await websocket.send(data_to_send)
                    current_sent_images_with_no_response+=1
                    logging.info(" sending images...")
                else:
                    logging.info("already send max allowed images will wait for response from server before sending again...")
            """else:
                queue.put_nowait((received_data[0],{}))
            """
        except Exception as e:
            logging.error(e)
        
    logging.error("send images coroutine is dead...")

websocket=None
async def start_all(source, in_url):
    global websocket
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)

    queue = asyncio.Queue()

    streams=LoadStreams(source)
    
    receive_task = asyncio.create_task(storeReceivedData(queue,in_url,streams))
   
    #wait for the storeReceive coroutine to start connection  
    await asyncio.sleep(2)
    ###
    send_task = asyncio.create_task(sendImages(streams,  queue, in_url))
    
    await asyncio.sleep(1)
    draw_save_task=asyncio.create_task(draw_save_results())
    while not exit:
        await asyncio.sleep(1)
        
   
    await asyncio.sleep(5)
    # terminate the workers
    send_task.cancel()

    await asyncio.sleep(1)
    receive_task.cancel()
    
    await asyncio.sleep(1)
    draw_save_task.cancel()









if(__name__=="__main__"):
    
    load_dotenv()  # take environment variables from .env.
    source=os.getenv("SOURCE") # Get the configuration file for all the streams
    display_or_save=os.getenv("DISPLAY_OR_SAVE","DISPLAY")
    logLevel=os.getenv("LOG_LEVEL","WARN")
    url=os.getenv("URL","ws://172.21.0.4:8081")
    #logging.basicConfig(filename='logs.log', level=getattr(logging,logLevel.upper()))
    logging.basicConfig( level=getattr(logging, logLevel.upper()))

    if(display_or_save=="DISPLAY"):
        cv2.namedWindow("results",cv2.WINDOW_NORMAL)
    asyncio.get_event_loop().run_until_complete(start_all(source, url))

