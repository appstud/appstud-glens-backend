import cv2
import numpy as np
import pdb
import os 
import json
import time
import argparse
pts_img=[]
point_chosen=False

def get_pixel_coordinates(event, x, y, flags, param):
    global point_chosen,img
    if event == cv2.EVENT_LBUTTONDOWN:
        pts_img.append([x,y])
        point_chosen=True
        cv2.drawMarker(img,(x,y),(0,0,255),markerType=cv2.MARKER_CROSS,markerSize=60,thickness=4)
        print(x,y)




def read_config_from_file(filename="config.json"):
    if(not os.path.exists(filename)):
        open(filename,'a').close()
    try:
        with open(filename,'r') as f:    
                data=json.loads(f.read())
    except Exception as e:
        print(e)
        return dict()

    return data

def write_config_to_file(data,output="config.json"):
    if(not os.path.exists(output)):
        open(output,'a').close()

    with open(output,'w') as f:
        json.dump(data,f,ensure_ascii=False,indent=4)

def get_config_from_redis(cam_id,r):
    dataForCamera=json.loads(r.get("config"))[cam_id]
    return dataForCamera


def getIntrinsicExtrinsicMatrix(dataForCamera):
    if('alpha_v' in dataForCamera.keys()):
        #dataForCamera=json.loads(r.get(message["CAM_ID"]))["cam1"]
        alpha_u=dataForCamera['alpha_u']
        alpha_v=dataForCamera['alpha_v']
        c_x=dataForCamera['c_x']
        c_y=dataForCamera['c_y']
        distCoeff=np.array(dataForCamera["distCoeff"])
        intrinsicMatrix=np.array([[alpha_u,0,c_x],[0,alpha_v,c_y],[0,0,1]])
    else:
        intrinsicMatrix=None
        distCoeff=None
        rigidTransformToWCS=np.eye(4)

    if('tx'  in dataForCamera.keys() and "roll"  in dataForCamera.keys()):
        ###Notice the minus sign for pitch
        rigidTransformToWCS=getExtrinsicMatrix(tx=dataForCamera["tx"],ty=dataForCamera["ty"],tz=dataForCamera["tz"],yaw=dataForCamera["yaw"],pitch=-dataForCamera["pitch"],roll=dataForCamera["roll"])
        rigidTransformToWCS=np.vtack((extriniscMatrix,np.array([0,0,0,1])))
    else:
        rigidTransformToWCS=np.eye(4)

    return intrinsicMatrix,distCoeff,rigidTransformToWCS


def estimate_homography(cam_id="0",filename="config.json"):
    """script to calculate the homography that maps from 3D world to image"""
    
    global point_chosen,img
    pts_world=[]
    
    finish=False
    data=read_config_from_file(filename)
    intrinsicMatrix,distCoeff,_=getIntrinsicExtrinsicMatrix(data[cam_id])
    mapx,mapy=cv2.initUndistortRectifyMap(intrinsicMatrix,distCoeff,None,intrinsicMatrix,img.shape[0:2][::-1],5)
    img=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
    cv2.imshow("img",img)    
    cv2.waitKey(1)

    while(True):
        print('Select on the screen a point (minimum number of 4 points must be chosen) followed by enter (while the image is in the front) or press q to finish selecting points')
        while(not point_chosen):
            q=cv2.waitKey(1000000)
            
            if(q & 0XFF==ord("q")):
                finish=True
                break
        if(finish):
            break
        point_chosen=False
        cv2.imshow("img",img)
        cv2.waitKey(1)
        inp=input('Enter its real coordinates separated by _ ( example 2.3_5.3)')
        x,y=list(map(float,inp.split('_')))
        pts_world.append([x,y])
    
    h_world_to_img, status = cv2.findHomography(np.array(pts_world).astype(np.float32), np.array(pts_img).astype(np.float32))
    
    max_x,max_y=np.max(pts_world,axis=0)
    
    min_resolution=100
    
    area_is_wider=max_y==min(max_y,max_x)
    scale_for_visualisation=min_resolution/min(max_x,max_y)
    
    aspect_ratio=max(max_x,max_y)/min(max_x,max_y)

    dst_img_shape=(min_resolution,int(min_resolution*aspect_ratio)) if not area_is_wider else (int(min_resolution*aspect_ratio),min_resolution)
    
    im_dst = cv2.warpPerspective(img,np.dot(np.array([[scale_for_visualisation,0,0],[0,scale_for_visualisation,0],[0,0,1]]),np.linalg.inv(h_world_to_img)), dst_img_shape)
    cv2.imshow("out",im_dst)
    cv2.waitKey(100000000)

    
    """try:
        data[cam_id]["Homography_3D_to_img"]=h_world_to_img.tolist()

    except Exception as e:
        print("creating data for cam_id {0}".format(cam_id),e)
        data[cam_id]=dict()
        data[cam_id]["Homography_3D_to_img"]=h_world_to_img.tolist()
        data[cam_id]["Homography_img_to_3D"]=np.linalg.inv(h_world_to_img).tolist()
    """
    if(data[cam_id]):
        intrinsicMatrix,_,_=getIntrinsicExtrinsicMatrix(data[cam_id])
        camera_to_plane_3D_transform,plane_to_camera_3D_transform=extractPoseParametersFromHomography(h_world_to_img,np.linalg.inv(intrinsicMatrix))
        data[cam_id]["Homography_3D_to_img"]=h_world_to_img.tolist()
        data[cam_id]["Homography_img_to_3D"]=np.linalg.inv(h_world_to_img).tolist()
        data[cam_id]["camera_to_plane_3D_transform"]=camera_to_plane_3D_transform.tolist()
        data[cam_id]["plane_to_camera_3D_transform"]=plane_to_camera_3D_transform.tolist()
    else:
        print("error: no data for this cam")
    print("camera_to_plane_3D",camera_to_plane_3D_transform,"plane_to_camera_3D_transform",plane_to_camera_3D_transform) 
    print("dot assert:",np.dot(camera_to_plane_3D_transform,plane_to_camera_3D_transform))
    
    write_config_to_file(data,filename)
    testExtrinsicParametersCalculation(cam_id,filename)
     

def testExtrinsicParametersCalculation(cam_id,filename):
    data=read_config_from_file(filename)
    print(data)
    intrinsicMatrix,_,_=getIntrinsicExtrinsicMatrix(data[cam_id])
    extrinsicMatrix=np.array(data[cam_id]["plane_to_camera_3D_transform"])
    #drawCoordinateSystems(intrinsicMatrix,extrinsicMatrix[0:3,:],img, _3Dpoints=np.array([[3,0,0],[15, 0, 0], [3, 12, 0], [3, 0, 6]]))
    drawCoordinateSystems(intrinsicMatrix,extrinsicMatrix[0:3,:],img, _3Dpoints=np.array([[0,0,0],[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
    cv2.imshow("img",img)
    cv2.waitKey(100000)

def drawCoordinateSystems(intrinsicMatrix, extrinsicMatrix, image, _3Dpoints=np.array([[0,0,0],[0.08, 0, 0], [0, 0.08, 0], [0, 0, 0.08]])):
    _2DprojectedPoints=intrinsicMatrix @ extrinsicMatrix @ np.transpose(np.hstack((_3Dpoints,np.ones([len(_3Dpoints),1]))))
    _2DprojectedPoints=_2DprojectedPoints/_2DprojectedPoints[2,:]
    _2DprojectedPoints=np.array(_2DprojectedPoints).astype(int)
    print(_2DprojectedPoints)
    cv2.arrowedLine(image, tuple(_2DprojectedPoints[0:2,0]), tuple(_2DprojectedPoints[0:2,1]), (0,255,0), 3, 0)
    cv2.arrowedLine(image, tuple(_2DprojectedPoints[0:2,0]), tuple(_2DprojectedPoints[0:2,2]),(255,0,0),3, 0)
    cv2.arrowedLine(image, tuple(_2DprojectedPoints[0:2,0]), tuple(_2DprojectedPoints[0:2,3]), (0,0,255),3, 0)

    cv2.putText(image,'X' ,tuple(_2DprojectedPoints[0:2,1]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0))
    cv2.putText(image,'Y',tuple(_2DprojectedPoints[0:2,2]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0))
    cv2.putText(image,'-Z', tuple(_2DprojectedPoints[0:2,3]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0))
    
    return image,_2DprojectedPoints


def orthogonalizationCram_Schmidt(R):
    
    R=np.array(R)
    r1= R[0,:]
    r2= R[1,:]
    r3= R[2,:]

    r1=r1/np.linalg.norm(r1)
    r2=r2-np.dot(r1,r2)*r1
    r2=r2/np.linalg.norm(r2)
    r3=np.cross(r1,r2)
    
    return np.vstack((np.vstack((r1,r2)),r3))

def orthogonalizeRotationMatrix(R):
    R=np.array(R)
    R1= R[:,0]
    R2= R[:,1]
    R3= R[:,2]

    C=(1/3.0)*(R1/np.linalg.norm(R1)+R2/np.linalg.norm(R2)+R3/np.linalg.norm(R3))
    #print C
    modC=np.linalg.norm(C)
    P=C/(np.sqrt(3)*modC)

    Xp=(modC/(np.sqrt(3)*np.dot(R1,C)))*R1
    Yp=(modC/(np.sqrt(3)*np.dot(R2,C)))*R2
    Zp=(modC/(np.sqrt(3)*np.dot(R3,C)))*R3

    Fixy=np.arccos(np.dot(Xp-P,Yp-P)/(np.linalg.norm(Xp-P)*np.linalg.norm(Yp-P)))
    Fiyz=np.arccos(np.dot(Yp-P,Zp-P)/(np.linalg.norm(Yp-P)*np.linalg.norm(Zp-P)))
    thetax=(2*Fixy+Fiyz-2*np.pi)/3

    v1=(Xp-P)/np.linalg.norm(Xp-P)
    v3=C/np.linalg.norm(C)
    v2=np.cross(v3,v1)
    
    v1=np.transpose(np.matrix(v1))
    v3=np.transpose(np.matrix(v3))
    v2=np.transpose(np.matrix(v2))

    Raux=np.matrix(np.hstack((np.hstack((v1,v2)),v3)))
    #Raux=np.matrix(np.hstack())##
    X=np.transpose(np.matrix(P))+np.sqrt(2/3.0)*Raux*np.transpose(np.matrix([np.cos(thetax),np.sin(thetax),0]))
    Y=np.transpose(np.matrix(P))+np.sqrt(2/3.0)*Raux*np.transpose(np.matrix([np.cos(thetax+(2*np.pi/3)),np.sin(thetax+(2*np.pi/3)),0]))
    Z=np.transpose(np.matrix(P))+np.sqrt(2/3.0)*Raux*np.transpose(np.matrix([np.cos(thetax+(4*np.pi/3)),np.sin(thetax +(4*np.pi/3)),0]))
    Rcorrected=np.hstack((np.hstack((X,Y)),Z))

    return Rcorrected 


def extractPoseParametersFromHomography(H,inverseOfIntrinsicMatrix):
    #H is the Homography matrix
    #translation is the traslation vector
    #retval is a vector containing the euler angles
    r1=inverseOfIntrinsicMatrix*np.transpose(np.matrix(H[:,0]))
    rr1=np.copy(r1)
    lambda1=np.sqrt((r1[0]**2+r1[1]**2+r1[2]**2))
    
    r2=inverseOfIntrinsicMatrix*np.transpose(np.matrix(H[:,1]))
    rr2=np.copy(r2) 
    lambda2=np.sqrt((r2[0]**2+r2[1]**2+r2[2]**2))
    
    lambdaa=0.5*(lambda1+lambda2)
    print("lambdaa",lambdaa)
    rr3=np.transpose(np.matrix(np.cross(np.array(rr1)[:,0],np.array(rr2)[:,0])))
    
    matrixToOrthogonalize=np.transpose(np.matrix(np.hstack((np.hstack((rr1,rr2)),rr3 ))))
    #Rortho=orthogonalizeRotationMatrix(matrixToOrthogonalize)
    Rortho=orthogonalizationCram_Schmidt(matrixToOrthogonalize)
    r1=np.array(Rortho[0,:])
    r2=np.array(Rortho[1,:])
    r3=np.array(Rortho[2,:])


    translation=inverseOfIntrinsicMatrix*np.transpose(np.matrix(H[:,2]))
    translation=np.array(translation/lambdaa)
    R=np.transpose(np.matrix(np.vstack((r1,r2,r3))))
    extrinsicMatrix=np.array(np.vstack((np.hstack((R,translation)),np.array([0,0,0,1]))))
    #roll,yaw,pitch=self.getEulerAnglesFrom3DTransform(R)
    #retval=(roll,yaw,pitch)
    ###################################################################################################
    inverseOfExtrinsicMatrix=np.zeros([4,4])
    inverseOfExtrinsicMatrix[0:3,0:3]=np.transpose(R)
    inverseOfExtrinsicMatrix[0:3,3]=np.array(np.matrix(-inverseOfExtrinsicMatrix[0:3,0:3])*(np.matrix(translation))).reshape(-1)
    inverseOfExtrinsicMatrix[3,3]=1
    #print np.matrix(inverseOfExtrinsicMatrix)*np.matrix(np.vstack((extrinsicMatrix,np.array([0,0,0,1]))))


    return inverseOfExtrinsicMatrix,extrinsicMatrix



if(__name__=="__main__"):
    parser = argparse.ArgumentParser(description='Python script for camera calibration')

    img=cv2.imread("calib.jpg")
    cv2.namedWindow("img",cv2.WINDOW_NORMAL)
    cv2.namedWindow("out",cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("img", get_pixel_coordinates)
    
    parser.add_argument('--cam_id',type=str, nargs=1,help='id of cam')
    parser.add_argument('--output',type=str, nargs=1,help='Output file to save the data')
    args = parser.parse_args()    
    #testExtrinsicParametersCalculation("0","cam_bay.json")
    estimate_homography(args.cam_id[0],args.output[0])
