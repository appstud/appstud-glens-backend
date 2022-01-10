import numpy as np
import cv2
from math import cos , sin

def drawCoordinateSystems(intrinsicMatrix, extrinsicMatrix, image, _3Dpoints=np.array([[0,0,0],[0.08, 0, 0], [0, -0.08, 0], [0, 0, 0.08]])):
    _2DprojectedPoints=np.matrix(intrinsicMatrix)*np.matrix(extrinsicMatrix)*np.transpose(np.matrix(np.hstack((_3Dpoints,np.ones([len(_3Dpoints),1])))))
    _2DprojectedPoints=_2DprojectedPoints/_2DprojectedPoints[2,:]
    cv2.arrowedLine(image, tuple(_2DprojectedPoints[0:2,0]), tuple(_2DprojectedPoints[0:2,1]), (0,255,0), 3, 0)
    cv2.arrowedLine(image, tuple(_2DprojectedPoints[0:2,0]), tuple(_2DprojectedPoints[0:2,2]),(255,0,0),3, 0)
    cv2.arrowedLine(image, tuple(_2DprojectedPoints[0:2,0]), tuple(_2DprojectedPoints[0:2,3]), (0,0,255),3, 0)
    
    cv2.putText(image,'X' ,tuple(_2DprojectedPoints[0:2,1]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0))
    cv2.putText(image,'Y',tuple(_2DprojectedPoints[0:2,2]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0))
    cv2.putText(image,'-Z', tuple(_2DprojectedPoints[0:2,3]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0))
    return image,_2DprojectedPoints


def plot_pose_cube(img, yaw, pitch, roll, tdx=None, tdy=None, size=150.):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]

    p = pitch * np.pi / 180
    y = (-yaw * np.pi / 180)
    r = roll * np.pi / 180
    if tdx != None and tdy != None:
        face_x = tdx - 0.50 * size
        face_y = tdy - 0.50 * size
    else:
        height, width = img.shape[:2]

        face_x = width / 2 - 0.5 * size
        face_y=0
        #face_x = width / 2 - 0.5 * size
        #face_y = height / 2 - 0.5 * size

    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = 4*size * (sin(y)) + face_x
    y3 = 4*size * (-cos(y) * sin(p)) + face_y

    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img



def plot_pose_cube_full_perspective(intrinsicMatrix,extrinsicMatrix,img,distCoeff=None, _3Dpoints=np.array([[0,0,0],[0.08, 0, 0], [0, 0.08, 0], [0, 0, -0.20]])):
    # Input is a cv2 image
    # pose_params: (pitch, yaw, roll, tdx, tdy)
    # Where (tdx, tdy) is the translation of the face.
    # For pose we have [pitch yaw roll tdx tdy tdz scale_factor]
    #_3Dpoints=_3Dpoints-np.array([0,0.16,0])
    #_2DprojectedPoints=np.matrix(intrinsicMatrix)*np.matrix(extrinsicMatrix)*np.transpose(np.matrix(np.hstack((_3Dpoints,np.ones([len(_3Dpoints),1])))))
    #_2DprojectedPoints=_2DprojectedPoints/_2DprojectedPoints[2,:]
    _2DprojectedPoints=cv2.projectPoints(_3Dpoints,cv2.Rodrigues(extrinsicMatrix[0:3,0:3])[0],extrinsicMatrix[:,3],intrinsicMatrix,distCoeff)
    _2DprojectedPoints=_2DprojectedPoints[0][:,0,:]
    face_x,face_y=tuple(_2DprojectedPoints[0,:])
    x1,y1=_2DprojectedPoints[1,:]
    x2,y2=_2DprojectedPoints[2,:]
    x3,y3=_2DprojectedPoints[3,:]

    """
    face_x,face_y=tuple(_2DprojectedPoints[0:2,0])
    x1,y1=_2DprojectedPoints[0:2,1]
    x3,y3=_2DprojectedPoints[0:2,2]
    x2,y2=_2DprojectedPoints[0:2,3]
    """
    """
    x1 = size * (cos(y) * cos(r)) + face_x
    y1 = size * (cos(p) * sin(r) + cos(r) * sin(p) * sin(y)) + face_y
    x2 = size * (-cos(y) * sin(r)) + face_x
    y2 = size * (cos(p) * cos(r) - sin(p) * sin(y) * sin(r)) + face_y
    x3 = 4*size * (sin(y)) + face_x
    y3 = 4*size * (-cos(y) * sin(p)) + face_y
    """
    # Draw base in red
    cv2.line(img, (int(face_x), int(face_y)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(face_x), int(face_y)), (int(x2),int(y2)),(0,0,255),3)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x1-face_x),int(y2+y1-face_y)),(0,0,255),3)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x2-face_x),int(y1+y2-face_y)),(0,0,255),3)
    # Draw pillars in blue
    cv2.line(img, (int(face_x), int(face_y)), (int(x3),int(y3)),(255,0,0),2)
    cv2.line(img, (int(x1), int(y1)), (int(x1+x3-face_x),int(y1+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2), int(y2)), (int(x2+x3-face_x),int(y2+y3-face_y)),(255,0,0),2)
    cv2.line(img, (int(x2+x1-face_x),int(y2+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(255,0,0),2)
    # Draw top in green
    cv2.line(img, (int(x3+x1-face_x),int(y3+y1-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x2+x3-face_x),int(y2+y3-face_y)), (int(x3+x1+x2-2*face_x),int(y3+y2+y1-2*face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x1-face_x),int(y3+y1-face_y)),(0,255,0),2)
    cv2.line(img, (int(x3), int(y3)), (int(x3+x2-face_x),int(y3+y2-face_y)),(0,255,0),2)

    return img



def drawCSWeakPerspective(image,alpha_u,alpha_v,c_x,c_y,extrinsicMatrix,_3DPoints=np.array([[0,0,0],[0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.2]])):
    
    _2DPoints=np.dot(extrinsicMatrix,np.vstack((_3DPoints.T,np.ones([1,_3DPoints.shape[0]]))))
    _2DPoints[0,:]=alpha_u*(_2DPoints[0,:]/float(extrinsicMatrix[2,3]))+c_x
    _2DPoints[1,:]=alpha_v*(_2DPoints[1,:]/float(extrinsicMatrix[2,3]))+c_y
    
    cv2.line(image, tuple(_2DPoints[0:2,0]), tuple(_2DPoints[0:2,1]), (255,255,0), 3, 0)
    cv2.line(image, tuple(_2DPoints[0:2,0]), tuple(_2DPoints[0:2,2]),(255,0,255),3, 0)
    cv2.line(image, tuple(_2DPoints[0:2,0]), tuple(_2DPoints[0:2,3]), (0,255,255),3, 0)
    cv2.putText(image,'X' ,tuple(_2DPoints[0:2,1]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0))
    cv2.putText(image,'Y',tuple(_2DPoints[0:2,2]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0))
    cv2.putText(image,'-Z', tuple(_2DPoints[0:2,3]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0))

    return image



def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 250):

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width //2
        tdy = height // 2

   #print(yaw,roll,pitch)
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(0,255,0),3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(255,0,0),2)

    return img


