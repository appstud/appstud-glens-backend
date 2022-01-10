import numpy as np
import cv2
from math import cos , sin
from utils.pose_estimation_utils import *
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from utils.constants import *
from utils.draw_utils_keypoints import * 
import logging
fig = Figure()
canvas = FigureCanvas(fig)
ax = fig.gca()




def draw_predictions(draw_image,data):

    FPS_dict={k:v for k,v  in data.items() if k.startswith("FPS") }
    clusters_img=np.zeros([640,420,3],dtype=np.uint8)

    data=data["data"]
    for i in data.keys():
        if (isinstance(data[i],dict) and ("bbox" in data[i].keys() or "person_bbox" in data[i].keys())):
            draw_image=draw_data(draw_image, data[i],i,intrinsicMatrix=None,draw_weak_perspective=True, **FPS_dict)
            if("landmarks" in data[i] and isinstance(data[i]["landmarks"],dict)):
                draw_image = draw_skel_and_kp(draw_image, np.array([1]), np.array([[1]*len(PART_NAMES)]), np.array([[data[i]["landmarks"][PART_NAMES[j]] for j in range(len(PART_NAMES))]]),[[]],[[]],min_pose_score=0.2, min_part_score=0.2)

    if("clusters" in data.keys()):
        clusters_img=draw_clusters(data["clusters"])
    return draw_image,clusters_img







def draw_clusters(clusters_dict,maxMarkerSize=10):
    color=['yellow', 'blue', 'brown', 'green','olive','cyan','grey']    
    ax.clear()
    for i ,key in enumerate(clusters_dict.keys()):
         
        data_points=np.array(clusters_dict[key]["members"])
        #ax.plot(data_points[:,0],data_points[:,1],"*")
        if(not data_points.shape[0]):
            continue
        if(key=="not_a_cluster"):
            ax.plot(-data_points[:,1]+12,data_points[:,0],"*",markersize=2,color="black")
            
        else:
            markerSize=2+int(clusters_dict[key]["dangerous_rate"]*maxMarkerSize)
            if(clusters_dict[key]["alarm"]):
                ax.plot(-data_points[:,1]+12,data_points[:,0],"*",markersize=maxMarkerSize,color="red")
            else:
                ax.plot(-data_points[:,1]+12,data_points[:,0],"*",markersize=markerSize,color=color[i%len(color)])
        ax.set_xlim(-4,20)        
        ax.set_ylim(-4,20)        
        #ax.text(0.0,0.0,"Test", fontsize=45)

    canvas.draw()       # draw the canvas, cache the renderer


    width, height = fig.get_size_inches() * fig.get_dpi()
    img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
    return img[:,:,::-1]
     
def draw_data(img, d,ID ,intrinsicMatrix=None,FPS_posenet=None,FPS_clustering=None, FPS_face_detection=None,FPS_object_detection=None, FPS_tracking=None,FPS_TF_MODELS=None,draw_weak_perspective=True):
    pen_thickness=min(img.shape[:2])//200
    font_scale=min(img.shape[:2])//900+1.2
    offset=int(20*pen_thickness)
    mult=1
    try:
        if(FPS_posenet): 
            cv2.putText(img,'FPS posenet:'+str(FPS_posenet), (5,10+mult*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0),thickness=pen_thickness ,lineType=cv2.LINE_AA)
            mult+=1

        if(FPS_face_detection): 
            cv2.putText(img,'FPS face:'+str(FPS_face_detection), (5,10+mult*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0),thickness=pen_thickness ,lineType=cv2.LINE_AA)
            mult+=1
        if(FPS_object_detection):
            cv2.putText(img,'FPS object detection:'+str(FPS_object_detection), (5,10+mult*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0),thickness=pen_thickness ,lineType=cv2.LINE_AA)
            mult+=1
        if(FPS_TF_MODELS):
            cv2.putText(img,'FPS tf models:'+str(FPS_TF_MODELS), (5,10+mult*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness=pen_thickness, lineType=cv2.LINE_AA)
            mult+=1
        if(FPS_tracking):
            cv2.putText(img,'FPS tracking:'+str(FPS_tracking), (5,10+mult*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness=pen_thickness ,lineType=cv2.LINE_AA)
            mult+=1 
        if(FPS_clustering):
            cv2.putText(img,'FPS clustering:'+str(FPS_clustering), (5,10+mult*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness=pen_thickness, lineType=cv2.LINE_AA)
    except:
        pass
    
    mult=0
    f=list(map(int,d["bbox"] if "bbox" in d else [])) # face bbox available
    p=list(map(int,d["person_bbox"] if "person_bbox" in d else [])) # person bbox available
    #l=landmarks[i]
    try:
        if( "landmarks" in d and not isinstance(d["landmarks"],dict)):
            l=np.array(d["landmarks"])
            img=cv2.polylines(img,np.int32(l.reshape((-1,1,2))),True,(0,0,255),3)
    except:
        pass
    
    if(len(p)==4):
        """try:
            if("person_pose_data" in d):
                cv2.putText(img,'pose:'+str(list(map(lambda x:"{:.2f}".format(x) if not isinstance(x,str) else x,d["person_pose_data"]))), (p[0]+p[2],p[1]+mult*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),thickness=pen_thickness  ,lineType=cv2.LINE_AA)
                mult+=1
             
        except Exception as e:
            pass
        """
        img=cv2.rectangle(img,(p[0],p[1]),(p[2]+p[0],p[3]+p[1]),(0,255,0),pen_thickness)   
        img=cv2.putText(img,'ID:'+str(ID), (p[0],p[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness=pen_thickness, lineType=cv2.LINE_AA)
        try:
            # for debugging 3D pose
            #cv2.putText(img,'person_pose_data:'+str(list(map(lambda x:"{:.2f}".format(x) if not isinstance(x,str) else x,d["person_pose_data"][0]))), (p[0],p[1]+mult*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),thickness=pen_thickness , lineType=cv2.LINE_AA)
            mult+=1
        except Exception as e:
            pass

    
    elif(len(f)==4):
        img=cv2.putText(img,'ID:'+str(ID), (f[0],f[1]), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0),thickness=pen_thickness,  lineType=cv2.LINE_AA)
    if(len(f)==4):
        img=cv2.rectangle(img,(f[0],f[1]),(f[2]+f[0],f[3]+f[1]),(0,0,255), pen_thickness)   
        #img=cv2.circle(img,(int(0.5*f[2]+f[0]),f[3]+f[1]),4,(0,0,255),3) 
        try:
            if(str(d["beard_mask"])=="mask on"):
                img=cv2.rectangle(img,(f[0],f[1]),(f[2]+f[0],f[3]+f[1]),(0,255,0), pen_thickness)   
            else:
                img=cv2.rectangle(img,(f[0],f[1]),(f[2]+f[0],f[3]+f[1]),(0,0,255), pen_thickness)   

            
        except:
            pass
        try: 
                        
            cv2.putText(img,'age:'+str(int(d["age"])), (f[0]+f[2],f[1]+mult*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),thickness=pen_thickness ,lineType=cv2.LINE_AA)
            mult+=1
        except Exception as e:
            pass

        try:
            cv2.putText(img,'sex:'+d["sex"], (f[0]+f[2],f[1]+mult*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness=pen_thickness, lineType=cv2.LINE_AA)
            mult+=1
        except Exception as e:
            pass

        try:
            cv2.putText(img,'glasses:'+d["glasses"], (f[0]+f[2],f[1]+mult*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),thickness=pen_thickness  ,lineType=cv2.LINE_AA)
            mult+=1
        except Exception as e:
            pass
        
        try:
            if("pose_data" in d):
                cv2.putText(img,'pose:'+str(list(map(lambda x:"{:.2f}".format(x) if not isinstance(x,str) else x,d["pose_data"]))), (f[0]+f[2],f[1]+mult*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),thickness=pen_thickness  ,lineType=cv2.LINE_AA)
                mult+=1
                if(not draw_weak_perspective):
                    extrinsicMatrix=getExtrinsicMatrix(*d["pose_data"])

                    img,_=drawCoordinateSystems(intrinsicMatrix, np.array(extrinsicMatrix), img, _3Dpoints=np.array([[0,0,0],[0.15, 0, 0], [0, 0.15, 0], [0, 0, -0.15]]))
                else:
                    img=draw_axis(img, float(d["pose_data"][3]), float(d["pose_data"][4]), float(d["pose_data"][5]), tdx=f[0]+f[2]/2, tdy=f[1]+f[3]/2, size = f[2])
                    #img=draw_axis(img, d[5][0], d[5][1], d[5][2], tdx=l[0,0], tdy=l[0,1], size = 150)
                    #img=draw_axis(img, d[5][0], d[5][1], d[5][2], tdx=l[1,0], tdy=l[1,1], size = 150)
        except Exception as e:
            #cv2.putText(img,'pose:'+str(d["pose_data"]), (f[0]+f[2],f[1]+60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), lineType=cv2.LINE_AA)
            pass
        
       
        try:
            cv2.putText(img,'hair:'+str(d["hairColor"]), (f[0]+f[2],f[1]+mult*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255),thickness=pen_thickness , lineType=cv2.LINE_AA)
            mult+=1
        except Exception as e:
            pass
        try:
            cv2.putText(img,'beard_mask:'+str(d["beard_mask"]), (f[0]+f[2],f[1]+mult*offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness=pen_thickness ,lineType=cv2.LINE_AA)
            mult+=1
        except Exception as e:
            pass

        #cv2.putText(img,'mask:'+str(d[3]), (f[0]+f[2],f[1]+100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),thickness=2, lineType=cv2.LINE_AA)
    return img



def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Draw a point
def draw_point(img, p, color ) :
    cv2.circle( img, p, 2, color,thickness=5 )


# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        #print getBarycentricCoordinates(t,np.array([0,0]))
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, 0)
    return img


def drawCoordinateSystems(intrinsicMatrix, extrinsicMatrix, image, _3Dpoints=np.array([[0,0,0],[0.08, 0, 0], [0, 0.08, 0], [0, 0, 0.08]])):
    _2DprojectedPoints=intrinsicMatrix.dot(extrinsicMatrix).dot(np.transpose(np.hstack((_3Dpoints,np.ones([len(_3Dpoints),1])))))
    _2DprojectedPoints=_2DprojectedPoints/_2DprojectedPoints[2,:]
    _2DprojectedPoints=np.array(_2DprojectedPoints).astype(int)
    
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

    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x1),int(y1)),(0,0,255),3)
    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x2),int(y2)),(255,0,0),3)
    cv2.arrowedLine(img, (int(tdx), int(tdy)), (int(x3),int(y3)),(0,255,0),2)
    
    return img


