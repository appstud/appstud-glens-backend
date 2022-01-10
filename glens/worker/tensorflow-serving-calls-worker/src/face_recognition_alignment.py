import numpy as np
import cv2

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
OUTER_EYES_AND_NOSE = [36, 45, 33]



def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated, M


###use for lightCNN
def alignRotationAndScaleLightCNN( img,faceROIs=[], landmarks=[], face_detection="mtcnn",ec_mc_y=48,crop_size=128,ec_y=48):
    """img must be RGB!!!!
    """
    #aligned=cropFaceForPoseEstimation(img,faceROIs,ad=0.3,output_shape=(200,200))
    #aligned=list(map(lambda x: cv2.cvtColor(x,cv2.COLOR_BGR2GRAY),aligned))
    aligned=len(landmarks)*[cv2.cvtColor(img,cv2.COLOR_BGR2RGB)]
    #aligned=len(landmarks)*[img]
    
    for i,l in enumerate(landmarks):
        if(face_detection=="mtcnn"):
            eye_l=l[0,:].astype(np.float32)
            eye_r=l[1,:].astype(np.float32)
            mouth_l=l[3,:].astype(np.float32)
            mouth_r=l[4,:].astype(np.float32)
        elif(face_detection=="pose_estimation"):
            ### getting some estimate for mouth position in case of pose estimation from eyes and nose positions
            eye_l=np.array(l["leftEye"],dtype=np.float32)[::-1]
            eye_r=np.array(l["rightEye"],dtype=np.float32)[::-1]
            nose=np.array(l["nose"],dtype=np.float32)[::-1]
            line_nose_mid_eye=nose-(eye_l+eye_r)/2
            mouth=(eye_l+eye_r)/2.0+2.0*line_nose_mid_eye
            mouth_l=mouth ### not correct but all I care for is the midpoint to be compatible with the rest of this function
            mouth_r=mouth ### not correct but all I care for is the midpoint to be compatible with the rest of this function
        else:
            ##check indices later
            eye_l=((l[42,:]+l[45,:])/2).astype(np.float32)
            eye_r=((l[39,:]+l[36,:])/2).astype(np.float32)
            mouth_l=l[54,:].astype(np.float32)
            mouth_r=l[48,:].astype(np.float32)


        ang_tan=(eye_l[1]-eye_r[1])/(eye_l[0]-eye_r[0])
        ang=np.arctan(ang_tan)*(180/np.pi)
        
        center=(faceROIs[i][0]+faceROIs[i][2]//2,faceROIs[i][1]+faceROIs[i][3]//2)
        aligned[i],M=rotate(aligned[i],ang,center=center)
        
        eyec=cv2.transform(np.array([[((eye_l+eye_r)/2.0)]]),M)[0][0]
        mouthc=cv2.transform(np.array([[((mouth_l+mouth_r)/2.0)]]),M)[0][0]
        
        h,w,_=aligned[i].shape
        resize_scale=ec_mc_y/(mouthc[1]-eyec[1])
        aligned[i]=cv2.resize(aligned[i],(int(resize_scale*w),int(resize_scale*h)))
         
        h,w,_=aligned[i].shape
        eyec2=(eyec*np.array([resize_scale, resize_scale])).astype(int)
        

        left=int(eyec2[0]-crop_size//2)
        right=int(left+crop_size-1)
        top=int(eyec2[1]-ec_y)
        bottom=int(top+crop_size-1)

        crop_y=max(0,top)
        crop_y_end=min(bottom,h)
        crop_x=max(0,left)
        crop_x_end=min(right,w)
        
        #aligned[i]= cv2.copyMakeBorder(aligned[i][crop_y:crop_y_end + 1, crop_x:crop_x_end + 1, :],0 if top>=0 else -top, 0 if bottom <=h-1 else bottom-(h-1) ,0 if left>=0 else -left ,0 if right < w-1 else right-(w-1),borderType=cv2.BORDER_CONSTANT)
        aligned[i]= cv2.copyMakeBorder(aligned[i][crop_y:crop_y_end + 1, crop_x:crop_x_end + 1, :],0 if top>=0 else -top, 0 if bottom <=h-1 else bottom-(h-1) ,0 if left>=0 else -left ,0 if right < w-1 else right-(w-1),borderType=cv2.BORDER_CONSTANT)

    return aligned



def align(imgDim, rgbImg, landmarks=[], landmarkIndices=OUTER_EYES_AND_NOSE,face_detection="mtcnn"):
    
    assert imgDim is not None
    assert rgbImg is not None
    assert landmarkIndices is not None

    npLandmarkIndices = np.array(landmarkIndices)
    alignedFaces=[]       
    
    if(face_detection=="mtcnn"):
        centers_eye=(imgDim*MINMAX_TEMPLATE[[42,39],:]+imgDim*MINMAX_TEMPLATE[[45,36],:])/2.0
        #bottom_lip=imgDim*MINMAX_TEMPLATE[57,:]
        nose=imgDim*MINMAX_TEMPLATE[33,:]
        target_pts=np.vstack((centers_eye,nose))
    else:
        target_pts=imgDim * MINMAX_TEMPLATE[npLandmarkIndices]

    for l in landmarks:
        if(face_detection=="mtcnn"):
            input_pts=l[0:3,:].astype(np.float32)
            
            #lips_mean=(l[3,:]+l[4,:])/2.0
            #input_pts=np.vstack((input_pts,lips_mean))
            H = cv2.getAffineTransform(input_pts.astype(np.float32),target_pts.astype(np.float32))
        else:
            H = cv2.getAffineTransform(l[npLandmarkIndices,:],target)
        alignedFaces.append(cv2.warpAffine(rgbImg, H, (imgDim, imgDim)).astype(np.uint8))

    return alignedFaces

