import numpy as np
import cv2
import glob
from numpy import array
from kalman import Kalman
from tracker import Tracker
import imutils

# read camera information
with open('src/calibration/cameraInfo.txt', 'r') as f:
    cameraInfo = f.read()
cameraInfo = eval(cameraInfo)
mtx1 = cameraInfo['mtx1']
mtx2 = cameraInfo['mtx2']
T = cameraInfo['T']

path_str = 'rectified/Stereo_conveyor_without_occlusions/'
imgs_left = sorted(glob.glob(path_str+'left/*.png'))
imgs_right = sorted(glob.glob(path_str+'right/*.png'))
assert imgs_left,imgs_right


kalman = Kalman()
tracker1 = Tracker(80)
tracker2 = Tracker(80)


def extract_keypoints(img1, img2):

    p1 = []
    p2 = []
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None) #Keypoints and descriptions
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if len(kp1)!=0 and len(kp2)!=0:
        bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1,des2,k=2)

        # Apply ratio test
        good = []
        for m in matches:
            if len(m) == 2 and m[0].distance < 0.75*m[1].distance:
                good.append(m[0])

        # cv.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  
        cv2.imshow('match', img3)

        p1 = [kp1[m.queryIdx].pt for m in good]
        p2 = [kp2[m.trainIdx].pt for m in good]

        p1 = np.array(p1)
        p2 = np.array(p2)

    return p1,p2


def triangulate(p1, p2, mtx1, mtx2, T):
    #project the feature points to 3D with triangulation
     
    #projection matrix for Left and Right Image
    M_left = mtx1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    M_rght = mtx2 @ np.hstack((np.eye(3), T))

    # homogeneous coordinates
    p1_flip = np.vstack((p1.T, np.ones((1, p1.shape[0])) ))
    p2_flip = np.vstack((p2.T, np.ones((1, p2.shape[0])) ))

    P = cv2.triangulatePoints(M_left, M_rght, p1_flip[:2], p2_flip[:2])

    # Normalize homogeneous coordinates (P->Nx4  [N,4] is the normalizer/scale)
    P = P / P[3]
    land_points = P[:3]
    return land_points.T

frame_num = 0
for f1,f2 in zip(imgs_left,imgs_right):
    img1 = cv2.imread(f1)
    img2 = cv2.imread(f2)
    img1_size = img1.shape[:2][::-1]
    img2 = cv2.resize(img2, img1_size)
    text = "Undetected"

    if not (tracker1.train(img1) and tracker2.train(img2)):
        continue

    ret1, Object1, _,_ = tracker1.detect(img1)
    ret2, Object2, _,_ = tracker2.detect(img2)

    if ret1 and ret2:
        text = "detected"
        frame_num +=1
        (x1, y1, w1, h1) = cv2.boundingRect(Object1)
        (x2, y2, w2, h2) = cv2.boundingRect(Object2)
        # if beyond the conveyor, reset the kalman filter and frame_num
        if x1==0:
            kalman.reset() 
            frame_num = 0   
        if frame_num<25: 
            pass     
            #cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)
        else:    
            #cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi1 = img1[y1:y1+h1, x1:x1+w1]
            roi2 = img2[y2:y2+h2, x2:x2+w2]
            kp_roi1, kp_roi2 = extract_keypoints(roi1, roi2)

            #extract_keypoints(img1, img2, mtx1, mtx2, T)

            if len(kp_roi1)!=0 and len(kp_roi2)!=0:
                kp1=[]
                kp2=[]
                for i in range(len(kp_roi1)):
                    p1 = (int(kp_roi1[i][0]),int(kp_roi1[i][1]))
                    p2 = (int(kp_roi2[i][0]),int(kp_roi2[i][1]))
                    kp_img1 = (p1[0]+x1, p1[1]+y1)
                    kp_img2 = (p2[0]+x2, p2[1]+y2)
                    #cv2.circle(roi, kp, 2, (0, 255, 0),2)
                    cv2.circle(img1, kp_img1, 2, (0, 255, 0),2)
                    cv2.circle(img2, kp_img2, 2, (0, 255, 0),2)
                    kp1.append(kp_img1)
                    kp2.append(kp_img2)

                kp1 = np.array(kp1)
                kp2 = np.array(kp2)
                P = triangulate(kp1, kp2, mtx1, mtx2, T)   
                print(P)


    cv2.putText(img1, "Status: {}".format(text), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # cv2.imshow('thresh', thresh)
    # cv2.imshow('foreground mask',  fg_mask )   
    # cv2.imshow('img1',img1)
    # cv2.imshow('img2',img2)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27: break

cv2.destroyAllWindows()
