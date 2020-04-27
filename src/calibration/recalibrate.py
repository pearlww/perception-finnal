'''
    Recalibrate based on undistortted images
'''
import numpy as np
import cv2
import glob
import os
from matplotlib import pyplot as plt


path_str = 'undistorted/Stereo_calibration_images/'

imgs1_path = glob.glob(path_str+'left/*')
imgs2_path = glob.glob(path_str+'right/*')
assert imgs1_path
assert imgs2_path

#Implement the number of vertical and horizontal corners
nb_vert = 9
nb_hori = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nb_hori*nb_vert,3), np.float32)
objp[:,:2] = np.mgrid[0:nb_vert,0:nb_hori].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints1 = [] # 2d points in image plane.
imgpoints2 = [] # 2d points in image plane.


for f1,f2 in zip(imgs1_path, imgs2_path):
    img1 = cv2.imread(f1)
    img2 = cv2.imread(f2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #Implement findChessboardCorners here
    ret1, corners1 = cv2.findChessboardCorners(gray1,(nb_vert,nb_hori), None)
    ret2, corners2 = cv2.findChessboardCorners(gray2,(nb_vert,nb_hori), None)

    # If found, add object points, image points (after refining them)
    if ret1 == True and ret2 == True:
        objpoints.append(objp)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)

#         # Draw and display the corners
#         img = cv2.drawChessboardCorners(img1, (nb_vert,nb_hori), corners1,ret1)
#         cv2.imshow('img1',img)
#         cv2.waitKey(100)
#         img = cv2.drawChessboardCorners(img2, (nb_vert,nb_hori), corners2,ret2)
#         cv2.imshow('img2',img)
#         cv2.waitKey(100)

# cv2.destroyAllWindows()

imagesize = gray1.shape[::-1]
_, mtx1, dist1,_ ,_ = cv2.calibrateCamera(objpoints, imgpoints1, imagesize, None, None)
_, mtx2, dist2,_ ,_= cv2.calibrateCamera(objpoints, imgpoints2, imagesize, None, None)


stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

ret, mtx1, dist1, mtx2, dist2, R, T,_ ,_ = cv2.stereoCalibrate(
    objpoints, 
    imgpoints1, 
    imgpoints2, 
    mtx1, dist1, 
    mtx2, dist2, 
    imagesize,
    flags=cv2.CALIB_FIX_INTRINSIC,
    criteria = stereocalib_criteria
    )

if ret:
    # camera information
    cameraInfo = {'mtx1':mtx1, 'dist1':dist1, 'mtx2':mtx2, 'dist2':dist2, 'R':R, 'T':T}
    with open('src/cameraInfo.txt','w') as f:
        f.write(str(cameraInfo))
else:
    print("calibration fail")     