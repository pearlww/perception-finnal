import numpy as np
import cv2
import glob
import os
from calib_utils import *
from matplotlib import pyplot as plt


#------------------------ calibrate and undistort -----------------------------#
path_str = 'raw/Stereo_calibration_images/'
grid_size = 0.336  #33.6 mm
mtx1, dist1, mtx2, dist2, _,_,_ = calibrate(path_str,grid_size)

path_str = 'raw/Stereo_calibration_images/'
output_path = 'undistorted/Stereo_calibration_images/'
undistort(path_str+'left*.png', output_path +'left/', mtx1, dist1)
undistort(path_str+'right*.png', output_path + 'right/', mtx2, dist2)


path_str = 'raw/Stereo_conveyor_with_occlusions/'
output_path = 'undistorted/Stereo_conveyor_with_occlusions/'
undistort(path_str+'left/*', output_path +'left/', mtx1, dist1)
undistort(path_str+'right/*', output_path + 'right/', mtx2, dist2)

path_str = 'raw/Stereo_conveyor_without_occlusions/'
output_path = 'undistorted/Stereo_conveyor_without_occlusions/'
undistort(path_str+'left/*', output_path +'left/', mtx1, dist1)
undistort(path_str+'right/*', output_path + 'right/', mtx2, dist2)


#-------------------- Recalibrate based on undistortted images------------------------------#
path_str = 'undistorted/Stereo_calibration_images/'
mtx1, dist1, mtx2, dist2, objpoints, imgpoints1, imgpoints2 = calibrate(path_str)

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
imagesize = (1280, 720)

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

# save camera information
if ret:
    cameraInfo = {'mtx1':mtx1, 'dist1':dist1, 'mtx2':mtx2, 'dist2':dist2, 'R':R, 'T':T}
    with open('src/cameraInfo.txt','w') as f:
        f.write(str(cameraInfo))
else:
    print("calibration fail")  

# --------------------------------- Rectify ------------------------------------------#
R1, R2, P1, P2, Q, roi1, roi2= cv2.stereoRectify(
    mtx1, dist1, 
    mtx2, dist2,
    imagesize, 
    R, 
    T,
    0) # 0=full crop, 1=no crop

path_str = 'undistorted/Stereo_conveyor_with_occlusions/'
output_path = 'rectified/Stereo_conveyor_with_occlusions/'
rectify(path_str+'left/', output_path +'left/', mtx1, dist1, R1, P1, roi1)
rectify(path_str+'right/', output_path +'right/', mtx2, dist2, R2, P2, roi2)

path_str = 'undistorted/Stereo_conveyor_without_occlusions/'
output_path = 'rectified/Stereo_conveyor_without_occlusions/'
rectify(path_str+'left/', output_path +'left/', mtx1, dist1, R1, P1, roi1)
rectify(path_str+'right/', output_path +'right/', mtx2, dist2, R2, P2, roi2)


#-------------------------- draw the lines --------------------------------------------#
left_path = 'rectified/Stereo_conveyor_without_occlusions/left/1000.png'
right_path = 'rectified/Stereo_conveyor_without_occlusions/right/1000.png'
imgl = cv2.imread(left_path)
imgr = cv2.imread(right_path)
calLines(imgl, imgr, numMatches=300, plot=True)