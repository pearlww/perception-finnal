import numpy as np
import cv2
import glob
import os
from numpy import array
from matplotlib import pyplot as plt

# read camera information
with open('src/calibration/cameraInfo.txt', 'r') as f:
    cameraInfo = f.read()
cameraInfo = eval(cameraInfo)
mtx1 = cameraInfo['mtx1']
mtx2 = cameraInfo['mtx2']
dist1 = cameraInfo['dist1']
dist2 = cameraInfo['dist2']
R = cameraInfo['R']
T = cameraInfo['T']


def rectify(input_path, output_path, mtx, dist, R, P, roi):
    num=0

    # no order
    imgs_path = glob.glob(input_path + "*")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # the function 'cv2.initUndistortRectifyMap' not work well for cameras with large radial distortion
    # more information to see: https://answers.opencv.org/question/28438/undistortion-at-far-edges-of-image/
    for i in range(len(imgs_path)):
        # read by order
        img = cv2.imread(input_path + str(i)+ ".png")
        x,y,w,h = roi
        img  = img[y:y+h, x:x+w]
        croped_size = img.shape[:2][::-1]
        map = cv2.initUndistortRectifyMap(mtx, dist, R, P, croped_size, cv2.CV_32FC1)
        dst = cv2.remap(img, map[0], map[1], cv2.INTER_LINEAR)
        cv2.imwrite(output_path + f"{num:04d}.png", dst)
        num+=1


imagesize = (1280, 720)
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


# for line in range(int(new_right.shape[0]/20)):
#     new_left[line * 20, :] = (0, 0, 255)
#     new_right[line * 20, :] = (0, 0, 255)

# for line in range(int(undisleft.shape[0]/20)):
#     undisleft[line * 20, :] = (0, 0, 255)
#     undisright[line * 20, :] = (0, 0, 255)

# original_left = 'raw/Stereo_conveyor_without_occlusions/left/1585434284_949631929_Left.png'
# original_right = 'raw/Stereo_conveyor_without_occlusions/right/1585434284_949631929_right.png'
# undistorted_and_rectified_left =
# undistorted_and_rectified_right =

# plt.figure(figsize=(15,10))
# plt.subplot(221), plt.imshow(new_left), plt.title('original left')
# plt.subplot(222), plt.imshow(new_right), plt.title('original right')
# plt.subplot(223), plt.imshow(undisleft), plt.title('undistorted and rectified left')
# plt.subplot(224), plt.imshow(undisright), plt.title('undistorted and rectified right')
# plt.show()