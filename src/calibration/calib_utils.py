import numpy as np
import cv2
import glob
import os
from numpy import array
from matplotlib import pyplot as plt


def calibrate(path_str, grid_size):
    imgs1_path = glob.glob(path_str+'left*.png')
    imgs2_path = glob.glob(path_str+'right*.png')
    assert imgs1_path
    assert imgs2_path

    #Implement the number of vertical and horizontal corners
    nb_vert = 9
    nb_hori = 6

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0)
    objp = np.zeros((nb_hori*nb_vert,3), np.float32)
    objp[:,:2] = np.mgrid[0:nb_vert, 0:nb_hori].T.reshape(-1,2)
    # multiple with grid_size of checkerboard
    objp = objp * grid_size
	
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints1 = [] # 2d points in image plane.
    imgpoints2 = [] # 2d points in image plane.

    # 原图的标号方法 (0001,0002, ...)可以自动排序
    for f1,f2 in zip(imgs1_path, imgs2_path):
        img1 = cv2.imread(f1)
        img2 = cv2.imread(f2)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        #Implement findChessboardCorners here
        ret1, corners1 = cv2.findChessboardCorners(gray1,(nb_vert,nb_hori), None)
        ret2, corners2 = cv2.findChessboardCorners(gray2,(nb_vert,nb_hori), None)

        # If found, add object points, image points (after refining them)
        # must add left and right together !!!
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

    return mtx1, dist1, mtx2, dist2, objpoints,imgpoints1,imgpoints2


def undistort(input_path, output_path, mtx, dist):
    num=0

    # do sort here to get ordered undistort images
    imgs_path = sorted(glob.glob(input_path))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # the function 'getOptimalNewCameraMatrix' not work well for cameras with large radial distortion
    # more information to see: https://answers.opencv.org/question/28438/undistortion-at-far-edges-of-image/
    for fname in imgs_path:
        img = cv2.imread(fname)
        #newcameramtx1, roi1 = cv2.getOptimalNewCameraMatrix(mtx1,dist1,imagesize,1,imagesize)
        dst = cv2.undistort(img, mtx, dist)
        cv2.imwrite(output_path + str(num) + '.png', dst)
        num+=1


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



def drawlines(img1, img2, lines, pts1, pts2):
    '''
    draw epipolar lines
    '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0,y0),(x1,y1),color,2)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

def calLines(imgl, imgr, numMatches=300, plot=False):
    '''
    draw epipolar lines
    '''
    imgl = cv2.cvtColor(imgl,cv2.COLOR_RGB2GRAY)
    imgr = cv2.cvtColor(imgr, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d_SIFT.create()
    kp1, des1 = sift.detectAndCompute(imgl, None)
    kp2, des2 = sift.detectAndCompute(imgr, None)
    matcher = cv2.FlannBasedMatcher()
    match = matcher.match(des1, des2)
    # select the most relevant matches
    match = sorted(match, key=lambda x:x.distance)
    pts1 = []
    pts2 = []
    for m in match[:numMatches]:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    # select inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Find epilines corresponding to points in right image (second image) and drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2 ,F)
    lines1 = lines1.reshape(-1, 3)
    imgl2, imgr1 = drawlines(imgl, imgr, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    imgr2, imgl1 = drawlines(imgr, imgl, lines2, pts2, pts1)

    if plot:
        plt.figure(figsize=(15,15))
        plt.subplot(221), plt.imshow(imgl1), plt.title('left keypoints')
        plt.subplot(222), plt.imshow(imgr1), plt.title('right keypoints')
        plt.subplot(223), plt.imshow(imgl2), plt.title('left epipolar lines')
        plt.subplot(224), plt.imshow(imgr2), plt.title('right epipolar lines')
        plt.show()
