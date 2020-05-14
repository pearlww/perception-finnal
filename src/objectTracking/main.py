import numpy as np
import cv2
import glob
from numpy import array

from kalman import Kalman
from tracker import Tracker
from classifier import Classifier
from track_utils import *

# read camera information
with open('src/calibration/cameraInfo.txt', 'r') as f:
    cameraInfo = f.read()
cameraInfo = eval(cameraInfo)
mtx1 = cameraInfo['mtx1']
mtx2 = cameraInfo['mtx2']
T = cameraInfo['T']

# load the classifier
clf = Classifier('training_dataset/dataset1/processed/*','training_dataset/dataset1/lable.txt')
#clf = Classifier('training_dataset/dataset2/video/*','training_dataset/dataset2/lable.txt')
clf.loadModel('src/classification/Classifier.pkl')

path_str = 'rectified/Stereo_conveyor_with_occlusions/'
imgs_left = sorted(glob.glob(path_str+'left/*.png'))
imgs_right = sorted(glob.glob(path_str+'right/*.png'))
assert imgs_left,imgs_right

kalman1 = Kalman()
kalman2 = Kalman()
tracker1 = Tracker(80)
tracker2 = Tracker(80)
tracker1.train(path_str+'left/*.png')
tracker2.train(path_str+'right/*.png')
#-------------------------------------------------------------------------------#

for f1,f2 in zip(imgs_left,imgs_right):
    img1 = cv2.imread(f1)
    img2 = cv2.imread(f2)
    img_size = img1.shape[:2][::-1]
    img2 = cv2.resize(img2, img_size)

    ret1, roi1, center1,rearest1  = tracker1.detect(img1)
    ret2, roi2, center2,rearest2  = tracker2.detect(img2)

    # if object is detected
    if ret1 and ret2:

        # if beyond the conveyor, reset the kalman filter
        if (center2[0]<250 or center2[0]>1100):
            kalman1.reset() 
            kalman2.reset()     
            cv2.rectangle(img1, roi1[0:2], roi1[2:4], (0, 0, 255), 2)
            cv2.rectangle(img2, roi2[0:2], roi2[2:4], (0, 0, 255), 2)
        else:
            cv2.rectangle(img1, roi1[0:2], roi1[2:4], (0, 255, 0), 2)
            cv2.rectangle(img2, roi2[0:2], roi2[2:4], (0, 255, 0), 2)
            cv2.circle(img1,(int(center1[0]),int(center1[1])) , 2, (0, 255, 0),2)
            cv2.circle(img2,(int(center2[0]),int(center2[1])) , 2, (0, 255, 0),2)

            # calculate 3D position
            #img1, img2, point3D = Track3D(img1,img2, roi1,roi2, center1, center2, mtx1, mtx2, T)
            P = triangulate(center1 , center2 , mtx1, mtx2, T)
            point3D = P[0]
            text = "Position: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(point3D[0]), float(point3D[1]),float(point3D[2]))
            cv2.putText(img2, text, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

            # do classification
            crop_img2 = img2[roi2[1]:roi2[3], roi2[0]:roi2[2]] # y:y+h, x:x+w
            #cv2.imshow("object", crop_img2)
            y_predict = clf.predictNewImage(crop_img2)
            cv2.putText(img2, "{}".format(y_predict), (roi2[0]+20, roi2[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  
            
            # Update kalman filter, use the rearest point as measurement.
            # z1 = int(center1[0]),int(center1[1])
            # z2 = int(center2[0]),int(center2[1])
            z1 = int(rearest1[0]),int(rearest1[1])
            z2 = int(rearest2[0]),int(rearest2[1])
            #cv2.circle(img1,(z1[0],z1[1]) , 2, (0, 255, 0),2)
            #cv2.circle(img2,(z2[0],z2[1]) , 2, (0, 255, 0),2)
            Z1 = np.array([[z1[0]],[z1[1]]])
            Z2 = np.array([[z2[0]],[z2[1]]])
            kalman1.update(Z1)
            kalman1.predict()
            kalman2.update(Z2)
            kalman2.predict()
                  
    # if object is undetected   
    else:
        # Predict and show the position
        X1 = kalman1.predict()
        X2 = kalman2.predict()
        pre_pos1 = np.array([X1[0], X1[3]]).T[0]
        pre_pos2 = np.array([X2[0], X2[3]]).T[0]
        cv2.circle(img1, (int(X1[0]), int(X1[3])), 10, (255, 255, 255), -1)
        cv2.circle(img2, (int(X2[0]), int(X2[3])), 10, (255, 255, 255), -1)

        P = triangulate(pre_pos1, pre_pos2, mtx1, mtx2, T) 
        point3D = P[0]  # choose the positive z
        text2 = "Position: x ={0:02f}m y = {1:02f}m z = {2:02f}m".format(float(point3D[0]), float(point3D[1]),float(point3D[2]))
        cv2.putText(img2, text2, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        #if beyond the image, reset the kalman filter 
        if pre_pos2[0]<0 or pre_pos2[0]>img_size[0] or pre_pos2[1]<0 or pre_pos2[1]>img_size[1]:
            kalman1.reset()
            kalman2.reset()  


    cv2.imshow('left img',img1)
    cv2.imshow('right img',img2)

    k = cv2.waitKey(5) & 0xFF
    if k == 27: break

cv2.destroyAllWindows()
