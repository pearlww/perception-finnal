import numpy as np
import cv2
import glob
from kalman import Kalman
from tracker import Tracker

path_str = 'rectified/Stereo_conveyor_with_occlusions/'
imgs_path = sorted(glob.glob(path_str+'right/*.png'))
assert imgs_path

kalman = Kalman()
tracker = Tracker(his = 80)


frame_num = 0

for fname in imgs_path:
    img = cv2.imread(fname)
    img_size = img.shape[:2][::-1]
    text = "Undetected"

    # train the  tracker with pure background
    if not tracker.train(img):
        continue

    ret, Object,fg_mask, thresh = tracker.detect(img)

    # if object is detected
    if ret:
        text = "detected"
        frame_num +=1
        (x, y, w, h) = cv2.boundingRect(Object)

        # if beyond the conveyor, reset the kalman filter and frame_num
        if x==0:
            kalman.reset() 
            frame_num = 0   
        if frame_num<25:      
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2) # red rectangle
        else:    
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) # green rectangle

            # find the rearest point
            maxx = -1
            for point in Object:
                if point[0][0]>maxx:
                    maxx = point[0][0]
                    z = point[0]

            # use this point as measurement.
            cv2.circle(img,(z[0],z[1]) , 2, (0, 255, 0),2)
            Z = np.array([[z[0]],[z[1]]])
            kalman.update(Z)
            kalman.predict()

    # if object is not detected         
    else:
        # Predict and show the position
        X = kalman.predict()
        pre_pos = (int(X[0]), int(X[3]))
        cv2.rectangle(img, (X[0]-150, X[3]-50), (X[0], X[3]+50), (255, 0, 0), 2)
        cv2.circle(img, pre_pos, 5, (0, 0, 255),2)

        #if beyond the image, reset the kalman filter 
        if pre_pos[0]<0 or pre_pos[0]>img_size[0] or pre_pos[1]<0 or pre_pos[1]>img_size[1]:
            kalman.reset() 

    cv2.putText(img, "Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow('thresh', thresh)
    cv2.imshow('foreground mask',  fg_mask )   
    cv2.imshow('conveyor_with_occlusions',img)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27: break

cv2.destroyAllWindows()
