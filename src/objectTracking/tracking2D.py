import numpy as np
import cv2
import glob
import sys
sys.path.append('src/utils')
from tracker import Tracker
from sklearn.svm import SVC
from classifier import Classifier

# load the classifier
clf = Classifier()
clf.loadModel('src/classification/Classifier.pkl')

path_str = 'rectified/Stereo_conveyor_without_occlusions/'
imgs_path = sorted(glob.glob(path_str+'right/*.png'))
assert imgs_path

tracker = Tracker(80)

for fname in imgs_path:

    img = cv2.imread(fname)
    text = "Undetected"
    if not tracker.train(img):
        continue

    ret, Object,fg_mask, thresh = tracker.detect(img)

    if ret:
        (x, y, w, h) = cv2.boundingRect(Object)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = img[y:y+h, x:x+w]
        y_predict = clf.predict(roi)
        text = y_predict
        cv2.putText(img, "class: {}".format(text), (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)   
        
      
    cv2.imshow('thresh', thresh)
    cv2.imshow('foreground mask',  fg_mask )   
    cv2.imshow('conveyor_with_occlusions',img)
    cv2.imshow('roi', roi)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: break
