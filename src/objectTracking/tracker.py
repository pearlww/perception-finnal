import cv2
import imutils
import glob
import numpy as np

class Tracker(object):

    def __init__(self, his):
        self.history = his 
        self.bs = cv2.createBackgroundSubtractorKNN(detectShadows=False) 
        self.bs.setHistory(self.history)
        self.trainTimes = 0

    def train(self,img_path):
        paths = sorted(glob.glob(img_path))
        for file in paths:
            img = cv2.imread(file)
            self.bs.apply(img)
            self.trainTimes += 1
            if self.trainTimes > self.history:
                break

    def detect(self, img):
        #img_blured = cv2.GaussianBlur(img,(21,21),0)
        fg_mask = self.bs.apply(img)
        
        mask = cv2.dilate(fg_mask, None, iterations=10)
        mask = cv2.erode(mask, None, iterations=10)

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        
        ret = False 
        roi = []
        center = []
        rearest = []
        # loop over the contours
        maxArea = -1
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) > maxArea and cv2.contourArea(c) > 2000:
                maxArea = cv2.contourArea(c)
                Object = c
                ret = True
        if ret:           
            (x, y, w, h) = cv2.boundingRect(Object)
            roi = (x,y, x+w,y+h)
            center = np.array([x+w/2,y+h/2])

            maxx = -1
            for point in Object:
                if point[0][0]>maxx:
                    maxx = point[0][0]
                    rearest = point[0]

        return ret, roi, center,rearest 