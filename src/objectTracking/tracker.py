import cv2
import imutils

class Tracker(object):

    def __init__(self, his):
        self.history = his 
        self.bs = cv2.createBackgroundSubtractorKNN(detectShadows=False) 
        self.bs.setHistory(self.history)
        self.trainTimes = 0

    def train(self,img):
        self.bs.apply(img)
        self.trainTimes += 1
        if self.trainTimes > self.history:
            return 1
        return 0 

    def detect(self, img):
        #img_blured = cv2.GaussianBlur(img,(21,21),0)
        fg_mask = self.bs.apply(img)
        thresh = cv2.threshold(fg_mask, 50, 255, cv2.THRESH_BINARY)[1] 
        
        thresh = cv2.dilate(thresh, None, iterations=4)
        #thresh = cv2.erode(thresh, None, iterations=3)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        maxArea = 0
        ret = 0 
        Object = 0           
        # loop over the contours
        for c in cnts:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) > maxArea and cv2.contourArea(c) > 2000:
                maxArea = cv2.contourArea(c)
                Object = c
                ret = 1

        return ret, Object, fg_mask, thresh     