import numpy as np
import cv2
import glob
from kalman import Kalman
from tracker import Tracker

path_str = 'rectified/Stereo_conveyor_with_occlusions/'
imgs_path = sorted(glob.glob(path_str+'right/*.png'))
assert imgs_path


kalman = Kalman()
tracker = Tracker(80)


def extract_keypoints(img):
    #kps = extract_keypoints(roi, img)
    orb = cv2.ORB_create()
    kps, des = orb.detectAndCompute(img, None)
    kps = np.float32([kp.pt for kp in kps])

    kp=[]
    for i in range(len(kps)):
        kp_roi = (int(kps[i][0]),int(kps[i][1]))
        kp_img = (kp_roi[0]+x,kp_roi[1]+y)
        #cv2.circle(roi, kp, 2, (0, 255, 0),2)
        #cv2.circle(img, kp_img, 2, (0, 255, 0),2)
        kp.append(kp_img)

    kp = np.float32(kp)    
    kp = np.array(kp)

    return kp

def featureTracking(img_1, img_2, p1):

    # params = dict(winSize=(21, 21),
    #              maxLevel=3,
    #              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    # the p1 must be float32, not float64
    p2, status, _ = cv2.calcOpticalFlowPyrLK(img_1, img_2, p1, None)

    index= np.array([ i==1 for i in status ])
    #print((index == False).any())
    
    #p1 = p1[index.flatten()]
    p2 = p2[index.flatten()]
    return p2

pre_img = cv2.imread(imgs_path[0])
pre_kp = np.array([])
frame_num = 0
for fname in imgs_path:

    img = cv2.imread(fname)
    img_size = img.shape[:2][::-1]
    text = "Undetected"

    if not tracker.train(img):
        continue

    ret, Object,fg_mask, thresh = tracker.detect(img)

    if ret:
        text = "detected"
        frame_num +=1
        (x, y, w, h) = cv2.boundingRect(Object)
        # if beyond the conveyor, reset the kalman filter and frame_num
        if x==0:
            kalman.reset() 
            frame_num = 0   

        if frame_num<25:      
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi = img[y:y+h, x:x+w]
            kp = extract_keypoints(roi)
            for pi in kp:
                p = (int(pi[0]),int(pi[1]))
                cv2.circle(img, p , 2, (0, 255, 0),2)            
            pre_kp = kp
            pre_img = img
        else:    
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if len(pre_kp)!=0:
                tracked_kp = featureTracking(pre_img, img, pre_kp)
                for pi in tracked_kp:
                    p = (int(pi[0]),int(pi[1]))
                    cv2.circle(img, p , 2, (0, 255, 0),2)


            # Z = np.array([[cx],[cy]])
            # kalman.update(Z)

            pre_img = img
            pre_kp = tracked_kp

    ### Predict the next state
    X = kalman.predict()
    pre_pos = (int(X[0]), int(X[3]))
    cv2.rectangle(img, (X[0]-50, X[3]-50), (X[0]+50, X[3]+50), (255, 0, 0), 2)
    #cv2.circle(img, pre_pos, 5, (0, 0, 255),2)
    if pre_pos[0]<0 or pre_pos[0]>img_size[0] or pre_pos[1]<0 or pre_pos[1]>img_size[1]:
        kalman.reset() 

    cv2.putText(img, "Status: {}".format(text), (10, 20),
    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    cv2.imshow('thresh', thresh)
    cv2.imshow('foreground mask',  fg_mask )   
    cv2.imshow('conveyor_with_occlusions',img)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27: break

cv2.destroyAllWindows()
