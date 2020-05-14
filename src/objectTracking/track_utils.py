import cv2
import numpy as np

def extract_keypoints(img1, img2):

    p1 = []
    p2 = []
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None) #Keypoints and descriptions
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if len(kp1)!=0 and len(kp2)!=0:
        bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1,des2,k=2)

        # Apply ratio test
        good = []
        for m in matches:
            if len(m) == 2 and m[0].distance < 0.75*m[1].distance:
                good.append(m[0])

        # cv.drawMatchesKnn expects list of lists as matches.
        # img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)  
        # cv2.imshow('match', img3)

        p1 = [kp1[m.queryIdx].pt for m in good]
        p2 = [kp2[m.trainIdx].pt for m in good]

        p1 = np.array(p1)
        p2 = np.array(p2)

    return p1,p2


def triangulate(p1, p2, mtx1, mtx2, T):
    #project the feature points to 3D with triangulation
     
    #projection matrix for Left and Right Image
    M_left = mtx1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    M_rght = mtx2 @ np.hstack((np.eye(3), T))

    # homogeneous coordinates
    p1_flip = np.vstack((p1.T, np.ones((1, p1.shape[0])) ))
    p2_flip = np.vstack((p2.T, np.ones((1, p2.shape[0])) ))

    P = cv2.triangulatePoints(M_left, M_rght, p1_flip[:2], p2_flip[:2])

    # Normalize homogeneous coordinates (P->Nx4  [N,4] is the normalizer/scale)
    P = P / P[3]
    land_points = P[:3]
    return land_points.T



def Track3D(img1, img2, roi1,roi2, center1, center2, mtx1, mtx2, T):
    # calculate 3D position
    crop_img1 = img1[roi1[1]:roi1[3], roi1[0]:roi1[2]] # y:y+h, x:x+w
    crop_img2 = img2[roi2[1]:roi2[3], roi2[0]:roi2[2]] # y:y+h, x:x+w

    kp_roi1, kp_roi2 = extract_keypoints(crop_img1,crop_img2)
    if len(kp_roi1)!=0 and len(kp_roi2)!=0:
        kp1=[]
        kp2=[]
        for i in range(len(kp_roi1)):
            p1 = (int(kp_roi1[i][0]),int(kp_roi1[i][1]))
            p2 = (int(kp_roi2[i][0]),int(kp_roi2[i][1]))
            kp_img1 = (p1[0]+roi1[0], p1[1]+roi1[1])
            kp_img2 = (p2[0]+roi2[0], p2[1]+roi2[1])
            cv2.circle(img1, kp_img1, 2, (0, 255, 0),2)
            cv2.circle(img2, kp_img2, 2, (0, 255, 0),2)
            kp1.append(kp_img1)
            kp2.append(kp_img2)
        kp1 = np.array(kp1)
        kp2 = np.array(kp2)
        P = triangulate(kp1 , kp1 , mtx1, mtx2, T)
        point3D = np.mean(P,axis = 0)
    else:
        P = triangulate(center1 , center2 , mtx1, mtx2, T)
        point3D = P[0]
     
    return  img1, img2, point3D