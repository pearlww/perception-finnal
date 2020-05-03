import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.externals import joblib


class Classifier(object):
    def __init__(self):      
        self.K = 20
        path_str = 'training_dataset/imgs_mxl/*'
        imgs_path = sorted(glob.glob(path_str))
        label_path = 'training_dataset/lable_mxl.txt'
        self.loadTrainingData(imgs_path,label_path)
        self.PCs = self.PCA()
        self.mean = self.X.mean(0)
        self.std = self.X.std(0)

    def loadModel(self,modelPath):
        self.clf = joblib.load(modelPath)

    def loadTrainingData(self,imgs_path,label_path):
        images = []
        for file in imgs_path:
            img = cv2.imread(file)
            (h,w,d) = img.shape
            # To apply a classifier on this data, we need to flatten the image, to turn the data in a (samples, feature) matrix:
            images.append(img.reshape(h*w*d))
        # read labels
        with open(label_path, 'r') as f:
            lable = f.read()     
        lable = eval(lable)
        #lable = np.loadtxt('training_dataset/lable_mxl.txt',dtype=str)

        self.X = np.array(images)
        self.y = lable

    def PCA(self):
        # PCA by computing SVD of Y
        U,S,Vh = np.linalg.svd(self.X, full_matrices=False)
        V = Vh.T
        PCs = V[:,range(self.K)]
        return PCs

    def predict(self,image):
        img_resized = cv2.resize(image,(256,256))
        x = img_resized.reshape(-1,256*256*3)
        xs = (x - self.mean)/self.std
        x_pca = xs @ self.PCs
        y_predict = self.clf.predict(x_pca)
        return y_predict
