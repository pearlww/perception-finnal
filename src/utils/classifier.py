import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.decomposition import PCA

class Classifier(object):
    def __init__(self, imgs_path,label_path):      
        self.n_components = 20        
        self.loadTrainingData(imgs_path,label_path)
        self.pca = PCA(n_components=self.n_components, svd_solver='randomized', whiten=True).fit(self.X)
        self.scaler = preprocessing.MinMaxScaler().fit(self.X)

    def train(self):
        param_grid = {'C': [0.1, 1, 10, 100, 1000, 10000],
                    'gamma': [0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.1, 1]}
        self.clf = GridSearchCV(SVC(kernel='rbf'), param_grid = param_grid)
        self.clf.fit(self.X,self.y) 

    def loadModel(self,modelPath):
        self.clf = joblib.load(modelPath)

    def loadTrainingData(self,train_imgs_path,train_lable_path):
        images = []
        imgs_path = sorted(glob.glob(train_imgs_path))
        for file in imgs_path:
            img = cv2.imread(file)
            (h,w,d) = img.shape
            # To apply a classifier on this data, we need to flatten the image, to turn the data in a (samples, feature) matrix:
            images.append(img.reshape(h*w*d))
        # read labels
        lable = np.loadtxt(train_lable_path,dtype='str')
        self.X = np.array(images)
        self.y = lable

    def predict(self,image):
        img_resized = cv2.resize(image,(256,256))
        x = img_resized.reshape(-1,256*256*3)
        xs = self.scaler.transform(x)
        x_pca = self.pca.transform(xs)
        y_predict = self.clf.predict(x_pca)
        return y_predict
