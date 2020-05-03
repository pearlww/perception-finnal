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


def show_some_results(images, targets, sample_size, imgSize):

    nsamples=sample_size
    rand_idx = np.random.choice(images.shape[0], nsamples)
    images_and_labels = list(zip(images[rand_idx], targets[rand_idx]))

    plt.figure(1, figsize=(10, 8), dpi=160)
    for index, (image, label) in enumerate(images_and_labels):
        plt.subplot(np.ceil(nsamples/6.0), 6, index + 1)
        plt.axis('off')
        #each image is flat, we have to reshape to 2D array 
        plt.imshow(image.reshape(imgSize), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(label)
    
def standarderlization(X):    
    N,D = X.shape  
    # Center the data (subtract mean column values) 
    Xc = X - np.ones((N,1))*X.mean(0)
    Xs= Xc/Xc.std(0)
    return Xs

def PCA(X, K):

    # PCA by computing SVD of Y
    U,S,Vh = np.linalg.svd(X, full_matrices=False)
    V = Vh.T
    # Compute variance explained by principal components
    rho = (S*S) / (S*S).sum() 
    PCs = V[:,range(K)]
    return PCs, rho

def loadTrainingData(imgs_path,label_path):
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

    X = np.array(images)
    y = lable
    return X,y


if __name__ == "__main__":
    
    # read images
    path_str = 'training_dataset/imgs_mxl/*'
    imgs_path = sorted(glob.glob(path_str))
    label_path = 'training_dataset/lable_mxl.txt'

    X, y = loadTrainingData(imgs_path,label_path)
    classNames = np.unique(y)
    dic = {'cup':1,'box':2,'book':3}
    classIndex = []
    for i in y:
        classIndex.append(dic[i])
    K = 20

    # doing pca
    X = standarderlization(X)
    PCs, rho = PCA(X, K)

    # do classification by SVM
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train_pca = X_train @ PCs
    X_test_pca = X_test @ PCs
    clf = SVC(C=0.5,gamma=0.00001)
    # param_grid = {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #             'gamma': [0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.1]}
    # clf = GridSearchCV(SVC(kernel='rbf'), param_grid = param_grid)
    clf.fit(X_train_pca,y_train) 
    y_predict = clf.predict(X_test_pca)

    #print result
    #print(clf.best_estimator_)
    print("training score\n:", clf.score(X_train_pca, y_train))
    print("testing score\n:", clf.score(X_test_pca, y_test))
    print("test tesult:\n", classification_report(y_test, y_predict, target_names=classNames))
    print("confusion matrix:\n", confusion_matrix(y_test, y_predict, labels=classNames))

    # Plot variance explained
    plt.figure()
    plt.plot(rho,'o-')
    plt.title('Variance explained by principal components')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained value')

    # Plot PCA1 and PCA2 of the data
    plt.figure()
    plt.title('Projected on PCs')
    X_pca = X @ PCs
    plt.scatter(X_pca[:,0], X_pca[:,1],c=classIndex)
    plt.legend(classIndex)
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    #plot some classify result
    #show_some_results(X_test, y_predict, 12, (h,w,d))

    #save the model
    joblib.dump(clf, 'src/classification/Classifier.pkl') 
    #np.savetxt('src/classification/pcs.txt', PCs[:,range(K)])

    plt.show()

    
