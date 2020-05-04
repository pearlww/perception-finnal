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


def show_some_results(images, targets, sample_size, imgshape):

    nsamples=sample_size
    rand_idx = np.random.choice(images.shape[0], nsamples)
    images_and_labels = list(zip(images[rand_idx], targets[rand_idx]))

    plt.figure(3)
    for index, (image, label) in enumerate(images_and_labels):
        plt.subplot(np.ceil(nsamples/6.0), 6, index + 1)
        plt.axis('off')
        #each image is flat, we have to reshape to 2D array 
        plt.imshow(image.reshape(imgshape))
        plt.title(label)
    

def loadTrainingData(imgs_path,label_path):
    # read images
    imgs_path = sorted(glob.glob(imgs_path))
    images = []
    for file in imgs_path:
        img = cv2.imread(file)
        (h,w,d) = img.shape
        # To apply a classifier on this data, we need to flatten the image, to turn the data in a (samples, feature) matrix:
        images.append(img.reshape(h*w*d))
    # read labels
    lable = np.loadtxt(label_path, dtype = 'str')

    X = np.array(images)
    y = lable
    return X, y, (h,w,d)


if __name__ == "__main__":
    
    
    imgs_path = 'training_dataset/processed/imgs/*'
    label_path = 'training_dataset/processed/lable.txt'

    X, y ,imgshape = loadTrainingData(imgs_path,label_path)
    classNames = np.unique(y)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

    # doing nomalization (not standardelization)
    scaler = preprocessing.MinMaxScaler()
    #scaler = preprocessing.StandardScaler()
    X_train_scal = scaler.fit_transform(X_train)
    X_test_scal = scaler.transform(X_test)

    # doing pca
    n_components = 20
    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    X_train_pca = pca.fit_transform(X_train_scal)
    X_test_pca = pca.transform(X_test_scal)
    

    # do classification by SVM
    #clf = SVC(C=0.5,gamma=0.00001)
    param_grid = {'C': [0.1, 1, 10, 100, 1000, 10000],
                'gamma': [0.000001, 0.00001, 0.0001, 0.001, 0.005, 0.01, 0.1, 1]}
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid = param_grid)
    clf.fit(X_train_pca,y_train) 
    y_predict = clf.predict(X_test_pca)

    #print result
    print(clf.best_estimator_)
    print("training score\n:", clf.score(X_train_pca, y_train))
    print("testing score\n:", clf.score(X_test_pca, y_test))
    print("test tesult:\n", classification_report(y_test, y_predict, target_names=classNames))
    print("confusion matrix:\n", confusion_matrix(y_test, y_predict, labels=classNames))

    # # Plot variance explained
    plt.figure(1)
    plt.plot(pca.explained_variance_ratio_,'o-')
    plt.title('Variance explained by principal components')
    plt.xlabel('Principal component')
    plt.ylabel('Variance explained value')

    # Plot PCA1 and PCA2 of the data
    dic = {'cup':1,'box':2,'book':3}
    classIndex = []
    for i in y_train:
        classIndex.append(dic[i]) 
    plt.figure(2)
    plt.title('Projected on PCs')
    plt.scatter(X_train_pca[:,0], X_train_pca[:,1],c=classIndex)
    plt.legend(classIndex)
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    #plot some classify result
    show_some_results(X_test, y_predict, 12, imgshape)

    #save the model
    joblib.dump(clf, 'src/classification/Classifier.pkl') 
    #np.savetxt('src/classification/pcs.txt', PCs[:,range(K)])

    plt.show()

    
