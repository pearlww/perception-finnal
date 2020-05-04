
import cv2
import random
import numpy as np
import glob
import os


def nameAndResize(input_path,output_path, size, lable):

    imgs_path = sorted(glob.glob(input_path))
    num = 0
    for file in imgs_path:
        image = cv2.imread(file)
        img_resized = cv2.resize(image, size)
        cv2.imwrite(output_path + lable + '_' + f"{num:04d}.jpg", img_resized)
        num+=1

def random_rotation(image):
    rows, cols,c = image.shape
    degree = random.uniform(-25, 25)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),  degree, 1)
    return cv2.warpAffine(image, M, (cols, rows))

def addeptive_gaussian_noise(image):
    h,s,v=cv2.split(image)
    s = cv2.adaptiveThreshold(s, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    h = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    v = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return cv2.merge([h,s,v])

def add_light(image, gamma=1.0):
    # gamma>=1:light, gamma<1:dark
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def add_light_color(image, color, gamma=1.0):
    # gamma>=1:light color, gamma<1:dark color
    invGamma = 1.0 / gamma
    image = (color - image)
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def saturation_image(image,saturation):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = image[:, :, 2]
    v = np.where(v <= 255 - saturation, v + saturation, 255)
    image[:, :, 2] = v
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


def hue_image(image,saturation):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = image[:, :, 2]
    v = np.where(v <= 255 + saturation, v - saturation, 255)
    image[:, :, 2] = v
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


def horizontal_flip(image):
    # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
    return image[:, ::-1]

def image_augment(input_path, output_path): 

    imgs_path = sorted(glob.glob(input_path))

    path = random.sample(imgs_path,5)
    for file in path:
        image = cv2.imread(file)
        img_rot = random_rotation(image)
        path, name = os.path.split(file)
        cv2.imwrite(output_path+'%s' %str(name[:-4]) + '_rot.jpg', img_rot)

    path = random.sample(imgs_path,5)  
    for file in path:
        image = cv2.imread(file)
        img_flip = horizontal_flip(image)
        path, name = os.path.split(file)
        cv2.imwrite(output_path+'%s' %str(name[:-4]) + '_flip.jpg', img_flip)

    path = random.sample(imgs_path,5)
    for file in path:
        image = cv2.imread(file)
        img_noise = addeptive_gaussian_noise(image)                    
        path, name = os.path.split(file)
        cv2.imwrite(output_path+'%s' %str(name[:-4]) + '_noise.jpg', img_noise)

    path = random.sample(imgs_path,5)
    for file in path:
        image = cv2.imread(file)
        img_light = add_light(image,gamma = 1.5)                    
        path, name = os.path.split(file)
        cv2.imwrite(output_path+'%s' %str(name[:-4]) + '_light.jpg', img_light)

    path = random.sample(imgs_path,5)
    for file in path:
        image = cv2.imread(file)
        img_colorlight = add_light_color(image,50, gamma=1.5)                    
        path, name = os.path.split(file)
        cv2.imwrite(output_path+'%s' %str(name[:-4]) + '_colorlight.jpg', img_colorlight)

def getLable(input_path,output_path):
    lable = []
    imgs_path = sorted(glob.glob(input_path))
    for file in imgs_path:
        path, name = os.path.split(file)
        lab = name[:-9]
        lable.append(lab)
    print(lable)
    np.savetxt(output_path + 'lable.txt', lable, fmt="%s")

    
if __name__ == "__main__":
    
    # ---------------------- name and resize -------------------------------------#
    path_str1 = 'training_dataset/raw/book/*'
    path_str2 = 'training_dataset/raw/cup/*'
    path_str3 = 'training_dataset/raw/box/*'
    output_path = 'training_dataset/processed/imgs/'
    size = (256,256)
    nameAndResize(path_str1,output_path, size, 'book')
    nameAndResize(path_str2,output_path, size, 'cup')
    nameAndResize(path_str3,output_path, size, 'box')


    #----------------------------- augment ----------------------------------------#
    input_path = 'training_dataset/processed/imgs/*'
    output_path = 'training_dataset/processed/augmented/'
    image_augment(input_path, output_path)
   
    
    # # --------------------------- get lable --------------------------------------#
    input_path = 'training_dataset/processed/imgs/*'
    output_path =  'training_dataset/processed/'
    getLable(input_path,output_path)

