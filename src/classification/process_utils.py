
import cv2
import random
import numpy as np
import glob
import os


def processing(input_path,output_path, size, lable):

    imgs_path = sorted(glob.glob(input_path))
    num = 0
    for file in imgs_path:
        image = cv2.imread(file)
        image = cv2.resize(image, size)    
        cv2.imwrite(output_path + lable + '_' + f"{num:04d}.jpg", image)
        num+=1

    image_augment(output_path +'*', output_path, num, lable,size)


def random_rotation(image):
    rows, cols,c = image.shape
    degree = random.choice([90,-90])
    M = cv2.getRotationMatrix2D((cols/2,rows/2),  degree, 1)
    return cv2.warpAffine(image, M, (cols, rows))

def crop_image(image, origin_size):
    image=image[0:200,0:200]
    image = cv2.resize(image,origin_size)
    return image

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

def gausian_blur(image, kernel):
    return cv2.GaussianBlur(image,kernel,1)


def averageing_blur(image,kernel):
    return  cv2.blur(image,kernel)

def median_blur(image):
    return cv2.medianBlur(image,5)

def salt_and_paper_image(image,p,a):
    noisy=image
    #salt
    num_salt = np.ceil(a * image.size * p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
    noisy[coords] = 1
    #paper
    num_pepper = np.ceil(a * image.size * (1. - p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
    noisy[coords] = 0
    return image

    
def image_augment(input_path, output_path, num, lable,size): 

    imgs_path = sorted(glob.glob(input_path))

    path = random.sample(imgs_path,5)
    for file in path:
        image = cv2.imread(file)
        image = random_rotation(image)
        cv2.imwrite(output_path + lable + '_' + f"{num:04d}.jpg", image)
        num+=1

    path = random.sample(imgs_path,10)
    for file in path:
        image = cv2.imread(file)
        image = crop_image(image,size)
        cv2.imwrite(output_path + lable + '_' + f"{num:04d}.jpg", image)
        num+=1

    path = random.sample(imgs_path,5)  
    for file in path:
        image = cv2.imread(file)
        image = horizontal_flip(image)
        cv2.imwrite(output_path + lable + '_' + f"{num:04d}.jpg", image)
        num+=1

    path = random.sample(imgs_path,5)
    for file in path:
        image = cv2.imread(file)
        image = addeptive_gaussian_noise(image)                    
        cv2.imwrite(output_path + lable + '_' + f"{num:04d}.jpg", image)
        num+=1

    path = random.sample(imgs_path,5)
    for file in path:
        image = cv2.imread(file)
        image = salt_and_paper_image(image,0.5,0.09)                    
        cv2.imwrite(output_path + lable + '_' + f"{num:04d}.jpg", image)
        num+=1

    path = random.sample(imgs_path,5)
    for file in path:
        image = cv2.imread(file)
        image = add_light(image,gamma = 1.5)                    
        cv2.imwrite(output_path + lable + '_' + f"{num:04d}.jpg", image)
        num+=1

    path = random.sample(imgs_path,5)
    for file in path:
        image = cv2.imread(file)
        image = add_light_color(image,255, gamma=1.5)                    
        cv2.imwrite(output_path + lable + '_' + f"{num:04d}.jpg", image)
        num+=1

    path = random.sample(imgs_path,5)
    for file in path:
        image = cv2.imread(file)
        image = gausian_blur(image, (9,9))
        cv2.imwrite(output_path + lable + '_' + f"{num:04d}.jpg", image)
        num+=1
        path = random.sample(imgs_path,10)

    for file in path:
        image = cv2.imread(file)
        image = averageing_blur(image, (5,5))
        cv2.imwrite(output_path + lable + '_' + f"{num:04d}.jpg", image)
        num+=1


def getLable_name(input_path,output_path):
    lable = []
    imgs_path = sorted(glob.glob(input_path))
    for file in imgs_path:
        path, name = os.path.split(file)
        lab = name[:-9]
        lable.append(lab)
    print(len(lable))
    np.savetxt(output_path + 'lable.txt', lable, fmt="%s")

def getLable_video(output_path):
    lable = []
    for i in range(0,383):
        lable.append('box') 
    for i in range(383,962):
        lable.append('book') 
    for i in range(962,1345):
        lable.append('cup')             
    print(len(lable))

    np.savetxt(output_path + 'lable2.txt', lable, fmt="%s")

def getLable_number(output_path):
    lable = []
    for i in range(0,90):
        lable.append('box') 
    for i in range(90,190):
        lable.append('book') 
    for i in range(190,283):
        lable.append('cup')             
    print(len(lable))

    np.savetxt(output_path + 'lable.txt', lable, fmt="%s")

if __name__ == "__main__":
    
    # ---------------------- name and resize -------------------------------------#
    # path_str1 = 'training_dataset/raw/book/*'
    # path_str2 = 'training_dataset/raw/cup/*'
    # path_str3 = 'training_dataset/raw/box/*'
    # output_path = 'training_dataset/processed/'
    # size = (256,256)
    # processing(path_str1,output_path+'book/', size, 'book')
    # processing(path_str2,output_path+'cup/', size, 'cup')
    # processing(path_str3,output_path+'box/', size, 'box')
 
    
    # # # --------------------------- get lable --------------------------------------#
    input_path = 'training_dataset/processed/all/*'
    output_path =  'training_dataset/'
    #getLable_name(input_path,output_path)
    getLable_number(output_path)

