
import cv2
import random
import numpy as np
import glob
import os



# def __init__(self, file):
#     head, tail = os.path.split(file)
#     self.path =  head
#     self.name = tail
#     print(self.name)
#     self.image = cv2.imread(file)

# def resize(self, image, size):
#     image = cv2.resize(image,size)

#     return image

# def random_rotate(save_path, image):
#     img = self.image.copy()

#     angle = random.uniform(-25, 25)
#     h,w = image.shape
#     #rotate matrix
#     M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale=1.0)
#     #rotate
#     image = cv2.warpAffine(image,M,(w,h)
#     cv2.imwrite(save_path+'%s' %str(name_int)+'_flip.jpg', image)

# def flip(save_path, image, vflip=False, hflip=False):
#     img = self.image.copy()
#     if hflip or vflip:
#         if hflip and vflip:
#             c = -1
#         else:
#             c = 0 if vflip else 1
#         image = cv2.flip(image, flipCode=c)



# def image_augment(self, save_path): 

#     img = self.image.copy()
#     img_resize = self.resize(img,(256,256))
#     img_flip = self.flip(img, vflip=True, hflip=False)
#     img_rot = self.rotate(img)
#     #img_gaussian = self.add_GaussianNoise(img)
    
#     name_int = self.name[:len(self.name)-4]
#     cv2.imwrite(save_path+'%s' %str(name_int)+'_resize.jpg', img_resize)
#     cv2.imwrite(save_path+'%s' %str(name_int)+'_vflip.jpg', img_flip)
#     cv2.imwrite(save_path+'%s' %str(name_int)+'_rot.jpg', img_rot)
#     #cv2.imwrite(save_path+'%s' %str(name_int)+'_GaussianNoise.jpg', img_gaussian)
    
    
if __name__ == "__main__":
    
    # ---------------------- name and resize -------------------------------------#
    # path_str1 = 'training_dataset/books/*'
    # path_str2 = 'training_dataset/cups/*'
    # path_str3 = 'training_dataset/boxes/*'
    # output_path = 'training_dataset/test/'
    # imgs_path = sorted(glob.glob(path_str3))
    # print(imgs_path)

    # num = 0
    # name1 = 'book'
    # name2 = 'cup'
    # name3 = 'box'
    # for file in imgs_path:
    #     image = cv2.imread(file)
    #     img_resized = cv2.resize(image,(256,256))
    #     cv2.imwrite(output_path + name3 + '_' + f"{num:04d}.jpg", img_resized)
    #     num+=1

    #----------------------------- augment ----------------------------------------#
    
    # path_str = 'training_dataset/test/*'
    # imgs_path = sorted(glob.glob(path_str))

    # # = random.choice(imgs_path)

    # # --------------------------- get lable --------------------------------------#

    # lable = []
    # for file in imgs_path:
    #     path, name = os.path.split(file)
    #     lab = name[:-9]
    #     lable.append(lab)

    # print(lable)
    # with open('training_dataset/lable.txt','w') as f:
    #     f.write(str(lable))

    #----------------------------- get mxl lable -----------------------------------------#
    lable_mxl = []
    for i in range(100):
        lable_mxl.append('cup') 
    for i in range(100):
        lable_mxl.append('box') 
    for i in range(100):
        lable_mxl.append('book')             
    print(lable_mxl)
    
    with open('training_dataset/lable_mxl.txt','w') as f:
        f.write(str(lable_mxl))
    #np.savetxt( 'training_dataset/lable_mxl.csv', lable_mxl)