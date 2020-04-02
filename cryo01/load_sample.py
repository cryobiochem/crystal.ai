from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import numpy as np

# image_path = "D:\\github\\pudim\\cryo01\\datasets\\crystal_microscopy\\água\\P1010174_edit.JPG"

# img = plt.imread(image_path).astype(float)

# img = img /255

# img = (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3

# img_border = img > 0.999
# img_area = img > 0.9

# plt.imshow(img_border)
# plt.imshow(img_area)

# total_area = img_area.sum()
# print(img_area.sum()/img.sum())

def load_samples(image_size=(64,64,3),data_dir = 'datasets\\crystal_microscopy',dic=False):
    #data_dir = 'datasets\\crystal_microscopy'
    os.listdir(data_dir)
    image_size = image_size
    
    n= 0
    dic = {}
    for folder_path in os.listdir(data_dir):
      print('current folder: '+folder_path)
      dic.update({n : folder_path})
      
      for image_path in os.listdir(data_dir+'/'+folder_path):
        fullpath = data_dir+'/'+folder_path+'/'+image_path
        print(fullpath)
        img = image.load_img(fullpath,target_size=image_size)
        current_batch = np.array(img)
        current_batch = current_batch[np.newaxis,:]
        current_label = np.array([n],dtype=str)
        current_label = current_label.reshape((1,1))
        if'X' in locals():
            X = np.concatenate((X,current_batch),axis=0)
            Y = np.concatenate((Y,current_label),axis=0)
            
        else:
            X = current_batch
            Y = current_label
      n += 1      
    
    
    from tensorflow.keras.utils import to_categorical
    classes = to_categorical(Y[:,0],3)
    
    if dic == True:
        return dic, Y, X, classes
    else:
        return Y, X, classes

