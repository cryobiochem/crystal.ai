from load_sample import *
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard, ReduceLROnPlateau
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad

L = 148

img_shape        = (L,L,3)
epochs           = 200
steps_per_epoch  = 1024
validation_steps = 32


time1 = time.time()

filename,dic, Y, X, classes = load_samples(image_size=img_shape)

from borderline2 import ImageProperties

print('Starting process: borderline2.py')

 
X = np.divide(np.sum(X,axis=3),255*3)
area_threshold = 0.8
X_Area  = X >= area_threshold

for n in range(X.shape[0]):
    dil =  ImageProperties(
        n              =   n,
        img_array      =   X,
        area_array     =   X_Area,
        area_threshold =   0.8,
        save           =   0,
        n_openings     =   1,
        n_erosions     =   1,
        n_dilations    =   1,
        filename       =   filename    
        )        
    if n == 0:
        X_dil = dil
    else:
        X_dil = np.concatenate((X_dil,dil),axis=0)                                    

time2 = time.time()

deltatime = time2 - time1
print('\nTotal Runtime: '+str(round(deltatime,1))+' seconds.')

X = X[:,:,:,np.newaxis]
X_dil = X_dil[:,:,:,np.newaxis]

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD



input_img = Input(shape=(L, L, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

autoencoder.fit(X, X_dil, epochs=5000)


for n in range(10):
    sample_prediction = autoencoder.predict(X[n].reshape(1,X.shape[1],X.shape[2],1))
    
    fig, (ax2,ax3,ax1) = plt.subplots(1,3)
    
    ax1.set_title('Predicted')
    im1 = ax1.imshow(sample_prediction.reshape((L,L)))
    ax2.set_title('Original')
    im2 = ax2.imshow(X[n].reshape(L,L))
    ax3.set_title('Img Processing')
    im3 = ax3.imshow(X_dil[n].reshape(L,L))
