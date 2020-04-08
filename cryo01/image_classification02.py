from load_sample import *
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard, ReduceLROnPlateau
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad
from tensorflow.keras.utils import to_categorical

img_shape=(512,512,3)
epochs = 200

filename,dic, Y, X, classes = load_samples(image_size=img_shape)

img_shape=(512,512,1)

#H5py data load
from h5py import File
h5f = File('new_area_data.h5','r')
X = h5f['X_new'][:]
h5f.close()

# X = X[:,:,:,np.newaxis]

#img_shape=(512,512,1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

plt.imshow(X_train[0].reshape(512,512))

unique, counts = np.unique(y_train, return_counts=True)
print('Train Set: ')
print (np.asarray((unique, counts)).T)

unique, counts = np.unique(y_test, return_counts=True)
print('Test Set: ')
print (np.asarray((unique, counts)).T)

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

#X_train = X_train / 255
#X_test = X_test / 255

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(X_train)

#model

model = Sequential()

#layer01    
model.add(Conv2D(
    filters=32, 
    kernel_size=(5,5),
    strides=3,
    padding='same',
    input_shape=img_shape, 
    #kernel_regularizer=regularizers.l2(0.01),
    #activity_regularizer=regularizers.l2(0.01),
    kernel_initializer='he_normal'
    ))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))

#layer02
model.add(Conv2D(
    filters=32, 
    kernel_size=(5,5),
    strides=3,
    padding='same',
    #kernel_regularizer=regularizers.l2(0.01),
    #activity_regularizer=regularizers.l2(0.01),
    kernel_initializer='he_normal'
    ))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))


#Dense Layer
model.add(Flatten())
model.add(Dense(
    128
    #,kernel_regularizer=regularizers.l2(0.01)
    #,activity_regularizer=regularizers.l2(0.01)
    )) 
#model.add(BatchNormalization(axis=-1, center=True, scale=False)) 
model.add(Activation("relu"))
model.add(Dropout(0.5))
    
    
model.add(Dense(3,activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.summary()

early_stop = EarlyStopping(monitor='val_loss',patience=8,restore_best_weights=True)
reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=0)


model.fit_generator(
    datagen.flow(X_train, y_train),
    validation_data=datagen.flow(X_test,y_test),
    validation_steps = 64,
    steps_per_epoch=128,
    epochs=epochs,
    callbacks=[
        early_stop,
        reduceLR],
    shuffle=True
    )

metrics = pd.DataFrame(model.history.history)
metrics[['loss','val_loss']].plot()

pred = model.predict(X_test)

for n in range(pred.shape[0]):
  i = np.argmax(pred[n])
  pred[n] = 0
  pred[n][i] = 1
  
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,pred))

cm = confusion_matrix(
    y_test.argmax(axis=1), pred.argmax(axis=1))

import seaborn as sns
plt.figure(figsize=(10,6))
test_eval = sns.heatmap(cm,annot=True,cmap='viridis')


pred2 = model.predict(X_train)

for n in range(pred2.shape[0]):
  i = np.argmax(pred2[n])
  pred2[n] = 0
  pred2[n][i] = 1
  
print(classification_report(y_test,pred))

cm2 = confusion_matrix(
    y_train.argmax(axis=1), pred2.argmax(axis=1))

plt.figure(figsize=(10,6))
train_eval = sns.heatmap(cm2,annot=True,cmap='viridis')
