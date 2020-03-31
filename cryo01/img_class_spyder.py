import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import imageio
from copy import deepcopy
import os
from matplotlib.image import imread
import h5py
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Convolution2D
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers, initializers
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad
import time
ts = time.time()
import datetime
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H%M%S')

# Directories and Image Sizes
data_dir = 'datasets\\crystal_microscopy'
save_dir = 'ML\\Test\\gen_data'
model_dir = 'ML\\Test\\saved_models'
log_dir = 'ML\\Test\\logs\\fit'
img_shape = (32,32,3)

# Model Design
model = Sequential()

model.add(Conv2D(
    filters=32, 
    kernel_size=(3,3),
    strides=1,
    padding='same',
    input_shape=img_shape, 
    use_bias=False,
    kernel_regularizer=regularizers.l2(0.01),
    activity_regularizer=regularizers.l2(0.01),
    kernel_initializer='he_normal'
    ))
model.add(BatchNormalization(axis=-1, center=True, scale=False))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Conv2D(
    filters=32, 
    kernel_size=(3,3),
    strides=1,
    padding='same',
    use_bias=False,
    kernel_regularizer=regularizers.l2(0.01),
    activity_regularizer=regularizers.l2(0.01),
    kernel_initializer='he_normal'
    ))
model.add(BatchNormalization(axis=-1, center=True, scale=False))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(
    filters=64, 
    kernel_size=(3,3),
    strides=1,
    padding='same',
    input_shape=img_shape, 
    use_bias=False,
    kernel_regularizer=regularizers.l2(0.01),
    activity_regularizer=regularizers.l2(0.01),
    kernel_initializer='he_normal'
    ))
model.add(BatchNormalization(axis=-1, center=True, scale=False))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Conv2D(
    filters=64, 
    kernel_size=(3,3),
    strides=1,
    padding='same',
    use_bias=False,
    kernel_regularizer=regularizers.l2(0.01),
    activity_regularizer=regularizers.l2(0.01),
    kernel_initializer='he_normal'
    ))
model.add(BatchNormalization(axis=-1, center=True, scale=False))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(
    filters=128, 
    kernel_size=(3,3),
    strides=1,
    padding='same',
    input_shape=img_shape, 
    use_bias=False,
    kernel_regularizer=regularizers.l2(0.01),
    activity_regularizer=regularizers.l2(0.01),
    kernel_initializer='he_normal'
    ))
model.add(BatchNormalization(axis=-1, center=True, scale=False))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Conv2D(
    filters=128, 
    kernel_size=(3,3),
    strides=1,
    padding='same',
    use_bias=False,
    kernel_regularizer=regularizers.l2(0.01),
    activity_regularizer=regularizers.l2(0.01),
    kernel_initializer='he_normal'
    ))
model.add(BatchNormalization(axis=-1, center=True, scale=False))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Flatten())

model.add(Dense(
    1024,
    kernel_regularizer=regularizers.l2(0.01),
    activity_regularizer=regularizers.l2(0.01)))   
model.add(BatchNormalization(axis=-1, center=True, scale=False)) 
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(
    1024,
    kernel_regularizer=regularizers.l2(0.01),
    activity_regularizer=regularizers.l2(0.01)))   
model.add(BatchNormalization(axis=-1, center=True, scale=False)) 
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(
    1024,
    kernel_regularizer=regularizers.l2(0.01),
    activity_regularizer=regularizers.l2(0.01)))   
model.add(BatchNormalization(axis=-1, center=True, scale=False)) 
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(
    128,
    kernel_regularizer=regularizers.l2(0.01),
    activity_regularizer=regularizers.l2(0.01)))   
model.add(BatchNormalization(axis=-1, center=True, scale=False)) 
model.add(Activation("relu"))
model.add(Dropout(0.5))


model.add(Dense(3,activation='softmax'))



model.compile(loss='categorical_crossentropy',
              optimizer=Adagrad(learning_rate=0.01),
              metrics=['accuracy'])

model.summary()

early_stop = EarlyStopping(monitor='val_loss',restore_best_weights=False,patience=2)

board = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True,
    update_freq='epoch',
    profile_batch=2,
    embeddings_freq=1)


#Data Augmentation/Generator
train_datagen  = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1/255, #255 max intensity in a pixel
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.18)





train_generator  = train_datagen.flow_from_directory(data_dir
                                              ,target_size=img_shape[:2]
                                              ,color_mode='rgb'
                                              ,batch_size=32
                                              ,class_mode='categorical'
                                              ,subset='training'
                                              ,shuffle=True
                                              #,save_to_dir=save_dir+'/train'
                                              )


validation_generator  = train_datagen.flow_from_directory(data_dir
                                              ,target_size=img_shape[:2]
                                              ,color_mode='rgb'
                                              ,class_mode='categorical'
                                              ,subset='validation'
                                              #,save_to_dir=save_dir+'/test'
                                               )



model.fit_generator(
    train_generator,
    steps_per_epoch=512,
    epochs=100,
    validation_data=validation_generator,
    validation_steps= 32,
    #callbacks=[early_stop,board]
    callbacks=[early_stop]
    )

model.save(model_dir+'/'+str(st)+'_model_epoch'+'.h5')
train_generator.reset()

# Model Evaluation
metricas = pd.DataFrame(model.history.history)

plot = metricas[['loss','val_loss']].plot()
#x1,x2,y1,y2 = plot.axis()
#plt.axis((x1,x2,y1,3))


model.save(model_dir+'/model.h5') 

model.metrics_names
model.evaluate_generator(validation_generator, verbose=1)

y_prob = model.predict(train_generator) 
y_prob = np.argmax(y_prob,axis=1)
label_map = (train_generator.class_indices)
y_true_labels = train_generator.classes

print(label_map)
print(classification_report(y_true_labels,y_prob))
cm = confusion_matrix(y_true_labels,y_prob)
plt.figure(figsize=(10,6))
sns.heatmap(cm,annot=True,cmap='viridis')
