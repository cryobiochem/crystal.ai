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
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.utils import to_categorical


# Variables
data_dir = 'datasets\\crystal_microscopy'
save_dir = 'ML\\Test\\gen_data'
model_dir = 'ML\\Test\\saved_models'
log_dir = 'ML\\Test\\logs\\fit'
log_dir = 'ML\\Test\\history'
img_shape = (64,64,3)
epochs = 1
steps_per_epoch = 5
validation_steps = 4

import model_design

model, train_generator, validation_generator, model_summary = model_design.designer(
    img_shape=img_shape,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    log_dir=log_dir,
    data_dir=data_dir)


# Model Evaluation
metrics = pd.DataFrame(model.history.history)

plot = metrics[['loss','val_loss']].plot()
#x1,x2,y1,y2 = plot.axis()
#plt.axis((x1,x2,y1,3))


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


#model.save(model_dir+'/model.h5')


#Other stuff
import time
ts = time.time()
import datetime
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H%M%S')
import os
history_folder = 'ML\\Test\\history\\'+st
os.mkdir(history_folder)

file = open(history_folder+'\\model_summary.txt','w')
file.write(model_summary)
file.close()