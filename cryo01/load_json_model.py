import h5py
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.preprocessing import image as iii
import numpy as np
import matplotlib.pyplot as plt
#from load_sample import *
import joblib
#filename,dic, Y, X, classes = load_samples(image_size=(512,512,3))

#model load

# # load json and create model
# json_file = open('model2.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model2.h5")
# #print("Loaded model from disk")
#loaded_model.summary()

from tensorflow.keras.models import load_model

model = load_model('GPU_image_class_model.h5')
labels = joblib.load('Label_dic.pkl')
image_path = 'sample_image.JPG'

img = iii.load_img(image_path,target_size=(512,512))
img = np.array(img)
img = np.sum(img,axis=2)

plt.imshow(img)
img = img[np.newaxis,:,:,np.newaxis]

pred = model.predict(img)
pred = np.argmax(pred)
pred = labels[pred]

metrics = joblib.load('metrics.pkl')

print(f"this image is predicted as {pred}")
#metrics.plot()
