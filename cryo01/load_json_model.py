import h5py
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.preprocessing import image as iii
import numpy as np
import matplotlib.pyplot as plt
from load_sample import *
filename,dic, Y, X, classes = load_samples(image_size=(512,512,3))
#model load

# load json and create model
json_file = open('model2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model2.h5")
print("Loaded model from disk")

#loaded_model.summary()

image_path = 'D:\\github\\pudim\\cryo01\\datasets\\crystal_microscopy\\polisacárido teste\\P1010156_edit.JPG'

img = iii.load_img(image_path,target_size=(512,512))
img = np.array(img)
img = np.sum(img,axis=2)

plt.imshow(img)
img = img[np.newaxis,:,:,np.newaxis]

pred = loaded_model.predict(img)
pred = np.argmax(pred)

print('I predict that the image is from '+dic[pred])
