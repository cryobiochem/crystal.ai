from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import joblib

def return_prediction(model,labels,sample_json):

    sample_image = sample_json['pathing']
    #test = "sample_image.jpg"
    sample_image = image.load_img(sample_image,target_size=(512,512))
    sample_image = np.array(sample_image)
    sample_image = np.sum(sample_image,axis=2)
    sample_image = sample_image / (255*3)
    sample_image = sample_image[np.newaxis,:,:,np.newaxis]
    label_dic = labels
    pred = model.predict(sample_image.astype('float64'))
    pred_2 = np.argmax(pred)
    pred_2 = label_dic[pred_2]
    
    probabilities =''
    new_Dic = {}
    for n in range(pred.shape[1]):
        current = {labels[n] : 100*pred[0][n]}
        new_Dic.update(current)
        if n == pred.shape[1] - 1:
            probabilities += str(labels[n])+': '+str(100*pred[0][n])+' % '
        else:
            probabilities += str(labels[n])+': '+str(100*pred[0][n])+' % ' + '; '
        
    i = 0
    j = 0
    for k,v in new_Dic.items():
        if v >= 80:
            i = 1
        if v <= 60:
            j += 1
    if i < 1 and j < 3:
        low_acc_flag = 'Predictive Status: Low accuracy prediction'
    elif i < 1 and j == 3:
        low_acc_flag = 'Predictive Status: No match'
    else:
        low_acc_flag = 'Predictive Status: OK'
    

    
    return low_acc_flag, pred_2,probabilities


app = Flask(__name__)

@app.route("/")

def index():
	return """

	<h1> Flask app is running </h1>
    
    
    
	"""

model = load_model('GPU_image_class_model.h5')
labels = joblib.load('Label_dic.pkl')

@app.route("/api/model",methods=['POST'])

def image_prediction():
	content = request.json
	results = return_prediction(model,labels,content)

	return jsonify(results)

if __name__ == '__main__':
	app.run()