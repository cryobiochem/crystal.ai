from flask import Flask, render_template,session,url_for,redirect
import numpy as np
from wtforms import TextField, SubmitField
from flask_wtf import FlaskForm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import joblib

def return_prediction(model,labels,sample_json):

    sample_image = sample_json['image_pathing']
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
    low_acc_flag = ''
    for k,v in new_Dic.items():
        if v >= 80:
            i = 1
        if v <= 60:
            j += 1
    if i < 1 and j < 3:
        low_acc_flag = 'Low accuracy prediction'
    elif i < 1 and j == 3:
        low_acc_flag = 'No match'
    else:
        low_acc_flag = 'OK'
    
    return pred_2, probabilities, low_acc_flag


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

class CryoForm(FlaskForm):

    image_path = TextField("Path of the image")

    submit = SubmitField("Predict")

@app.route("/", methods=['GET','POST'])

def index():
	
    form = CryoForm()

    if form.validate_on_submit():

        session['image_path'] = form.image_path.data

        return redirect(url_for("prediction"))

    return render_template('home.html',form=form)

model = load_model('GPU_image_class_model.h5')
labels = joblib.load('Label_dic.pkl')

@app.route('/prediction')

def prediction():
    
    content = {}

    content['image_pathing'] = session['image_path']

    pred_2, probabilities, status = return_prediction(model,labels,content)

    return render_template(
        'prediction.html',
        pred_2=pred_2,
        probabilities=probabilities,
        status=status)


if __name__ == '__main__':
	app.run()