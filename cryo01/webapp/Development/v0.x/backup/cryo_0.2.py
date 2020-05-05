#Libraries
import joblib
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from flask import Flask, render_template, url_for, request
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FileField, RadioField
from werkzeug import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'MY_KEY'

class InfoForm(FlaskForm):
    test_string  = StringField('Test String:')
    test_radio = RadioField('Radio here',choices=[('Ape','Ape'),('Dog','Dog')])
    test_image = FileField('Image Here')
    submit = SubmitField('Submit')


@app.route('/')
def index():
    return render_template("basic.html")


@app.route('/model_selection')
def model_sel():

    model_dir = 'model'
    lista = []
    for models in os.listdir(model_dir):
        lista.append(models)

    return render_template("model_selection.html",model_dir=lista)

@app.route('/model_prediction')
def prediction_model():
    selected_model_dir = request.args.get('selected_model')


    model_path = os.listdir('model/'+str(selected_model_dir))
    for model_file in model_path:
        if model_file[-2:] == 'h5':
            model = load_model('model/'+selected_model_dir+'/'+model_file)
            break
    #model.predict(image_file_name)
    print(model)
    return render_template("model_prediction.html",selected_model_dir=selected_model_dir)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404





if __name__ == '__main__':
    app.run(debug=True)
