#Libraries
import joblib
import os
import numpy as np
from tensorflow.keras.preprocessing import image as LoadImage
from tensorflow.keras.models import load_model
from flask import Flask, render_template, url_for, request, session,redirect,url_for, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FileField, RadioField
from werkzeug import secure_filename
from flask_uploads import configure_uploads, IMAGES, UploadSet
from flask_wtf.file import FileAllowed
from wtforms.validators import DataRequired

app = Flask(__name__)
app.config['SECRET_KEY'] = 'MY_KEY'
app.config['UPLOADED_IMAGES_DEST'] = 'static/images'
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024

images = UploadSet('images', IMAGES)
configure_uploads(app, images)

class InfoForm(FlaskForm):


    model_dir = 'model'
    model_choices = []
    for models in os.listdir(model_dir):
        select = (models,models)
        model_choices.append(select)
    model_radio = RadioField('Please choose your model:',choices=model_choices, validators=[DataRequired()])
    image = FileField('Image',
        validators=[
            FileAllowed(['PNG','JPG','jpg','png','JPEG','jpeg']),
            DataRequired()])
    submit = SubmitField('Submit')


@app.route('/')
def index():
    return render_template("basic.html")


@app.route('/model_selection',methods=['GET','POST'])
def model_selection():

    form = InfoForm()

    if form.validate_on_submit():
        filename = images.save(form.image.data)
        session['image'] = filename
        session['model_directory'] = form.model_radio.data

        return redirect(url_for('model_prediction'))

    return render_template("model_selection.html",form=form)

@app.route('/model_prediction')
def model_prediction():

    selected_model_dir = session.get('model_directory',None)
    image_file_name = session.get('image',None)
    image = LoadImage.load_img('static/images/'+image_file_name,target_size=(512,512))
    image = np.array(image)
    image = image[np.newaxis,:]
    image = np.divide(np.sum(image,axis=3),255*3)
    image = image.reshape(1,512,512,1)

    model_path = os.listdir('model/'+str(selected_model_dir))
    for model_file in model_path:
        if model_file[-2:] == 'h5':
            model = load_model('model/'+selected_model_dir+'/'+model_file)

        elif model_file == 'Label_dic.pkl':
            labels = joblib.load('model/'+selected_model_dir+'/'+model_file)

        else:
            break

    ## image prediction ##
    prediction = model.predict(image)

    full_prediction = ''

    for i in range(len(labels)):
        full_prediction += labels[i]+': '+str(round(100*prediction[0][i],2))+'%  '

    prev = labels[np.argmax(prediction)]




    return render_template("model_prediction.html",full_prediction=full_prediction,prev=prev)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404





if __name__ == '__main__':
    app.run(debug=True)
