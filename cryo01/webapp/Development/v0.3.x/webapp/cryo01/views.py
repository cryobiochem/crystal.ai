import os
from flask import Blueprint, render_template,redirect,url_for
from webapp import db
#from webapp.models import Cryo01
from webapp.cryo01.forms import Cryo01Form
from flask import Flask, render_template, url_for, request, session,redirect,url_for, flash, session
from flask_uploads import configure_uploads, IMAGES, UploadSet
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as LoadImage
import joblib
import numpy as np
images = UploadSet('images', IMAGES)


cryo01_blueprints = Blueprint('cryo01',__name__,template_folder='templates/cryo01')

@cryo01_blueprints.route('/cryo01')
def cryo01():

    return render_template("cryo01.html")


@cryo01_blueprints.route('/cryo01/cryo01_selection',methods=['GET','POST'])
def cryo01_selection():


    form = Cryo01Form()

    if form.validate_on_submit():

        filename = images.save(form.image.data)
        session['image'] = filename
        session['model_directory'] = form.model_radio.data

        return redirect(url_for('cryo01.cryo01_prediction'))

    return render_template("cryo01_selection.html",form=form)

@cryo01_blueprints.route('/cryo01/cryo01_prediction')
def cryo01_prediction():

    primary_working_folder = os.getcwd()
    selected_model_dir = session.get('model_directory',None)
    image_file_name = session.get('image',None)
    current_user = session.get('username',None)
    os.chdir(os.getcwd()+'//webapp//uploads')

    from pathlib import Path
    Path(current_user).mkdir(parents=True,exist_ok=True)
    import shutil
    shutil.move(image_file_name,current_user+'\\'+image_file_name)

    image_source = LoadImage.load_img(current_user+'\\'+image_file_name,target_size=(512,512))
    image = np.array(image_source)
    image = image[np.newaxis,:]
    image = np.divide(np.sum(image,axis=3),255*3)
    image = image.reshape(1,512,512,1)

    os.chdir(primary_working_folder+'\\webapp\\cryo01\\')


    model_path = os.listdir('data/model/'+str(selected_model_dir))
    for model_file in model_path:
        if model_file[-2:] == 'h5':
            model = load_model('data/model/'+selected_model_dir+'/'+model_file)

        elif model_file == 'Label_dic.pkl':
            labels = joblib.load('data/model/'+selected_model_dir+'/'+model_file)

        else:
            break

    ## image prediction ##
    prediction = model.predict(image)

    full_prediction = ''

    for i in range(len(labels)):
        full_prediction += labels[i]+': '+str(round(100*prediction[0][i],2))+'%  '

    prev = labels[np.argmax(prediction)]

    os.chdir(primary_working_folder)


    return render_template("cryo01_prediction.html",full_prediction=full_prediction,prev=prev)
