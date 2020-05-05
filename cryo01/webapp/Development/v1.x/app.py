import joblib
import os
from flask import Flask, render_template, url_for, request, session,redirect,url_for, flash, session
from flask_uploads import configure_uploads, IMAGES, UploadSet
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import datetime
import email_validator
from flask_wtf import FlaskForm
import os
from wtforms import StringField, SubmitField, FileField, RadioField
from wtforms.validators import DataRequired
from flask_wtf.file import FileAllowed

import numpy as np
from tensorflow.keras.preprocessing import image as LoadImage
from tensorflow.keras.models import load_model
from werkzeug import secure_filename


app = Flask(__name__)
app.config['SECRET_KEY'] = 'MY_KEY'
app.config['UPLOADED_IMAGES_DEST'] = 'static/images'
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 # 4 MB max file
image_formats  = ['PNG','JPG','jpg','png','JPEG','jpeg']
images = UploadSet('images', IMAGES)
configure_uploads(app, images)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'+os.path.join(basedir,'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#import db_model
db = SQLAlchemy(app)
Migrate(app,db)

##### DB MODELS #####


class user(db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer,primary_key=True)
    username = db.Column(db.Text)
    email = db.Column(db.Text)
    password = db.Column(db.Text)
    created_on = db.Column(db.DateTime)
    last_change = db.Column(db.DateTime)

    def __init__(self,username,email,password,created_on,last_change):
        self.username = username
        self.email = email
        self.password = password
        self.created_on = created_on
        self.last_change = last_change

    def __repr__(self):
        return self.id,self.username,self.email,self.created_on,self.last_change

@app.route('/')
def index():


    return render_template("homepage.html")

@app.route('/admin')
def admin():

    if session['username'] == 'admin':
        users = user.query.all()

        for u in users:
            print(f"{u.id},{u.username},{u.email},{u.created_on},{u.last_change}")
        return render_template("homepage.html")

    else:
        return render_template("404.html")

@app.route('/signup',methods=['GET','POST'])
def signup():

    from signup_form import SignupForm
    form = SignupForm()

    if form.validate_on_submit():
        session.pop('signup_status',None)
        username = form.username.data
        email = form.email.data
        password = form.password.data
        created_on = datetime.datetime.now()
        last_change = created_on

        users = user.query.all()
        for u in users:
            if u.username == username:
                session['signup_status'] = 'Username already in use.'
                return redirect(url_for('signup'))
            else:

                new_user = user(username,email,password,created_on,last_change)
                db.session.add(new_user)
                db.session.commit()

                return redirect(url_for('login'))

    return render_template('signup.html',form=form)

@app.route('/login',methods=['GET','POST'])
def login():

    from signup_form import LoginForm
    form = LoginForm()

    if form.validate_on_submit():

        username = form.username.data
        password = form.password.data

        try:
            user_db = user.query.filter_by(username=username).first()

            if user_db.password == password:
                session['username'] = username
                session['login_status'] = 'OK'
                print('ok')
                return redirect(url_for('index'))
            else:
                session['login_status'] = 'Invalid user / password'
                print('not ok')
                return redirect(url_for('login'))
        except:
            session['login_status'] = 'Invalid user / password'

    return render_template('login.html',form=form)

@app.route('/logout')
def logout():
    session.pop('username',None)
    return redirect(url_for('index'))

@app.route('/cryo01')
def cryo01():

    return render_template("projects/cryo01/cryo01.html")

@app.route('/cryo01/selection',methods=['GET','POST'])
def cryo01_selection():

    from cryo01_form import InfoForm

    form = InfoForm()

    if form.validate_on_submit():
        filename = images.save(form.image.data)
        session['image'] = filename
        session['model_directory'] = form.model_radio.data

        return redirect(url_for('cryo01_prediction'))

    return render_template("projects/cryo01/cryo01_selection.html",form=form)

@app.route('/cryo01/prediction')
def cryo01_prediction():

    selected_model_dir = session.get('model_directory',None)
    image_file_name = session.get('image',None)
    image = LoadImage.load_img('static/images/'+image_file_name,target_size=(512,512))
    image = np.array(image)
    image = image[np.newaxis,:]
    image = np.divide(np.sum(image,axis=3),255*3)
    image = image.reshape(1,512,512,1)

    model_path = os.listdir('static\project_data\CRYO01\model\'+str(selected_model_dir))
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

    return render_template("projects/cryo01/cryo01_prediction.html",full_prediction=full_prediction,prev=prev)


if __name__ == '__main__':
    app.run(debug=True)
