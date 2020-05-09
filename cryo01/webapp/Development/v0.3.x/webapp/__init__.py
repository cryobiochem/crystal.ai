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

basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
app.config['SECRET_KEY'] = 'MysecretKey'
app.config['UPLOADED_IMAGES_DEST'] = os.path.join(basedir, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024 # 4 MB max file
image_formats  = ['PNG','JPG','jpg','png','JPEG','jpeg']


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///'+os.path.join(basedir,'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
Migrate(app,db)

from webapp.cryo01.views import cryo01_blueprints, images
from webapp.user.views import user_blueprints
app.register_blueprint(cryo01_blueprints,url_pref='/cryo01')
app.register_blueprint(user_blueprints,url_pref='/user')
configure_uploads(app, images)
