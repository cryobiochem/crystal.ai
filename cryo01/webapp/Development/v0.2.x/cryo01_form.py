from flask_wtf import FlaskForm
import os
from wtforms import StringField, SubmitField, FileField, RadioField
from wtforms.validators import DataRequired
from flask_wtf.file import FileAllowed

class InfoForm(FlaskForm):


    model_dir = 'static\project_data\CRYO01\model'
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
