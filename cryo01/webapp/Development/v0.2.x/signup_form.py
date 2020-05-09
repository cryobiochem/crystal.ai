from flask_wtf import FlaskForm
from wtforms import StringField,IntegerField,SubmitField,PasswordField
from wtforms.validators import DataRequired, Email, EqualTo

class SignupForm(FlaskForm):

    username = StringField('Username: ',
        validators=
            [DataRequired()])
    email = StringField('Email: ',
        validators=
            [DataRequired(),
            Email()]
        )
    password = PasswordField('Password: ',
        validators=
            [DataRequired(),
            EqualTo('password_confirm', message='Passwords do not match.')])
    password_confirm = PasswordField('Confirm Password: ')

    submit = SubmitField('submit')

class LoginForm(FlaskForm):

    username = StringField('Username: ',
        validators=
            [DataRequired()])

    password = PasswordField('Password: ',
        validators=
            [DataRequired()])

    submit = SubmitField('Login')
