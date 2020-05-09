from flask import Blueprint, render_template,redirect,url_for
from webapp import db
from flask import Flask, render_template, url_for, request, session,redirect,url_for, flash, session
from webapp.user.forms import SignupForm, LoginForm
from webapp.models import user
import datetime

user_blueprints = Blueprint('user',__name__,template_folder='templates/user')


@user_blueprints.route('/signup',methods=['GET','POST'])
def signup():

    form = SignupForm()

    if form.validate_on_submit():
        session.pop('signup_status',None)
        username = form.username.data
        email = form.email.data
        password = form.password.data
        created_on = datetime.datetime.now()
        last_change = created_on
        print('form_submit')

        users = user.query.all()

        if len(users) > 0:
            for u in users:
                if  u.username == username:
                    session['signup_status'] = 'Username already in use.'
                    return redirect(url_for('user.signup'))
                    print('not ok')
                else:
                    new_user = user(username,email,password,created_on,last_change)
                    db.session.add(new_user)
                    db.session.commit()
                    print('ok')

                    return redirect(url_for('user.login'))

        else:
            new_user = user(username,email,password,created_on,last_change)
            db.session.add(new_user)
            db.session.commit()
            print('not ok')

            return redirect(url_for('user.login'))

    return render_template('signup.html',form=form)

@user_blueprints.route('/login',methods=['GET','POST'])
def login():

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
                return redirect(url_for('user.login'))
        except:
            session['login_status'] = 'Invalid user / password'

    return render_template('login.html',form=form)

@user_blueprints.route('/logout')
def logout():
    session.pop('username',None)
    return redirect(url_for('index'))
