from webapp import app
import joblib
import os
from flask import Flask, render_template, url_for, request, session,redirect,url_for, flash, session
from webapp.models import user

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



if __name__ == '__main__':
    app.run(debug=True)
