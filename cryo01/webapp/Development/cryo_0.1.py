#Libraries
import joblib
import os
from tensorflow.keras.models import load_model
from flask import Flask, render_template



app = Flask(__name__)

@app.route('/')

def index():
    return render_template("basic.html")


@app.route('/model_selection')

def model_sel():

    model_dir = 'assets/model/'
    lista = []
    for models in os.listdir(model_dir):
        lista.append(models)

    return render_template("model_selection.html",model_dir=lista)

@app.route('/model_prediction')

def prediction_model():

    return render_template("model_prediction.html")


if __name__ == '__main__':
    app.run(debug=True)
