#importing libraries
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
import pandas as pd

#creating instance of the class
app=Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
    return flask.render_template('index.html')


cols = ['Type', 'Sex', 'Size', 'Intake Condition', 'First_Color',
       'Second_Color', 'First Breed', 'Second Breed', 'Age']

#prediction function
def ValuePredictor(to_predict_list):
    to_predict_dict = {k: v for k, v in zip(cols, to_predict_list)}
    to_predict = pd.DataFrame(data=to_predict_dict, columns=cols, index=[0])
    loaded_model = pickle.load(open("clf_model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    prob = loaded_model.predict_proba(to_predict)[:,0]
    return(result[0], prob[0])


@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        outcome = ValuePredictor(to_predict_list)
        
        if outcome[0]== "ADOPTION":
            prediction = 'ADOPTION'
            probability = '{:.2%}'.format(np.round(outcome[1], decimals=2))
            
            
        elif outcome[0] == "EUTHANIZE":
            prediction='EUTHANIZE'
            probability = '{:.2%}'.format(np.round(outcome[1], decimals=2))
            
        return render_template("result.html", prediction = prediction, probability = probability)