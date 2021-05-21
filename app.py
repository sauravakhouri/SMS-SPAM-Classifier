import joblib
from flask import Flask, request, render_template

from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

app=Flask(__name__)

# Loading the MNB model Count Vectorizer from the disk

vector=joblib.load('SpamClassifier_vector.pkl')
model=joblib.load('SpamClassifier_model.pkl')

def Classification(sms):
    cv=vector.transform([sms])
    my_pred=model.predict(cv)

    if my_pred==1:
        return (['This message is a SPAM'])
    else:
        return (['This message is not a SPAM '])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Classify',methods=['POST'])
def Classify():
    msg=request.form['SMS']
    result=Classification(msg)

    return render_template('index.html',classify_text=result)

if __name__=="__main__":
    app.run(debug=True)

