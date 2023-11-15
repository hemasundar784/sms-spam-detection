from flask import Flask,render_template,url_for,request
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score,classification_report
import pickle

filename='spam_model.pkl'
mul=pickle.load(open(filename,'rb'))
trans=pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if(request.method=='POST'):
	    message=request.form["message"]
	    data=[message]
	    vect=trans.transform(data).toarray()
	    my_prediction=mul.predict(vect)
    return render_template('result.html',prediction = my_prediction[0])


if __name__ == '__main__':
    app.run(debug=True)