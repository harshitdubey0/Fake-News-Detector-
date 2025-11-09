# @Harshit
# github - harshitdubey0

from flask import Flask, render_template, request
import pandas as pd
import sklearn
import itertools
import numpy as np
import seaborn as sb
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from matplotlib import pyplot as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os

# Ensure stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__, template_folder='./templates', static_folder='./static')

# Load model and vectorizer
loaded_model = pickle.load(open("model.pkl", 'rb'))
vector = pickle.load(open("vector.pkl", 'rb'))

lemmatizer = WordNetLemmatizer()
stpwrds = set(stopwords.words('english'))

def fake_news_det(news):
    review = re.sub(r'[^a-zA-Z\s]', '', news)
    review = review.lower()
    review = nltk.word_tokenize(review)
    corpus = [lemmatizer.lemmatize(word) for word in review if word not in stpwrds]
    input_data = [' '.join(corpus)]
    vectorized_input_data = vector.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        message = request.form['news']
        pred = fake_news_det(message)

        if pred[0] == 1:
            result = "Prediction of the News : Looking Fake News 📰"
        else:
            result = "Prediction of the News : Looking Real News 📰"

        return render_template("prediction.html", prediction_text=result)
    else:
        return render_template('prediction.html', prediction_text="Something went wrong")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
