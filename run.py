from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle
import pandas as pd
from nltk.stem.snowball import EnglishStemmer
import re

def stem_tokenizer(text):
    stemmer = EnglishStemmer(ignore_stopwords=True)
    words = re.sub(r"[^A-Za-z0-9\-]", " ", text).lower().split()
    words = [stemmer.stem(word) for word in words]
    return words 

with open("lgclassifier.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])

def index():
	Id_to_Indus={1: 'IT Consutlant software service',
           2:'Retail, manufacturing',
           3: 'Finance, insurance',
           4: 'Federal, department and law',
           5: 'Healthcare pharmaceutical',
           6: 'High Tech'}
		   
	pred_indus = ""
	if request.method == "POST":
		JD = request.form["JD"]
		X = pd.Series(JD)
		pred_labelID = model.predict(X)[0]
		pred_indus = Id_to_Indus[pred_labelID]
	return render_template("index.html", pred=pred_indus)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.8', port=5000)
