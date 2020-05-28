from flask import Flask, render_template, flash ,redirect, request, url_for, session , logging
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction import stop_words
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import pickle
import os
import shutil
from shutil import copyfile
from models import Models

app = Flask(__name__)
app.config['UPLOAD_PATH'] = 'files/raw'

m = Models()

#init MYSQL


class ArticleForm(Form):
    title = StringField('Title')#,[validators.Length(min=1,max=200)])
    body = TextAreaField('Body')#,[validators.Length(min=30)])

@app.route('/',methods=["POST","GET"])
def index():
    form = ArticleForm(request.form)
    if request.method == 'POST' and form.validate():
        title = form.title.data
        body = form.body.data
        return redirect(url_for('article', body = body))

    return render_template('add_article.html', form=form)

@app.route('/article_catogerized')
def article():
    a = m.predict(request.args['body'])
    return render_template('article.html',  article = request.args['body'],preditions = a, ensemble = max(set(a), key = a.count))


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/articles',methods=['POST','GET'])
def articles():
    if request.method == 'POST':
        global file_names
        file_names = []
        for f in request.files.getlist('files'):
            f.save(os.path.join(app.config['UPLOAD_PATH'], f.filename))
            file_names.append(f.filename)
        sel_mod  = request.form.get("sel_mod")
        return redirect(url_for('classifing',sel_mod=sel_mod))
    models = ['bag of words knn','bag of words SVM' , 'bag of words NB','n grams SVM' , 'n grams NB ','n grams KNN','CNN']
    return render_template('articles.html',models=models)

@app.route('/classifing')
def classifing():
    sel_mod = request.args['sel_mod']
    for file in file_names:
        # print(file)
        f = open('files/raw/'+file)
        x=m.predict(f.read())
        print(x)
        copyfile('files/raw/'+file,'files/'+str(m.predict(f.read())[3])+"/"+file)
        # os.remove('files/raw/'+file)
        # print(f.read())
    return redirect(url_for('classified'))

@app.route('/classified')
def classified():
    files = [[],[],[],[]]
    files[0] = os.listdir('files/1')
    files[1] = os.listdir('files/2')
    files[2] = os.listdir('files/3')
    files[3] = os.listdir('files/4')
    return render_template('dashboard.html',files=files)

@app.route('/classify/<string:filename>', methods=['GET','POST'])
def classify(filename):
    f = open('files/raw/'+filename)
    body  = f.read()
    a = m.predict(body)
    return render_template('article.html',  article = body,preditions = a, ensemble = max(set(a), key = a.count))

@app.route('/delete',methods=['GET','POST'])
def delete():
    fol = Path('files/raw')
    if fol.exists():
        shutil.rmtree('files/raw')
    fol = Path('files/1')
    if fol.exists():
        shutil.rmtree('files/1')
    fol = Path('files/2')
    if fol.exists():
        shutil.rmtree('files/2')
    fol = Path('files/3')
    if fol.exists():
        shutil.rmtree('files/3')
    fol = Path('files/4')
    if fol.exists():
        shutil.rmtree('files/4')
    os.makedirs('files/raw')
    os.makedirs('files/1')
    os.makedirs('files/2')
    os.makedirs('files/3')
    os.makedirs('files/4')
    return redirect(url_for('articles'))


if __name__ == '__main__':

    app.secret_key='secret123'
    app.run(debug=True)
    app.run(host='0.0.0.0',port=5005)
