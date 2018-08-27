#!/usr/bin/env python3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.linear_model.logistic import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
    
from pathlib import Path

from core.wrapper.scraper import get_ticker_news

import pandas as pd
import string
import re

def processText(text):
    # clean up text through processes
    path = Path('./core/training/all_tickers.csv')
    tickers = pd.read_csv(path,header=None)
    nltk_stops = stopwords.words('english')
    avoid_words = set(['URL','user'] + 
                  list(string.punctuation)).union(nltk_stops)
    lemma = WordNetLemmatizer()
    x = re.sub("\d+|[^a-zA-Z0-9]"," ",text)
    return ' '.join([lemma.lemmatize(word.lower()) 
                     for word in x.split() 
                         if word not in set(tickers[0].tolist())
                         if word not in set(avoid_words)
                    ])

def trainingData():
    path = Path('./core/training/twt_sample.csv')
    df = pd.read_csv(path,header=None,names=['created_at','text', 'label'])
    df['label'] = df.label.map({'positive':1,'negative':0})
    df = df.drop(['created_at'],axis=1)
    df['text'] = df['text'].apply(processText)
    df = df.drop_duplicates('text')
    df = df[df['text'].str.split().str.len() > 3]
    # split the X data(text) and y data(label)
    path = Path('./core/training/train.csv')
    with open(path,'a',newline='') as f:
        df.to_csv(f, header=False, index=False)
    return df['text'], df['label']

def vectorization(X,T):
    vect  = CountVectorizer()
    X_dtm = vect.fit_transform(X)   
    X_tst = vect.transform(T)
    return X_dtm, X_tst

def news_data(ticker):
    # Read Training File into Pandas using a relative path
    table = get_ticker_news(ticker)
    df = pd.DataFrame(table, columns=['date', 'text'])
    df['text'] = df['text'].apply(processText)
    df = df.drop_duplicates('text')
    df = df[df['text'].str.split().str.len() > 3]
    return df.text

def predict(ticker):
    X , y = trainingData()
    T = news_data(ticker)
    X_dtm, X_tst = vectorization(X,T)
    nb = MultinomialNB()
    nb.fit(X_dtm,y)
    # machine prediction results for X_test
    y_pred_class = nb.predict(X_tst)
    print(y_pred_class)


predict('msft')

