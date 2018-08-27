#!/usr/bin/env python3
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
    
from spacy.tokens import Doc

from pathlib import Path
from core.wrapper.scraper import get_ticker_news

import spacy
import pandas as pd
import string
import re

def processText(text):
    # Clean up text with stopword, punctuation, and ticker removal
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
                         if word not in set(avoid_words)])

def sent_polarity(text,score):
    # Lexicon based sentiment polarity, used for confirmation purposes
    def pol_scores(doc):
        return SentimentIntensityAnalyzer().polarity_scores(doc.text)
    Doc.set_extension('polarity_scores',getter=pol_scores,force=True)
    nlp = spacy.load('en')
    lst = list(nlp(text)._.polarity_scores.values())
    if score == 0.0: # Negative [0]
        if lst[0] == 0.0 and lst[3] >= 0.50: # Negative & Compound
            # Negative is in fact a Positive -> Transformation...
            return pd.Series({'new_sent':  1.0,'polarity': lst[2]}) 
        if lst[0] == 0.0 and lst[3] <  0.50: # Negative & Compound
            return pd.Series({'new_sent':score,'polarity': lst[3]}) 
        else: 
            return pd.Series({'new_sent':score,'polarity': lst[0]}) 
    if score == 1.0: # Positive [2]
        if lst[2] == 0.0 and lst[3] <  0.0: # Positive & Compound
            # Positive is in fact a Negative -> Transformation...
            return pd.Series({'new_sent':  0.0,'polarity': lst[0]}) 
        else: 
            return pd.Series({'new_sent':score,'polarity': lst[2]}) 

def trainingData():
    # Acquire machine learning training data for model
    path = Path('./core/training/train.csv')
    df = pd.read_csv(path,header=None,names=['text', 'label'])
    return df['text'], df['label']

def vectorization(X,T):
    # Model learns/transforms the dataset into document term matrices
    vect  = CountVectorizer()
    X_dtm = vect.fit_transform(X)   
    X_tst = vect.transform(T)
    return X_dtm, X_tst

def news_data(ticker):
    # Acquire news_data scraping results for desired ticker symbol
    table = get_ticker_news(ticker)
    df = pd.DataFrame(table, columns=['date', 'text'])
    df['text'] = df['text'].apply(processText)
    df = df.drop_duplicates('text')
    df = df[df['text'].str.split().str.len() > 3]
    return df

def predict_model(ticker):
    # Machine learning based prediction using Multinomial NB model
    X , y = trainingData()
    df = news_data(ticker)
    X_dtm, X_tst = vectorization(X,df.text)
    nb = MultinomialNB()
    nb.fit(X_dtm,y)
    y_pred = nb.predict(X_tst)
    df['sentiment'] = pd.Series(y_pred.tolist(), index=df.index)
    df = df.merge(df.apply(lambda row:  sent_polarity(row['text'],
                                        row['sentiment']),axis=1),
                                        left_index=True, right_index=True)
    path = Path('./core/training/results.csv')
    with open(path,'w',newline='') as f:
        df.to_csv(f, header=False, index=False)
    df['date'] = pd.to_datetime(df['date'])
    df_pos = df.loc[df['new_sent'] == 1]
    df_neg = df.loc[df['new_sent'] == 0]
    df_pos = df_pos.resample('1d',on='date'
                            ).sum().fillna(0).rolling(window=1).mean()
    df_neg = df_neg.resample('1d',on='date'
                            ).sum().fillna(0).rolling(window=1).mean()
    from matplotlib import pyplot as plt
    plt.plot(df_pos.index,df_pos['polarity'],'g',
            df_neg.index,df_neg['polarity'],'r')
    plt.show()


predict_model('ge')

