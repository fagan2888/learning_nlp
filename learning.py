from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import datasets
import pytumblr
import os
import pandas as pd
import numpy as np
import nltk

# Authenticate via OAuth
client = pytumblr.TumblrRestClient(
'NZGjtPxebWwy8Ny5tfzlOHic5yu3cOb3mSyIdfOkarBpQIzAZu',
'9ECCX9z3AFf6Wv9dfxn78AvsShoYOJDJPG8RFeagezMVF6OEOp',
'419Gt8xArVBe3E70Z9fU4twuBx2bcGaZHo6Y8iPuPPRL2oHOy3',
'9o8tTukN8VoQulcMkEepioPyXBZh3fUEkl8nRYw10q3vTo1834'
)

request = (client.posts('kennypolcari.tumblr.com',type='text',limit='1',filter='text'))['posts']

vect = CountVectorizer()
posts = [post['body'].replace('\n', ' ') for post in request] # Format that We Want = ['text','more text']!!!
posts_lst = [post for post in posts]

posts_dtm = vect.fit_transform(posts)
x = pd.DataFrame(posts_dtm.toarray(), columns=vect.get_feature_names())
print(posts_dtm)

# iris = datasets.load_iris()
# print(iris)

# split X and y into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
