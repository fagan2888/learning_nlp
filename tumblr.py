#!/usr/local/bin/env python3
from textblob import TextBlob

import pytumblr

# Authenticate via OAuth
client = pytumblr.TumblrRestClient(
'NZGjtPxebWwy8Ny5tfzlOHic5yu3cOb3mSyIdfOkarBpQIzAZu',
'9ECCX9z3AFf6Wv9dfxn78AvsShoYOJDJPG8RFeagezMVF6OEOp',
'419Gt8xArVBe3E70Z9fU4twuBx2bcGaZHo6Y8iPuPPRL2oHOy3',
'9o8tTukN8VoQulcMkEepioPyXBZh3fUEkl8nRYw10q3vTo1834'
)
import scipy


request = (client.posts('kennypolcari.tumblr.com',type='text',limit='30',filter='text'))['posts']
posts = [post['body'] for post in request]
def sys():
    lst = []
    for post in posts:
        pos_count = 0
        for line in post.split(' '):
            analysis = TextBlob(line)
            if analysis.sentiment.polarity >= 0.55:
                pos_count +=1

        neg_count = 0
        for line in post.split(' '):
            analysis = TextBlob(line)
            if analysis.sentiment.polarity <= -0.4:
                neg_count +=1
        lst.append([pos_count,neg_count])
        print(pos_count,neg_count)
    return lst

lst = sys()

from matplotlib import pyplot as plt
import pandas as pd 

days = pd.bdate_range('2018-06-13','2018-07-24')

x = [x for x,y in reversed(lst)]
y = [y for y in days]
plt.plot(y,x)
plt.show()
