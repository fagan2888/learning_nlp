import requests

url = ('https://newsapi.org/v2/everything?'
       'q=Apple&'
       'from=2018-08-04&'
       'sortBy=popularity&'
       'apiKey=d6305c0eb2ed4248b01e84e3043f7b3a')

response = requests.get(url).json()
x = response['articles']
y = [(post['description'],post['publishedAt']) for post in x]
print(y)

