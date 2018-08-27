import pytumblr

# Authenticate via OAuth
client = pytumblr.TumblrRestClient(
'NZGjtPxebWwy8Ny5tfzlOHic5yu3cOb3mSyIdfOkarBpQIzAZu',
'9ECCX9z3AFf6Wv9dfxn78AvsShoYOJDJPG8RFeagezMVF6OEOp',
'419Gt8xArVBe3E70Z9fU4twuBx2bcGaZHo6Y8iPuPPRL2oHOy3',
'9o8tTukN8VoQulcMkEepioPyXBZh3fUEkl8nRYw10q3vTo1834'
)

request = (client.posts('kennypolcari.tumblr.com',type='text',limit='2',filter='text'))['posts']
posts = [post['body'] for post in request]
print(posts[1])



    