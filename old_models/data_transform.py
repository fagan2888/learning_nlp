import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Read Training File into Pandas using a relative path
path = 'data/example.tsv'
data = pd.read_table(path, header=None, names=['text', 'label'])

# Spit dataframe into: Labels - 'X table' | Documents - 'y table'
X = data.text
y = data.label

# Read One_line File into Pandas
path = 'data/one_line.tsv'
data = pd.read_table(path, header=None, names=['text', 'label'])

T = data.text
print(T)

# Convert X dataframe into: Document Term Matrix - 'X_train_dtm'
vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X)
T_test_dtm  = vect.transform(T)

nb = MultinomialNB()
nb.fit(X_train_dtm,y)

y_pred_class = nb.predict(T_test_dtm)

print(y_pred_class)
