# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 15:36:06 2018

@author: Mecit
"""
import scipy.io as sc
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from sklearn.metrics import f1_score
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test_author = pd.read_csv('test_author.txt', header=None)
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train.author.values)
y_test = lbl_enc.fit_transform(test_author.values)
xtrain, xvalid, ytrain, yvalid = train_test_split(train.text.values, y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)

tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(train.text.values))
xtrain_tfv =  tfv.transform(train.text.values) 
xvalid_tfv = tfv.transform(test.text.values)
# Fitting a simple Logistic Regression on TFIDF
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, y)
predictions = clf.predict(xvalid_tfv)
print(f1_score(y_test, predictions, average='weighted'))
#Let's see the difference when considering the tfidf within the test data
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
tfv.fit(list(train.text.values) + list(test.text.values))
xtrain_tfv =  tfv.transform(train.text.values) 
xvalid_tfv = tfv.transform(test.text.values)
# Fitting a simple Logistic Regression on TFIDF
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, y)
predictions = clf.predict(xvalid_tfv)
print(f1_score(y_test, predictions, average='weighted'))

##############################################################################
#Work on the bag of words
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')

# Fitting Count Vectorizer to both training and test sets (semi-supervised learning)
ctv.fit(list(train.text.values))
xtrain_ctv =  ctv.transform(train.text.values) 
xvalid_ctv = ctv.transform(test.text.values)

clf = LogisticRegression(C=1.0)
clf.fit(xtrain_ctv, y)
predictions = clf.predict(xvalid_ctv)
print(f1_score(y_test, predictions, average='weighted'))

#Consider the test dataset
#Work on the bag of words
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')

# Fitting Count Vectorizer to both training and test sets (semi-supervised learning)
ctv.fit(list(train.text.values)+ list(test.text.values))
xtrain_ctv =  ctv.transform(train.text.values) 
xvalid_ctv = ctv.transform(test.text.values)

clf = LogisticRegression(C=1.0)
clf.fit(xtrain_ctv, y)
predictions = clf.predict(xvalid_ctv)
print(f1_score(y_test, predictions, average='micro'))
# Fitting a simple Naive Bayes on TFIDF
clf = MultinomialNB()
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
print(f1_score(y_test, predictions, average='micro'))
# Fitting a simple Naive Bayes on Counts
clf = MultinomialNB()
clf.fit(xtrain_ctv, ytrain)
predictions = clf.predict(xvalid_ctv)
print(f1_score(y_test, predictions, average='micro'))
###############################################################################
#Import the 50 authors dataset and do the same tests
train_df = pd.read_csv("train.csv",encoding = "ISO-8859-1")
test_df = pd.read_csv("test.csv",encoding = "ISO-8859-1")
test_author = sc.loadmat("test_author.mat")["test_author"]
y = train_df.author.values
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')

# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(train_df.text.values))
xtrain_tfv =  tfv.transform(train_df.text.values) 
xvalid_tfv = tfv.transform(test_df.text.values)
# Fitting a simple Logistic Regression on TFIDF
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, y)
predictions = clf.predict(xvalid_tfv)
print(f1_score(test_author, predictions, average='weighted'))
#Let's see the difference when considering the tfidf within the test data
tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
tfv.fit(list(train_df.text.values) + list(test_df.text.values))
xtrain_tfv =  tfv.transform(train_df.text.values) 
xvalid_tfv = tfv.transform(test_df.text.values)
# Fitting a simple Logistic Regression on TFIDF
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, y)
predictions = clf.predict(xvalid_tfv)
print(f1_score(test_author, predictions, average='micro'))
# Fitting a simple Naive Bayes on TFIDF
clf = MultinomialNB()
clf.fit(xtrain_tfv, y)
predictions = clf.predict(xvalid_tfv)
print(f1_score(test_author, predictions, average='micro'))
# Fitting a simple Naive Bayes on Counts
clf = MultinomialNB()
clf.fit(xtrain_ctv, y)

predictions = clf.predict(xvalid_ctv)
print(f1_score(test_author, predictions, average='micro'))
# Apply SVD, I chose 120 components. 120-200 components are good enough for SVM model.
svd = decomposition.TruncatedSVD(n_components=120)
svd.fit(xtrain_tfv)
xtrain_svd = svd.transform(xtrain_tfv)
xvalid_svd = svd.transform(xvalid_tfv)

# Scale the data obtained from SVD. Renaming variable to reuse without scaling.
scl = preprocessing.StandardScaler()
scl.fit(xtrain_svd)
xtrain_svd_scl = scl.transform(xtrain_svd)
xvalid_svd_scl = scl.transform(xvalid_svd)
clf = SVC(C=1.0) 
clf.fit(xtrain_svd_scl, y)
predictions = clf.predict(xvalid_svd_scl)
print(f1_score(test_author, predictions, average='micro'))

# Fitting a simple xgboost on tf-idf
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_tfv.tocsc(), y)
predictions = clf.predict_proba(xvalid_tfv.tocsc())

print (f1_score(test_author, predictions, average='micro'))

# Fitting a simple xgboost on tf-idf
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_ctv.tocsc(), ytrain)
predictions = clf.predict_proba(xvalid_ctv.tocsc())


# Fitting a simple xgboost on tf-idf svd features
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
clf.fit(xtrain_svd, y)
predictions = clf.predict(xvalid_svd)

print (f1_score(test_author, predictions, average='micro'))

# Fitting a simple xgboost on tf-idf svd features
clf = xgb.XGBClassifier(nthread=10)
clf.fit(xtrain_svd, ytrain)
predictions = clf.predict_proba(xvalid_svd)

"""
dir = r'C:\Program Files\mingw-w64\x86_64-7.2.0-posix-seh-rt_v5-rev0\mingw64\bin'
import os 
os.environ['PATH'].count(dir)
os.environ['PATH'].find(dir)
os.environ['PATH'] = dir + ';' + os.environ['PATH']
"""
y_pred = sc.loadmat("conf.mat")["at"]
cm = y_pred.astype('float') / y_pred.sum(axis=1)[:, np.newaxis]
import seaborn as sns
import matplotlib.pyplot as plt

        # Sample figsize in inches
mask = np.zeros_like(cm)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    fig, ax = plt.subplots(figsize=(10,10))
    ax = sns.heatmap(cm, mask=mask, vmax=.3, square=True)

fscore = clf.booster().get_fscore()
import operator
sorted_fscore = sorted(fscore.items(), key=operator.itemgetter(1), reverse= True)
top_20 = []
top_20_names = []
for i in range(20):
    top_20.append(sorted_fscore[i][1])
    top_20_names.append(sorted_fscore[i][0][1:])
    
fig, ax = plt.subplots(figsize=(10,10))

y_pos = np.arange(len(top_20_names))

ax.barh(y_pos, top_20, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(top_20_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('F Score Weight')
ax.set_title('Feature Importance')

plt.show()