# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
dir = r'C:\Program Files\mingw-w64\x86_64-7.2.0-posix-seh-rt_v5-rev0\mingw64\bin'
import os 
os.environ['PATH'].count(dir)
os.environ['PATH'].find(dir)
os.environ['PATH'] = dir + ';' + os.environ['PATH']
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
import scipy.io as sc
from sklearn.linear_model import LogisticRegression
test_author = sc.loadmat("test_author.mat")["test_author"]

from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk import word_tokenize
train = pd.read_csv('train1.csv')
test = pd.read_csv('test1.csv')
ytrain = train.author.values
embeddings_index = {}
#Download glove embeddings 
#http://www-nlp.stanford.edu/data/glove.840B.300d.zip
f = open('glove.840B.300d.txt')
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# this function creates a normalized vector for the whole sentence
def sent2vec(s):
    words = str(s).lower().decode('utf-8')
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())

# create sentence vectors using the above function for training and validation set
xtrain_glove = [sent2vec(x) for x in tqdm(train.text.values)]
xtest_glove = [sent2vec(x) for x in tqdm(test.text.values)]

xtrain_glove = np.array(xtrain_glove)
xtest_glove = np.array(xtest_glove)

# Fitting a simple xgboost on glove features
clf = xgb.XGBClassifier(nthread=10, silent=False)
clf.fit(xtrain_glove, ytrain)
predictions = clf.predict(xtest_glove)
print("f1 Score")
print(f1_score(test_author, predictions, average='micro'))
print("Accuracy")
print(accuracy_score(test_author, predictions))

# Fitting a simple xgboost on glove features
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1, silent=False)
clf.fit(xtrain_glove, ytrain)
predictions = clf.predict(xtest_glove)
print("f1 Score")
print(f1_score(test_author, predictions, average='micro'))
print("Accuracy")
print(accuracy_score(test_author, predictions))
# Fitting a simple Logistic Regression on word2vec
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_glove, ytrain)
predictions = clf.predict(xtest_glove)
print("Logistic Regression")
print("f1 Score")
print(f1_score(test_author, predictions, average='micro'))
print("Accuracy")
print(accuracy_score(test_author, predictions))
#SVM
# Fitting a simple SVM
clf = SVC(C=1.0, probability=False) 
clf.fit(xtrain_glove, ytrain)
predictions = clf.predict(xtest_glove)
print("SVM")
print("f1 Score")
print(f1_score(test_author, predictions, average='micro'))
print("Accuracy")
print(accuracy_score(test_author, predictions))

raw_data = ''
for element in tqdm(train.text.values):
    raw_data = raw_data + element + ' '
    
for element in tqdm(test.text.values):
    raw_data = raw_data + element + ' '

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer()
transformed = tfidf_vec.fit_transform(raw_documents=[raw_data])
index_value={i[1]:i[0] for i in tfidf_vec.vocabulary_.items()}

fully_indexed = []
for row in transformed:
    fully_indexed.append({index_value[column]:value for (column,value) in zip(row.indices,row.data)})