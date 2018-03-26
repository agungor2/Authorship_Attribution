# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 11:03:55 2018
Doc2vec Example
@author: mgungor
dir = r'C:\Program Files\mingw-w64\x86_64-7.2.0-posix-seh-rt_v5-rev0\mingw64\bin'
import os 
os.environ['PATH'].count(dir)
os.environ['PATH'].find(dir)
os.environ['PATH'] = dir + ';' + os.environ['PATH']

If you can't do CPU computing do this setting
1.  pip uninstall gensim
2.  pip uninstall scipy 

3. pip install --no-cache-dir scipy==0.15.1
4. pip install --no-cache-dir gensim==0.12.1

"""


# Import libraries

from gensim.models import doc2vec
from collections import namedtuple
import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import scipy.io as sc
from sklearn.linear_model import LogisticRegression
import multiprocessing
import gensim
cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION
# Load data
train = pd.read_csv('train1.csv')
test = pd.read_csv('test1.csv')
ytrain = train.author.values
test_author = sc.loadmat("test_author.mat")["test_author"]

# Transform data (you can add more data preprocessing steps) 

train_docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in tqdm(enumerate(train.text.values)):
    words = text.lower().split()
    tags = [i]
    train_docs.append(analyzedDocument(words, tags))

# Train model (set min_count = 1, if you want the model to work with the provided example data set)

model_train = doc2vec.Doc2Vec(train_docs, size = 300, window = 300, min_count = 1, workers = cores)

test_docs = []
analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
for i, text in tqdm(enumerate(test.text.values)):
    words = text.lower().split()
    tags = [i]
    test_docs.append(analyzedDocument(words, tags))

# Train model (set min_count = 1, if you want the model to work with the provided example data set)

model_test = doc2vec.Doc2Vec(test_docs, size = 300, window = 300, min_count = 1, workers = cores)

xtrain = np.zeros((len(train.text.values), 300))
for i in range(len(train.text.values)):
    xtrain[i,:] = model_train.docvecs[i]
# Get the vectors
xtest = np.zeros((len(test.text.values), 300))
for i in range(len(test.text.values)):
    xtest[i,:] = model_test.docvecs[i]

# Fitting a simple xgboost 
clf = xgb.XGBClassifier(nthread=10, silent=False)
clf.fit(xtrain, ytrain)
predictions = clf.predict(xtest)
print("f1 Score")
print(f1_score(test_author, predictions, average='micro'))
print("Accuracy")
print(accuracy_score(test_author, predictions))

# Fitting a simple xgboost on glove features
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1, silent=False)
clf.fit(xtrain, ytrain)
predictions = clf.predict(xtest)
print("f1 Score")
print(f1_score(test_author, predictions, average='micro'))
print("Accuracy")
print(accuracy_score(test_author, predictions))
# Fitting a simple Logistic Regression on word2vec
clf = LogisticRegression(C=1.0)
clf.fit(xtrain, ytrain)
predictions = clf.predict(xtest)
print("Logistic Regression")
print("f1 Score")
print(f1_score(test_author, predictions, average='micro'))
print("Accuracy")
print(accuracy_score(test_author, predictions))


###############################################################################
#Second method 
"""
Doc2Vec function contains alpha and min_alpha parameters, but that means that 
the learning rate decays during one epoch from alpha to min_alpha. 
Train several epochs, set the learning rate manually
"""
import random

alpha_val = 0.025        # Initial learning rate
min_alpha_val = 1e-4     # Minimum for linear learning rate decay
passes = 15              # Number of passes of one document during training

alpha_delta = (alpha_val - min_alpha_val) / (passes - 1)

model_train = doc2vec.Doc2Vec( size = 300 # Model initialization
    , window = 300
    , min_count = 1
    , workers = cores)

model_train.build_vocab(train_docs) # Building vocabulary

model_test = doc2vec.Doc2Vec( size = 300 # Model initialization
    , window = 300
    , min_count = 1
    , workers = cores)

model_test.build_vocab(test_docs) # Building vocabulary

for epoch in tqdm(range(passes)):

    # Shuffling gets better results

    random.shuffle(train_docs)
    random.shuffle(test_docs)
    # Train

    model_train.alpha, model_train.min_alpha = alpha_val, alpha_val

    model_train.train(train_docs,total_examples=model_train.corpus_count,epochs=model_train.iter)
    
    #Test
    model_test.alpha, model_test.min_alpha = alpha_val, alpha_val

    model_test.train(test_docs,total_examples=model_test.corpus_count,epochs=model_test.iter)

    # Logs

    print('Completed pass %i at alpha %f' % (epoch + 1, alpha_val))

    # Next run alpha

    alpha_val -= alpha_delta
    
    
# Get the vectors

xtrain = np.zeros((len(train.text.values), 300))
for i in range(len(train.text.values)):
    xtrain[i,:] = model_train.docvecs[i]
# Get the vectors
xtest = np.zeros((len(test.text.values), 300))
for i in range(len(test.text.values)):
    xtest[i,:] = model_test.docvecs[i]

# Fitting a simple xgboost on glove features
clf = xgb.XGBClassifier(nthread=10, silent=False)
clf.fit(xtrain, ytrain)
predictions = clf.predict(xtest)
print("f1 Score")
print(f1_score(test_author, predictions, average='micro'))
print("Accuracy")
print(accuracy_score(test_author, predictions))

# Fitting a simple xgboost 
clf = xgb.XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1, silent=False)
clf.fit(xtrain, ytrain)
predictions = clf.predict(xtest)
print("f1 Score")
print(f1_score(test_author, predictions, average='micro'))
print("Accuracy")
print(accuracy_score(test_author, predictions))
# Fitting a simple Logistic Regression on word2vec
clf = LogisticRegression(C=1.0)
clf.fit(xtrain, ytrain)
predictions = clf.predict(xtest)
print("Logistic Regression")
print("f1 Score")
print(f1_score(test_author, predictions, average='micro'))
print("Accuracy")
print(accuracy_score(test_author, predictions))