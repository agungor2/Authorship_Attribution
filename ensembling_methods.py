# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 17:44:23 2018

@author: mgungor
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, accuracy_score
import scipy.io as sc

models = [('MultiNB', MultinomialNB(alpha=0.03)),
          ('Calibrated MultiNB', CalibratedClassifierCV(
              MultinomialNB(alpha=0.03), method='isotonic')),
          ('Calibrated BernoulliNB', CalibratedClassifierCV(
              BernoulliNB(alpha=0.03), method='isotonic')),
          ('Calibrated Huber', CalibratedClassifierCV(
              SGDClassifier(loss='modified_huber', alpha=1e-4,
                            max_iter=10000, tol=1e-4), method='sigmoid')),
          ('Logit', LogisticRegression(C=30))]

train = pd.read_csv('train1.csv')
vectorizer=TfidfVectorizer(token_pattern=r'\w{1,}', sublinear_tf=True, ngram_range=(1,2))
clf = VotingClassifier(models, voting='soft', weights=[3,3,3,1,1])
X_train = vectorizer.fit_transform(train.text.values)
authors = train.author
clf.fit(X_train, authors)

test = pd.read_csv('test1.csv', index_col=0)
X_test = vectorizer.transform(test.text.values)
results = clf.predict_proba(X_test)
test_author = sc.loadmat("test_author.mat")["test_author"]
print("f1 Score")
print(f1_score(test_author, results, average='micro'))
print("Accuracy")
print(accuracy_score(test_author, results))