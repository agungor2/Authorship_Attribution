# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 12:09:35 2018

@author: mgungor
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

train_txt = pd.read_csv('train.csv')
test_txt = pd.read_csv('test.csv')

vec = CountVectorizer()
train_bow = vec.fit_transform(train_txt["text"]).toarray()
test_bow = vec.fit_transform(test_txt["text"]).toarray()
print(train_bow[0])
print(train_bow.shape)
print(test_bow[0])
print(test_bow.shape)

##Check if the word is in the vocabulary list
vec.vocabulary_.get('frankenstein')
#Out: 6301