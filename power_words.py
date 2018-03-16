# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 01:03:34 2018

@author: mgungor
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


train_df = pd.read_csv("train.csv",encoding = "ISO-8859-1")
power_words = pd.read_csv("power_words_in_database.txt", header = None, names = ["words"])
power_words = power_words["words"].tolist()

df = {}

for author_i in train_df.author.unique():
    cv = CountVectorizer(vocabulary=power_words)
    df[author_i] = sum(sum(cv.fit_transform(train_df[train_df.author == author_i]["text"]).toarray())) / len(train_df[train_df.author == author_i])
    print(author_i)
    
plt.bar(range(len(df)), list(df.values()), align='center')

plt.show()

print(max(df, key=df.get))
print(min(df, key=df.get))