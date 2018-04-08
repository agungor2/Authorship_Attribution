# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:22:30 2018

@author: Mecit
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
train_df = pd.read_csv("train1.csv")
function_words = pd.read_csv("function_word_list.txt", header = None, names = ["words"])
function_words = function_words["words"].tolist()

df = {}

for author_i in train_df.author.unique():
    cv = CountVectorizer(vocabulary=function_words)
    df[author_i] = sum(sum(cv.fit_transform(train_df[train_df.author == author_i]["text"]).toarray())) / len(train_df[train_df.author == author_i])
    print(author_i)
"""    
plt.bar(range(len(df)), list(df.values()), align='center')

plt.show()
"""
author_list = [key for key in df.keys()]
author_list_values = [df[key] for key in df.keys()]
plt.figure(figsize=(8,4))
sns.barplot(author_list, author_list_values, alpha=0.8)
plt.ylabel('Function Word Occurence', fontsize=12)
plt.xlabel('Author Name', fontsize=12)
plt.show()

print(max(df, key=df.get))
print(min(df, key=df.get))