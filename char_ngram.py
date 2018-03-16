# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 16:19:14 2018

@author: mgungor
"""

import pandas as pd 
from collections import Counter
train_df = pd.read_csv("train.csv",encoding = "ISO-8859-1")

def Char_ngram(string, n=3):
    return([string[i:i+n] for i in range(len(string)-n+1)])
    
author_list = [1, 2, 3, 4, 6]
#For n=3
for author_i in author_list:
    print(author_i)
    at = train_df[train_df.author == author_i].text.values
    tmp_dict = {}
    for i in range(len(at)):
        for element in at[i].split():
            tmp = Char_ngram(element, n=3)
            if (tmp):
                for tmp1 in tmp:
                    if tmp1 in tmp_dict:
                        tmp_dict[tmp1] += 1
                    else:
                        tmp_dict[tmp1] = 1
    d = Counter(tmp_dict)
    for k, v in d.most_common(5):
        print('%s: %i' % (k, v))
    print("\n")

#for n=4 grams
for author_i in author_list:
    print(author_i)
    at = train_df[train_df.author == author_i].text.values
    tmp_dict = {}
    for i in range(len(at)):
        for element in at[i].split():
            tmp = Char_ngram(element, n=4)
            if (tmp):
                for tmp1 in tmp:
                    if tmp1 in tmp_dict:
                        tmp_dict[tmp1] += 1
                    else:
                        tmp_dict[tmp1] = 1
    d = Counter(tmp_dict)
    for k, v in d.most_common(5):
        print('%s: %i' % (k, v))
    print("\n")